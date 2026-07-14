from __future__ import annotations

import hashlib
import subprocess
import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import backup_repo


def _write_text(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _extract_7z_archive(archive_path: Path, dest_dir: Path) -> set[str]:
    subprocess.run(
        ["7zz", "x", "-y", f"-o{dest_dir}", str(archive_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    root = dest_dir / archive_path.stem.replace("-repo", "").replace("-data", "")
    return {
        str(path.relative_to(dest_dir).as_posix())
        for path in root.rglob("*")
        if path.is_file()
    }


class BackupRepoTests(unittest.TestCase):
    def test_backup_splits_data_and_repo_and_applies_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "prostate_train"
            root.mkdir()

            _write_text(root / ".git" / "config", "[core]\nrepositoryformatversion = 0\n")
            _write_text(root / ".git" / "HEAD", "ref: refs/heads/main\n")
            _write_text(root / ".venv" / "pyvenv.cfg", "home = /tmp\n")
            _write_text(root / "README.md", "# repo\n")
            _write_text(root / "data" / "raw" / "scan.nii.gz", "nii\n")
            _write_text(root / "data" / "raw" / "notes.txt", "keep\n")
            _write_text(root / "results" / "summary.csv", "a,b\n1,2\n")
            _write_text(root / "work_dir_picai" / "keep.txt", "keep\n")
            _write_text(root / "work_dir_picai" / "metrics.json", "{}\n")
            _write_text(root / "work_dir_picai" / "image.nii.gz", "nii\n")
            _write_text(root / "work_dir_picai" / "checkpoint.pth", "binary\n")
            _write_text(root / "work_dir_picai" / "nested" / "report.csv", "ok\n")
            _write_text(root / "work_dir_picai" / "nested" / "weights.pt", "binary\n")
            _write_text(root / "backups" / "old.txt", "ignore me\n")

            artifacts = backup_repo.create_backup_archives(root, root / "backups")

            self.assertTrue(artifacts.data_archive.is_file())
            self.assertTrue(artifacts.repo_archive.is_file())
            self.assertTrue(artifacts.data_sha256.is_file())
            self.assertTrue(artifacts.repo_sha256.is_file())
            self.assertEqual(artifacts.repo_archive.suffix, ".7z")
            self.assertEqual(artifacts.data_archive.suffix, ".7z")

            data_names = _extract_7z_archive(artifacts.data_archive, Path(tmp) / "data_extract")
            repo_names = _extract_7z_archive(artifacts.repo_archive, Path(tmp) / "repo_extract")

            prefix = root.name + "/"
            self.assertIn(prefix + "data/raw/scan.nii.gz", data_names)
            self.assertIn(prefix + "data/raw/notes.txt", data_names)
            self.assertNotIn(prefix + "README.md", data_names)
            self.assertNotIn(prefix + ".git/config", data_names)

            self.assertIn(prefix + "README.md", repo_names)
            self.assertIn(prefix + ".git/config", repo_names)
            self.assertIn(prefix + ".git/HEAD", repo_names)
            self.assertIn(prefix + "results/summary.csv", repo_names)
            self.assertIn(prefix + "work_dir_picai/keep.txt", repo_names)
            self.assertIn(prefix + "work_dir_picai/nested/report.csv", repo_names)
            self.assertNotIn(prefix + ".venv/pyvenv.cfg", repo_names)
            self.assertNotIn(prefix + "data/raw/notes.txt", repo_names)
            self.assertNotIn(prefix + "data/raw/scan.nii.gz", repo_names)
            self.assertNotIn(prefix + "work_dir_picai/metrics.json", repo_names)
            self.assertNotIn(prefix + "work_dir_picai/image.nii.gz", repo_names)
            self.assertNotIn(prefix + "work_dir_picai/checkpoint.pth", repo_names)
            self.assertNotIn(prefix + "work_dir_picai/nested/weights.pt", repo_names)
            self.assertNotIn(prefix + "backups/old.txt", repo_names)

            expected_data_sha = hashlib.sha256(artifacts.data_archive.read_bytes()).hexdigest()
            expected_repo_sha = hashlib.sha256(artifacts.repo_archive.read_bytes()).hexdigest()
            self.assertEqual(artifacts.data_sha256.read_text(encoding="utf-8"), f"{expected_data_sha}  {artifacts.data_archive.name}\n")
            self.assertEqual(artifacts.repo_sha256.read_text(encoding="utf-8"), f"{expected_repo_sha}  {artifacts.repo_archive.name}\n")

    def test_backup_archives_repo_before_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            _write_text(root / "data" / "a.txt", "1\n")
            _write_text(root / "README.md", "# repo\n")

            order: list[str] = []

            def fake_write(root_arg: Path, archive_path: Path, archive_kind: str, output_dir: Path | None = None, *, store: bool = False, progress: bool = False):
                order.append(f"{archive_kind}:{archive_path.name}")
                return []

            with patch.object(backup_repo, "_write_7z_archive", side_effect=fake_write), patch.object(
                backup_repo, "_write_checksum_file", side_effect=lambda path: path.with_suffix(path.suffix + ".sha256")
            ):
                backup_repo.create_backup_archives(root, root / "backups")

            self.assertEqual(order, ["repo:repo-repo.7z", "data:repo-data.7z"])

    def test_collect_files_applies_work_dir_suffix_rule_only_inside_work_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            _write_text(root / "work_dir_x" / "kept.csv", "ok\n")
            _write_text(root / "work_dir_x" / "drop.json", "{}\n")
            _write_text(root / "outside.json", "{}\n")

            repo_files = backup_repo._collect_files(root, archive_kind="repo")
            rels = {p.as_posix() for p in repo_files}

            self.assertIn("work_dir_x/kept.csv", rels)
            self.assertNotIn("work_dir_x/drop.json", rels)
            self.assertIn("outside.json", rels)

    def test_collect_files_returns_only_data_tree_for_data_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            _write_text(root / "data" / "a.txt", "1\n")
            _write_text(root / "results" / "b.txt", "2\n")
            data_files = backup_repo._collect_files(root, archive_kind="data")

            self.assertEqual([p.as_posix() for p in data_files], ["data/a.txt"])


if __name__ == "__main__":
    unittest.main()
