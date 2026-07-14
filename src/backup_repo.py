"""Create portable 7z backups of this repository.

The backup is split into two archives:

- ``<name>-repo.7z`` contains the rest of the tree.
- ``<name>-data.7z`` contains only ``data/`` and uses store mode.

The repo archive keeps ``.git/`` but skips ``.venv/`` and the large transient
artifacts in ``work_dir_*`` that are not useful for laptop analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

EXCLUDED_CACHE_DIRS = {
    "__pycache__",
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
}
EXCLUDED_WORK_DIR_SUFFIXES = (".json", ".nii.gz", ".b2nd", ".pkl", ".pth", ".pt", ".npz")
BACKUP_SUBDIR = "backups"
DATA_DIR_NAME = "data"


@dataclass(frozen=True)
class BackupArtifacts:
    data_archive: Path
    repo_archive: Path
    data_sha256: Path
    repo_sha256: Path


def _has_excluded_work_dir_suffix(rel_path: Path) -> bool:
    name = rel_path.name
    return any(name.endswith(suffix) for suffix in EXCLUDED_WORK_DIR_SUFFIXES)


def _is_under_work_dir(rel_path: Path) -> bool:
    return any(part.startswith("work_dir") for part in rel_path.parts)


def _is_within(rel_path: Path, root: Path) -> bool:
    try:
        rel_path.relative_to(root)
    except ValueError:
        return False
    return True


def _archive_relpath(root_name: str, rel_path: Path) -> str:
    return (Path(root_name) / rel_path).as_posix()


def _should_prune_dir(rel_dir: Path, child_name: str, archive_kind: str, output_rel: Path | None) -> bool:
    candidate = rel_dir / child_name if rel_dir != Path(".") else Path(child_name)

    if output_rel is not None and _is_within(candidate, output_rel):
        return True

    if child_name in EXCLUDED_CACHE_DIRS or child_name == ".venv":
        return True

    if archive_kind == "data":
        return candidate.parts[:1] != (DATA_DIR_NAME,)

    if archive_kind == "repo":
        return candidate.parts[:1] == (DATA_DIR_NAME,)

    raise ValueError(f"Unknown archive kind: {archive_kind}")


def _should_include_file(rel_file: Path, archive_kind: str, output_rel: Path | None) -> bool:
    if output_rel is not None and _is_within(rel_file, output_rel):
        return False

    if archive_kind == "data":
        return rel_file.parts[:1] == (DATA_DIR_NAME,)

    if archive_kind == "repo":
        if rel_file.parts[:1] == (DATA_DIR_NAME,):
            return False
        if _is_under_work_dir(rel_file) and _has_excluded_work_dir_suffix(rel_file):
            return False
        return True

    raise ValueError(f"Unknown archive kind: {archive_kind}")


def _collect_files(root: Path, archive_kind: str, output_dir: Path | None = None) -> list[Path]:
    root = root.resolve()
    output_rel = None
    if output_dir is not None:
        try:
            output_rel = output_dir.resolve().relative_to(root)
        except ValueError:
            output_rel = None

    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        rel_dir = Path(dirpath).relative_to(root)
        dirnames[:] = [d for d in sorted(dirnames) if not _should_prune_dir(rel_dir, d, archive_kind, output_rel)]
        for filename in sorted(filenames):
            rel_file = rel_dir / filename if rel_dir != Path(".") else Path(filename)
            if _should_include_file(rel_file, archive_kind, output_rel):
                files.append(rel_file)

    return sorted(files)


def _write_listfile(selected_files: list[Path], *, root_name: str) -> Path:
    listfile = tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", delete=False)
    try:
        for rel_file in selected_files:
            listfile.write(_archive_relpath(root_name, rel_file))
            listfile.write("\n")
    finally:
        listfile.close()
    return Path(listfile.name)


def _write_7z_archive(
    root: Path,
    archive_path: Path,
    archive_kind: str,
    output_dir: Path | None = None,
    *,
    store: bool = False,
    progress: bool = False,
) -> list[Path]:
    selected_files = _collect_files(root, archive_kind=archive_kind, output_dir=output_dir)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.unlink(missing_ok=True)
    listfile_path = _write_listfile(selected_files, root_name=root.name)
    try:
        cmd = ["7zz", "a", "-t7z", "-y"]
        if store:
            cmd.append("-mx=0")
        cmd.extend([str(archive_path), f"@{listfile_path}"])

        run_kwargs = {"cwd": root.parent, "text": True}
        if not progress:
            run_kwargs["capture_output"] = True

        try:
            subprocess.run(cmd, check=True, **run_kwargs)
        except subprocess.CalledProcessError as exc:
            output = exc.stderr or exc.stdout or ""
            raise RuntimeError(f"7zz failed while writing {archive_path.name}:\n{output}") from exc
    finally:
        listfile_path.unlink(missing_ok=True)

    return selected_files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_checksum_file(archive_path: Path) -> Path:
    checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    checksum = _sha256(archive_path)
    checksum_path.write_text(f"{checksum}  {archive_path.name}\n", encoding="utf-8")
    return checksum_path


def create_backup_archives(
    root: Path,
    output_dir: Path,
    base_name: str | None = None,
    *,
    progress: bool | None = None,
) -> BackupArtifacts:
    root = root.resolve()
    output_dir = output_dir.resolve()
    base_name = base_name or root.name
    if progress is None:
        progress = sys.stderr.isatty()

    repo_archive = output_dir / f"{base_name}-repo.7z"
    data_archive = output_dir / f"{base_name}-data.7z"

    _write_7z_archive(
        root,
        repo_archive,
        archive_kind="repo",
        output_dir=output_dir,
        progress=progress,
    )
    _write_7z_archive(
        root,
        data_archive,
        archive_kind="data",
        output_dir=output_dir,
        store=True,
        progress=progress,
    )

    data_sha256 = _write_checksum_file(data_archive)
    repo_sha256 = _write_checksum_file(repo_archive)

    return BackupArtifacts(
        data_archive=data_archive,
        repo_archive=repo_archive,
        data_sha256=data_sha256,
        repo_sha256=repo_sha256,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root to archive.")
    parser.add_argument("--output-dir", type=Path, default=Path(BACKUP_SUBDIR), help="Directory to write the archives into.")
    parser.add_argument("--base-name", type=str, default=None, help="Base filename for the archives. Defaults to the repo folder name.")
    parser.add_argument("--progress", dest="progress", action="store_true", help="Force interactive progress bars.")
    parser.add_argument("--no-progress", dest="progress", action="store_false", help="Disable progress bars.")
    parser.set_defaults(progress=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = create_backup_archives(args.root, args.output_dir, base_name=args.base_name, progress=args.progress)

    manifest = {
        "data_archive": str(artifacts.data_archive),
        "repo_archive": str(artifacts.repo_archive),
        "data_sha256": str(artifacts.data_sha256),
        "repo_sha256": str(artifacts.repo_sha256),
    }
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
