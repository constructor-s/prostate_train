from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import download_picai


def write_fake_picai_fold_zip(zip_path: Path, fold: int) -> None:
    with ZipFile(zip_path, "w") as zip_file:
        zip_file.writestr(f"fold_{fold}/sample_{fold}.txt", f"fold_{fold} data\n")


class DownloadPicaiTests(unittest.TestCase):
    def test_download_writes_reproducible_raw_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"

            def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess[str]:
                if command[0] == sys.executable and "zenodo_get" in command:
                    output_idx = command.index("-o")
                    tmp_dir = Path(command[output_idx + 1])
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    for fold in range(5):
                        write_fake_picai_fold_zip(tmp_dir / f"picai_public_images_fold{fold}.zip", fold)
                return subprocess.CompletedProcess(command, 0)

            with patch("download_picai.subprocess.run", side_effect=fake_run) as mocked_run:
                dataset_dir = download_picai.download_picai(data_root, progress=False)

            expected_dir = data_root / "raw" / "zenodo" / "picai" / "record-6624726"
            self.assertEqual(dataset_dir, expected_dir)
            self.assertTrue((expected_dir / "images").is_dir())
            self.assertTrue((expected_dir / ".complete").is_file())

            manifest = json.loads((expected_dir / "download_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["dataset"], "picai")
            self.assertEqual(manifest["source"], "zenodo")
            self.assertEqual(manifest["record_id"], 6624726)
            self.assertEqual(manifest["num_folds"], 5)
            self.assertEqual(mocked_run.call_count, 2)  # 1 zenodo_get + 1 git clone

    def test_download_reuses_complete_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"
            dataset_dir = data_root / "raw" / "zenodo" / "picai" / "record-6624726"
            (dataset_dir / "images").mkdir(parents=True)
            (dataset_dir / "download_manifest.json").write_text("{}\n", encoding="utf-8")
            (dataset_dir / ".complete").touch()
            labels_dir = data_root / "raw" / "zenodo" / "picai" / "picai_labels"
            (labels_dir / ".git").mkdir(parents=True)

            with patch("download_picai.subprocess.run") as mocked_run:
                result = download_picai.download_picai(data_root, progress=False)

            self.assertEqual(result, dataset_dir)
            mocked_run.assert_not_called()

    def test_download_rejects_partial_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"
            dataset_dir = data_root / "raw" / "zenodo" / "picai" / "record-6624726"
            dataset_dir.mkdir(parents=True)

            with self.assertRaises(RuntimeError):
                download_picai.download_picai(data_root, progress=False)


if __name__ == "__main__":
    unittest.main()
