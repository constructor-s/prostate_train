from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import download_prostate158


def write_fake_prostate158_zip(zip_path: Path) -> None:
    with ZipFile(zip_path, "w") as zip_file:
        zip_file.writestr("prostate158_train/train.csv", "image,label\n1,2\n")
        zip_file.writestr("prostate158_train/valid.csv", "image,label\n3,4\n")
        zip_file.writestr("prostate158_train/train/sample.txt", "ok\n")


class DownloadProstate158Tests(unittest.TestCase):
    def test_download_writes_reproducible_raw_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"

            def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess[str]:
                self.assertTrue(check)
                self.assertEqual(command[0], sys.executable)
                self.assertEqual(command[1:3], ["-m", "zenodo_get"])
                output_dir = Path(command[command.index("-o") + 1])
                write_fake_prostate158_zip(output_dir / "prostate158.zip")
                return subprocess.CompletedProcess(command, 0)

            with patch("download_prostate158.subprocess.run", side_effect=fake_run) as mocked_run:
                dataset_dir = download_prostate158.download_prostate158(data_root, progress=False)

            expected_dir = data_root / "raw" / "zenodo" / "prostate158" / "record-6481141" / "prostate158_train"
            self.assertEqual(dataset_dir, expected_dir)
            self.assertTrue((expected_dir / "train.csv").is_file())
            self.assertTrue((expected_dir / "valid.csv").is_file())
            self.assertTrue((expected_dir / "train" / "sample.txt").is_file())
            self.assertTrue((expected_dir / ".complete").is_file())

            manifest = json.loads((expected_dir / "download_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["dataset"], "prostate158")
            self.assertEqual(manifest["source"], "zenodo")
            self.assertEqual(manifest["record_id"], 6481141)
            mocked_run.assert_called_once()

    def test_download_reuses_complete_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"
            dataset_dir = data_root / "raw" / "zenodo" / "prostate158" / "record-6481141" / "prostate158_train"
            (dataset_dir / "train").mkdir(parents=True)
            (dataset_dir / "train.csv").write_text("image,label\n1,2\n", encoding="utf-8")
            (dataset_dir / "valid.csv").write_text("image,label\n3,4\n", encoding="utf-8")
            (dataset_dir / "train" / "sample.txt").write_text("ok\n", encoding="utf-8")
            (dataset_dir / "download_manifest.json").write_text("{}\n", encoding="utf-8")
            (dataset_dir / ".complete").touch()

            with patch("download_prostate158.subprocess.run") as mocked_run:
                result = download_prostate158.download_prostate158(data_root, progress=False)

            self.assertEqual(result, dataset_dir)
            mocked_run.assert_not_called()

    def test_download_rejects_partial_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"
            dataset_dir = data_root / "raw" / "zenodo" / "prostate158" / "record-6481141" / "prostate158_train"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "train.csv").write_text("image,label\n1,2\n", encoding="utf-8")

            with self.assertRaises(RuntimeError):
                download_prostate158.download_prostate158(data_root, progress=False)


if __name__ == "__main__":
    unittest.main()