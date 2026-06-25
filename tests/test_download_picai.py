from __future__ import annotations

import csv
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


class BuildPicaiInferCsvTests(unittest.TestCase):
    def _make_tree(self, root: Path, patients: list[str], label_dirs: list[str]) -> None:
        """Create a fake images/ tree and label directories under root."""
        for patient_id in patients:
            patient_dir = root / "images" / patient_id
            patient_dir.mkdir(parents=True)
            (patient_dir / f"{patient_id}_1000001_t2w.mha").touch()
            (patient_dir / f"{patient_id}_1000001_adc.mha").touch()

        for rel_label_dir in label_dirs:
            label_dir = root / rel_label_dir
            label_dir.mkdir(parents=True)
            for patient_id in patients:
                (label_dir / f"{patient_id}_1000001.nii.gz").touch()

    def test_basic_csv_without_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_tree(root, ["10001", "10002"], [])
            output_csv = root / "out.csv"

            download_picai.build_picai_infer_csv(root / "images", output_csv)

            with output_csv.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r["ID"] for r in rows], ["10001", "10002"])
            self.assertTrue(rows[0]["t2"].endswith("_t2w.mha"))
            self.assertNotIn("adc", rows[0]["t2"])

    def test_csv_with_label_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            label_dirs = ["labels/AI/Bosma22b", "labels/AI/Guerbet23"]
            self._make_tree(root, ["10001", "10002"], label_dirs)
            output_csv = root / "out.csv"

            download_picai.build_picai_infer_csv(
                root / "images",
                output_csv,
                label_dirs=[root / d for d in label_dirs],
            )

            with output_csv.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(list(rows[0].keys()), ["ID", "t2", "labels_ai_bosma22b", "labels_ai_guerbet23"])
            self.assertTrue(rows[0]["labels_ai_bosma22b"].endswith(".nii.gz"))
            self.assertTrue(rows[1]["labels_ai_guerbet23"].endswith(".nii.gz"))

    def test_missing_label_file_is_empty_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_tree(root, ["10001", "10002"], [])
            label_dir = root / "labels" / "AI" / "Bosma22b"
            label_dir.mkdir(parents=True)
            # Only create a label for 10001, not 10002
            (label_dir / "10001_1000001.nii.gz").touch()
            output_csv = root / "out.csv"

            download_picai.build_picai_infer_csv(
                root / "images",
                output_csv,
                label_dirs=[label_dir],
            )

            with output_csv.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["labels_ai_bosma22b"], str(Path("labels/AI/Bosma22b/10001_1000001.nii.gz")))
            self.assertEqual(rows[1]["labels_ai_bosma22b"], "")

    def test_paths_are_relative_to_csv_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_tree(root, ["10001"], ["labels/AI/Bosma22b"])
            output_csv = root / "out.csv"

            download_picai.build_picai_infer_csv(
                root / "images",
                output_csv,
                label_dirs=[root / "labels/AI/Bosma22b"],
            )

            with output_csv.open() as f:
                rows = list(csv.DictReader(f))
            # All paths must be relative (no leading slash)
            for col in ("t2", "labels_ai_bosma22b"):
                self.assertFalse(rows[0][col].startswith("/"), f"{col} path should be relative")


if __name__ == "__main__":
    unittest.main()
