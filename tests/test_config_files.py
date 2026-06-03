from __future__ import annotations

import json
from pathlib import Path
import unittest

import nibabel as nib
import numpy as np
import yaml


class ConfigFilesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parent.parent
        self.yaml_path = self.repo_root / "configs" / "prostate158_input.yaml"
        self.json_path = self.repo_root / "configs" / "prostate158_folds.json"

    def test_input_yaml_is_valid(self) -> None:
        self.assertTrue(self.yaml_path.is_file(), f"Missing config file: {self.yaml_path}")
        config = yaml.safe_load(self.yaml_path.read_text(encoding="utf-8"))
        self.assertIsInstance(config, dict)

        expected_keys = {
            "modality",
            "datalist",
            "dataroot",
            "dataset_name_or_id",
            "nnunet_preprocessed",
            "nnunet_raw",
            "nnunet_results",
        }
        self.assertTrue(expected_keys.issubset(config.keys()))
        self.assertEqual(config["modality"], "MRI")
        # self.assertEqual(config["dataset_name_or_id"], "Prostate158")

        datalist_value = config["datalist"]
        self.assertTrue(datalist_value.startswith("./configs/"), "Datalist should be stored under configs/")

        for field in ["nnunet_preprocessed", "nnunet_raw", "nnunet_results"]:
            self.assertIsInstance(config[field], str)
            self.assertTrue(config[field].startswith("./"), f"{field} should be a repo-relative path")

    def test_datalist_json_is_valid(self) -> None:
        self.assertTrue(self.json_path.is_file(), f"Missing datalist file: {self.json_path}")
        data = json.loads(self.json_path.read_text(encoding="utf-8"))
        self.assertIn("training", data)
        self.assertIn("testing", data)
        self.assertIsInstance(data["training"], list)
        self.assertIsInstance(data["testing"], list)

        for split_name in ["training", "testing"]:
            with self.subTest(split=split_name):
                for item in data[split_name]:
                    self.assertIsInstance(item, dict)
                    self.assertIn("image", item)
                    self.assertIn("label", item)
                    self.assertIn("id", item)
                    self.assertIsInstance(item["id"], str)
                    self.assertIsInstance(item["image"], str)
                    self.assertIsInstance(item["label"], str)
                    self.assertEqual(Path(item["image"]).name, "t2.nii.gz")
                    self.assertEqual(Path(item["label"]).name, "t2_anatomy_reader1.nii.gz")

        dataroot = yaml.safe_load(self.yaml_path.read_text(encoding="utf-8"))["dataroot"]
        data_root_path = (self.repo_root / dataroot).resolve()
        if data_root_path.exists():
            for split_name in ["training", "testing"]:
                with self.subTest(split=split_name, existence_check=True):
                    for item in data[split_name]:
                        image_path = data_root_path / item["image"]
                        label_path = data_root_path / item["label"]
                        self.assertTrue(image_path.exists(), f"Missing image file: {image_path}")
                        self.assertTrue(label_path.exists(), f"Missing label file: {label_path}")
                        img = nib.load(label_path)
                        arr = np.asarray(img.dataobj)
                        self.assertGreater(arr.nbytes, 0, f"Label file is empty: {label_path}")
                        self.assertTrue(np.any(arr != 0), f"Label mask has no non-zero voxels: {label_path}")


    def _load_config(self) -> dict:
        return yaml.safe_load(self.yaml_path.read_text(encoding="utf-8"))

    def test_nnunet_work_dir_structure(self) -> None:
        work_dir = (self.repo_root / "work_dir").resolve()
        if not work_dir.exists():
            self.skipTest("work_dir not present")
        config = self._load_config()
        for field in ["nnunet_preprocessed", "nnunet_raw", "nnunet_results"]:
            path = (self.repo_root / config[field]).resolve()
            self.assertTrue(path.is_dir(), f"{field} directory missing: {path}")

    def test_nnunet_raw_data_structure(self) -> None:
        config = self._load_config()
        raw_root = (self.repo_root / config["nnunet_raw"]).resolve()
        if not raw_root.exists():
            self.skipTest("nnUNet raw data not present")
        dataset_id = str(config["dataset_name_or_id"])
        dataset_dirs = list(raw_root.glob(f"Dataset{dataset_id}_*"))
        self.assertEqual(
            len(dataset_dirs), 1,
            f"Expected exactly one Dataset{dataset_id}_* dir in {raw_root}, found {len(dataset_dirs)}",
        )
        dataset_dir = dataset_dirs[0]

        dataset_json_path = dataset_dir / "dataset.json"
        self.assertTrue(dataset_json_path.is_file(), f"Missing {dataset_json_path}")
        meta = json.loads(dataset_json_path.read_text(encoding="utf-8"))
        for key in ["channel_names", "labels", "numTraining"]:
            self.assertIn(key, meta, f"dataset.json missing key: {key}")

        num_training = meta["numTraining"]
        for subdir in ["imagesTr", "labelsTr"]:
            d = dataset_dir / subdir
            self.assertTrue(d.is_dir(), f"Missing {subdir}: {d}")
            count = len(list(d.glob("*.nii.gz")))
            self.assertEqual(
                count, num_training,
                f"{subdir} has {count} files, expected {num_training}",
            )

    def test_nnunet_preprocessed_structure(self) -> None:
        config = self._load_config()
        preprocessed_root = (self.repo_root / config["nnunet_preprocessed"]).resolve()
        if not preprocessed_root.exists():
            self.skipTest("nnUNet preprocessed data not present")
        dataset_id = str(config["dataset_name_or_id"])
        dataset_dirs = list(preprocessed_root.glob(f"Dataset{dataset_id}_*"))
        self.assertEqual(
            len(dataset_dirs), 1,
            f"Expected exactly one Dataset{dataset_id}_* dir in {preprocessed_root}, found {len(dataset_dirs)}",
        )
        dataset_dir = dataset_dirs[0]

        plans_path = dataset_dir / "nnUNetPlans.json"
        self.assertTrue(plans_path.is_file(), f"Missing nnUNetPlans.json: {plans_path}")
        plans = json.loads(plans_path.read_text(encoding="utf-8"))
        self.assertIn("configurations", plans, "nnUNetPlans.json missing 'configurations'")
        known_configs = {"2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"}
        self.assertTrue(
            known_configs & set(plans["configurations"].keys()),
            f"nnUNetPlans.json has no recognized configuration; got {list(plans['configurations'].keys())}",
        )

        # splits_final.json is created by the first training run, not preprocessing
        splits_path = dataset_dir / "splits_final.json"
        if splits_path.is_file():
            splits = json.loads(splits_path.read_text(encoding="utf-8"))
            self.assertIsInstance(splits, list)
            self.assertGreater(len(splits), 0, "splits_final.json contains no folds")
            for i, fold in enumerate(splits):
                with self.subTest(fold=i):
                    self.assertIn("train", fold, f"Fold {i} missing 'train'")
                    self.assertIn("val", fold, f"Fold {i} missing 'val'")
                    self.assertGreater(len(fold["train"]), 0, f"Fold {i} has no training cases")
                    self.assertGreater(len(fold["val"]), 0, f"Fold {i} has no validation cases")


    def test_nnunet_label_files_not_empty(self) -> None:
        config = self._load_config()
        dataset_id = str(config["dataset_name_or_id"])

        label_dirs: list[Path] = []

        raw_root = (self.repo_root / config["nnunet_raw"]).resolve()
        if raw_root.exists():
            for match in raw_root.glob(f"Dataset{dataset_id}_*"):
                label_dirs.append(match / "labelsTr")
                label_dirs.append(match / "labelsTs")

        preprocessed_root = (self.repo_root / config["nnunet_preprocessed"]).resolve()
        if preprocessed_root.exists():
            for match in preprocessed_root.glob(f"Dataset{dataset_id}_*"):
                label_dirs.append(match / "gt_segmentations")

        if not label_dirs:
            self.skipTest("No work_dir label directories present")

        for label_dir in label_dirs:
            if not label_dir.is_dir():
                continue
            rel = label_dir.relative_to(self.repo_root)
            for label_path in sorted(label_dir.glob("*.nii.gz")):
                with self.subTest(path=str(label_path.relative_to(label_dir.parent))):
                    arr = np.asarray(nib.load(label_path).dataobj)
                    self.assertTrue(
                        np.any(arr != 0),
                        f"Label mask is all-zero: {rel / label_path.name}",
                    )


if __name__ == "__main__":
    unittest.main()
