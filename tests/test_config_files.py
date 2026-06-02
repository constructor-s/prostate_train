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


if __name__ == "__main__":
    unittest.main()
