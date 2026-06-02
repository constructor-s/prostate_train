from __future__ import annotations

import json
from pathlib import Path
import unittest

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
        self.assertEqual(config["dataset_name_or_id"], "Prostate158")

        datalist_value = config["datalist"]
        self.assertTrue(datalist_value.startswith("./configs/"), "Datalist should be stored under configs/")

        for field in ["nnunet_preprocessed", "nnunet_raw", "nnunet_results"]:
            self.assertIsInstance(config[field], str)
            self.assertTrue(config[field].startswith("./"), f"{field} should be a repo-relative path")

    def test_datalist_json_is_valid(self) -> None:
        self.assertTrue(self.json_path.is_file(), f"Missing datalist file: {self.json_path}")
        data = json.loads(self.json_path.read_text(encoding="utf-8"))
        self.assertIn("training", data)
        self.assertIn("validation", data)
        self.assertIsInstance(data["training"], list)
        self.assertIsInstance(data["validation"], list)

        for split_name in ["training", "validation"]:
            with self.subTest(split=split_name):
                for item in data[split_name]:
                    self.assertIsInstance(item, dict)
                    self.assertIn("image", item)
                    self.assertIn("label", item)
                    self.assertIn("id", item)
                    self.assertIsInstance(item["id"], str)
                    self.assertIsInstance(item["label"], str)
                    self.assertIsInstance(item["image"], list)
                    self.assertEqual(len(item["image"]), 1, "Config should use only the t2 image")
                    self.assertTrue(item["image"][0].endswith("t2.nii.gz"), "Only t2 images should be used")

        dataroot = yaml.safe_load(self.yaml_path.read_text(encoding="utf-8"))["dataroot"]
        data_root_path = (self.repo_root / dataroot).resolve()
        if data_root_path.exists():
            for split_name in ["training", "validation"]:
                with self.subTest(split=split_name, existence_check=True):
                    for item in data[split_name]:
                        image_path = data_root_path / item["image"][0]
                        label_path = data_root_path / item["label"]
                        self.assertTrue(image_path.exists(), f"Missing image file: {image_path}")
                        self.assertTrue(label_path.exists(), f"Missing label file: {label_path}")


if __name__ == "__main__":
    unittest.main()
