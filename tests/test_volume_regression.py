from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import blosc2
import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "volume_regression"))

from common import (
    DEFAULT_SPATIAL_SIZE,
    center_crop_or_pad,
    compute_mask_volume_ml,
    load_nnunet_patch_size,
    load_preprocessed_image,
    regression_metrics,
)
from evaluate import evaluate_predictions
from generate_manifest import build_manifest_rows, copy_manifest_metadata
from train import build_split_frames, resolve_spatial_size


class VolumeRegressionTests(unittest.TestCase):
    def test_compute_mask_volume_ml_uses_original_spacing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            label_path = Path(tmp) / "label.nii.gz"
            arr = np.zeros((4, 5, 6), dtype=np.uint8)
            arr[0:2, 0:3, 0:4] = 1
            affine = np.diag([2.0, 3.0, 4.0, 1.0])
            nib.save(nib.Nifti1Image(arr, affine), label_path)

            volume_ml = compute_mask_volume_ml(label_path)

            self.assertAlmostEqual(volume_ml, 24 * 2 * 3 * 4 / 1000)

    def test_center_crop_or_pad_returns_requested_shape(self) -> None:
        arr = np.ones((4, 6, 2), dtype=np.float32)
        out = center_crop_or_pad(arr, (6, 4, 4))
        self.assertEqual(out.shape, (6, 4, 4))
        self.assertGreater(out.sum(), 0)

    def test_default_spatial_size_matches_nnunet_dhw_order(self) -> None:
        arr = np.ones((29, 270, 270), dtype=np.float32)

        out = center_crop_or_pad(arr, DEFAULT_SPATIAL_SIZE)

        self.assertEqual(DEFAULT_SPATIAL_SIZE, (32, 160, 160))
        self.assertEqual(out.shape, (32, 160, 160))
        self.assertGreater(out.sum(), 0)


    def test_nnunet_patch_size_loads_from_plans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            preprocessed_dir = Path(tmp) / "preprocessed" / "Dataset274_raw"
            preprocessed_dir.mkdir(parents=True)
            plans_path = preprocessed_dir / "nnUNetPlans.json"
            plans_path.write_text(
                json.dumps({"configurations": {"3d_fullres": {"patch_size": [28, 256, 256]}}}),
                encoding="utf-8",
            )

            self.assertEqual(load_nnunet_patch_size(plans_path), (28, 256, 256))

    def test_train_spatial_size_defaults_to_preprocessed_plans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            preprocessed_dir = Path(tmp) / "preprocessed" / "Dataset274_raw"
            preprocessed_dir.mkdir(parents=True)
            (preprocessed_dir / "nnUNetPlans.json").write_text(
                json.dumps({"configurations": {"3d_fullres": {"patch_size": [30, 180, 180]}}}),
                encoding="utf-8",
            )
            args = type("Args", (), {"spatial_size": None, "plans": None, "nnunet_config": "3d_fullres"})()

            self.assertEqual(resolve_spatial_size(args, preprocessed_dir), (30, 180, 180))

    def test_load_preprocessed_image_supports_b2nd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "case_0.b2nd"
            arr = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
            blosc2.asarray(arr).save(path)

            loaded = load_preprocessed_image(path)

            self.assertEqual(loaded.shape, (2, 3, 4))
            np.testing.assert_array_equal(loaded, arr[0])

    def test_load_preprocessed_image_supports_nifti(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "case_0.nii.gz"
            arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            nib.save(nib.Nifti1Image(arr, np.eye(4)), path)

            loaded = load_preprocessed_image(path)

            self.assertEqual(loaded.shape, (2, 3, 4))
            np.testing.assert_array_equal(loaded, arr)

    def test_manifest_generation_uses_raw_nnunet_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dataset_dir = root / "raw" / "Dataset274_raw"
            for subdir in ["imagesTr", "imagesTs", "labelsTr", "labelsTs"]:
                (raw_dataset_dir / subdir).mkdir(parents=True, exist_ok=True)

            train_image = raw_dataset_dir / "imagesTr" / "case_0.nii.gz"
            train_label = raw_dataset_dir / "labelsTr" / "case_0.nii.gz"
            test_image = raw_dataset_dir / "imagesTs" / "case_1.nii.gz"
            test_label = raw_dataset_dir / "labelsTs" / "case_1.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros((3, 3, 3), dtype=np.float32), np.eye(4)), train_image)
            nib.save(nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.uint8), np.diag([1.0, 2.0, 3.0, 1.0])), train_label)
            nib.save(nib.Nifti1Image(np.zeros((3, 3, 3), dtype=np.float32), np.eye(4)), test_image)
            nib.save(nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.uint8), np.diag([2.0, 2.0, 2.0, 1.0])), test_label)
            (raw_dataset_dir / "datalist.json").write_text(
                json.dumps(
                    {
                        "training": [
                            {"id": "source_0", "new_name": "case_0", "image": str(train_image), "label": str(train_label)},
                        ],
                        "testing": [
                            {"id": "source_1", "new_name": "case_1", "image": str(test_image), "label": str(test_label)},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (raw_dataset_dir / "dataset.json").write_text("{}\n", encoding="utf-8")

            rows = build_manifest_rows(
                raw_root=root / "raw",
                dataset_id=274,
            )

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["case_id"], "case_0")
            self.assertEqual(rows[0]["split"], "training")
            self.assertEqual(rows[0]["image"], str(train_image))
            self.assertEqual(rows[0]["label"], str(train_label))
            self.assertEqual(rows[1]["case_id"], "case_1")
            self.assertEqual(rows[1]["split"], "testing")
            self.assertEqual(rows[1]["image"], str(test_image))
            self.assertEqual(rows[1]["label"], str(test_label))
            self.assertAlmostEqual(rows[0]["volume_ml"], 27 * 1 * 2 * 3 / 1000)
            self.assertAlmostEqual(rows[0]["log_volume_ml"], math.log(rows[0]["volume_ml"]))

            preprocessed_dataset_dir = root / "preprocessed" / "Dataset274_raw"
            preprocessed_dataset_dir.mkdir(parents=True)
            (preprocessed_dataset_dir / "nnUNetPlans.json").write_text(
                json.dumps({"configurations": {"3d_fullres": {"patch_size": [3, 3, 3]}}}),
                encoding="utf-8",
            )
            (preprocessed_dataset_dir / "splits_final.json").write_text(
                json.dumps([{"train": ["case_0"], "val": []}]),
                encoding="utf-8",
            )

            copied = copy_manifest_metadata(raw_dataset_dir, root / "manifest_out", preprocessed_dataset_dir)
            self.assertEqual(
                {p.name for p in copied},
                {"dataset.json", "datalist.json", "nnUNetPlans.json", "splits_final.json"},
            )
            self.assertEqual(load_nnunet_patch_size(root / "manifest_out" / "nnUNetPlans.json"), (3, 3, 3))

    def test_train_build_split_frames_uses_raw_training_fold_and_holdout_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dataset_dir = root / "raw" / "Dataset274_raw"
            for subdir in ["imagesTr", "imagesTs", "labelsTr", "labelsTs"]:
                (raw_dataset_dir / subdir).mkdir(parents=True, exist_ok=True)

            def make_case(case_name: str, image_dir: str, label_dir: str, spacing: tuple[float, float, float]) -> tuple[Path, Path]:
                image_path = raw_dataset_dir / image_dir / f"{case_name}.nii.gz"
                label_path = raw_dataset_dir / label_dir / f"{case_name}.nii.gz"
                nib.save(nib.Nifti1Image(np.zeros((3, 3, 3), dtype=np.float32), np.eye(4)), image_path)
                label = np.zeros((3, 3, 3), dtype=np.uint8)
                label[0, :, :] = 1
                nib.save(nib.Nifti1Image(label, np.diag([*spacing, 1.0])), label_path)
                return image_path, label_path

            train_image_0, train_label_0 = make_case("case_0", "imagesTr", "labelsTr", (1.0, 1.0, 1.0))
            train_image_1, train_label_1 = make_case("case_1", "imagesTr", "labelsTr", (2.0, 1.0, 1.0))
            test_image_2, test_label_2 = make_case("case_2", "imagesTs", "labelsTs", (3.0, 1.0, 1.0))

            (raw_dataset_dir / "datalist.json").write_text(
                json.dumps(
                    {
                        "training": [
                            {"id": "source_0", "new_name": "case_0", "image": str(train_image_0), "label": str(train_label_0)},
                            {"id": "source_1", "new_name": "case_1", "image": str(train_image_1), "label": str(train_label_1)},
                        ],
                        "testing": [
                            {"id": "source_2", "new_name": "case_2", "image": str(test_image_2), "label": str(test_label_2)},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            preprocessed_dataset_dir = root / "preprocessed" / "Dataset274_raw"
            preprocessed_dataset_dir.mkdir(parents=True)
            (preprocessed_dataset_dir / "nnUNetPlans.json").write_text(
                json.dumps({"configurations": {"3d_fullres": {"patch_size": [3, 3, 3]}}}),
                encoding="utf-8",
            )
            (preprocessed_dataset_dir / "splits_final.json").write_text(
                json.dumps([{"train": ["case_0"], "val": ["case_1"]}]),
                encoding="utf-8",
            )

            train_df, val_df, test_df = build_split_frames(
                raw_root=root / "raw",
                preprocessed_root=root / "preprocessed",
                dataset_id=274,
                fold=0,
            )

            self.assertEqual(train_df["case_id"].tolist(), ["case_0"])
            self.assertEqual(val_df["case_id"].tolist(), ["case_1"])
            self.assertEqual(test_df["case_id"].tolist(), ["case_2"])
            self.assertEqual(train_df["split"].unique().tolist(), ["training"])
            self.assertEqual(val_df["split"].unique().tolist(), ["validation"])
            self.assertEqual(test_df["split"].unique().tolist(), ["testing"])
            self.assertAlmostEqual(train_df.iloc[0]["volume_ml"], 9 / 1000)
            self.assertAlmostEqual(val_df.iloc[0]["volume_ml"], 18 / 1000)
            self.assertAlmostEqual(test_df.iloc[0]["volume_ml"], 27 / 1000)

    def test_regression_metrics_and_evaluate_include_median_baseline(self) -> None:
        metrics = regression_metrics([10, 20, 30], [12, 18, 33])
        self.assertAlmostEqual(metrics["mae_ml"], 7 / 3)
        self.assertEqual(metrics["n"], 3)

        with tempfile.TemporaryDirectory() as tmp:
            pred_path = Path(tmp) / "pred.csv"
            pd.DataFrame(
                [
                    {"volume_ml_true": 10.0, "volume_ml_pred": 12.0, "median_volume_ml_pred": 20.0},
                    {"volume_ml_true": 20.0, "volume_ml_pred": 18.0, "median_volume_ml_pred": 20.0},
                ]
            ).to_csv(pred_path, index=False)

            result = evaluate_predictions(pred_path)

            self.assertIn("direct_regression", result)
            self.assertIn("median_volume_baseline", result)
            self.assertAlmostEqual(result["direct_regression"]["mae_ml"], 2.0)
            self.assertAlmostEqual(result["median_volume_baseline"]["mae_ml"], 5.0)


if __name__ == "__main__":
    unittest.main()
