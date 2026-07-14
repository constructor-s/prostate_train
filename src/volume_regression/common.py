"""Shared utilities for direct prostate volume regression.

This module supports the plain scripts in ``src/volume_regression``. It is not
intended to be run directly.

Inputs and assumptions:
- Regression inputs are nnUNet v2 image volumes stored as either preprocessed
  arrays or raw NIfTI files.
- Segmentation targets are NIfTI masks, usually
  ``Dataset274_raw/gt_segmentations/case_*.nii.gz``.
- Whole-gland target volume is computed as ``count(label > 0) * voxel_volume``
  from the target label image geometry and converted from mm3 to mL.
- ``case_*_seg.b2nd`` files are segmentation tensors and must not be used as
  regression model inputs.

Key helpers:
- ``load_preprocessed_image`` reads ``.b2nd``, ``.npy``, ``.npz``, and NIfTI
  volumes and returns the first image channel as a 3D float32 array.
- ``load_nnunet_raw_cases`` reads the raw nnUNet ``datalist.json`` and returns
  a dataframe keyed by ``case_id``.
- ``load_splits`` reads nnUNet ``splits_final.json`` and maps cases to the
  script convention: ``training`` for train cases and ``testing`` for val cases.
- ``regression_metrics`` computes volume-error metrics in mL plus correlation
  and Bland-Altman summaries.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
# nnUNet preprocessed 3D tensors are channel-first (C, D, H, W). After
# selecting the image channel, all spatial sizes remain in (D, H, W) order.
DEFAULT_SPATIAL_SIZE = (32, 160, 160)


def resolve_path(path: str | Path, base: str | Path = REPO_ROOT) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(base) / path


def load_folds(path: str | Path) -> dict[str, list[dict]]:
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    for split in ("training", "testing"):
        if split not in data or not isinstance(data[split], list):
            raise ValueError(f"{path} must contain a list field named {split!r}")
    return data


def compute_mask_volume_ml(label_path: str | Path) -> float:
    img = nib.load(str(label_path))
    arr = np.asarray(img.dataobj)
    voxels = int(np.count_nonzero(arr > 0))
    zooms = img.header.get_zooms()[:3]
    if len(zooms) != 3:
        raise ValueError(f"Expected a 3D label image, got zooms={zooms} for {label_path}")
    volume_ml = voxels * float(np.prod(zooms)) / 1000.0
    if not math.isfinite(volume_ml) or volume_ml <= 0:
        raise ValueError(f"Computed non-positive volume {volume_ml} mL for {label_path}")
    return volume_ml


def find_dataset_dir(preprocessed_root: str | Path, dataset_id: str | int | None = None) -> Path:
    root = Path(preprocessed_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Missing nnUNet preprocessed root: {root}")
    pattern = f"Dataset{dataset_id}_*" if dataset_id is not None else "Dataset*"
    matches = sorted(p for p in root.glob(pattern) if p.is_dir())
    if not matches:
        raise FileNotFoundError(f"No {pattern} directory found under {root}")
    if len(matches) > 1:
        raise ValueError(f"Expected one {pattern} directory under {root}, found {matches}")
    return matches[0]


def load_nnunet_datalist(dataset_dir: str | Path) -> dict[str, list[dict]]:
    datalist_path = Path(dataset_dir) / "datalist.json"
    if not datalist_path.is_file():
        raise FileNotFoundError(f"Missing nnUNet datalist file: {datalist_path}")

    datalist = json.loads(datalist_path.read_text(encoding="utf-8"))
    for split in ("training", "testing"):
        if split not in datalist or not isinstance(datalist[split], list):
            raise ValueError(f"{datalist_path} must contain a list field named {split!r}")
    return datalist


def load_nnunet_raw_cases(dataset_dir: str | Path) -> pd.DataFrame:
    datalist = load_nnunet_datalist(dataset_dir)
    rows: list[dict[str, str]] = []
    for split in ("training", "testing"):
        for item in datalist[split]:
            case_id = str(item.get("new_name") or item.get("id"))
            source_id = str(item.get("id", case_id))
            rows.append(
                {
                    "case_id": case_id,
                    "source_id": source_id,
                    "split": split,
                    "image": str(resolve_path(item["image"])),
                    "label": str(resolve_path(item["label"])),
                }
            )
    return pd.DataFrame(rows)


def config_data_dir(dataset_dir: str | Path, nnunet_config: str = "3d_fullres") -> Path:
    dataset_dir = Path(dataset_dir)
    candidates = sorted(p for p in dataset_dir.glob(f"*_{nnunet_config}") if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No nnUNet preprocessed config dir '*_{nnunet_config}' under {dataset_dir}")
    if len(candidates) > 1:
        raise ValueError(f"Expected one '*_{nnunet_config}' directory under {dataset_dir}, found {candidates}")
    return candidates[0]


def iter_preprocessed_image_files(dataset_dir: str | Path, nnunet_config: str = "3d_fullres") -> list[Path]:
    data_dir = config_data_dir(dataset_dir, nnunet_config)
    paths: list[Path] = []
    for suffix in (".b2nd", ".npy", ".npz"):
        paths.extend(data_dir.glob(f"case_*{suffix}"))
    return sorted(p for p in paths if not p.name.endswith("_seg.b2nd") and "_seg." not in p.name)


def load_splits(dataset_dir: str | Path, fold: int = 0) -> dict[str, str]:
    splits_path = Path(dataset_dir) / "splits_final.json"
    if not splits_path.is_file():
        raise FileNotFoundError(f"Missing nnUNet splits file: {splits_path}")
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    if fold < 0 or fold >= len(splits):
        raise ValueError(f"Fold {fold} is out of range for {splits_path}; found {len(splits)} folds")
    selected = splits[fold]
    case_to_split = {case_id: "training" for case_id in selected["train"]}
    case_to_split.update({case_id: "testing" for case_id in selected["val"]})
    return case_to_split


def load_nnunet_patch_size(plans_path: str | Path, nnunet_config: str = "3d_fullres") -> tuple[int, int, int]:
    plans_path = Path(plans_path)
    if not plans_path.is_file():
        raise FileNotFoundError(f"Missing nnUNet plans file: {plans_path}")

    plans = json.loads(plans_path.read_text(encoding="utf-8"))
    configurations = plans.get("configurations")
    if not isinstance(configurations, dict) or nnunet_config not in configurations:
        raise ValueError(f"{plans_path} does not contain configurations.{nnunet_config}")

    patch_size = configurations[nnunet_config].get("patch_size")
    if not isinstance(patch_size, list) or len(patch_size) != 3:
        raise ValueError(f"{plans_path} configurations.{nnunet_config}.patch_size must be a list of 3 integers")
    return tuple(int(v) for v in patch_size)


def load_preprocessed_image(path: str | Path) -> np.ndarray:
    return load_image_volume(path)


def load_image_volume(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".b2nd":
        import blosc2

        arr = blosc2.open(str(path), mode="r")[:]
    elif path.name.endswith(".nii") or path.name.endswith(".nii.gz"):
        img = nib.load(str(path))
        arr = np.asarray(img.dataobj)
    elif path.suffix == ".npz":
        loaded = np.load(path)
        key = "data" if "data" in loaded.files else loaded.files[0]
        arr = loaded[key]
    else:
        arr = np.load(path, mmap_mode="r")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image or channel-first 4D array, got shape {arr.shape} from {path}")
    return arr


def center_crop_or_pad(arr: np.ndarray, spatial_size: Iterable[int]) -> np.ndarray:
    target = tuple(int(v) for v in spatial_size)
    if len(target) != arr.ndim:
        raise ValueError(f"Target size {target} does not match array shape {arr.shape}")

    result = arr
    slices = []
    pad_width = []
    axis_names = ("D", "H", "W") if arr.ndim == 3 else tuple(f"axis{axis}" for axis in range(arr.ndim))
    for axis_name, size, out_size in zip(axis_names, result.shape, target):
        if size > out_size:
            # print(f"Cropping {axis_name} from {size} to {out_size}")
            start = (size - out_size) // 2
            slices.append(slice(start, start + out_size))
            pad_width.append((0, 0))
        elif size < out_size:
            # print(f"Padding {axis_name} from {size} to {out_size}")
            slices.append(slice(None))
            before = (out_size - size) // 2
            after = out_size - size - before
            pad_width.append((before, after))
        else:
            slices.append(slice(None))
            pad_width.append((0, 0))

    result = result[tuple(slices)]
    if any(before or after for before, after in pad_width):
        result = np.pad(result, pad_width, mode="constant")
    return result.astype(np.float32, copy=False)


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    true = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: true={true.shape}, pred={pred.shape}")
    if true.size == 0:
        raise ValueError("Cannot compute metrics for empty arrays")

    err = pred - true
    abs_err = np.abs(err)
    denom = np.where(true != 0, np.abs(true), np.nan)
    pct = abs_err / denom * 100.0
    avg = (true + pred) / 2.0
    ba_diff = err
    bias = float(np.mean(ba_diff))
    sd = float(np.std(ba_diff, ddof=1)) if true.size > 1 else float("nan")
    correlations_defined = true.size > 1 and np.std(true) > 0 and np.std(pred) > 0

    metrics = {
        "n": int(true.size),
        "mae_ml": float(np.mean(abs_err)),
        "median_ae_ml": float(np.median(abs_err)),
        "rmse_ml": float(np.sqrt(np.mean(err ** 2))),
        "mape_percent": float(np.nanmean(pct)),
        "pearson_r": float(np.corrcoef(true, pred)[0, 1]) if correlations_defined else float("nan"),
        "spearman_r": float(pd.Series(true).corr(pd.Series(pred), method="spearman")) if correlations_defined else float("nan"),
        "bland_altman_bias_ml": bias,
        "bland_altman_lower_ml": bias - 1.96 * sd if math.isfinite(sd) else float("nan"),
        "bland_altman_upper_ml": bias + 1.96 * sd if math.isfinite(sd) else float("nan"),
        "mean_true_ml": float(np.mean(true)),
        "mean_pred_ml": float(np.mean(pred)),
        "mean_bland_altman_average_ml": float(np.mean(avg)),
    }
    return metrics


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_dicts_csv(path: str | Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
