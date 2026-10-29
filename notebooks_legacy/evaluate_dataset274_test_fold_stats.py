#!/usr/bin/env python3

# %%
"""Evaluate Dataset274 test-fold foreground segmentation and volume error.

This file uses Jupytext-compatible percent-cell syntax so it can be opened as a
notebook or executed as a plain Python script.
"""

from __future__ import annotations

import math
import sys
import re
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:  # pragma: no cover - plain Python fallback
    def display(*objects: object) -> None:
        for obj in objects:
            print(obj)


REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from seg_metrics import compute_segmentation_metrics


# %%
PRED_DIR = Path("work_dir_pooled2/nnunet_results/Dataset274_raw/pred_3d_fullres")
GT_DIR = Path("work_dir_pooled2/nnunet_raw/Dataset274_raw/labelsTs")
CASE_PATTERN = re.compile(r"case_(\d+)\.nii\.gz$")


def case_key(path: Path) -> tuple[int, str]:
    match = CASE_PATTERN.match(path.name)
    if match:
        return int(match.group(1)), path.name
    return math.inf, path.name


def load_binary_mask(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = nib.load(str(path))
    arr = np.asarray(img.dataobj)
    spacing = tuple(float(v) for v in img.header.get_zooms()[:3])
    if len(spacing) != 3:
        raise ValueError(f"Expected 3D image for {path}, got spacing {spacing}")
    return (arr > 0), spacing


def volume_ml(mask: np.ndarray, spacing_mm: tuple[float, float, float]) -> float:
    return float(mask.sum() * np.prod(spacing_mm) / 1000.0)


def summarize_diffs(diffs: pd.Series) -> dict[str, float]:
    bias = float(diffs.mean())
    sd = float(diffs.std(ddof=1))
    loa_delta = 1.96 * sd if math.isfinite(sd) else float("nan")
    return {
        "bias_ml": bias,
        "loa_lower_ml": bias - loa_delta,
        "loa_upper_ml": bias + loa_delta,
        "diff_sd_ml": sd,
    }


def load_case_pairs(pred_dir: Path, gt_dir: Path) -> list[tuple[Path, Path]]:
    pred_files = sorted(pred_dir.glob("case_*.nii.gz"), key=case_key)
    gt_files = sorted(gt_dir.glob("case_*.nii.gz"), key=case_key)

    if not pred_files:
        raise FileNotFoundError(f"No prediction files found under {pred_dir}")
    if not gt_files:
        raise FileNotFoundError(f"No ground-truth files found under {gt_dir}")

    pred_ids = {p.name for p in pred_files}
    gt_ids = {p.name for p in gt_files}
    missing_pred = sorted(gt_ids - pred_ids)
    missing_gt = sorted(pred_ids - gt_ids)
    if missing_pred or missing_gt:
        raise ValueError(
            "Prediction/ground-truth case mismatch: "
            f"missing predictions={missing_pred[:5]}, missing labels={missing_gt[:5]}"
        )

    return [(pred_dir / name, gt_dir / name) for name in sorted(pred_ids, key=lambda n: case_key(Path(n)))]


# %%
case_pairs = load_case_pairs(PRED_DIR, GT_DIR)
rows: list[dict[str, float | str]] = []

for pred_path, gt_path in case_pairs:
    pred_mask, pred_spacing = load_binary_mask(pred_path)
    gt_mask, gt_spacing = load_binary_mask(gt_path)

    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"Shape mismatch for {pred_path.name}: {pred_mask.shape} vs {gt_mask.shape}")
    if not np.allclose(pred_spacing, gt_spacing):
        raise ValueError(f"Spacing mismatch for {pred_path.name}: {pred_spacing} vs {gt_spacing}")

    metrics = compute_segmentation_metrics(pred_mask, gt_mask, pred_spacing)
    pred_ml = volume_ml(pred_mask, pred_spacing)
    gt_ml = volume_ml(gt_mask, gt_spacing)
    diff_ml = pred_ml - gt_ml

    rows.append(
        {
            "case_id": pred_path.name.replace(".nii.gz", ""),
            "dsc": metrics["dsc"],
            "pred_ml": pred_ml,
            "gt_ml": gt_ml,
            "diff_ml": diff_ml,
            "abs_error_ml": abs(diff_ml),
            "sq_error_ml2": diff_ml**2,
        }
    )

per_case = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)


# %%
summary = {
    "n_cases": int(len(per_case)),
    "foreground_dsc_mean": float(per_case["dsc"].mean()),
    "foreground_dsc_sd": float(per_case["dsc"].std(ddof=1)),
    "mean_pred_ml": float(per_case["pred_ml"].mean()),
    "mean_gt_ml": float(per_case["gt_ml"].mean()),
    "mae_ml": float(per_case["abs_error_ml"].mean()),
    "rmse_ml": float(np.sqrt(per_case["sq_error_ml2"].mean())),
}
summary.update(summarize_diffs(per_case["diff_ml"]))

summary_df = pd.DataFrame([summary]).assign(
    loa_text=lambda df: df.apply(
        lambda row: f"{row['loa_lower_ml']:.3f} to {row['loa_upper_ml']:.3f}",
        axis=1,
    )
)
distribution_df = per_case[["dsc", "pred_ml", "gt_ml", "diff_ml", "abs_error_ml"]].describe(
    percentiles=[0.025, 0.25, 0.5, 0.75, 0.975]
).T


# %%
print(f"Evaluated {summary['n_cases']} matched cases")
display(per_case.head())
display(summary_df[[
    "n_cases",
    "foreground_dsc_mean",
    "foreground_dsc_sd",
    "bias_ml",
    "loa_lower_ml",
    "loa_upper_ml",
    "mae_ml",
    "rmse_ml",
]])
display(distribution_df)


# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].scatter(per_case["gt_ml"], per_case["diff_ml"], s=35, alpha=0.85)
axes[0].axhline(summary["bias_ml"], color="black", linestyle="--", linewidth=1)
axes[0].axhline(summary["loa_lower_ml"], color="tab:red", linestyle=":", linewidth=1)
axes[0].axhline(summary["loa_upper_ml"], color="tab:red", linestyle=":", linewidth=1)
axes[0].set_xlabel("Ground truth volume (mL)")
axes[0].set_ylabel("Prediction minus ground truth (mL)")
axes[0].set_title("Bland-Altman volume difference")

axes[1].hist(per_case["dsc"], bins=np.linspace(0, 1, 21), color="black", alpha=0.75)
axes[1].set_xlabel("Foreground DSC")
axes[1].set_ylabel("Count")
axes[1].set_title("Foreground DSC distribution")

fig.tight_layout()
plt.show()


# %%
print("Summary")
for key in [
    "foreground_dsc_mean",
    "foreground_dsc_sd",
    "bias_ml",
    "loa_lower_ml",
    "loa_upper_ml",
    "mae_ml",
    "rmse_ml",
]:
    print(f"{key}: {summary[key]:.4f}")
