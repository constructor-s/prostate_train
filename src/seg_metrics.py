"""Binary segmentation metrics for medical imaging.

All functions accept boolean (or 0/1 int) numpy arrays of identical shape.
``spacing_mm`` is a (D, H, W) or (H, W) tuple of voxel sizes in millimetres,
used to convert voxel counts to physical volumes and to compute surface
distances in mm.

Overlap metrics (DSC, IoU, Precision, Recall, Specificity) delegate to
torchmetrics for correctness and testability.  Surface-distance metrics
(HD95, ASSD) use a custom implementation because torchmetrics does not
support anisotropic per-axis spacing.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.int32)).flatten()


def _surface_distances_mm(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing_mm: tuple[float, ...],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Compute directed surface distances in mm between border voxels.

    Returns (pred→gt distances, gt→pred distances), or (None, None) when
    either mask has no border (empty or fully-filled volume).
    """
    scale = np.array(spacing_mm, dtype=float)

    def border(mask: np.ndarray) -> np.ndarray:
        return mask & ~binary_erosion(mask)

    pred_b = border(pred.astype(bool))
    gt_b   = border(gt.astype(bool))

    if not pred_b.any() or not gt_b.any():
        return None, None

    pred_pts = np.argwhere(pred_b) * scale
    gt_pts   = np.argwhere(gt_b)   * scale

    tree_gt   = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)

    d_p2g, _ = tree_gt.query(pred_pts,  workers=-1)
    d_g2p, _ = tree_pred.query(gt_pts,  workers=-1)
    return d_p2g, d_g2p


def compute_segmentation_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing_mm: tuple[float, ...],
) -> dict[str, float]:
    """Return a flat dict of segmentation metrics for a binary mask pair.

    Args:
        pred: Predicted binary mask (bool or 0/1 int), any shape.
        gt:   Ground-truth binary mask, same shape as *pred*.
        spacing_mm: Voxel size in mm, one value per spatial dimension
            (e.g. ``(slice_thickness, row_spacing, col_spacing)``).
            Used for volume calculations and surface-distance metrics.

    Returns:
        Dict with keys: dsc, iou, precision, recall, specificity,
        hd95_mm, assd_mm, vol_sim,
        tp_vol_mm3, fp_vol_mm3, fn_vol_mm3, tn_vol_mm3,
        pred_vol_mm3, gt_vol_mm3.
    """
    pred_t = _to_tensor(pred)
    gt_t   = _to_tensor(gt)

    metrics = {
        "dsc":         BinaryF1Score()(pred_t, gt_t).item(),
        "iou":         BinaryJaccardIndex()(pred_t, gt_t).item(),
        "precision":   BinaryPrecision()(pred_t, gt_t).item(),
        "recall":      BinaryRecall()(pred_t, gt_t).item(),
        "specificity": BinarySpecificity()(pred_t, gt_t).item(),
    }

    # Volume metrics derived from confusion counts
    voxel_vol = float(np.prod(spacing_mm))
    pred_b = pred.astype(bool)
    gt_b   = gt.astype(bool)

    tp = int((pred_b & gt_b).sum())
    fp = int((pred_b & ~gt_b).sum())
    fn = int((~pred_b & gt_b).sum())
    tn = int((~pred_b & ~gt_b).sum())

    pred_vol = (tp + fp) * voxel_vol
    gt_vol   = (tp + fn) * voxel_vol

    metrics["tp_vol_mm3"]  = tp * voxel_vol
    metrics["fp_vol_mm3"]  = fp * voxel_vol
    metrics["fn_vol_mm3"]  = fn * voxel_vol
    metrics["tn_vol_mm3"]  = tn * voxel_vol
    metrics["pred_vol_mm3"] = pred_vol
    metrics["gt_vol_mm3"]   = gt_vol
    metrics["vol_sim"]      = abs(pred_vol - gt_vol) / gt_vol if gt_vol > 0 else float("nan")

    # Surface-distance metrics in mm (spacing-aware)
    d_p2g, d_g2p = _surface_distances_mm(pred_b, gt_b, spacing_mm)
    if d_p2g is not None:
        all_d = np.concatenate([d_p2g, d_g2p])
        metrics["hd95_mm"] = float(np.percentile(all_d, 95))
        metrics["assd_mm"] = float((d_p2g.mean() + d_g2p.mean()) / 2)
    else:
        metrics["hd95_mm"] = float("nan")
        metrics["assd_mm"] = float("nan")

    return metrics
