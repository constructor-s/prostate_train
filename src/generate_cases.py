"""Build paired (image, ground-truth, prediction) case lists for Prostate158.

Intended to be imported by evaluation notebooks and scripts so that CSV
parsing and path resolution logic is not duplicated across files.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd


def build_prostate158_anatomy_cases(
    csv_path: str | Path,
    dataset_dir: str | Path,
    results_dir: str | Path,
) -> list[dict]:
    """Return a list of cases pairing T2 image, anatomy GT, and anatomy prediction.

    Reads ``train.csv`` or ``valid.csv`` from *dataset_dir* and resolves paths
    relative to *dataset_dir* (image + GT) and *results_dir* (prediction).

    The prediction path mirrors the input T2 path:
        ``results_dir / <split>/<case_id>/t2/t2_trans.nii.gz``

    Cases where the GT or prediction file does not exist are skipped with a
    warning so the caller can proceed with the available subset.

    Args:
        csv_path:    Path to train.csv or valid.csv.
        dataset_dir: Root directory that the CSV paths are relative to
                     (e.g. ``prostate158_train/``).
        results_dir: Root directory containing anatomy predictions
                     (e.g. ``results/anatomy/``).

    Returns:
        List of dicts, each with keys:
            ``case_id``    – integer case ID from the CSV
            ``image_path`` – Path to t2.nii.gz
            ``gt_path``    – Path to t2_anatomy_reader1.nii.gz
            ``pred_path``  – Path to predicted t2_trans.nii.gz
    """
    csv_path = Path(csv_path)
    dataset_dir = Path(dataset_dir)
    results_dir = Path(results_dir)

    df = pd.read_csv(csv_path)

    cases = []
    for _, row in df.iterrows():
        case_id = int(row["ID"])
        image_path = dataset_dir / row["t2"]
        gt_path = dataset_dir / row["t2_anatomy_reader1"]

        # Prediction lives under results_dir/<split>/<case_id>/t2/t2_trans.nii.gz.
        # The CSV "t2" column is e.g. "train/024/t2.nii.gz"; stripping both
        # suffixes gives "train/024/t2" which is the bundle's output sub-folder.
        t2_stem = Path(row["t2"]).with_suffix("").with_suffix("")  # train/024/t2
        pred_path = results_dir / t2_stem / "t2_trans.nii.gz"

        missing = [p for p in (gt_path, pred_path) if not p.exists()]
        if missing:
            warnings.warn(f"Case {case_id}: skipping, missing {[str(p) for p in missing]}")
            continue

        cases.append(
            {
                "case_id": case_id,
                "image_path": image_path,
                "gt_path": gt_path,
                "pred_path": pred_path,
            }
        )

    return cases
