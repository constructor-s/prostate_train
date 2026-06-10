"""Build paired (image, ground-truth, prediction) case lists for Prostate158.

Intended to be imported by evaluation notebooks and scripts so that CSV
parsing and path resolution logic is not duplicated across files.

Run as a script to generate a folds JSON from CSVs:

    python src/generate_cases.py \\
        --training data/.../training_data.csv \\
        --testing  data/.../livechallenge_test_data.csv \\
        --output   configs/promise12_folds.json
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd


def build_prostate158_anatomy_cases(
    csv_path: str | Path,
    dataset_dir: str | Path,
    preds_dir: str | Path,
) -> list[dict]:
    """Return a list of cases pairing T2 image, anatomy GT, and anatomy prediction.

    Reads ``train.csv`` or ``valid.csv`` from *dataset_dir* and resolves paths
    relative to *dataset_dir* (image + GT) and *preds_dir* (prediction).

    The prediction path mirrors the input T2 path:
        ``preds_dir / <split>/<case_id>/t2/t2_trans.nii.gz``

    Cases where the GT or prediction file does not exist are skipped with a
    warning so the caller can proceed with the available subset.

    Args:
        csv_path:    Path to train.csv or valid.csv.
        dataset_dir: Root directory that the CSV paths are relative to
                     (e.g. ``prostate158_train/``).
        preds_dir: Root directory containing anatomy predictions
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
    preds_dir = Path(preds_dir)

    df = pd.read_csv(csv_path)

    cases = []
    for _, row in df.iterrows():
        case_id = int(row["ID"])
        image_path = dataset_dir / row["t2"]
        gt_path = dataset_dir / row["t2_anatomy_reader1"]

        # Prediction lives under preds_dir/<split>/<case_id>/t2/t2_trans.nii.gz.
        # The CSV "t2" column is e.g. "train/024/t2.nii.gz"; stripping both
        # suffixes gives "train/024/t2" which is the bundle's output sub-folder.
        t2_stem = Path(row["t2"]).with_suffix("").with_suffix("")  # train/024/t2
        pred_path = preds_dir / t2_stem / "t2_trans.nii.gz"

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


def build_promise12_cases(
    csv_path: str | Path,
    dataset_dir: str | Path,
    preds_dir: str | Path,
) -> list[dict]:
    """Return a list of cases pairing T2 image, segmentation GT, and prediction.

    Reads ``livechallenge_test_data.csv`` from *dataset_dir* and resolves paths
    relative to *dataset_dir* (image + GT) and *preds_dir* (prediction).

    The prediction path mirrors the input T2 stem with a ``.nii.gz`` extension:
        ``preds_dir / <case_stem>.nii.gz``  (e.g. ``Case00.nii.gz``)

    Cases where the GT or prediction file does not exist are skipped with a
    warning so the caller can proceed with the available subset.

    Args:
        csv_path:    Path to livechallenge_test_data.csv.
        dataset_dir: Root directory that the CSV paths are relative to.
        preds_dir:   Root directory containing segmentation predictions.

    Returns:
        List of dicts, each with keys:
            ``case_id``    – integer case ID from the CSV
            ``image_path`` – Path to the T2 ``.mhd`` image
            ``gt_path``    – Path to the ``_segmentation.mhd`` ground truth
            ``pred_path``  – Path to the predicted ``.nii.gz`` file
    """
    csv_path = Path(csv_path)
    dataset_dir = Path(dataset_dir)
    preds_dir = Path(preds_dir)

    df = pd.read_csv(csv_path)

    cases = []
    for _, row in df.iterrows():
        case_id = int(row["ID"])
        image_path = dataset_dir / row["t2"]
        gt_path = dataset_dir / row["segmentation"]

        # Prediction: preds_dir/<case_stem>.nii.gz (e.g. Case00.nii.gz)
        case_stem = Path(row["t2"]).stem  # e.g. "Case00"
        pred_path = preds_dir / case_stem / f"{case_stem}_trans.nii.gz"

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


def build_nnunet_test_cases(
    images_dir: str | Path,
    labels_dir: str | Path,
    preds_dir: str | Path,
) -> list[dict]:
    """Return a list of cases pairing image, GT label, and nnUNet ensemble prediction.

    Enumerates ``labels_dir`` for ``*.nii.gz`` files and resolves the matching
    image (``<stem>_0000.nii.gz``) and prediction (``<stem>.nii.gz``) paths.

    Args:
        images_dir: Directory containing ``case_XXX_0000.nii.gz`` input images.
        labels_dir: Directory containing ``case_XXX.nii.gz`` ground-truth labels.
        preds_dir:  Directory containing ``case_XXX.nii.gz`` ensemble predictions.

    Returns:
        List of dicts, each with keys:
            ``case_id``    – integer extracted from the case stem
            ``image_path`` – Path to the input image
            ``gt_path``    – Path to the ground-truth label
            ``pred_path``  – Path to the ensemble prediction
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    preds_dir = Path(preds_dir)

    cases = []
    for gt_path in sorted(labels_dir.glob("*.nii.gz")):
        stem = gt_path.name.replace(".nii.gz", "")  # e.g. "case_119"
        case_id = int(stem.split("_")[-1])
        image_path = images_dir / f"{stem}_0000.nii.gz"
        pred_path = preds_dir / f"{stem}.nii.gz"

        missing = [p for p in (image_path, pred_path) if not p.exists()]
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


def generate_folds_json(
    splits: list[tuple[str, Path]],
    image_col: str = "t2",
    label_col: str = "segmentation",
    id_col: str = "ID",
) -> dict:
    """Build a folds dict from a list of (split_name, csv_path) pairs.

    Args:
        splits:    List of (split_name, csv_path) pairs, e.g. [("training", Path("train.csv"))].
        image_col: CSV column name for the image path.
        label_col: CSV column name for the label/segmentation path.
        id_col:    CSV column name for the integer case ID.

    Returns:
        Dict mapping split names to lists of ``{"image": ..., "label": ..., "id": ...}`` dicts.
    """
    result: dict[str, list[dict]] = {}
    for split_name, csv_path in splits:
        df = pd.read_csv(csv_path)
        result[split_name] = [
            {"image": row[image_col], "label": row[label_col], "id": str(int(row[id_col]))}
            for _, row in df.iterrows()
        ]
    return result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Generate a folds JSON (MONAI datalist format) from CSV files."
    )
    parser.add_argument("--training", type=Path, required=True, metavar="CSV", help="CSV for the training split.")
    parser.add_argument("--testing", type=Path, required=True, metavar="CSV", help="CSV for the testing split.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path.")
    parser.add_argument("--image-col", default="t2", help="CSV column for image paths (default: t2).")
    parser.add_argument("--label-col", default="segmentation", help="CSV column for label paths (default: segmentation).")
    parser.add_argument("--id-col", default="ID", help="CSV column for case IDs (default: ID).")
    args = parser.parse_args()

    folds = generate_folds_json(
        splits=[("training", args.training), ("testing", args.testing)],
        image_col=args.image_col,
        label_col=args.label_col,
        id_col=args.id_col,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(folds, indent=4))
    total = sum(len(v) for v in folds.values())
    print(f"Wrote {total} cases to {args.output}")
