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


def build_prostate158_nnunet_cases(
    csv_path: str | Path,
    preds_dir: str | Path,
) -> list[dict]:
    """Return cases pairing T2 image, anatomy GT, and nnUNet segmentation prediction.

    Like :func:`build_prostate158_anatomy_cases` but infers *dataset_dir* from
    *csv_path* and expects predictions named ``<case_id>.nii.gz`` in *preds_dir*
    (the default nnUNet ensemble output layout).

    Args:
        csv_path:  Path to train.csv or valid.csv; its parent is used as dataset_dir.
        preds_dir: Directory containing ``<case_id>.nii.gz`` predictions.

    Returns:
        List of dicts with keys ``case_id``, ``image_path``, ``gt_path``, ``pred_path``.
    """
    csv_path = Path(csv_path)
    dataset_dir = csv_path.parent
    preds_dir = Path(preds_dir)

    df = pd.read_csv(csv_path)

    cases = []
    for _, row in df.iterrows():
        case_id = int(row["ID"])
        image_path = dataset_dir / row["t2"]
        gt_path = dataset_dir / row["t2_anatomy_reader1"]
        pred_path = preds_dir / f"{case_id}.nii.gz"

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


def build_promise12_nnunet_cases(
    csv_path: str | Path,
    preds_dir: str | Path,
) -> list[dict]:
    """Return cases pairing T2 image, segmentation GT, and nnUNet prediction.

    Like :func:`build_promise12_cases` but infers *dataset_dir* from *csv_path*
    and expects predictions named ``<case_id>.nii.gz`` in *preds_dir*.

    Args:
        csv_path:  Path to training_data.csv or livechallenge_test_data.csv;
                   its parent is used as dataset_dir.
        preds_dir: Directory containing ``<case_id>.nii.gz`` predictions.

    Returns:
        List of dicts with keys ``case_id``, ``image_path``, ``gt_path``, ``pred_path``.
    """
    csv_path = Path(csv_path)
    dataset_dir = csv_path.parent
    preds_dir = Path(preds_dir)

    df = pd.read_csv(csv_path)

    cases = []
    for _, row in df.iterrows():
        case_id = int(row["ID"])
        image_path = dataset_dir / row["t2"]
        gt_path = dataset_dir / row["segmentation"]
        pred_path = preds_dir / f"{case_id}.nii.gz"

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


def build_promise12_nnunet_runner_cases(
    csv_path: str | Path,
    preds_dir: str | Path,
    segmentation_key: str = "segmentation"
) -> list[dict]:
    """Return cases pairing T2 image, segmentation GT, and nnUNetV2Runner prediction.

    Matches CSV rows (sorted by ID) to prediction files (sorted by case index)
    positionally, so the offset introduced by MONAI's sequential naming is handled
    automatically without requiring knowledge of the training set size.

    Args:
        csv_path:  Path to test_data.csv; its parent is used as dataset_dir.
        preds_dir: Directory containing ``case_<N>.nii.gz`` predictions.

    Returns:
        List of dicts with keys ``case_id``, ``image_path``, ``gt_path``, ``pred_path``.
    """
    csv_path = Path(csv_path)
    dataset_dir = csv_path.parent
    preds_dir = Path(preds_dir)

    df = pd.read_csv(csv_path).sort_values("ID").reset_index(drop=True)
    pred_files = sorted(preds_dir.glob("*.nii.gz"))

    if len(pred_files) != len(df):
        warnings.warn(
            f"Prediction count ({len(pred_files)}) differs from CSV rows ({len(df)}); "
            "pairing by position may be incorrect."
        )

    cases = []
    for (_, row), pred_path in zip(df.iterrows(), pred_files):
        case_id = int(row["ID"])
        image_path = dataset_dir / row["t2"]
        gt_path = dataset_dir / row[segmentation_key]

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
