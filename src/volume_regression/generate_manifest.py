"""Generate a regression manifest from the raw nnUNet v2 prostate dataset.

Purpose:
    Create a CSV keyed by nnUNet case IDs such as ``case_0`` rather than
    original dataset IDs.

Inputs:
    ``--raw-root`` points to an nnUNet raw root containing a dataset directory
    such as ``Dataset274_raw``. Within that directory, this script expects:
    - ``datalist.json`` with ``training`` and ``testing`` entries.
    - ``imagesTr`` / ``imagesTs`` raw image volumes.
    - ``labelsTr`` / ``labelsTs`` target masks.

Output:
    A CSV with columns ``case_id``, ``source_id``, ``split``, ``image``,
    ``label``, ``volume_ml``, and ``log_volume_ml``. ``split`` is the raw
    nnUNet train/test split from ``datalist.json``.

Assumptions:
    Whole gland is ``label > 0``. Target volume is computed from raw label image
    geometry.

Example:
    uv run python src/volume_regression/generate_manifest.py \
      --raw-root work_dir_pooled2/nnunet_raw \
      --preprocessed-root work_dir_pooled2/nnunet_preprocessed \
      --dataset-id 274 \
      --output work_dir_volume_regression_resnet/resnet18_pooled2/manifest.csv
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import pandas as pd

from common import (
    REPO_ROOT,
    compute_mask_volume_ml,
    find_dataset_dir,
    load_nnunet_raw_cases,
    resolve_path,
)


def build_manifest_rows(
    raw_root: str | Path,
    dataset_id: str | int | None = 274,
) -> list[dict]:
    raw_root = resolve_path(raw_root)
    dataset_dir = find_dataset_dir(raw_root, dataset_id)
    cases = load_nnunet_raw_cases(dataset_dir)
    rows: list[dict] = []
    for row in cases.to_dict(orient="records"):
        target_label = Path(row["label"])
        if not target_label.is_file():
            raise FileNotFoundError(f"Missing target label for {row['case_id']}: {target_label}")
        volume_ml = compute_mask_volume_ml(target_label)
        rows.append(
            {
                "case_id": row["case_id"],
                "source_id": row["source_id"],
                "split": row["split"],
                "image": row["image"],
                "label": str(target_label),
                "volume_ml": volume_ml,
                "log_volume_ml": math.log(volume_ml),
            }
        )
    if not rows:
        raise ValueError(f"No manifest rows found in {dataset_dir}")
    return rows


def copy_manifest_metadata(
    raw_dataset_dir: str | Path,
    output_dir: str | Path,
    preprocessed_dataset_dir: str | Path | None = None,
) -> list[Path]:
    raw_dataset_dir = Path(raw_dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for name in ("dataset.json", "datalist.json"):
        source = raw_dataset_dir / name
        if source.is_file():
            destination = output_dir / name
            shutil.copy2(source, destination)
            copied.append(destination)
    if preprocessed_dataset_dir is not None:
        preprocessed_dataset_dir = Path(preprocessed_dataset_dir)
        for name in ("nnUNetPlans.json", "splits_final.json"):
            source = preprocessed_dataset_dir / name
            if source.is_file():
                destination = output_dir / name
                shutil.copy2(source, destination)
                copied.append(destination)
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a prostate volume regression manifest from raw nnUNet data.")
    parser.add_argument("--raw-root", default="work_dir_pooled2/nnunet_raw")
    parser.add_argument("--preprocessed-root", default="work_dir_pooled2/nnunet_preprocessed")
    parser.add_argument("--dataset-id", default="274")
    parser.add_argument("--output", default="results/volume_regression/resnet18_pooled2/manifest.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = resolve_path(args.raw_root)
    raw_dataset_dir = find_dataset_dir(raw_root, args.dataset_id)
    preprocessed_root = resolve_path(args.preprocessed_root) if args.preprocessed_root else None
    preprocessed_dataset_dir = (
        find_dataset_dir(preprocessed_root, args.dataset_id)
        if preprocessed_root is not None and preprocessed_root.is_dir()
        else None
    )
    rows = build_manifest_rows(
        raw_root=raw_root,
        dataset_id=args.dataset_id,
    )
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    copied = copy_manifest_metadata(raw_dataset_dir, output.parent, preprocessed_dataset_dir)

    display = output.relative_to(REPO_ROOT) if output.is_relative_to(REPO_ROOT) else output
    copied_display = [p.relative_to(REPO_ROOT) if p.is_relative_to(REPO_ROOT) else p for p in copied]
    print(f"Wrote {len(rows)} rows to {display}")
    print(f"Copied metadata: {', '.join(str(p) for p in copied_display)}")


if __name__ == "__main__":
    main()
