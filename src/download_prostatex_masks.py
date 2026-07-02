"""Download the PROSTATEx_masks dataset (prostate and lesion NIfTI masks) from GitHub."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/rcuocolo/PROSTATEx_masks.git"
SOURCE_NAME = "github"
DATASET_NAME = "prostatex_masks"
COMPLETION_SENTINEL = ".complete"
MANIFEST_NAME = "download_manifest.json"
REQUIRED_PATHS = ("Files/prostate/mask_prostate", "Files/lesions/Masks")


def is_complete(dataset_dir: Path) -> bool:
    sentinel = dataset_dir / COMPLETION_SENTINEL
    return sentinel.is_file() and all((dataset_dir / p).exists() for p in REQUIRED_PATHS)


def has_required_contents(dataset_dir: Path) -> bool:
    return dataset_dir.is_dir() and all((dataset_dir / p).exists() for p in REQUIRED_PATHS)


def dataset_root(data_root: Path) -> Path:
    return Path(data_root) / "raw" / SOURCE_NAME / DATASET_NAME


def _mask_lookup(mask_dir: Path, suffix: str) -> dict[str, Path]:
    """Return {4-digit-number -> path}, handling any prefix and any digit padding."""
    lookup = {}
    for p in sorted(mask_dir.glob("*.nii.gz")):
        stem = p.name.removesuffix(".nii.gz").removesuffix(suffix)
        m = re.search(r"(\d+)$", stem)
        if m:
            lookup[f"{int(m.group(1)):04d}"] = p
    return lookup


def build_index_csv(dataset_dir: Path) -> Path:
    image_list = dataset_dir / "Files" / "prostate" / "image_list.csv"
    images_dir = dataset_dir / "Files" / "prostate" / "Images"
    wg_lookup = _mask_lookup(dataset_dir / "Files" / "prostate" / "mask_prostate", "")
    tz_lookup = _mask_lookup(dataset_dir / "Files" / "prostate" / "mask_tz", "_tz")
    pz_lookup = _mask_lookup(dataset_dir / "Files" / "prostate" / "mask_pz", "_pz")

    rows = []
    with open(image_list, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            series = row["T2"]
            m = re.match(r"(ProstateX-)(\d+)", series)
            if not m:
                continue
            patient_id = f"{m.group(1)}{int(m.group(2)):04d}"
            num = f"{int(m.group(2)):04d}"
            t2_path = images_dir / f"{series}.nii.gz"
            missing = [k for k, lut in [("label_wg", wg_lookup), ("label_tz", tz_lookup), ("label_pz", pz_lookup)] if num not in lut]
            if not t2_path.exists():
                missing.insert(0, "t2")
            if missing:
                print(f"WARNING: {patient_id} missing {missing}, skipping", file=sys.stderr)
                continue
            rows.append(
                {
                    "ID": patient_id,
                    "t2": os.path.relpath(t2_path, dataset_dir),
                    "label_wg": os.path.relpath(wg_lookup[num], dataset_dir),
                    "label_tz": os.path.relpath(tz_lookup[num], dataset_dir),
                    "label_pz": os.path.relpath(pz_lookup[num], dataset_dir),
                }
            )

    rows.sort(key=lambda r: r["ID"])
    index_csv = dataset_dir / "index.csv"
    with open(index_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "t2", "label_wg", "label_tz", "label_pz"])
        writer.writeheader()
        writer.writerows(rows)
    return index_csv


def write_manifest(dataset_dir: Path) -> None:
    manifest = {
        "dataset": DATASET_NAME,
        "source": SOURCE_NAME,
        "repo_url": REPO_URL,
        "required_paths": list(REQUIRED_PATHS),
    }
    (dataset_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (dataset_dir / COMPLETION_SENTINEL).touch()


def download_dataset(data_root: Path, progress: bool = True) -> Path:
    data_root = Path(data_root)
    dataset_dir = dataset_root(data_root)

    if is_complete(dataset_dir):
        return dataset_dir

    if dataset_dir.exists():
        raise RuntimeError(
            f"Found an existing but incomplete dataset folder at {dataset_dir}. "
            "Delete it or finish the clone before retrying."
        )

    dataset_dir.parent.mkdir(parents=True, exist_ok=True)

    command = ["git", "clone", "--depth=1", REPO_URL, str(dataset_dir)]
    if not progress:
        command.insert(2, "--quiet")
    subprocess.run(command, check=True)

    if not has_required_contents(dataset_dir):
        raise RuntimeError(
            f"Clone completed but expected files were not found in {dataset_dir}."
        )

    write_manifest(dataset_dir)
    build_index_csv(dataset_dir)

    return dataset_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path("data"), help="Root folder for raw data."
    )
    parser.add_argument("--no-progress", action="store_true", help="Hide clone progress output.")
    parser.add_argument("--rebuild-csv", action="store_true", help="Regenerate index.csv without re-cloning.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rebuild_csv:
        dataset_dir = dataset_root(args.data_root)
        if not has_required_contents(dataset_dir):
            raise SystemExit(f"Dataset not found at {dataset_dir}. Run without --rebuild-csv first.")
        index_csv = build_index_csv(dataset_dir)
        print(f"Wrote {index_csv}")
        return
    dataset_dir = download_dataset(args.data_root, progress=not args.no_progress)
    print(f"Downloaded dataset to {dataset_dir}")


if __name__ == "__main__":
    main()
