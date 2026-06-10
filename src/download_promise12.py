"""Download the PROMISE12 prostate MR segmentation dataset from Zenodo."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

DEFAULT_RECORD_ID = 8026660
DEFAULT_FOLDER = "promise12"
# Each zip extracts into the dataset root; check one file from each archive.
REQUIRED_PATHS = (
    "training_data",
    "test_data",
    "livechallenge_test_data",
)
SOURCE_NAME = "zenodo"
DATASET_NAME = "promise12"
COMPLETION_SENTINEL = ".complete"
MANIFEST_NAME = "download_manifest.json"

EXPECTED_ZIPS = {
    "training_data.zip",
    "test_data.zip",
    "livechallenge_test_data.zip",
}


def is_complete(dataset_dir: Path) -> bool:
    sentinel = dataset_dir / COMPLETION_SENTINEL
    return sentinel.is_file() and all((dataset_dir / p).exists() for p in REQUIRED_PATHS)


def has_required_contents(dataset_dir: Path) -> bool:
    return dataset_dir.is_dir() and all((dataset_dir / p).exists() for p in REQUIRED_PATHS)


def dataset_root(data_root: Path, record_id: int) -> Path:
    return Path(data_root) / "raw" / SOURCE_NAME / DATASET_NAME / f"record-{record_id}"


def write_manifest(dataset_dir: Path, record_id: int) -> None:
    manifest = {
        "dataset": DATASET_NAME,
        "source": SOURCE_NAME,
        "record_id": record_id,
        "extracted_folder": DEFAULT_FOLDER,
        "required_paths": list(REQUIRED_PATHS),
    }
    (dataset_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (dataset_dir / COMPLETION_SENTINEL).touch()


def download_dataset(
    data_root: Path, record_id: int = DEFAULT_RECORD_ID, progress: bool = True
) -> Path:
    data_root = Path(data_root)
    record_root = dataset_root(data_root, record_id)
    dataset_dir = record_root / DEFAULT_FOLDER

    if is_complete(dataset_dir):
        return dataset_dir

    if dataset_dir.exists():
        raise RuntimeError(
            f"Found an existing but incomplete dataset folder at {dataset_dir}. "
            "Delete it or finish the extraction before retrying."
        )

    record_root.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = data_root / "tmp" / SOURCE_NAME / DATASET_NAME / f"record-{record_id}"
    tmp_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp_dir:
        command = [
            sys.executable,
            "-m",
            "zenodo_get",
            str(record_id),
            "-o",
            tmp_dir,
            "-g",
            "*.zip",
            "-n",
            "-v",
            "2" if progress else "0",
        ]
        subprocess.run(command, check=True)

        archives = sorted(Path(tmp_dir).glob("*.zip"))
        found_names = {a.name for a in archives}
        missing = EXPECTED_ZIPS - found_names
        if missing:
            raise RuntimeError(
                f"Expected zip files not found for record {record_id}: {missing}. "
                f"Found: {found_names}"
            )

        for archive in archives:
            dest = dataset_dir / archive.stem
            dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(dest)

    if not has_required_contents(dataset_dir):
        raise RuntimeError(
            f"Download completed but expected files were not found in {dataset_dir}."
        )

    write_manifest(dataset_dir, record_id)

    if not is_complete(dataset_dir):
        raise RuntimeError(
            f"Download completed but dataset was not marked complete at {dataset_dir}."
        )

    return dataset_dir


def generate_csvs(SPLITS = ("training_data", "test_data", "livechallenge_test_data")) -> None:
    """Generate t2/segmentation CSV files for each PROMISE12 split."""
    import csv
    import re

    dataset_dir = dataset_root(Path("data"), DEFAULT_RECORD_ID) / DEFAULT_FOLDER
    for split in SPLITS:
        split_dir = dataset_dir / split
        if not split_dir.is_dir():
            print(f"Skipping {split}: directory not found at {split_dir}")
            continue

        images = sorted(
            (f for f in split_dir.glob("Case??.mhd") if "_segmentation" not in f.name),
            key=lambda f: int(re.search(r"(\d+)", f.stem).group(1)),
        )

        rows = []
        for img in images:
            case_id = int(re.search(r"(\d+)", img.stem).group(1))
            seg = split_dir / f"{img.stem}_segmentation.mhd"
            rows.append({
                "ID": case_id,
                "t2": f"{split}/{img.name}",
                "segmentation": f"{split}/{seg.name}" if seg.exists() else "",
            })

        out_csv = dataset_dir / f"{split}.csv"
        with out_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["ID", "t2", "segmentation"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} cases to {out_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path("data"), help="Root folder for raw and temporary data."
    )
    parser.add_argument("--record-id", type=int, default=DEFAULT_RECORD_ID, help="Zenodo record id.")
    parser.add_argument("--no-progress", action="store_true", help="Hide download progress output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = download_dataset(
        args.data_root, record_id=args.record_id, progress=not args.no_progress
    )
    print(f"Downloaded dataset to {dataset_dir}")
    generate_csvs()
    print("Generated CSV files for each split.")


if __name__ == "__main__":
    main()
