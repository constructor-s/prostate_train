"""Download the public Prostate158 dataset from Zenodo."""

from __future__ import annotations

import argparse
import json
import tempfile
import subprocess
import zipfile
import sys
from pathlib import Path

DEFAULT_RECORD_ID = 6481141
DEFAULT_FOLDER = "prostate158_train"
REQUIRED_PATHS = ("train.csv", "valid.csv", "train")
SOURCE_NAME = "zenodo"
DATASET_NAME = "prostate158"
COMPLETION_SENTINEL = ".complete"
MANIFEST_NAME = "download_manifest.json"


def is_complete(dataset_dir: Path) -> bool:
    sentinel = dataset_dir / COMPLETION_SENTINEL
    return sentinel.is_file() and all((dataset_dir / relative_path).exists() for relative_path in REQUIRED_PATHS)


def has_required_contents(dataset_dir: Path) -> bool:
    return dataset_dir.is_dir() and all((dataset_dir / relative_path).exists() for relative_path in REQUIRED_PATHS)


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
    (dataset_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (dataset_dir / COMPLETION_SENTINEL).touch()


def download_prostate158(data_root: Path, record_id: int = DEFAULT_RECORD_ID, progress: bool = True) -> Path:
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
        if len(archives) != 1:
            raise RuntimeError(f"Expected exactly one downloaded zip for record {record_id}, found {len(archives)}.")

        archive_path = archives[0]
        with zipfile.ZipFile(archive_path) as zip_file:
            zip_file.extractall(record_root)

    if not has_required_contents(dataset_dir):
        raise RuntimeError(f"Download completed but expected files were not found in {dataset_dir}.")

    write_manifest(dataset_dir, record_id)

    if not is_complete(dataset_dir):
        raise RuntimeError(f"Download completed but dataset was not marked complete at {dataset_dir}.")

    return dataset_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Root folder for raw and temporary data.")
    parser.add_argument("--record-id", type=int, default=DEFAULT_RECORD_ID, help="Zenodo record id.")
    parser.add_argument("--no-progress", action="store_true", help="Hide download progress output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = download_prostate158(args.data_root, record_id=args.record_id, progress=not args.no_progress)
    print(f"Downloaded Prostate158 to {dataset_dir}")


if __name__ == "__main__":
    main()