"""Download the public PI-CAI dataset from Zenodo and clone picai_labels."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

DEFAULT_RECORD_ID = 6624726
SOURCE_NAME = "zenodo"
DATASET_NAME = "picai"
NUM_FOLDS = 5
IMAGES_SUBDIR = "images"
LABELS_REPO = "https://github.com/DIAGNijmegen/picai_labels"
LABELS_DIR_NAME = "picai_labels"
REQUIRED_PATHS = ("images",)
COMPLETION_SENTINEL = ".complete"
MANIFEST_NAME = "download_manifest.json"


def is_complete(dataset_dir: Path) -> bool:
    sentinel = dataset_dir / COMPLETION_SENTINEL
    return sentinel.is_file() and all((dataset_dir / relative_path).exists() for relative_path in REQUIRED_PATHS)


def has_required_contents(dataset_dir: Path) -> bool:
    return dataset_dir.is_dir() and all((dataset_dir / relative_path).exists() for relative_path in REQUIRED_PATHS)


def dataset_root(data_root: Path, record_id: int) -> Path:
    return Path(data_root) / "raw" / SOURCE_NAME / DATASET_NAME / f"record-{record_id}"


def labels_root(data_root: Path) -> Path:
    return Path(data_root) / "raw" / SOURCE_NAME / DATASET_NAME / LABELS_DIR_NAME


def write_manifest(dataset_dir: Path, record_id: int) -> None:
    manifest = {
        "dataset": DATASET_NAME,
        "source": SOURCE_NAME,
        "record_id": record_id,
        "num_folds": NUM_FOLDS,
        "images_subdir": IMAGES_SUBDIR,
        "required_paths": list(REQUIRED_PATHS),
    }
    (dataset_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (dataset_dir / COMPLETION_SENTINEL).touch()


def download_and_extract_folds(record_id: int, tmp_dir: Path, dataset_dir: Path, progress: bool = True) -> None:
    images_dir = dataset_dir / IMAGES_SUBDIR
    images_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "zenodo_get",
        str(record_id),
        "-o",
        str(tmp_dir),
        "-g",
        "*.zip",
        "-n",
        "-v",
        "2" if progress else "0",
    ]
    subprocess.run(command, check=True)

    archives = sorted(tmp_dir.glob("*.zip"))
    if len(archives) != NUM_FOLDS:
        raise RuntimeError(f"Expected {NUM_FOLDS} zip files, found {len(archives)}.")

    for archive_path in archives:
        with zipfile.ZipFile(archive_path) as zip_file:
            zip_file.extractall(images_dir)


def clone_labels(data_root: Path) -> None:
    target = labels_root(data_root)
    git_dir = target / ".git"

    if git_dir.exists():
        print(f"Labels repo already exists at {target}, skipping clone.")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", LABELS_REPO, str(target)], check=True)


def download_picai(data_root: Path, record_id: int = DEFAULT_RECORD_ID, progress: bool = True) -> Path:
    data_root = Path(data_root)
    record_root = dataset_root(data_root, record_id)
    dataset_dir = record_root

    if is_complete(dataset_dir):
        clone_labels(data_root)
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
        download_and_extract_folds(record_id, Path(tmp_dir), dataset_dir, progress=progress)

    if not has_required_contents(dataset_dir):
        raise RuntimeError(f"Download completed but expected files were not found in {dataset_dir}.")

    write_manifest(dataset_dir, record_id)
    clone_labels(data_root)

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
    dataset_dir = download_picai(args.data_root, record_id=args.record_id, progress=not args.no_progress)
    print(f"Downloaded PI-CAI to {dataset_dir}")


if __name__ == "__main__":
    main()
