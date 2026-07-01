"""Download the public PI-CAI dataset from Zenodo and clone picai_labels."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split

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


def build_picai_infer_csv(
    images_dir: str | Path,
    output_csv: str | Path,
    label_dirs: list[str | Path] | None = None,
) -> None:
    """Write a CSV index for all PICAI t2w images, optionally including label paths.

    Scans *images_dir* for patient subfolders and finds the ``*_t2w.mha`` file
    in each. Writes *output_csv* with paths relative to the CSV's parent directory.

    Column names for label directories are derived from the last two path parts
    joined by ``_`` and lowercased (e.g. ``AI/Bosma22b`` → ``ai_bosma22b``).
    If no matching file exists for a patient in a label directory, that cell is
    left empty.

    Args:
        images_dir: Directory of per-patient subfolders (e.g. ``record-6624726/images``).
        output_csv: Destination CSV path; all paths are relative to its parent.
        label_dirs: Optional list of label directories to include as extra columns.
    """
    images_dir = Path(images_dir).resolve()
    output_csv = Path(output_csv).resolve()
    csv_root = output_csv.parent

    label_cols = {}
    for d in (label_dirs or []):
        p = Path(d).resolve()
        col = "_".join(p.parts[-3:]).lower() # e.g. "picai_labels/anatomical_delineations/whole_gland/AI/Bosma22b" → "whole_gland_ai_bosma22b"
        label_cols[col] = p

    fieldnames = ["ID", "t2"] + list(label_cols)

    rows = []
    for t2w_file in sorted(images_dir.rglob("*_t2w.*")):
        # Get the part of the filename before _t2w
        id_ = t2w_file.stem.rsplit("_t2w", maxsplit=1)[0]
        row: dict[str, str] = {"ID": id_, "t2": os.path.relpath(t2w_file, csv_root)}
        for col, label_dir in label_cols.items():
            label_file = label_dir / f"{id_}.nii.gz"
            assert label_file.exists(), f"Expected label file {label_file} does not exist."
            row[col] = os.path.relpath(label_file, csv_root)
        rows.append(row)            

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


_ZONAL_SOURCES = {"HeviAI23": "zonal_pz_tz/AI/HeviAI23", "Yuan23": "zonal_pz_tz/AI/Yuan23"}
_ALL_SOURCES = ["Bosma22b", "Guerbet23", "HeviAI23", "Yuan23"]


def _convert_zonal_labels(delineations_root: Path) -> None:
    """Binarize zonal (PZ+TZ) labels into whole-gland labels; skip existing files."""
    for source, rel in _ZONAL_SOURCES.items():
        src_dir = delineations_root / rel
        dst_dir = delineations_root / "whole_gland" / "AI" / source
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src_file in sorted(src_dir.glob("*.nii.gz")):
            dst_file = dst_dir / src_file.name
            if dst_file.exists():
                continue
            img = nib.load(src_file)
            data = np.asarray(img.dataobj)
            binarized = nib.Nifti1Image((data > 0).astype(np.uint8), img.affine, img.header)
            nib.save(binarized, dst_file)
    print("Zonal label conversion complete.")


def generate_picai_folds(
    data_root: Path,
    csv_path: Path,
    output_json: Path,
    seed: int = 42,
    test_fraction: float = 0.2,
) -> None:
    """Generate configs/picai_folds.json from images_nii.csv.

    Randomly assigns one of 4 silver-standard raters per case, converts zonal
    labels to whole-gland binarizations first, then writes an 80/20 train/test
    JSON split compatible with picai_input.yaml.
    """
    data_root = Path(data_root)
    delineations_root = data_root / "picai_labels" / "anatomical_delineations"

    _convert_zonal_labels(delineations_root)

    df_rows = list(csv.DictReader(open(csv_path)))
    rng = np.random.default_rng(seed)
    chosen_sources = rng.choice(_ALL_SOURCES, size=len(df_rows))

    record_prefix = csv_path.parent.name  # "record-6624726"
    entries = []
    for row, source in zip(df_rows, chosen_sources):
        case_id = row["ID"]
        entries.append({
            "image": f"{record_prefix}/{row['t2']}",
            "label": f"picai_labels/anatomical_delineations/whole_gland/AI/{source}/{case_id}.nii.gz",
            "id": f"picai_{case_id}",
        })

    training, testing = train_test_split(entries, test_size=test_fraction, random_state=seed)
    output_json = Path(output_json)
    output_json.write_text(json.dumps({"training": training, "testing": testing}, indent=4))
    print(f"Wrote {len(training)} training and {len(testing)} testing entries to {output_json}")


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
