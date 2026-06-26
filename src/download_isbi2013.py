"""Download the NCI-ISBI 2013 Prostate Challenge dataset from TCIA.

Downloads segmentation labels (NRRD) and source MR images (DICOM) for all
three splits (training, leaderboard, test), then converts the DICOM series
to NIfTI (.nii.gz) in a sibling images_nii/ directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import zipfile
from pathlib import Path

import requests

SOURCE_NAME = "tcia"
DATASET_NAME = "isbi_mr_prostate_2013"
COMPLETION_SENTINEL = ".complete"
MANIFEST_NAME = "download_manifest.json"
SPLITS = ("training", "leaderboard", "test")
REQUIRED_PATHS = ("labels/training", "images_nii/training", "labels_nii/training")
LABEL_SUBDIR = {"training": "Training", "leaderboard": "Leaderboard", "test": "Test"}

LABEL_URLS = {
    "training":    "https://www.cancerimagingarchive.net/wp-content/uploads/NCI-ISBI-2013-Prostate-Challenge-Training.zip",
    "leaderboard": "https://www.cancerimagingarchive.net/wp-content/uploads/NCI-ISBI-2013-Prostate-Challenge-Leaderboard.zip",
    "test":        "https://www.cancerimagingarchive.net/wp-content/uploads/NCI-ISBI-2013-Prostate-Challenge-Test.zip",
}
TCIA_MANIFEST_URLS = {
    "training":    "https://www.cancerimagingarchive.net/wp-content/uploads/ISBI-Prostate-Challenge-Training.tcia",
    "leaderboard": "https://www.cancerimagingarchive.net/wp-content/uploads/ISBI-Prostate-Challenge-LeaderBoard.tcia",
    "test":        "https://www.cancerimagingarchive.net/wp-content/uploads/ISBI-Prostate-Challenge-Testing.tcia",
}


def dataset_root(data_root: Path) -> Path:
    return Path(data_root) / "raw" / SOURCE_NAME / DATASET_NAME


def is_complete(dataset_dir: Path) -> bool:
    sentinel = dataset_dir / COMPLETION_SENTINEL
    return sentinel.is_file() and all((dataset_dir / p).exists() for p in REQUIRED_PATHS)


def has_required_contents(dataset_dir: Path) -> bool:
    return dataset_dir.is_dir() and all((dataset_dir / p).exists() for p in REQUIRED_PATHS)


def write_manifest(dataset_dir: Path) -> None:
    manifest = {
        "dataset": DATASET_NAME,
        "source": SOURCE_NAME,
        "splits": list(SPLITS),
        "required_paths": list(REQUIRED_PATHS),
    }
    (dataset_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (dataset_dir / COMPLETION_SENTINEL).touch()


def _download_file(url: str, dest: Path, progress: bool) -> None:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                if progress and total:
                    downloaded += len(chunk)
                    pct = downloaded * 100 // total
                    print(f"\r  {dest.name}: {pct}%", end="", flush=True)
        if progress and total:
            print()


def _download_labels(split: str, dataset_dir: Path, tmp_dir: Path, progress: bool) -> None:
    dest_dir = dataset_dir / "labels" / split
    if dest_dir.is_dir() and any(dest_dir.iterdir()):
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = tmp_dir / f"labels_{split}.zip"
    if progress:
        print(f"Downloading {split} labels...")
    _download_file(LABEL_URLS[split], zip_path, progress)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def _download_images(split: str, dataset_dir: Path, tmp_dir: Path, progress: bool) -> None:
    from tcia_utils import nbia

    dest_dir = dataset_dir / "images" / split
    manifest_path = tmp_dir / f"images_{split}.tcia"
    if progress:
        print(f"Downloading {split} TCIA manifest...")
    _download_file(TCIA_MANIFEST_URLS[split], manifest_path, progress)
    if progress:
        print(f"Downloading {split} DICOM images...")
    nbia.downloadSeries(str(manifest_path), input_type="manifest", path=str(dest_dir))


def convert_to_nifti(dataset_dir: Path) -> None:
    """Convert downloaded DICOM series to NIfTI in images_nii/<split>/, named by PatientID."""
    import SimpleITK as sitk

    for split in SPLITS:
        images_dir = dataset_dir / "images" / split
        if not images_dir.is_dir():
            continue
        out_dir = dataset_dir / "images_nii" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for series_dir in sorted(d for d in images_dir.iterdir() if d.is_dir()):
            dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(series_dir))
            if not dicom_names:
                continue
            meta_reader = sitk.ImageFileReader()
            meta_reader.SetFileName(dicom_names[0])
            meta_reader.LoadPrivateTagsOn()
            meta_reader.ReadImageInformation()
            if not meta_reader.HasMetaDataKey("0010|0020"):
                continue
            patient_id = meta_reader.GetMetaData("0010|0020").strip()
            out_path = out_dir / f"{patient_id}.nii.gz"
            if out_path.exists():
                continue
            series_reader = sitk.ImageSeriesReader()
            series_reader.SetFileNames(dicom_names)
            sitk.WriteImage(series_reader.Execute(), str(out_path))
            count += 1
        if count:
            print(f"Converted {count} series to {out_dir}")


def convert_labels_to_nifti(dataset_dir: Path) -> None:
    """Convert NRRD labels to NIfTI in labels_nii/<split>/, named by PatientID."""
    import SimpleITK as sitk

    for split in SPLITS:
        label_dir = dataset_dir / "labels" / split / LABEL_SUBDIR[split]
        if not label_dir.is_dir():
            continue
        out_dir = dataset_dir / "labels_nii" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for nrrd_file in sorted(label_dir.glob("*.nrrd")):
            stem = nrrd_file.stem
            for suffix in ("_truth", "_correctedLabels"):
                stem = stem.replace(suffix, "")
            out_path = out_dir / f"{stem}.nii.gz"
            if out_path.exists():
                continue
            sitk.WriteImage(sitk.ReadImage(str(nrrd_file)), str(out_path))
            count += 1
        if count:
            print(f"Converted {count} labels to {out_dir}")


def convert_labels_to_nifti_whole_gland(dataset_dir: Path) -> None:
    """Binarize multi-label NRRDs (any label > 0) to NIfTI in labels_nii_wg/<split>/."""
    import SimpleITK as sitk

    for split in SPLITS:
        label_dir = dataset_dir / "labels" / split / LABEL_SUBDIR[split]
        if not label_dir.is_dir():
            continue
        out_dir = dataset_dir / "labels_nii_wg" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for nrrd_file in sorted(label_dir.glob("*.nrrd")):
            stem = nrrd_file.stem
            for suffix in ("_truth", "_correctedLabels"):
                stem = stem.replace(suffix, "")
            out_path = out_dir / f"{stem}.nii.gz"
            if out_path.exists():
                continue
            img = sitk.ReadImage(str(nrrd_file))
            wg = sitk.Cast(img > 0, sitk.sitkUInt8)
            sitk.WriteImage(wg, str(out_path))
            count += 1
        if count:
            print(f"Converted {count} whole-gland labels to {out_dir}")


def build_isbi2013_csvs(dataset_dir: Path) -> None:
    """Write per-split CSV indices pairing image, multi-label, and whole-gland NIfTI paths."""
    for split in SPLITS:
        nii_dir = dataset_dir / "images_nii" / split
        if not nii_dir.is_dir():
            continue
        label_nii_dir = dataset_dir / "labels_nii" / split
        label_wg_dir = dataset_dir / "labels_nii_wg" / split
        output_csv = dataset_dir / f"{split}.csv"
        rows = []
        for nii_file in sorted(nii_dir.glob("*.nii.gz")):
            patient_id = nii_file.stem.removesuffix(".nii")
            label_file = label_nii_dir / f"{patient_id}.nii.gz"
            label_wg_file = label_wg_dir / f"{patient_id}.nii.gz"
            assert label_file.exists(), f"Missing label: {label_file}"
            assert label_wg_file.exists(), f"Missing whole-gland label: {label_wg_file}"
            rows.append({
                "ID": patient_id,
                "t2": os.path.relpath(nii_file, dataset_dir),
                "label": os.path.relpath(label_file, dataset_dir),
                "label_wg": os.path.relpath(label_wg_file, dataset_dir),
            })
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["ID", "t2", "label", "label_wg"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {output_csv}")


def download_dataset(data_root: Path, progress: bool = True) -> Path:
    data_root = Path(data_root)
    dataset_dir = dataset_root(data_root)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if not is_complete(dataset_dir):
        tmp_root = data_root / "tmp" / SOURCE_NAME / DATASET_NAME
        tmp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp_dir:
            tmp_path = Path(tmp_dir)
            for split in SPLITS:
                _download_labels(split, dataset_dir, tmp_path, progress)
                _download_images(split, dataset_dir, tmp_path, progress)

    convert_to_nifti(dataset_dir)
    convert_labels_to_nifti(dataset_dir)
    convert_labels_to_nifti_whole_gland(dataset_dir)
    build_isbi2013_csvs(dataset_dir)

    if not has_required_contents(dataset_dir):
        raise RuntimeError(
            f"Expected files not found in {dataset_dir}."
        )

    write_manifest(dataset_dir)
    return dataset_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path("data"),
        help="Root folder for raw and temporary data.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Hide download progress.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = download_dataset(args.data_root, progress=not args.no_progress)
    print(f"Downloaded dataset to {dataset_dir}")


if __name__ == "__main__":
    main()
