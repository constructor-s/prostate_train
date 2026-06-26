"""Download the QIN-PROSTATE-Repeatability dataset from TCIA.

Downloads all DICOM series (MR, SEG, SR) for 15 subjects × 2 sessions
(test-retest mpMRI) from the QIN-PROSTATE-Repeatability collection, then
converts all MR series to NIfTI and all SEG series to per-segment NIfTI.

Output layout under dataset_dir:
  images/       raw DICOM (flat by SeriesInstanceUID)
  mr_nii/       all MR series as {PatientID}_{StudyDate}_{SeriesDescription}.nii.gz
  seg_nii/      all SEG segments as {PatientID}_{StudyDate}_{MR_desc}_{SegDesc}.nii.gz
  index.csv     one row per visit: ID, t2, t2_wholegland
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np

SOURCE_NAME = "tcia"
DATASET_NAME = "qin_prostate_repeatability"
COLLECTION = "QIN-PROSTATE-Repeatability"
COMPLETION_SENTINEL = ".complete"
MANIFEST_NAME = "download_manifest.json"

DESC_T2 = "T2 Weighted Axial"
DESC_WHOLEGLAND = "WholeGland"


def dataset_root(data_root: Path) -> Path:
    return Path(data_root) / "raw" / SOURCE_NAME / DATASET_NAME


def is_complete(dataset_dir: Path) -> bool:
    sentinel = dataset_dir / COMPLETION_SENTINEL
    images_dir = dataset_dir / "images"
    return sentinel.is_file() and images_dir.is_dir() and any(images_dir.iterdir())


def write_manifest(dataset_dir: Path, n_series: int) -> None:
    manifest = {
        "collection": COLLECTION,
        "dataset": DATASET_NAME,
        "n_series": n_series,
        "source": SOURCE_NAME,
    }
    (dataset_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (dataset_dir / COMPLETION_SENTINEL).touch()


def download_dataset(data_root: Path, progress: bool = True) -> Path:
    from tcia_utils import nbia

    data_root = Path(data_root)
    dataset_dir = dataset_root(data_root)

    if is_complete(dataset_dir):
        if progress:
            print(f"Dataset already downloaded at {dataset_dir}")
        return dataset_dir

    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if progress:
        print(f"Fetching series list for collection: {COLLECTION}")
    series_list = nbia.getSeries(collection=COLLECTION)

    if progress:
        print(f"Downloading {len(series_list)} series...")
    nbia.downloadSeries(series_list, path=str(images_dir))

    write_manifest(dataset_dir, n_series=len(series_list))
    if progress:
        print(f"Downloaded {len(series_list)} series to {images_dir}")

    return dataset_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize(s: str) -> str:
    """Lowercase, replace non-alphanumeric runs with a single underscore."""
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _build_series_index(images_dir: Path) -> list[dict]:
    """Read one DICOM header per series directory and return a list of dicts."""
    import pydicom

    index = []
    for series_dir in sorted(images_dir.iterdir()):
        dcm_files = sorted(f for f in series_dir.iterdir() if f.suffix == ".dcm")
        if not dcm_files:
            continue
        ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
        entry: dict = {
            "series_dir":       series_dir,
            "PatientID":        str(ds.get("PatientID", "")).strip(),
            "StudyInstanceUID": str(ds.get("StudyInstanceUID", "")).strip(),
            "StudyDate":        str(ds.get("StudyDate", "")).strip(),
            "SeriesInstanceUID": str(ds.get("SeriesInstanceUID", "")).strip(),
            "SeriesDescription": str(ds.get("SeriesDescription", "")).strip(),
            "Modality":         str(ds.get("Modality", "")).strip(),
            "n_files":          len(dcm_files),
            "ref_series_uid":   "",
        }
        if entry["Modality"] == "SEG" and hasattr(ds, "ReferencedSeriesSequence"):
            entry["ref_series_uid"] = str(
                ds.ReferencedSeriesSequence[0].SeriesInstanceUID
            ).strip()
        index.append(entry)
    return index


# ---------------------------------------------------------------------------
# MR DICOM → NIfTI
# ---------------------------------------------------------------------------

def _convert_mr_series(series_dir: Path, out_path: Path) -> None:
    import SimpleITK as sitk

    dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(series_dir))
    if not dicom_names:
        return
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_names)
    sitk.WriteImage(reader.Execute(), str(out_path))


def convert_mr_to_nifti(dataset_dir: Path, progress: bool = True) -> None:
    """Convert all MR series to NIfTI in mr_nii/."""
    index = _build_series_index(dataset_dir / "images")
    out_dir = dataset_dir / "mr_nii"
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for entry in index:
        if entry["Modality"] != "MR":
            continue
        if entry["SeriesDescription"] != DESC_T2:
            print(f"  WARN: skipping MR series with non-T2 description: {entry['SeriesDescription']}")
            continue
        stem = f"{entry['PatientID']}_{entry['StudyDate']}_{_sanitize(entry['SeriesDescription'])}"
        out_path = out_dir / f"{stem}.nii.gz"
        if out_path.exists():
            continue
        _convert_mr_series(entry["series_dir"], out_path)
        count += 1
        if progress:
            print(f"  MR → {out_path.name}")

    if progress:
        print(f"Converted {count} MR series → {out_dir}")


# ---------------------------------------------------------------------------
# SEG DICOM → NIfTI  (uses pydicom-seg for reliable coordinate handling)
# ---------------------------------------------------------------------------

def convert_seg_to_nifti(dataset_dir: Path, progress: bool = True) -> None:
    """Convert all SEG series to per-segment NIfTI in seg_nii/."""
    import pydicom
    import pydicom_seg
    import SimpleITK as sitk

    index = _build_series_index(dataset_dir / "images")
    out_dir = dataset_dir / "seg_nii"
    out_dir.mkdir(parents=True, exist_ok=True)
    mr_dir = dataset_dir / "mr_nii"

    uid_to_mr: dict[str, dict] = {
        e["SeriesInstanceUID"]: e for e in index if e["Modality"] == "MR"
    }

    count = 0
    for entry in index:
        if entry["Modality"] != "SEG":
            continue

        mr_entry = uid_to_mr.get(entry["ref_series_uid"])
        if mr_entry is None:
            if progress:
                print(f"  WARN: no MR found for SEG {entry['series_dir'].name}")
            continue

        mr_desc_san = _sanitize(mr_entry["SeriesDescription"])
        patient = entry["PatientID"]
        date = entry["StudyDate"]

        mr_nii = mr_dir / f"{patient}_{date}_{mr_desc_san}.nii.gz"
        if not mr_nii.exists():
            if progress:
                print(f"  WARN: MR NIfTI not found for resampling: {mr_nii.name}")
            continue

        seg_file = next(entry["series_dir"].glob("*.dcm"))
        seg_ds = pydicom.dcmread(str(seg_file))
        result = pydicom_seg.SegmentReader().read(seg_ds)
        desc_map = {
            int(seg.SegmentNumber): getattr(seg, "SegmentDescription", str(seg.SegmentNumber))
            for seg in seg_ds.SegmentSequence
        }

        all_exist = all(
            (out_dir / f"{patient}_{date}_{mr_desc_san}_{_sanitize(desc_map[n])}.nii.gz").exists()
            for n in result.available_segments
        )
        if all_exist:
            continue

        ref = sitk.ReadImage(str(mr_nii))

        for seg_num in result.available_segments:
            seg_desc = desc_map[seg_num]
            out_path = out_dir / f"{patient}_{date}_{mr_desc_san}_{_sanitize(seg_desc)}.nii.gz"
            if out_path.exists():
                continue
            seg_sitk = result.segment_image(seg_num)
            resampled = sitk.Resample(
                seg_sitk, ref, sitk.Transform(),
                sitk.sitkNearestNeighbor, 0, seg_sitk.GetPixelID(),
            )
            sitk.WriteImage(resampled, str(out_path))
            count += 1
            if progress:
                print(f"  SEG → {out_path.name}")

    if progress:
        print(f"Converted {count} SEG segments → {out_dir}")


# ---------------------------------------------------------------------------
# CSV index
# ---------------------------------------------------------------------------

def build_csv(dataset_dir: Path, progress: bool = True) -> None:
    """Write index.csv: one row per visit with ID, t2, t2_wholegland paths."""
    mr_dir = dataset_dir / "mr_nii"
    seg_dir = dataset_dir / "seg_nii"

    t2_san = _sanitize(DESC_T2)
    wg_san = _sanitize(DESC_WHOLEGLAND)

    # Collect all visits from MR NIfTI filenames
    visits: set[tuple[str, str]] = set()
    for f in mr_dir.glob("*.nii.gz"):
        parts = f.stem.removesuffix(".nii").split("_")
        # PatientID is PCAMPMRI-XXXXX (with hyphen), StudyDate is 8 digits
        # Filename: {PatientID}_{StudyDate}_{sanitized_desc}.nii.gz
        # PatientID contains a hyphen so it won't be split by our underscore scheme,
        # but we sanitize desc only. Reconstruct by matching known date pattern.
        for i, part in enumerate(parts):
            if len(part) == 8 and part.isdigit():
                patient = "_".join(parts[:i])
                date = part
                visits.add((patient, date))
                break

    rows = []
    for patient, date in sorted(visits):
        visit_id = f"{patient}_{date}"
        t2_path = mr_dir / f"{patient}_{date}_{t2_san}.nii.gz"
        wg_path = seg_dir / f"{patient}_{date}_{t2_san}_{wg_san}.nii.gz"
        rows.append({
            "ID": visit_id,
            "t2": os.path.relpath(t2_path, dataset_dir) if t2_path.exists() else "",
            "t2_wholegland": os.path.relpath(wg_path, dataset_dir) if wg_path.exists() else "",
        })

    output_csv = dataset_dir / "index.csv"
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "t2", "t2_wholegland"])
        writer.writeheader()
        writer.writerows(rows)
    if progress:
        print(f"Wrote {len(rows)} rows to {output_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path("data"),
        help="Root folder for raw and temporary data.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Hide progress output.")
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download and only run conversion (dataset already present).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    progress = not args.no_progress
    data_root = Path(args.data_root)
    dataset_dir = dataset_root(data_root)

    if not args.skip_download:
        download_dataset(data_root, progress=progress)

    convert_mr_to_nifti(dataset_dir, progress=progress)
    convert_seg_to_nifti(dataset_dir, progress=progress)
    build_csv(dataset_dir, progress=progress)
    print(f"Done. Dataset at {dataset_dir}")


if __name__ == "__main__":
    main()
