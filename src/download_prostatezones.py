"""Download ProstateZones NRRD annotations and clone the companion repository."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

DEFAULT_RECORD_ID = 10718469
SOURCE_NAME = "zenodo"
DATASET_NAME = "prostatezones"
COMPANION_REPO = "https://github.com/UMU-DDI/ProstateZones.git"
COMPANION_DIR_NAME = "ProstateZones"
ARCHIVE_NAME = "ProstateZones.zip"
COMPLETION_SENTINEL = ".complete"
MANIFEST_NAME = "download_manifest.json"
REQUIRED_PATHS = ("Singles", "Duplicates/R1", "Duplicates/R2", "README.txt", "summary_cspca.xlsx")


def dataset_root(data_root: Path, record_id: int) -> Path:
    return Path(data_root) / "raw" / SOURCE_NAME / DATASET_NAME / f"record-{record_id}"


def clone_companion(data_root: Path, progress: bool = True) -> None:
    target = Path(data_root) / "raw" / SOURCE_NAME / DATASET_NAME / COMPANION_DIR_NAME
    if (target / ".git").exists():
        if progress:
            print(f"Companion repo already exists at {target}, skipping clone.")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    command = ["git", "clone", "--depth", "1"]
    if not progress:
        command.append("--quiet")
    subprocess.run([*command, COMPANION_REPO, str(target)], check=True)


def has_required_contents(dataset_dir: Path) -> bool:
    """Check mask counts, unique identities, and matching duplicate readers."""
    if not all((dataset_dir / name).is_dir() for name in REQUIRED_PATHS[:3]):
        return False
    if not all((dataset_dir / name).is_file() for name in REQUIRED_PATHS[3:]):
        return False
    patient_sets = []
    for folder, suffix, expected in (
        ("Singles", "", 160), ("Duplicates/R1", "_R1", 40), ("Duplicates/R2", "_R2", 40),
    ):
        masks = sorted((dataset_dir / folder).glob("*.nrrd"))
        if len(masks) != expected:
            return False
        patients = set()
        for mask in masks:
            match = re.fullmatch(rf"Seg-(\d{{4}}){suffix}\.nrrd", mask.name)
            if match is None or not mask.is_file() or mask.stat().st_size == 0:
                return False
            patients.add(match.group(1))
        if len(patients) != expected:
            return False
        patient_sets.append(patients)
    singles, reader1, reader2 = patient_sets
    return reader1 == reader2 and singles.isdisjoint(reader1) and len(singles | reader1) == 200


def is_complete(dataset_dir: Path) -> bool:
    return (
        (dataset_dir / COMPLETION_SENTINEL).is_file()
        and (dataset_dir / MANIFEST_NAME).is_file()
        and has_required_contents(dataset_dir)
    )


def write_manifest(dataset_dir: Path, record_id: int) -> None:
    manifest = {
        "dataset": DATASET_NAME,
        "source": SOURCE_NAME,
        "record_id": record_id,
        "archive": ARCHIVE_NAME,
        "required_paths": list(REQUIRED_PATHS),
        "single_reader_masks": 160,
        "duplicate_masks_per_reader": 40,
        "unique_patients": 200,
        "total_masks": 240,
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
    if is_complete(record_root):
        if progress:
            print(f"Already complete; skipping download: {record_root}")
        clone_companion(data_root, progress=progress)
        return record_root
    if record_root.exists():
        raise RuntimeError(
            f"Found an existing but incomplete dataset folder at {record_root}. "
            "Delete it or finish the extraction before retrying."
        )

    tmp_root = data_root / "tmp" / SOURCE_NAME / DATASET_NAME / f"record-{record_id}"
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp_dir:
        subprocess.run(
            [sys.executable, "-m", "zenodo_get", str(record_id), "-o", tmp_dir,
             "-g", ARCHIVE_NAME, "-n", "-v", "2" if progress else "0"],
            check=True,
        )
        archive = Path(tmp_dir) / ARCHIVE_NAME
        if not archive.is_file():
            raise RuntimeError(f"Expected archive not downloaded: {ARCHIVE_NAME}")
        with zipfile.ZipFile(archive) as zf:
            # Reject archive paths that would escape the dataset directory.
            for member in zf.infolist():
                if not (record_root / member.filename).resolve().is_relative_to(record_root.resolve()):
                    raise RuntimeError(f"Archive member outside dataset: {member.filename}")
            record_root.mkdir(parents=True, exist_ok=False)
            zf.extractall(record_root)

    if not has_required_contents(record_root):
        raise RuntimeError(
            f"Invalid ProstateZones contents at {record_root}: expected 160 single masks, "
            "40 matching masks per duplicate reader, and 200 unique patients."
        )
    write_manifest(record_root, record_id)
    clone_companion(data_root, progress=progress)
    return record_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"),
                        help="Root folder for raw and temporary data.")
    parser.add_argument("--record-id", type=int, default=DEFAULT_RECORD_ID, help="Zenodo record id.")
    parser.add_argument("--no-progress", action="store_true", help="Hide download progress output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = download_dataset(args.data_root, record_id=args.record_id, progress=not args.no_progress)
    print(f"Dataset available at {dataset_dir}")


if __name__ == "__main__":
    main()
