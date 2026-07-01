"""Download the PROSTATE-DIAGNOSIS dataset from TCIA.

Downloads DICOM MR images (via TCIA manifest), clinical metadata (XLS),
and NRRD segmentations (multi-component 3D Slicer segmentations for 5
cases, plus NCI-ISBI 2013 Challenge central gland / peripheral zone
segmentations for 30 cases), then converts the DICOM series and NRRD
labels to NIfTI (.nii.gz).

Output layout under dataset_dir:
  images/               raw DICOM (by series)
  images_nii/           MR series as {PatientID}.nii.gz
  labels_nrrd/slicer/   extracted multi-component NRRD segmentations (5 cases)
  labels_nrrd/isbi/     extracted NCI-ISBI challenge NRRD segmentations (30 cases)
  labels_nii/slicer/    slicer NRRD segmentations converted to NIfTI
  labels_nii/isbi/      ISBI challenge NRRD segmentations converted to NIfTI
  metadata/             clinical metadata XLS
  index.csv             one row per subject with an ISBI segmentation available
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path

import requests

SOURCE_NAME = "tcia"
DATASET_NAME = "prostate_diagnosis"
COLLECTION = "PROSTATE-DIAGNOSIS"
COMPLETION_SENTINEL = ".complete"
MANIFEST_NAME = "download_manifest.json"
REQUIRED_PATHS = ("images_nii", "labels_nii/isbi")

TCIA_MANIFEST_URL = "https://www.cancerimagingarchive.net/wp-content/uploads/TCIA_PROSTATE-DIAGNOSIS_06-22-2015.tcia"
METADATA_URL = "https://www.cancerimagingarchive.net/wp-content/uploads/ProstateDiagnosis_metadata-05-07-2012.xlsx"
SLICER_NRRD_URL = "https://www.cancerimagingarchive.net/wp-content/uploads/ProstateDx-NRRD-T2W_TSE_AX-05-07-2012.zip"
ISBI_NRRD_URL = "https://wiki.cancerimagingarchive.net/download/attachments/6882545/NCI_ISBI_Challenge-ProstateDx_Training_Segmentations.zip?version=1&modificationDate=1364323551603&api=v2"


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
        "collection": COLLECTION,
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


def _download_images(dataset_dir: Path, tmp_dir: Path, progress: bool) -> None:
    from tcia_utils import nbia

    dest_dir = dataset_dir / "images"
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_dir / "images.tcia"
    if progress:
        print("Downloading TCIA manifest...")
    _download_file(TCIA_MANIFEST_URL, manifest_path, progress)
    if progress:
        print("Downloading DICOM images...")
    nbia.downloadSeries(str(manifest_path), input_type="manifest", path=str(dest_dir))


def _download_metadata(dataset_dir: Path, progress: bool) -> None:
    dest_dir = dataset_dir / "metadata"
    dest_file = dest_dir / "ProstateDiagnosis_metadata.xlsx"
    if dest_file.is_file():
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    if progress:
        print("Downloading clinical metadata...")
    _download_file(METADATA_URL, dest_file, progress)


def _download_and_extract_nrrd(url: str, dest_dir: Path, tmp_dir: Path, tmp_name: str, progress: bool) -> None:
    if dest_dir.is_dir() and any(dest_dir.iterdir()):
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = tmp_dir / tmp_name
    if progress:
        print(f"Downloading {tmp_name}...")
    _download_file(url, zip_path, progress)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def convert_images_to_nifti(dataset_dir: Path) -> None:
    """Convert downloaded DICOM series to NIfTI in images_nii/, named by PatientID."""
    import SimpleITK as sitk

    images_dir = dataset_dir / "images"
    if not images_dir.is_dir():
        return
    out_dir = dataset_dir / "images_nii"
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


def _convert_nrrd_dir(nrrd_dir: Path, out_dir: Path) -> int:
    import SimpleITK as sitk

    if not nrrd_dir.is_dir():
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for nrrd_file in sorted(nrrd_dir.rglob("*.nrrd")):
        out_path = out_dir / f"{nrrd_file.stem}.nii.gz"
        if out_path.exists():
            continue
        sitk.WriteImage(sitk.ReadImage(str(nrrd_file)), str(out_path))
        count += 1
    return count


def convert_labels_to_nifti(dataset_dir: Path) -> None:
    """Convert extracted NRRD segmentations to NIfTI in labels_nii/{slicer,isbi}/."""
    slicer_count = _convert_nrrd_dir(
        dataset_dir / "labels_nrrd" / "slicer", dataset_dir / "labels_nii" / "slicer"
    )
    if slicer_count:
        print(f"Converted {slicer_count} slicer labels to {dataset_dir / 'labels_nii' / 'slicer'}")

    isbi_count = _convert_nrrd_dir(
        dataset_dir / "labels_nrrd" / "isbi", dataset_dir / "labels_nii" / "isbi"
    )
    if isbi_count:
        print(f"Converted {isbi_count} isbi labels to {dataset_dir / 'labels_nii' / 'isbi'}")


def _case_suffix(stem: str) -> str | None:
    """Extract the 4-digit case number (e.g. '0006') from an NRRD/label stem, if present."""
    match = re.search(r"(\d{4})", stem)
    return match.group(1) if match else None


def build_index_csv(dataset_dir: Path) -> None:
    """Write index.csv pairing image and ISBI-challenge label NIfTI paths by case suffix."""
    images_dir = dataset_dir / "images_nii"
    labels_dir = dataset_dir / "labels_nii" / "isbi"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        return

    label_by_suffix = {}
    for label_file in sorted(labels_dir.glob("*.nii.gz")):
        suffix = _case_suffix(label_file.stem.removesuffix(".nii"))
        if suffix:
            label_by_suffix[suffix] = label_file

    rows = []
    for image_file in sorted(images_dir.glob("*.nii.gz")):
        patient_id = image_file.stem.removesuffix(".nii")
        suffix = _case_suffix(patient_id)
        label_file = label_by_suffix.get(suffix) if suffix else None
        if label_file is None:
            continue
        rows.append({
            "ID": patient_id,
            "t2": os.path.relpath(image_file, dataset_dir),
            "label": os.path.relpath(label_file, dataset_dir),
        })

    output_csv = dataset_dir / "index.csv"
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "t2", "label"])
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
            _download_images(dataset_dir, tmp_path, progress)
            _download_metadata(dataset_dir, progress)
            _download_and_extract_nrrd(
                SLICER_NRRD_URL, dataset_dir / "labels_nrrd" / "slicer", tmp_path,
                "labels_slicer.zip", progress,
            )
            _download_and_extract_nrrd(
                ISBI_NRRD_URL, dataset_dir / "labels_nrrd" / "isbi", tmp_path,
                "labels_isbi.zip", progress,
            )

    convert_images_to_nifti(dataset_dir)
    convert_labels_to_nifti(dataset_dir)
    build_index_csv(dataset_dir)

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

"""
# PROSTATE-DIAGNOSIS | PROSTATE-DIAGNOSIS

DOI: [10.7937/K9/TCIA.2015.FOQEUJVT](https://www.cancerimagingarchive.net/collection/prostate-diagnosis//#citations) | [Data Citation Required](https://www.cancerimagingarchive.net/collection/prostate-diagnosis//#citations) | [4.5k Views](https://commons.datacite.org/doi.org/10.7937/K9/TCIA.2015.FOQEUJVT) | [16 Citations](https://commons.datacite.org/doi.org/10.7937/K9/TCIA.2015.FOQEUJVT) | Image Collection

| Location | Species | Subjects | Data Types | Cancer Types | Size | External Resources | Status | Updated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prostate Gland | Human | 92 | MR, Treatment, Measurement, Classification, Segmentation | Prostate Cancer | 5.67GB | Clinical, Image Analyses | Complete | 2021/08/09 |

Prostate cancer T1- and T2-weighted magnetic resonance images (MRIs) were acquired on a 1.5 T Philips Achieva by combined surface and endorectal coil, including dynamic contrast-enhanced images obtained prior to, during and after I.V. administration of 0.1 mmol/kg body weight of Gadolinium-DTPA (pentetic acid). Corresponding clinical metadata (XLS format) and 3D segmentation files (NRRD format) are offered as a supplement to this image collection.  The XLS file contains pathology biopsy and excised gland tissue reports and the MRI radiology report for most subjects.  

The Multi-component NRRD Segmentations allow visualization and downstream analysis in 3D Slicer of the following prostate components: prostate gland boundary; internal capsule; central gland, peripheral zone; seminal vesicles; urethra; cancer – dominant nodule; neurovascular bundle; penile bulb; ejaculatory duct; veru-montanum; and rectum. See our tutorial on [Using 3D Slicer with the Prostate-Diagnosis data](https://wiki.cancerimagingarchive.net/display/Public/Hands-On+Tutorial%3A+Using+3D+Slicer+with+the+Prostate-Diagnosis+Data) if you are not familiar with using this kind of data.

The Seminal vesicles (SV) and neurovascular bundle (NVB) Segmentations delineate the neurovascular bundle and seminal vessicles as MHA files. These were provided as part of a planned challenge competition that did not materialize.

The Third Party Analysis dataset mentioned beneath the Data Access table was added later as part of the [NCI-ISBI 2013 Challenge - Automated Segmentation of Prostate Structures](https://www.cancerimagingarchive.net/analysis-result/isbi-mr-prostate-2013/). It includes segmentations for 30 Prostate-Diagnosis subjects in NRRD format which mark the boundaries of the central gland and peripheral zone were also provided 

### [Detailed Description](https://www.cancerimagingarchive.net/collection/prostate-diagnosis//#)

#### Metadata

Corresponding clinical metadata (XLS format) and 3D segmentation files (NRRD format) are offered as a supplement to this image collection.

-   [Prostate-Diagnosis metadata](https://www.cancerimagingarchive.net/wp-content/uploads/ProstateDiagnosis_metadata-05-07-2012.xlsx) (updated 2012-05-07) – The XLS file contains pathology biopsy and excised gland tissue reports and the MRI radiology report for most subjects.
-   NRRD 3D segmentations (2 separate sets of segmentations available)
    -   [NRRD segmentations](https://www.cancerimagingarchive.net/wp-content/uploads/ProstateDx-NRRD-T2W_TSE_AX-05-07-2012.zip) (updated 2012-05-07)- The software used to generate the NRRD files on the MR T2W\_TSE\_AX image sequences was [3DSlicer](http://www.slicer.org/). The 3DSlicer NRRD files allow visualization and downstream analysis of the following prostate components: prostate gland boundary; internal capsule; central gland, peripheral zone; seminal vesicles; urethra; cancer – dominant nodule; neurovascular bundle; penile bulb; ejaculatory duct; veru-montanum; and rectum. Presently, there are available mark-ups of 5 cases (case extension #’s 0006, 0014, 0019, 0021, 0048). These markups are made public courtesy (and copyrighted by) Dr. Nicolas Bloch as portions of his forthcoming online prostate cancer image atlas.
    -   [NCI\_ISBI\_Challenge-ProstateDx\_Training\_Segmentations.zip](https://wiki.cancerimagingarchive.net/download/attachments/6882545/NCI_ISBI_Challenge-ProstateDx_Training_Segmentations.zip?version=1&modificationDate=1364323551603&api=v2)– This file contains segmentations for 30 Prostate-Diagnosis subjects in NRRD format which mark the boundaries of the central gland and peripheral zone. This data was provided as part of the [NCI-ISBI 2013 Challenge – Automated Segmentation of Prostate Structures](https://wiki.cancerimagingarchive.net/display/Public/NCI-ISBI+2013+Challenge+-+Automated+Segmentation+of+Prostate+Structures).
    -   [ProstateDx\_1.5T\_Training\_Segmentations.zip](https://www.cancerimagingarchive.net/wp-content/uploads/ProstateDx_1.5T_Training_Segmentations.zip) – Segmentations of the neurovascular bundle and seminal vessicles are available as MHA files. These were provided as part of a planned follow up competition that did not materialize.
    -   **Note:** see our tutorial on [Using 3D Slicer with the Prostate-Diagnosis data](https://wiki.cancerimagingarchive.net/display/Public/Hands-On+Tutorial%3A+Using+3D+Slicer+with+the+Prostate-Diagnosis+Data) if you are not familiar with using this kind of data.

## Data Access

### Version 2: Updated 2021/08/09

A database mismatch in 4 series of PatientID **ProstateDx-01-0035** was updated so that PatientName, PatientID, and the image are now correct. No changes were made to UID, zips or Excel files.

| Title | Data Type | Format | Access Points | Subjects | Studies | Series | Images | License | Metadata |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Images | MR | DICOM | 
[Download (5.67gb)](https://www.cancerimagingarchive.net/wp-content/uploads/TCIA_PROSTATE-DIAGNOSIS_06-22-2015.tcia) [Search](https://nbia.cancerimagingarchive.net/nbia-search/?CollectionCriteria=PROSTATE-DIAGNOSIS)

Download requires [TCIA Data Retriever](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images)

 | 87 | 87 | 348 | 30,903 | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) | [View](https://www.cancerimagingarchive.net/wp-content/uploads/TCIA_PROSTATE-DIAGNOSIS_06-22-2015-nbia-digest.xlsx) |
| Clinical Metadata | Treatment, Measurement, Classification | XLS | 

[Download (59.13kb)](https://www.cancerimagingarchive.net/wp-content/uploads/ProstateDiagnosis_metadata-05-07-2012.xlsx)

 | 54 |  |  |  | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) | — |
| Multi-component NRRD Segmentations | Segmentation | NRRD and ZIP | 

[Download (73.22kb)](https://www.cancerimagingarchive.net/wp-content/uploads/ProstateDx-NRRD-T2W_TSE_AX-05-07-2012.zip)

 |  |  |  |  | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) | — |
| NCI ISBI Challenge - Segmentations of central gland and the peripheral zone | Segmentation | NRRD and ZIP | 

[Download (161.09kb)](https://www.cancerimagingarchive.net/wp-content/uploads/NCI_ISBI_Challenge-ProstateDx_Training_Segmentations.zip)

 |  |  |  |  | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) | — |
| Seminal vesicles (SV) and neurovascular bundle (NVB) Segmentations | Segmentation | MHA and ZIP | 

[Download (86.79kb)](https://www.cancerimagingarchive.net/wp-content/uploads/ProstateDx_1.5T_Training_Segmentations.zip)

 |  |  |  |  | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) | — |

Analysis Results Using This Collection

[

ISBI-MR-Prostate-2013

](https://www.cancerimagingarchive.net/analysis-result/isbi-mr-prostate-2013/)

Related Datasets

[

ISBI-MR-Prostate-2013

](https://www.cancerimagingarchive.net/analysis-result/isbi-mr-prostate-2013/)No related Collections found

Legend: Analysis Results| Collections

## External Resources

The NCI Cancer Research Data Commons (CRDC) provides access to additional data and a cloud-based data science infrastructure that connects data sets with analytics tools to allow users to share, integrate, analyze, and visualize cancer research data.

-   [Imaging Data Commons (IDC)](https://portal.imaging.datacommons.cancer.gov/explore/filters/?collection_id=prostate_diagnosis) (Imaging Data)

## Citations & Data Usage Policy

**Data Citation Required:** Users must abide by the [TCIA Data Usage Policy and Restrictions](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/). Attribution must include the following citation, including the Digital Object Identifier:

<table style="margin:0; border:none; background-color:#fbfbfb; "><tbody><tr><td style="margin: 50% 0; border: 0px; border-collapse: collapse;background-color: #62C6FF;"><i class="fa-solid fa-quote-left" style="font-size: 2em;"></i></td><td style="margin: 50% 0; border: 0px; border-collapse: collapse;"><h4>Data Citation</h4></td></tr><tr><td style="border: 0px; border-collapse: collapse;background-color: #62C6FF;"></td><td style="width: 100%; border: 0px; border-collapse: collapse;"><p>Bloch, B. N., Jain, A., &amp; Jaffe, C. C.. (2015). <strong>Data From PROSTATE-DIAGNOSIS [Dataset].</strong> The Cancer Imaging Archive. <a href="http://doi.org/10.7937/K9/TCIA.2015.FOQEUJVT">https://doi.org/10.7937/K9/TCIA.2015.FOQEUJVT</a></p></td></tr></tbody></table>
"""
