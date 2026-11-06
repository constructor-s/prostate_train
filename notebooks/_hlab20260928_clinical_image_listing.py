# %% [markdown]
# # T2 images with complete clinical measurements
# Filter the existing image listing by workbook clinical availability and the
# reference volume-study exclusion rule. Ground truth is median manual reader
# planimetry in mL (cm³), not reported or model-derived volume. Missing manual
# measurements leave ground truth blank and do not exclude an otherwise eligible
# image. The output also preserves the original reported volume in vol_report_ref,
# separately from the manual ground truth.
# Reference: prostate_volume_haider/notebooks/analyze_report_volume_coefficients.ipynb.
# Run cells in order, or: .venv/bin/python notebooks/hlab20260928_clinical_image_listing.py

# %% Imports and explicit input paths
from getpass import getpass
from io import BytesIO
from pathlib import Path

import msoffcrypto
import numpy as np
import pandas as pd

ROOT = (
    Path(__file__).resolve().parents[1]
    if "__file__" in globals()
    else next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "pyproject.toml").is_file())
)
DATA = ROOT / "data_internal/hlab20260928"
WORKBOOK = DATA / "PROSATELLMVOL20260703-12-WorkingBillShi.xlsx"
LISTING = DATA / "hlab20260928_t2_image_listing.csv"
OUTPUT = DATA / "hlab20260928_t2_image_listing_full_clinical.csv"
DIMENSIONS = ["dim1_report_ref", "dim2_report_ref", "dim3_report_ref"]
READERS = [
    "p_axt2_ch_vol", "p_axt2_ah_vol", "p_axt2_dd_vol", "p_axt2_ao_vol",
    "p_axt2_chepg_vol", "p_axt2_epgch_vol", "p_axt2_epg_vol",
]


def numeric(values):
    return pd.to_numeric(
        values.astype("string").str.replace(",", ".", regex=False)
        .str.replace(">", "", regex=False), errors="coerce",
    ).astype(float)


# %% Decrypt in memory; never write the password or decrypted workbook
with WORKBOOK.open("rb") as stream:
    office = msoffcrypto.OfficeFile(stream)
    if office.is_encrypted():
        password = getpass("Excel password: ")
        try:
            office.load_key(password=password, verify_password=True)
            decrypted = BytesIO()
            office.decrypt(decrypted)
        finally:
            del password
        decrypted.seek(0)
        source = pd.read_excel(decrypted, engine="openpyxl", dtype=object, keep_default_na=False)
        decrypted.close()
    else:
        source = pd.read_excel(WORKBOOK, engine="openpyxl", dtype=object, keep_default_na=False)

required = ["aano", *DIMENSIONS, "vol_report_ref", "reportmaxpirads_ref",
            "psa_report_ref", "psa", "excludeforvolumestudy_ref", *READERS]
missing = sorted(set(required) - set(source.columns))
if missing:
    raise ValueError(f"Missing workbook columns: {missing}")
if not source.columns.is_unique:
    raise ValueError("Workbook has duplicate column names")
source["aano"] = source["aano"].astype("string").str.strip()
if source["aano"].eq("").any() or source["aano"].isna().any() or source["aano"].duplicated().any():
    raise ValueError("Workbook AANO must be nonempty and unique")
source = source.set_index("aano")

# %% Clinical eligibility and reference manual consensus
dimensions = source[DIMENSIONS].apply(numeric)
reported = numeric(source["vol_report_ref"])
pirads = numeric(source["reportmaxpirads_ref"])
psa = numeric(source["psa_report_ref"]).fillna(numeric(source["psa"]))
checks = pd.DataFrame(index=source.index)
checks["dimensions_available"] = np.isfinite(dimensions).all(axis=1) & dimensions.gt(0).all(axis=1)
checks["reported_volume_available"] = np.isfinite(reported) & reported.gt(0)
checks["pirads_available"] = np.isfinite(pirads)
checks["psa_available"] = np.isfinite(psa) & psa.ge(0)
checks["study_allowed"] = source["excludeforvolumestudy_ref"].astype("string").str.strip().fillna("").isin(["", "Y-VOLSEGTRAIN"])

manual = source[READERS].apply(numeric)
# Match the reference correction: CH zero denotes an unavailable measurement.
manual.loc[manual["p_axt2_ch_vol"].eq(0), "p_axt2_ch_vol"] = np.nan
if (manual.notna() & (~np.isfinite(manual) | manual.le(0))).any().any():
    raise ValueError("Nonpositive or nonfinite manual reader volumes require review")
ground_truth = manual.median(axis=1).rename("ground_truth_volume_ml")

# %% Match examination suffixes and preserve original image order and paths
listing = pd.read_csv(LISTING, dtype=str, keep_default_na=False)
if list(listing.columns) != ["ID", "t2"] or listing["ID"].duplicated().any():
    raise ValueError("Expected unique image IDs and exactly ID,t2 columns")
aano = listing["ID"].str.rsplit("_", n=1).str[-1]
matched = aano.isin(source.index)
image_checks = checks.reindex(aano).set_axis(listing.index).fillna(False).astype(bool)
eligible = matched & image_checks.all(axis=1)
output = listing.loc[eligible].copy()
output["ground_truth_volume_ml"] = aano.loc[eligible].map(ground_truth)
output["vol_report_ref"] = aano.loc[eligible].map(source["vol_report_ref"])
missing_images = [p for p in output["t2"] if not (DATA / p).is_file()]
if missing_images:
    raise FileNotFoundError(f"{len(missing_images)} retained T2 paths do not exist")
assert output[["ID", "t2"]].equals(listing.loc[eligible])
assert output["ground_truth_volume_ml"].dropna().gt(0).all()

print(pd.Series({
    "input_images": len(listing), "matched": int(matched.sum()),
    "unmatched": int((~matched).sum()),
    "clinically_incomplete_matched": int((matched & ~image_checks.drop(columns="study_allowed").all(axis=1)).sum()),
    "study_excluded_matched": int((matched & ~image_checks["study_allowed"]).sum()),
    "retained": len(output),
    "retained_with_manual_ground_truth": int(output["ground_truth_volume_ml"].notna().sum()),
}))
print("Failed checks among matched images (counts may overlap):")
print((~image_checks.loc[matched]).sum())

# %% Export; original inputs remain unchanged
output.to_csv(OUTPUT, index=False)
print(f"Saved {len(output)} rows to {OUTPUT}")
