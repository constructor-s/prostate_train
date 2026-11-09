# %% [markdown]
# # T2 images with complete clinical measurements
# Filter the existing image listing by workbook clinical availability and the
# reference volume-study exclusion rule. Ground truth is median manual reader
# planimetry in mL (cm³), not reported or model-derived volume. Missing manual
# measurements leave ground truth blank and do not exclude an otherwise eligible
# image. The output also preserves the original reported volume in vol_report_ref,
# separately from the manual ground truth.
# Select one clinically eligible scan per patient using recorded cancer outcome,
# manual ground truth, then earliest date string, in that order.
# Reference: prostate_volume_haider/notebooks/analyze_report_volume_coefficients.ipynb.
# Run cells in order, or: .venv/bin/python notebooks/hlab20260928_clinical_image_listing.py

# %% Imports and explicit input paths
from getpass import getpass
from io import BytesIO
from pathlib import Path

import msoffcrypto
import numpy as np
import pandas as pd
from IPython.display import display

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

required = ["aano", "amrn", "CSCANCER", "adate", *DIMENSIONS, "vol_report_ref", "reportmaxpirads_ref",
            "psa_report_ref", "psa", "excludeforvolumestudy_ref", *READERS]
missing = sorted(set(required) - set(source.columns))
if missing:
    raise ValueError(f"Missing workbook columns: {missing}")
if not source.columns.is_unique:
    raise ValueError("Workbook has duplicate column names")
patient_ids = source["amrn"].astype("string").str.strip()
assert patient_ids.notna().all() and patient_ids.ne("").all(), "Workbook AMRN must be nonempty"
source["aano"] = source["aano"].astype("string").str.strip()
if source["aano"].eq("").any() or source["aano"].isna().any() or source["aano"].duplicated().any():
    raise ValueError("Workbook AANO must be nonempty and unique")
source = source.set_index("aano")

# %% Clinical eligibility and reference manual consensus
dimensions = source[DIMENSIONS].apply(numeric)
reported = numeric(source["vol_report_ref"])
pirads = numeric(source["reportmaxpirads_ref"])
psa_report = numeric(source["psa_report_ref"])
psa = numeric(source["psa"])

# %% Diagnostic: nonblank raw values that numeric() could not parse
# Entire workbook, before eligibility filtering; include manual reader fields.
# Blank/missing inputs are omitted. Parsed zero, negative, and infinite values
# are not parsing failures; their validity is checked separately below.
numeric_parse_failures_rows = []
for column in [*DIMENSIONS, "vol_report_ref", "reportmaxpirads_ref", "psa_report_ref", "psa", *READERS]:
    raw = source[column].astype("string")
    failed = raw.notna() & raw.str.strip().ne("") & numeric(source[column]).isna()
    for value, count in raw.loc[failed].value_counts().items():
        numeric_parse_failures_rows.append({
            "field": column, "raw_value_repr": repr(value), "count": int(count),
        })
numeric_parse_failures = pd.DataFrame(
    numeric_parse_failures_rows, columns=["field", "raw_value_repr", "count"],
)
print("Nonblank values not recognized by numeric() (entire workbook):")
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None):
    display(numeric_parse_failures)
if numeric_parse_failures.empty:
    print("All nonblank numeric inputs were recognized.")

# %% Clinical eligibility checks and reference manual consensus
checks = pd.DataFrame(index=source.index)
checks["dimensions_available"] = np.isfinite(dimensions).all(axis=1) & dimensions.gt(0).all(axis=1)
checks["reported_volume_available"] = np.isfinite(reported) & reported.gt(0)
checks["pirads_available"] = np.isfinite(pirads)
checks["psa_available"] = (
    (np.isfinite(psa_report) & psa_report.gt(0))
    | (np.isfinite(psa) & psa.gt(0))
)
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

# %% [markdown]
# ## Select one clinically eligible scan per patient
# Group by AMRN normalized to trimmed strings; other workbook fields remain unchanged.
# Eligibility requires finite positive reported volume and at least one finite,
# strictly positive PSA field (psa_report_ref OR psa), so those preferences are
# redundant here. First prefer a recorded CSCANCER outcome whose trimmed uppercase
# value starts with Y or N, then available manual ground truth (mL). Apply each
# preference only if at least one remaining scan for that patient satisfies it;
# otherwise keep all their candidates. Never eliminate the patient's last scan.
# Resolve remaining ties using ascending adate strings, assumed valid and sortable
# chronologically, then original image-list order. Dates are not parsed.
# Export exactly one scan per eligible patient, in original image-list order.

# %% Link all clinical fields and identify repeated patients
retained_review = listing.loc[eligible].copy()
retained_review["aano"] = aano.loc[eligible]
retained_review["ground_truth_volume_ml"] = retained_review["aano"].map(ground_truth)
retained_review = retained_review.merge(
    source.reset_index(), on="aano", how="left", sort=False,
    validate="many_to_one", suffixes=("", "_workbook"),
)
retained_review["amrn"] = retained_review["amrn"].astype("string").str.strip()
assert retained_review["amrn"].notna().all() and retained_review["amrn"].ne("").all()
patient_scan_counts = retained_review.groupby("amrn")["aano"].nunique()
retained_review["patient_scan_count"] = (
    retained_review["amrn"].map(patient_scan_counts).astype("Int64")
)
retained_review["image_row_order"] = np.arange(len(retained_review))

# %% Apply preferences within each patient and document candidate flow
selection_audit = retained_review.copy()
cscancer = selection_audit["CSCANCER"].astype("string").str.strip().str.upper().fillna("")
assert (cscancer.eq("") | cscancer.str.startswith(("Y", "N"))).all(), (
    "CSCANCER must be empty or start with Y or N"
)
selection_audit["has_cscancer"] = cscancer.str.startswith(("Y", "N"))
selection_audit["has_ground_truth"] = selection_audit["ground_truth_volume_ml"].notna()
selection_audit["date_sort_key"] = selection_audit["adate"].astype("string")
selection_audit["selected"] = True
selection_audit["elimination_stage"] = ""
selection_flow_rows = []
for stage, flag in [
    ("Eligible cohort", None),
    ("Prefer recorded CSCANCER", "has_cscancer"),
    ("Prefer manual ground truth", "has_ground_truth"),
    ("Earliest date string, then original image order", None),
]:
    candidates = selection_audit.loc[selection_audit["selected"]]
    before = len(candidates)
    if stage == "Eligible cohort":
        # Record the starting cohort without dropping any scans.
        dropped = candidates.index[:0]
    elif flag is not None:
        # For each patient, ask whether ANY remaining scan satisfies this preference.
        # transform broadcasts that patient-level answer back to every candidate row.
        patient_has_preferred = candidates.groupby("amrn")[flag].transform("any")
        # Drop a scan only if it fails the preference AND that patient has a scan
        # that passes. If none pass, keep all candidates so the patient is retained.
        dropped = candidates.index[patient_has_preferred & ~candidates[flag]]
    elif stage == "Earliest date string, then original image order":
        ordered = candidates.sort_values(
            ["date_sort_key", "image_row_order"], kind="stable",
        )
        dropped = ordered.index[ordered["amrn"].duplicated(keep="first")]
    else:
        raise AssertionError(f"Unknown selection stage: {stage!r}")
    selection_audit.loc[dropped, "selected"] = False
    selection_audit.loc[dropped, "elimination_stage"] = stage
    remaining = selection_audit.loc[selection_audit["selected"]]
    remaining_counts = remaining.groupby("amrn").size()
    assert set(remaining_counts.index) == set(patient_scan_counts.index), "Selection lost a patient"
    selection_flow_rows.append({
        "stage": stage, "candidate_scans_before": before,
        "candidate_scans_after": len(remaining), "scans_removed": before - len(remaining),
        "patients_remaining": len(remaining_counts),
        "patients_with_multiple_candidates": int(remaining_counts.gt(1).sum()),
    })

selection_flow = pd.DataFrame(selection_flow_rows)
selected_scans = selection_audit.loc[selection_audit["selected"]]
assert selected_scans.groupby("amrn").size().eq(1).all()
assert len(selected_scans) == len(patient_scan_counts)
selected_ids = selected_scans["ID"]
output = output.loc[output["ID"].isin(selected_ids)].copy()
assert len(output) == len(selected_scans)
assert list(output.columns) == ["ID", "t2", "ground_truth_volume_ml", "vol_report_ref"]
assert output["ID"].tolist() == selected_scans["ID"].tolist()

# %% Display the flow and candidates unresolved before the date tie-break
review_columns = [
    "patient_scan_count", "amrn", "aano", "selected",
    "elimination_stage", "has_cscancer", "has_ground_truth",
]
review_columns += [c for c in selection_audit.columns if c not in review_columns]
# Reconstruct candidates surviving both availability preferences. Include the
# eventual winner and date-eliminated scans, but omit earlier eliminated scans.
date_candidates = selection_audit.loc[
    selection_audit["selected"]
    | selection_audit["elimination_stage"].eq("Earliest date string, then original image order")
]
unresolved_patient_scans = date_candidates.loc[
    date_candidates.groupby("amrn")["ID"].transform("size").gt(1), review_columns,
].sort_values(["amrn", "date_sort_key", "image_row_order"], kind="stable")
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None,
    "display.max_colwidth", None, "display.expand_frame_repr", False,
):
    print("Selection flow (candidate scan counts refer to image rows):")
    display(selection_flow)
    print("Candidates unresolved after CSCANCER and ground-truth preferences (date tie-break shown):")
    display(unresolved_patient_scans)
    if unresolved_patient_scans.empty:
        print("No patients required the date tie-break.")
print(f"Selected {len(output)} scans for {len(selected_scans)} eligible patients")

# %% Export; original inputs remain unchanged
output.to_csv(OUTPUT, index=False)
print(f"Saved {len(output)} rows to {OUTPUT}")
