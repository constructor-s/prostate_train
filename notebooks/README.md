# Scientific notebooks

Run the Python cell notebooks from this directory (`notebooks/`); the repository
root is assumed to be its parent.

## Toronto JDMI manual ellipsoid failure analysis

[toronto_jdmi_manual_ellipsoid_failure.py](toronto_jdmi_manual_ellipsoid_failure.py)
is a focused follow-on to `toronto_jdmi_clinical_volume.py`. Run that notebook
first; this analysis reads its `report_volume_quality_audit.csv`,
`comparator_volumes.csv`, and `clinical_fields_audit.csv` outputs. It uses the
existing Below range, Above range, and Missing coefficient classifications to
compare flagged and within-range cases against median manual-reader planimetry
and the saved Multicenter and RUMC volumes. It also summarizes flagged-case
PSAD stratum changes and PI-RADS 3 crossings of 0.15 ng/mL². These flags are
review categories, not adjudicated errors, and the saved segmentation volumes
are comparators rather than independent truth.

Run cells in order, or after the source notebook from the repository root:

```bash
MPLBACKEND=Agg .venv/bin/python notebooks/toronto_jdmi_manual_ellipsoid_failure.py
```

Outputs are written to the Git-ignored
`results/toronto_jdmi_manual_ellipsoid_failure/`. Identifiers are retained only
in local case-level audit and review CSVs; displayed results contain aggregate
tables and an identifier-free flagged-case table.

## Toronto JDMI volume agreement

[toronto_jdmi_clinical_volume.py](toronto_jdmi_clinical_volume.py) compares clinical
manual ellipsoid, Multicenter Dataset274, and RUMC Dataset236 volumes against
median manual-reader planimetry. The cleaned cohort contains **383 scans from
383 patients**, after accession exclusions and requiring positive finite manual
reference volumes. Patient selection is performed upstream by
`_hlab20260928_clinical_image_listing.py`; the analysis preserves that selection.

Required private inputs under `data_internal/hlab20260928/` are
`hlab20260928_t2_image_listing_full_clinical.csv`, `exclude.json`, the listed T2
images, `PROSATELLMVOL20260703-12-WorkingBillShi.xlsx`, and saved masks plus dataset metadata in
`seg_out_Dataset274_raw_3d_fullres_5fold/` and
`seg_out_Dataset236_raw_3d_fullres_5fold/`. Supply the Excel password at runtime;
the workbook is decrypted in memory. No downloads, training, or inference occurs.
Run cells in order, or from the repository root:

```bash
MPLBACKEND=Agg .venv/bin/python notebooks/toronto_jdmi_clinical_volume.py
```

The notebook measures label-1 whole-gland volume on native grids in mL and
requires prediction/T2 corner displacement <= 0.5 mm. Empty predictions retain
zero volume; missing or invalid measurements stop analysis after a local audit.
It reproduces the PROSTATEx volume summaries, agreement plots, paired error
comparisons, and paired patient-level inference: 100,000 resamples, seed
20261102, BCa intervals, null-centered bootstrap-t RMSE tests, exact sign tests,
and Holm correction across the six primary tests. The training-derived JDMI
model is outside this version.

IDs have the structure `AIPR_{amrn}_{date}_{aano}`: `amrn` identifies the patient
and `aano` identifies the examination/accession. Workbook linkage uses trimmed
`aano` and verifies `amrn`, preserving selected scans and their order. The seven
manual-reader measurements reconstruct the listing reference (CH zero is missing).
Report dimensions in cm provide a separate π/6 ellipsoid diagnostic in mL.

The final analysis section, immediately before the reproducibility record,
checks clinical report quality by comparing that ellipsoid with `vol_report_ref`
using an inclusive range from `0.5 × (d1−0.1) × (d2−0.1) × (d3−0.1)` to
`π/6 × (d1+0.1) × (d2+0.1) × (d3+0.1)`, with dimensions in cm and volumes in mL.
This permits ±0.1 cm dimension uncertainty and coefficient choice; no extra
reported-volume rounding is allowed. An assertion requires every selected case
to have positive finite uncensored dimensions and valid calculated volumes.
The Bland–Altman plot distinguishes volumes below/above the
range by color and marker, retaining all selected cases in the tabulated bias
and descriptive limits of agreement. The plot shows only the zero-difference
line. It displays a log x-axis from 10–300 mL and
differences from −40 to +40 mL, annotating the number and percentage of selected
points outside the view. Clipped points remain in the statistics and audit.
Flags warrant review and do not exclude cases.
Outputs are `report_volume_quality_audit.csv` and
`report_volume_quality.png`/`.pdf` in the same local results directory.
After the plots, `report_volume_quality_table` displays every Below range,
Above range, and Missing coefficient case in cohort order, without identifiers,
and exports the same columns to `report_volume_quality_table.csv` without an index.
Columns are parsed `dim1`, `dim2`, `dim3` (cm), `recalculated_volume_ml`,
`recalculated_psad`, `report_volume_ml`, `report_psad`,
`manual_planimetry_volume_ml`, `manual_planimetry_psad`, `PIRADS`, and `CSCANCER`.
Recalculated volume uses π/6 × dim1 × dim2 × dim3; manual planimetry is the median
manual-reader reference. Each PSAD uses the same consolidated PSA divided by its
corresponding volume, in ng/mL². Missing PI-RADS and unknown outcomes remain in
the table. Display dimensions and volumes to two decimals and PSAD to three;
the dataframe and CSV retain full calculation precision. No flags yields an empty
table and a header-only CSV.
Immediately after the table, a printed summary compares reported with recalculated
PSAD among these flagged cases: changes across the four categories [0, 0.10),
[0.10, 0.15), [0.15, 0.20), and ≥0.20, plus PI-RADS 3 cases crossing 0.15 ng/mL².
Counts include both directions and their denominators. Comparisons use unrounded
PSAD, with equality at 0.15 on the ≥ side; missing PI-RADS is outside the PI-RADS 3 count.

PSA parsing follows the Haider analysis (decimal commas and `>` removal, with
censoring flagged). Preserve raw fields and apply its documented single-record
alternative-PSA correction to an analytical copy. Use positive finite
`psa_report_ref`, otherwise positive finite corrected `psa`; every selected case
must receive usable PSA. Calculate reference and comparator PSAD using the same
PSA divided by volume in mL, yielding ng/mL². Invalid density denominators stop
the density section after an audit, without removing zero predictions from
volume results. PI-RADS (`reportmaxpirads_ref`) and cancer outcome (`CSCANCER`)
are normalized for biopsy analysis; unavailable outcomes do not exclude
PSAD cases. Workbook key/linkage, reference, and PSA failures stop after audits.

All outputs stay in Git-ignored `results/toronto_jdmi_clinical_volume/`:
`cohort_audit.csv`, `cohort_flow.csv`, `measurement_audit.csv`,
`comparator_volumes.csv`, `comparator_summary.csv`,
`comparator_pairwise_per_case.csv`, `comparator_pairwise_counts.csv`,
`comparator_pairwise_statistics.csv`, and `comparator_agreement.png`/`.pdf`.
`run_metadata.json` records input SHA-256 checksums, saved model metadata,
parameters, package versions, and the notebook checksum. It also pins cohort
membership for subsequent runs; review and archive the prior output directory
before intentionally establishing a changed cohort. Patient-level audits and
metadata are private and must remain local; displayed tables and plots contain
aggregates plus the final individual-case review table, without patient identifiers.
Historical inference settings
are recorded only where saved metadata is available.

Additional local outputs are `workbook_key_audit.csv`, `clinical_fields_audit.csv`,
`psa_density_cohort_audit.csv`, and `psa_density_per_case.csv`. Metadata includes
the encrypted workbook checksum, PSA rule/source counts, correction and censoring
counts, and outcome availability. Passwords and decrypted workbooks are not saved.

The PSAD section reproduces PROSTATEx continuous summaries (bias, MAE, RMSE,
RMSPE, MAPE, Pearson correlation and absolute/relative limits of agreement),
logarithmic agreement plots, and empirical difference histograms. Reference PSAD
uses median manual-reader planimetry. Unrounded strata are [0, 0.10),
[0.10, 0.15), [0.15, 0.20), and [0.20, ∞); threshold comparisons use ≥0.10,
≥0.15 and ≥0.20. Confusion rows are reference strata, columns estimated strata.
Paired patient-level PSAD inference uses the same 100,000-resample settings as
volume inference. RMSE and closer tests share a Holm family; RMSPE and exact
McNemar stratum-accuracy tests have separate families. Sensitivity t-tests use
analogous separate families. Confidence intervals are pointwise, not adjusted.

Biopsy simulation selects normalized PI-RADS 3 only and recommends biopsy at
unrounded PSAD ≥0.15 ng/mL² for planimetry and each selected comparator.
`CSCANCER` Y/N supplies positive/negative outcomes; unknown outcomes are excluded
only here. No ISUP grades are inferred and there is no fallback to the full
cohort. Summaries report confusion counts, sensitivity, specificity, PPV, NPV,
accuracy, recommended/avoided biopsies, and positive outcomes detected/missed.
Confusion rows are CSCANCER outcomes and columns simulated decisions. This is a
retrospective threshold simulation, not observed biopsy management. Undefined
ratios are NaN; empty subgroups refresh zero-count tables and figures.

Additional `psa_density_` outputs are `summary.csv`, `agreement.png`/`.pdf`,
`confusion.csv`/`.png`/`.pdf`, `classification_summary.csv`, `per_stratum.csv`,
`threshold_statistics.csv`, `stratum_pairwise_statistics.csv`,
`pairwise_per_case.csv`, `pairwise_counts.csv`, and `pairwise_statistics.csv`.
Biopsy outputs use `psa_density_biopsy_` followed by `audit.csv`, `per_case.csv`,
`summary.csv`, and `confusion.csv`/`.png`/`.pdf`. Single-method runs refresh empty
pairwise outputs. Run metadata records density cutoffs, statistical families,
biopsy definitions and counts. Reported-marksheet-PSAD diagnostics are omitted.

## Toronto JDMI full cohort: 4,111 patients

[toronto_jdmi_4111_clinical_volume.py](toronto_jdmi_4111_clinical_volume.py)
replicates the volume, PSA-density, paired-inference, PI-RADS 3 biopsy simulation,
and report-quality analyses above using the **nnUNet702 automated surrogate
reference**. It measures agreement with saved automated masks rather than
accuracy against independent manual annotations. The existing 383-patient
manual-reference notebook remains a separate study.

Start from all **4,116 patients** in the same full clinical CSV and apply every
accession exclusion in `exclude.json`, leaving **4,111 patients**. Preserve the
upstream one-examination-per-patient selection and listing order; do not require
manual-reader volumes or substitute another visit. Available manual values are
checked against the workbook for provenance and never supply the analysis
reference.

In addition to the original notebook's workbook, T2 images, comparator predictions,
and comparator metadata, require one mask per eligible patient at
`data_internal/hlab20260928/PROSATEPSAD/{ID}/wp-pz-tz_axt2_nnunet-702.nii.gz`.
Validate finite integer labels drawn from **0, 1, 2, 3**. Whole gland is the
**union of labels 1, 2, 3**: count foreground voxels, multiply by the absolute
determinant of the spatial affine in mm³, and divide by 1,000 for mL. Record
observed labels and each label's voxel count. A mask may lack an individual zone
label but its whole-gland volume must be positive. The binary
`wp_axt2_nnunet-274.nii.gz` is not this study's reference.

Run cells in order, or from the repository root:

```bash
MPLBACKEND=Agg .venv/bin/python notebooks/toronto_jdmi_4111_clinical_volume.py
```

Supply the encrypted workbook password at runtime; it is not saved. No downloads,
training, or inference occur. Masks are measured on native grids with matching
shapes, valid affines, and ≤0.5 mm corner displacement. Explicit non-mm units
fail. The full inputs include 1,554 reference masks and 183 T2 images whose
spatial-unit code is unknown. Accept an unknown code only when its grid matches
a known-mm anchor within **0.01 mm** at corresponding voxel corners: use the T2
when marked mm, otherwise a selected comparator mask matching the T2 grid.
Record original units, the assumption, anchor path, and displacement in the
measurement audit, plus assumption counts and the rule in run metadata. The
0.01 mm tolerance allows observed coordinate differences up to 0.003 mm while
remaining stricter than the general 0.5 mm geometry tolerance. Inputs
are not modified. A missing anchor or looser match stops measurement.
Missing or invalid reference/comparator measurements stop after a local audit.
Zero comparator predictions stay in volume statistics and stop the PSAD section
after its audit. Other clinical validation, thresholds, statistical families,
100,000 bootstrap resamples, and seed `20261102` match the original analysis.

Outputs use the original study's filenames in the separate Git-ignored directory
`results/toronto_jdmi_4111_clinical_volume/`. `gt_volume_ml` and `gt_psad` refer to
the nnUNet702 surrogate; raw `ground_truth_volume_ml` and
`listing_manual_reference_ml` retain the optional manual CSV values. The
identifier-free report-quality table uses `reference_volume_ml` and
`reference_psad`. The reproducibility record hashes the reference masks,
predictions, T2 images, clinical inputs, and notebook; records label rules,
parameters, package versions, and available comparator metadata; and pins cohort
membership and the expected count. Review audits and archive previous outputs
before intentionally changing the cohort. Historical reference training and
inference settings are unknown. Patient-level outputs remain private and local.

## PI-CAI clinical volume and PSA density

[picai_clinical_volume.py](picai_clinical_volume.py) starts with the full PI-CAI
marksheet and selects one examination per patient before analysis exclusions.
Prefer exams with positive finite clinical volume and PSA, numeric PI-RADS, and
recorded biopsy results (including negative biopsies); choose the earliest MRI
date among complete exams, or the earliest overall if none are complete. Date
ties use original CSV row order. Reported PSAD is optional. No later analysis
substitutes another visit. The complete selection audit retains all 1,500 rows.

The notebook measures **Bosma22b automated surrogate reference** and nnUNet274
label-1 whole-gland volumes on their native grids in mL, retaining all 1,476
selected patients in `comparator_volumes_all_cases.csv`. Paired comparisons use
1,450 patients with valid clinical volume; recalculated PSA density uses 1,419.
It reproduces volume and density agreement, paired statistics, density strata,
reported-density diagnostics (1,032 patients), and the PI-RADS 3 biopsy simulation
(144 patients). All 139 suspicious reporting records remain eligible. These are
comparisons with an automated reference, not manual-reference accuracy estimates.

Inputs are the [clinical marksheet](../data/raw/zenodo/picai/picai_labels/clinical_information/marksheet.csv),
[Bosma22b masks](../data/raw/zenodo/picai/picai_labels/anatomical_delineations/whole_gland/AI/Bosma22b/),
and [existing nnUNet274 predictions](../results/picai_pred_pooled2_3d_fullres/).
Run cells in order; no downloads or inference are performed. Outputs, including
`patient_selection_audit.csv`, `cohort_audit.csv`, `cohort_flow.csv`, and the
comparison tables and figures, go to
[results/picai_clinical_volume](../results/picai_clinical_volume/).

## Linked PROSTATEx clinical-volume analysis

[prostatex_linked_clinical_volume.py](prostatex_linked_clinical_volume.py) reconstructs
the reviewed testing cohort (currently 157 cases), retains all clinical marksheet fields, and compares
clinical prostate volume with manual whole-gland segmentation volume in mL.
Outputs are written to [results/prostatex_linked_volume](../results/prostatex_linked_volume/).

Inputs:

- [Reviewed testing manifest](../configs/rumcnotprostatexnoclinical_folds.json):
  `testing` defines cohort membership and order; named `excluded_*` folds retain
  boundary-contact and mapping/clinical ineligibility exclusions.
- [PROSTATEx mask index](../data/raw/github/prostatex_masks/index.csv): image paths
  and manual whole-gland masks (`label_wg`), relative to the index directory.
- [PROSTATEx–PI-CAI mapping](../data/raw/zenodo/picai/picai_labels/additional_resources/ProstateX-mapping.json):
  links PROSTATEx IDs to PI-CAI patient/study IDs.
- [PI-CAI clinical marksheet](../data/raw/zenodo/picai/picai_labels/clinical_information/marksheet.csv):
  clinical `prostate_volume` (mL/cc) and all other clinical fields.

The top cells compute and display the cohort flow from the mapping JSON, with
examination and unique-patient counts and exclusions relative to each preceding stage:

| Stage | Examinations | Explanation |
|---|---:|---|
| PROSTATEx mapping JSON | 328 | 327 unique subjects; one has two mapped examinations |
| Local manual whole-gland mask available | 194 | 134 mapped examinations lack a local mask |
| Unambiguous linkage | 192 | Remove both examinations of ProstateX-0025 |
| Matched clinical marksheet | 192 | All remaining examinations match |
| Valid clinical volume before testing-fold selection | 190 | Two lack usable clinical volume |
| Reviewed testing fold: volume analysis | **157** | Exclude 33 otherwise eligible masks with superior/inferior boundary contact |
| PSA available: subsequent density analysis | **156** | ProstateX-0202 lacks PSA |

The testing fold is authoritative; metadata validates rather than expands it.
Duplicate or missing selected IDs, invalid clinical linkage or volume, and
image/manual-mask paths inconsistent with the index stop execution. The full
linkage audit retains cases outside the testing fold, including their
`exclusion_fold`. The approved manifest partitions 204 local cases into 157
`testing`, 33 `excluded_due_to_boundary_contact`, and 14
`excluded_due_to_ineligibility` cases. The latter includes four cases also showing
boundary contact; mapping/clinical ineligibility takes precedence. No case-ID
exception list is used by the linked analysis.
Clinical consistency flags remain review-only and do not remove selected cases.

Clinical consistency is assessed for the 150 complete valid PSA/volume/reported-PSAD
triples. Flag **|reported PSAD − PSA / clinical volume| > 0.01 ng/mL²** for review
only; equality passes, allowing numerical roundoff. The threshold is configurable
and is not documented source precision. Current inputs flag **19 cases**, including
ProstateX-0037. Missing/invalid clinical assessments have separate statuses. No review
flag or missing reported PSAD excludes a case from volume or recalculated density
analysis, and a flag does not establish which field, if any, is wrong.

The top cells export `cohort_flow.csv`, `cohort_linkage_audit.csv`, and the complete
157-case `clinical_consistency_audit.csv`. All 157 selected cases are measured once and
retained in both `comparator_volumes_all_cases.csv` and `comparator_volumes.csv`.
`clinical_suspicious_cases.csv` contains a compact table of every flagged case;
`clinical_case_review.csv` preserves their complete marksheet fields, identifiers,
paths, residuals, implied volume, manual-mask and comparator volumes and errors.
Review records are sorted by absolute residual and printed without truncation.
The notebook no longer generates `clinical_consistency_failed_cases.csv` or
`clinical_excluded_cases.csv`; any older copies are stale outputs.

## Dataset274 model (`pooled2`)

The trained segmentation model is **Dataset274_raw**, an nnUNet `3d_fullres` model
using `nnUNetTrainer_100epochs` and `nnUNetPlans`.

- Exported bundle: [Dataset274_raw_3d_fullres_5fold](../bundles/Dataset274_raw_3d_fullres_5fold/).
  Its [metadata](../bundles/Dataset274_raw_3d_fullres_5fold/metadata.json) configures
  a five-fold ensemble (folds 0–4) using `models/fold_<n>/checkpoint_final.pt`.
- Original training artifacts:
  [Dataset274_raw training directory](../work_dir_pooled2/nnunet_results/Dataset274_raw/),
  under `nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/`.
- Inference entrypoint: the bundle's [scripts/infer.py](../bundles/Dataset274_raw_3d_fullres_5fold/scripts/infer.py).
- Training configuration: [pooled2_input.yaml](../configs/pooled2_input.yaml).
  The authoritative source-data split is [pooled2_folds.json](../configs/pooled2_folds.json).

The split manifest contains these **scan counts**, not unique patient counts:

| Source dataset | Training/CV pool | Held-out test |
|---|---:|---:|
| Prostate158 | 119 | 20 |
| QIN prostate repeatability | 26 | 4 |
| PROSTATE-DIAGNOSIS | 27 | 3 |
| PROMISE12 | 67 | 8 |
| **Total** | **239** | **35** |

The pooled dataset contains 274 scans in total; QIN includes repeated scans.
The 239-scan pool is used for training/cross-validation, separate from the 35-scan
held-out test set. Neither PI-CAI nor PROSTATEx appears in this training manifest.

## Existing Dataset274 predictions

Verified locally on **2026-10-29**:

| Prediction dataset | Saved masks | Linked-cohort coverage | Location |
|---|---:|---:|---|
| PROSTATEx | 204 | 190/190 | [prostatex_pred_pooled2_3d_fullres](../results/prostatex_pred_pooled2_3d_fullres/) |
| PI-CAI | 1,500 | 190/190 | [picai_pred_pooled2_3d_fullres](../results/picai_pred_pooled2_3d_fullres/) |

PROSTATEx filenames are `ProstateX-####.nii.gz`; PI-CAI filenames are
`<patient_id>_<study_id>.nii.gz`. These are prediction masks; clinical volumes
come from the marksheet and ground-truth volumes come from the manual masks.

Both prediction directories contain a `plans.json` identifying `Dataset274_raw`,
plus `dataset.json` and `predict_from_raw_data_args.json`. This establishes the
saved dataset/model identity; these files do not independently establish which
historical checkpoints or ensemble folds produced the masks. The five-fold
settings above describe the exported bundle configuration.

For the linked PROSTATEx notebook, use the **PROSTATEx prediction directory** and
match by PROSTATEx ID. All 157 selected cases already have predictions, so no new inference
is needed. The notebook calculates Dataset274 and Dataset609 label-1 whole-gland volumes in
mL on each prediction's native grid, and compares both clinical and model volumes
against ground truth. Per-case exports include prediction paths, voxel counts,
voxel volumes, shape agreement, and maximum affine differences. Small affine
discrepancies are recorded without resampling or exclusion. Current output names
and the figure layout are described below.

## Selectable volume comparators

At the top of `prostatex_linked_clinical_volume.py`, set `SELECTED_COMPARATORS`
to any nonempty subset of `clinical`, `nnunet274`, `rumcnotprostatex609`, `bosma22b`,
`guerbet23`, `heviai23`, and `yuan23`. The current default selects clinical,
Dataset274 nnUNet, and Dataset609 nnUNet across all analyses. Plot labels are
**Manual ellipsoid**, **External (multi-site)** for Dataset274, and **Internal (RUMC)**
for Dataset609 trained on the RUMC cohort excluding PROSTATEx. Dataset609 predictions
come from [prostatex_pred_rumcnotprostatex609_3d_fullres](../results/prostatex_pred_rumcnotprostatex609_3d_fullres/)
and are matched by PROSTATEx ID. Both model sources validate dataset identity and
binary label metadata before measuring whole-gland label 1. Bosma22b remains
optional, labeled **Bosma22b**, with its own plotting style.
Clinical volume comes from the marksheet. HeviAI23 and Yuan23 whole gland is the union of labels 1 and 2.
Volume comparisons retain all 157 selected cases, including suspicious records; masks are
measured on native grids without resampling. This evaluates volume accuracy, not spatial overlap.

The generic outputs replace the former hardcoded comparison outputs:
`comparator_volumes.csv`, `comparator_summary.csv`, `comparator_agreement.png` and `.pdf`,
and (for multiple methods) `comparator_pairwise_per_case.csv` and
`comparator_pairwise_counts.csv`, under `results/prostatex_linked_volume/`.
Older output files may still exist and are not refreshed by this notebook.

Separate cells calculate the per-case errors and summary table, then construct
the figure from those results. The plotting cell groups figure dimensions,
relative-difference limits, method styles, and common legend options. Each method
has a column containing measured versus manual ground-truth volume, signed
difference, and percentage difference; y-axes are shared within each row. The
rightmost column contains unsmoothed count histograms with shared bins, using
half the pooled Freedman–Diaconis bin count (rounded up). Pearson's r uses the
original volumes. Axis labels and tick values remain visible on each method
panel; histogram axis titles are omitted.

## Paired post hoc volume statistics

The notebook also exports `comparator_pairwise_statistics.csv` in the same output
directory, comparing every selected method pair on the same patients. Positive
`rmse_improvement_ml` means RMSE(B) − RMSE(A) favors A. Primary significance uses
a two-sided, null-centered bootstrap-t test of paired squared-error differences;
equality of their population mean to zero is equivalent to equal population RMSE.
Paired BCa resampling gives pointwise 95% intervals for RMSE improvement. A paired
t-test on squared errors is reported separately as a sensitivity analysis.

The probability endpoint is `P(A closer | no tie)`, tested against 0.5 with an
exact two-sided binomial sign test and an exact 95% interval. Exactly equal
absolute errors are ties. Counts and all-patient proportions include
ties; the conditional probability excludes them. All-tie pairs have undefined
probability/interval and p=1. RMSE and win probability have separate conclusions:
lower RMSE need not imply winning on most patients.

Holm correction jointly covers both endpoints and all selected pairs (six tests
for the default three methods), without an omnibus gate. The sensitivity family
replaces bootstrap p-values with t-test p-values; undefined t-tests reserve their
slot as p=1 but are exported as undefined. Conclusions use adjusted p-values at
alpha=0.05; confidence intervals are pointwise and **not multiplicity adjusted**.

The settings are 100,000 resamples, seed 20261102 for paired BCa intervals, seed
20261103 for the null bootstrap, and batches of 1,000. Monte Carlo p-values use
`(extreme + 1) / (resamples + 1)` and cannot resolve values below approximately
1e-5. Zero-variance differences use an unstudentized centered-bootstrap fallback;
undefined BCa acceleration falls back to percentile intervals, explicitly labeled
in the output. Deterministic RMSE differences have point intervals. A single
selected method writes an empty statistics table rather than retaining old tests.

Predictions are fixed, and patients are treated as independent and representative
of the target population. The selected linked cohort is not a random population
sample; bootstrap resampling does not remove that limitation or capture training
variability. These are volume-accuracy comparisons, not spatial segmentation
metrics. Methodological background: [Ludbrook (1995), PubMed](https://pubmed.ncbi.nlm.nih.gov/7487733/).

## PSA-density agreement and strata

The same notebook continues with PSA density = PSA (ng/mL) / volume (mL), in
**ng/mL²**. The reference is PSA divided by manual whole-gland mask volume.
Each selected comparator uses that same PSA with its own volume measurement.
The marksheet's reported `psad` is examined separately at the end.

The density audit retains all 157 selected volume-analysis cases. At this step, remove only
missing/nonfinite PSA (currently ProstateX-0202), leaving **156 density cases**.
Cases with PSA but missing reported PSAD remain eligible, as do all
suspicious records. Finite PSA must be positive and reference and selected-method
volumes must be finite positive: unexpected nonpositive PSA or invalid/empty-volume
predictions are audited and stop execution rather than silently shrink the cohort.
These checks support log axes and relative errors. Nothing is imputed, and the
volume-analysis cohort remains unchanged.

The continuous summaries and agreement figures mirror the volume analysis:
measured versus reference density, signed and relative differences, and unsmoothed
count histograms. Density axes use log scales; difference rows show the full range.
Paired absolute-error counts, RMSE inference, and closer-probability inference use
the existing bootstrap settings, exact sign test, and Holm procedure. PSA-density
endpoints form a separate multiplicity family from volume endpoints. Fields ending
in `_psad` have units ng/mL²; `_psad2` denotes squared density units.

Strata use unrounded values: **[0, 0.10), [0.10, 0.15), [0.15, 0.20), [0.20, ∞)**.
Confusion-matrix rows are manual-mask reference strata and columns are comparator
strata. Figures show counts and row percentages on a shared color scale. Summaries
include accuracy, balanced accuracy, macro F1, quadratic-weighted kappa, and lower/
equal/higher stratum counts. Per-stratum tables include precision, recall,
specificity, F1, support, and confusion counts. Separate cutoff tables define
positive as density ≥ 0.10, ≥ 0.15, or ≥ 0.20 and report sensitivity, specificity,
PPV, NPV, accuracy, and TP/FP/FN/TN. Undefined ratios are NaN; macro averages use
defined components. These measure density-stratum agreement, not cancer detection.

Pairwise exact McNemar tests compare correct/incorrect stratum assignments on the
same eligible patients. Each pair uses a 2×2 table of both correct, A only correct,
B only correct, and both incorrect. Accuracy difference is
`100 × (accuracy A − accuracy B)` in percentage points; positive favors A.
Two-sided p-values receive Holm correction across all selected pairs as a separate
family from the continuous-error tests, with no omnibus gate and alpha 0.05.
No discordant correctness assignments gives p=1. No significant difference does
not establish equivalence. Quadratic weighted kappa remains descriptive; no
ordinal-error tests or additional bootstrap intervals are added.

New files under `results/prostatex_linked_volume/`:

- `psa_density_cohort_audit.csv`, `psa_density_per_case.csv`, and `psa_density_summary.csv`.
- `psa_density_agreement.png` / `.pdf` and `psa_density_confusion.png` / `.pdf`.
- `psa_density_confusion.csv`, `psa_density_classification_summary.csv`,
  `psa_density_per_stratum.csv`, and `psa_density_threshold_statistics.csv`.
- `psa_density_stratum_pairwise_statistics.csv`: paired correctness counts,
  accuracies, accuracy differences, raw/adjusted McNemar p-values, family size,
  alpha, and conclusions. Single-method runs replace this with an empty table
  retaining the same columns.
- `psa_density_pairwise_per_case.csv`, `psa_density_pairwise_counts.csv`, and
  `psa_density_pairwise_statistics.csv`. Single-method runs write empty pairwise tables.

## Reported marksheet PSA density

The reported-density cells use the complete 157-case volume cohort, including suspicious
records, for separately labeled reporting diagnostics. They compare reported
`psad` against **PSA / clinical volume** and
**PSA / manual-mask volume**. They also compare recalculated clinical density with
manual-mask density on the same complete cases. This subset is independent of
prediction-method selection; current inputs provide **150 complete reported-density
cases**. The audit records missing/invalid reports separately. Nonnegative
reported density is valid; a zero report has undefined implied volume and is
omitted from log-density display, but remains in statistics and strata.

Per-case residuals decompose total disagreement:

`reported − manual-mask = (reported − recalculated clinical) + (recalculated clinical − manual-mask)`.

The first term measures reporting inconsistency; the second measures the density
difference from using clinical volume with the same PSA. An absolute reporting
residual above **0.01 ng/mL²** is flagged for inspection. This configurable band
uses the same configured review threshold as the early clinical consistency assessment.
Source precision varies, and rounded PSA/volume or different source measurements
can also explain residuals. A flag is not confirmation of a reporting error and
does not affect analysis eligibility.

The final tables include signed and relative reporting residuals, volume implied
by reported density, largest discrepancies, stratum agreement/changes, and the
reporting-associated change in absolute manual-mask error (positive means a larger
error; negative means a smaller error). Agreement figures
compare all three measurement pairs; a residual histogram and component scatter
show how reporting residuals add to clinical-volume disagreement. Exported confusion
counts and heatmaps cover all three pairs on this same subset, using reference rows
and estimated columns as above.

Outputs use `psa_density_reported_` followed by `audit.csv`, `per_case.csv`,
`summary.csv`, `confusion.csv`, or `diagnostics.csv`; figures use `agreement`,
`components`, and `confusion`, each as PNG and PDF. All are in the same results directory.

## PI-RADS 3 biopsy-decision analysis

The July 2026 PI-CAI labels update adds `lesion_PIRADS`. The notebook's final
section parses comma-separated lesion scores and selects cases whose **maximum
numeric PI-RADS score is 3**. It ignores missing/`N/A` entries and rejects scores
outside integers 1–5. Cases with a PI-RADS 4/5 lesion are excluded even if another
lesion is PI-RADS 3. Only an absent PI-RADS column triggers the full-cohort
fallback; unavailable scores or an empty subgroup do not. Current local inputs
yield **23 eligible cases: 5 with csPCa and 18 without**.

The simulated decision recommends biopsy when **unrounded calculated PSAD is
≥ 0.15 ng/mL²**, including equality. It evaluates manual-mask density and all
`SELECTED_COMPARATORS` on the same patients against **`case_ISUP ≥ 2`** as the
outcome reference. ISUP 0–1 is negative; missing, nonfinite, noninteger, or
out-of-range ISUP values are audited and excluded. All volume cases appear in the
audit, including missing PSA and cases outside the subgroup. Clinical review
flags do not exclude cases, and reported marksheet PSAD is not used.

Summaries report TP/FP/FN/TN, sensitivity, specificity, PPV, NPV, accuracy,
biopsies recommended/avoided, and csPCa detected/missed. Sensitivity and the missed
csPCa fraction use `n_cspca`; specificity uses `n_non_cspca`; PPV/NPV use
`biopsies_recommended_n`/`biopsies_avoided_n`; accuracy and biopsy fractions use
`n`. Fractions range from 0 to 1; undefined ratios are NaN. Confusion matrices
have outcome rows and decision columns, showing counts and outcome-row
percentages. Empty analyses refresh zero-count summaries and figures and write
an empty per-case CSV.

Outputs under `results/prostatex_linked_volume/` use `psa_density_biopsy_` followed
by `audit.csv`, `per_case.csv`, `summary.csv`, and `confusion.csv`. Confusion
figures are also saved as PNG and PDF.

This is a retrospective threshold simulation. Scores mix PI-RADS v1/2/2.1, and
the source dataset was convenience sampled; some negatives are MRI-confirmed
without tissue sampling or follow-up. These results describe the selected cohort
and do not establish routine-care performance or prevalence.

## Study segmentation performance

The study notebook also compares supplied **Bosma22b** whole-gland label-1
masks with the same 157 ProstateX manual references using `ProstateX-mapping.json`.
It runs no inference for this comparator and exports mean DSC, MAE/RMSE (mL),
and MAPE/RMSPE (%). Missing or ambiguous mappings and invalid geometry are
audited; aggregates require complete coverage. Bosma22b masks are reoriented to
canonical axes, with nearest-neighbor resampling onto the manual grid when
resolutions or physical crops differ, using stored affines without fitting a
registration. Different field boundaries are permitted; nonoverlapping grids
are rejected. Voxels outside source coverage are background, so missed reference
anatomy contributes to DSC error. Volume errors use native-grid counts and voxel
volumes. All 157 selected cases evaluate successfully. Source-image diagnostics confirmed
that the 18 differing crops match their PI-CAI Bosma22b source grids; these were
not established patient-mapping errors.
Bosma22b training membership
is unverified, so these results carry no training designation. Other rows are NA.

[model_study_performance.py](model_study_performance.py) evaluates the five-fold
Dataset274 and Dataset236 ensembles. DSC, volume RMSE (mL), and case-count tables
use dataset/split rows and model columns:

| Dataset / manifest split | Expected scans | Dataset274 | Dataset236 |
|---|---:|---|---|
| pooled2 / training | 239 | Training/CV pool | Not requested |
| pooled2 / testing | 35 | Held-out test | External test |
| rumcnotprostatexnoclinical / training | 236 | Not requested | Training/CV pool |
| rumcnotprostatexnoclinical / testing | 193 | External test | Held-out test |

The RUMC testing split is the mapped PROSTATEx original training cohort, with
manual whole-gland masks. RUMC training uses **Bosma22b automated surrogate
reference masks**, so its metrics measure agreement with that automated reference.
Training-pool performance is not out-of-fold validation. Counts are scans, not
unique patients. Not-requested combinations remain `NA`; requested runs show
`evaluated/expected` even when predictions are missing.

The input YAMLs resolve the original split manifests and data roots. Dataset236
uses [its bundle inference entrypoint](../bundles/Dataset236_raw_3d_fullres_5fold/scripts/infer.py)
and [rumcnotprostatexnoclinical_input.yaml](../configs/rumcnotprostatexnoclinical_input.yaml)
for its training/testing splits, plus [pooled2_input.yaml](../configs/pooled2_input.yaml)
for the external testing split. Prediction folders are model-specific under the
study root. Existing Dataset274 folders are preserved: `pooled2_train`,
`pooled2_test`, and `prostatex_original_train_mapped`. Dataset236 uses `rumc_train`,
`prostatex_original_train_mapped`, and `pooled2_test` under its own model directory.
Run the cells in order, or from the repository root:

```bash
.venv/bin/python notebooks/model_study_performance.py
```

The notebook evaluates existing masks, prints all pending inference commands, and
asks once: `Run all commands above? [y/N]`. Enter `y` or `yes` to run the cohorts
sequentially; any other answer or unavailable input skips inference and exports
tables from available predictions. Complete cohorts are skipped. Commands use
`tile_step_size=0.8`, mirroring disabled, and `--device cuda`. Subprocesses inherit the
normal environment, with unbuffered Python output for timely logs. Existing
predictions in incomplete cohorts are overwritten.
Each bundle uses its five-fold ensemble, final checkpoints, and saved postprocessing.

Each running cohort shows a progress bar polling expected prediction files every
second. Overwritten files count only after changing; a full bar still waits for
inference and postprocessing to exit. Subprocess output goes to `inference.log`
in each prediction directory. Failure stops the queue and leaves that cohort's
metrics `NA`. Interrupting the notebook stops the launched process group.
Evaluation shows an `Evaluating <dataset>` progress bar counting every checked
case, including missing or invalid cases. The first evaluation identifies pending
cohorts; a second pass refreshes tables only after an execution attempt. Declining
approval or having complete cohorts reuses the first evaluation. Historical run
settings are not verified.

Outputs go to `results/20261105_tile0.8_nomirroring/tables/`: CSV/Markdown tables,
per-case metrics, and a cohort audit. Mean whole-gland DSC and volume RMSE (mL)
stay `NA` until every expected case evaluates successfully. Equal-shape masks
are accepted with corresponding voxel-corner displacement up to **0.5 mm**;
larger differences are rejected. Actual displacement is exported per case, and
masks are not resampled or modified. After the final evaluation pass, warning
tables list evaluated cases with displacement greater than **1e-3 mm** and
invalid cases with their errors. Smaller displacements remain in the per-case
CSV. Geometry comparisons use effective affines, not other header metadata. Reports include
model, cohort, and case ID in manifest order. Counts are
evaluated/expected scans. No command files, snapshots, or completion markers are
created by the notebook.

## Disagreement-based quality improvement

[volume_disagreement_quality_improvement.py](volume_disagreement_quality_improvement.py)
screens manual ellipsoid versus nnUNet274 volumes without using reference volume.
The primary screen requires signed disagreement outside mean ±1.96 sample SD
on **both** the mL and symmetric percentage scales, fitting each complete cohort
separately. Percentage disagreement uses the pair mean. Individual-scale and
union screens are descriptive secondary analyses.

Run the ProstateX source notebook through its agreement-figure cell first to
refresh `manual_automated_disagreement_flags.csv` and the highlighted PNG/PDF.
Then run:

```bash
MPLBACKEND=Agg .venv/bin/python notebooks/volume_disagreement_quality_improvement.py
```

Inputs are existing volume, PSAD, biopsy-audit, and provenance exports from
ProstateX, Toronto manual-reference, and expanded Toronto notebooks. The expanded
directory retains `4111` in its name; membership and counts follow the current
source notebook and saved metadata (currently 4,105). Input checks reject stale
membership, measurements, or eligibility exports. CSV floats use round-trip
parsing to preserve unrounded PSAD cutoff assignments.

Outputs under `results/volume_disagreement_quality_improvement/` include full
case flags and screening thresholds; volume metrics, review burden, paired
improvement/worsening summaries and primary inference; PSAD category transitions
and exact McNemar comparisons; biopsy eligibility, reference-decision agreement,
and known-outcome confusion counts; paired signed-error PNG/PDF; and input hashes.
The volume accuracy and review burden section prints every strict-intersection
outlier's ID, available image path, reference and manual volumes, and every
automated method volume available in the source export, without truncating rows
or columns. This local review table is also saved as `outlier_case_volumes.csv`;
reference type is labeled explicitly for each cohort.

Primary inference retains 100,000 resamples and seeds 20261102/20261103. RMSE and
sign tests form a two-test Holm family per cohort; RMSPE and PSAD-category tests
have separate one-test families. Thresholds and selected patients stay fixed
during resampling. Small groups and post-selection uncertainty limit inference.
Unknown cancer outcomes remain eligible for reference-decision comparisons but
are excluded from cancer-outcome metrics. Empty groups have zero counts and
undefined rates. Toronto cohorts overlap, and nnUNet702 is an automated surrogate,
so expanded results are neither manual-ground-truth validation nor independent
replication. The notebook performs no downloads, training, or inference.

## ProstateX manual-mask field-of-view contact

[prostatex_manual_mask_fov_contact.py](prostatex_manual_mask_fov_contact.py)
checks all 204 local manual whole-gland masks against their paired image geometry,
separating the reviewed testing cohort, explicitly excluded cases, and other local
cases. It reports contact area per outer face (mm²), terminal-slice gland volume
(mL), union boundary-volume fraction, and contact area relative to the largest
parallel gland cross-section. Canonical reorientation uses no interpolation.

Outputs are saved under `results/prostatex_manual_mask_fov_contact/` as
`case_contact.csv`, `face_contact.csv`, `summary.csv`, and
`normalized_summary.csv`. Absolute-area and gland-size-normalized thresholds are
descriptive review screens; contact alone does not establish truncation or justify
an automatic exclusion. No dataset membership or masks are changed.

## Candidate ProstateX cohort without manual-mask boundary contact

[_picai_rumc_incomplete_not_prostatex.py](_picai_rumc_incomplete_not_prostatex.py)
checks every local manual whole-gland mask against its paired image grid.
A foreground voxel (>0 voxels) on the superior or inferior outer face is boundary
contact; the affine determines anatomical orientation. Other faces are ignored.
There is no area threshold or case-specific exclusion list.

The candidate partitions all 204 local ProstateX cases into disjoint folds:

| Fold | Cases | Rule |
|---|---:|---|
| `testing` | 157 | Valid mapping/clinical volume and no S/I boundary contact |
| `excluded_due_to_boundary_contact` | 33 | Otherwise eligible, with S/I boundary contact |
| `excluded_due_to_ineligibility` | 14 | 11 unmapped, 1 ambiguous mapping, 2 invalid clinical volumes |

Mapping/clinical ineligibility takes precedence: four of the 14 ineligible cases
also have boundary contact, which remains recorded in the audit. The three folds
sum to 204 and preserve normal `id`, `image`, and manual `label` paths. The separate
`training` fold contains 236 unchanged PI-CAI RUMC cases and is not part of the
ProstateX total. Boundary contact is an operational rule, not proof of truncation.
Review and manually copy the candidate to the active manifest before refreshing
downstream analyses; generation does not modify the active manifest or results.
