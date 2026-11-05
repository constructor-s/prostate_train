# Scientific notebooks

Run the Python cell notebooks from this directory (`notebooks/`); the repository
root is assumed to be its parent.

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
the 190-case linked volume-analysis cohort, retains all clinical marksheet fields, and compares
clinical prostate volume with manual whole-gland segmentation volume in mL.
Outputs are written to [results/prostatex_linked_volume](../results/prostatex_linked_volume/).

Inputs:

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
| Valid clinical volume: volume analysis | **190** | Two lack usable clinical volume |
| PSA available: subsequent density analysis | **189** | ProstateX-0202 lacks PSA |

Clinical consistency is assessed for the 181 complete valid PSA/volume/reported-PSAD
triples. Flag **|reported PSAD − PSA / clinical volume| > 0.01 ng/mL²** for review
only; equality passes, allowing numerical roundoff. The threshold is configurable
and is not documented source precision. Current inputs flag **23 cases**, including
ProstateX-0037. Missing/invalid clinical assessments have separate statuses. No review
flag or missing reported PSAD excludes a case from volume or recalculated density
analysis, and a flag does not establish which field, if any, is wrong.

The top cells export `cohort_flow.csv`, `cohort_linkage_audit.csv`, and the complete
190-case `clinical_consistency_audit.csv`. All 190 cases are measured once and
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
match by PROSTATEx ID. All 190 cases already have predictions, so no new inference
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
Volume comparisons retain all 190 cases, including suspicious records; masks are
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

The density audit retains all 190 volume-analysis cases. At this step, remove only
missing/nonfinite PSA (currently ProstateX-0202), leaving **189 density cases**.
The eight cases with PSA but missing reported PSAD remain eligible, as do all
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

The reported-density cells use the complete 190-case volume cohort, including suspicious
records, for separately labeled reporting diagnostics. They compare reported
`psad` against **PSA / clinical volume** and
**PSA / manual-mask volume**. They also compare recalculated clinical density with
manual-mask density on the same complete cases. This subset is independent of
prediction-method selection; current inputs provide **181 complete reported-density
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
yield **26 eligible cases: 5 with csPCa and 21 without**.

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

[model_study_performance.py](model_study_performance.py) generates Dataset274
five-fold ensemble inference commands and evaluates the pooled training/CV pool,
pooled held-out test, and external manual PROSTATEx test from the configured
manifests (currently 239, 35, and 193 scans). Cohort keys and prediction folders
are `pooled2_train`, `pooled2_test`, and `prostatex_original_train_mapped`. Table
labels are **pooled2 train**, **pooled2 test**, and **PROSTATEx original train
(mapped)**. The latter describes the source cohort: all 193 cases have PI-CAI
mappings, and they remain external test cases for Dataset274. Training-pool performance is not
out-of-fold validation. Run the cells in order, or from the repository root:

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
The bundle uses its five-fold ensemble, final checkpoints, and saved postprocessing.

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
are accepted with corresponding voxel-corner displacement up to **0.25 mm**;
larger differences are rejected. Actual displacement is exported per case, and
masks are not resampled or modified. Counts are
evaluated/expected scans. No command files, snapshots, or completion markers are
created by the notebook.
