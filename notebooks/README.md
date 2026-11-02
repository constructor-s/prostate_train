# Scientific notebooks

Run the Python cell notebooks from this directory (`notebooks/`); the repository
root is assumed to be its parent.

## Linked PROSTATEx clinical-volume analysis

[prostatex_linked_clinical_volume.py](prostatex_linked_clinical_volume.py) reconstructs
the 190-case linked cohort, retains all clinical marksheet fields, and compares
clinical prostate volume with manual whole-gland segmentation volume in mL.
Outputs are written to [results/prostatex_linked_volume](../results/prostatex_linked_volume/).

Inputs:

- [PROSTATEx mask index](../data/raw/github/prostatex_masks/index.csv): image paths
  and manual whole-gland masks (`label_wg`), relative to the index directory.
- [PROSTATEx–PI-CAI mapping](../data/raw/zenodo/picai/picai_labels/additional_resources/ProstateX-mapping.json):
  links PROSTATEx IDs to PI-CAI patient/study IDs.
- [PI-CAI clinical marksheet](../data/raw/zenodo/picai/picai_labels/clinical_information/marksheet.csv):
  clinical `prostate_volume` (mL/cc) and all other clinical fields.

The cohort excludes ambiguous mappings, unmatched cases, and missing/nonpositive
clinical volumes. PSA availability is not required. The exported linkage audit
records exclusions.

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
is needed. The notebook now calculates Dataset274 label-1 whole-gland volumes in
mL on each prediction's native grid, and compares both clinical and model volumes
against ground truth. Per-case exports include prediction paths, voxel counts,
voxel volumes, shape agreement, and maximum affine differences. Small affine
discrepancies are recorded without resampling or exclusion. Current output names
and the figure layout are described below.

## Selectable volume comparators

At the top of `prostatex_linked_clinical_volume.py`, set `SELECTED_COMPARATORS`
to any nonempty subset of `clinical`, `nnunet274`, `bosma22b`, `guerbet23`,
`heviai23`, and `yuan23`. The current default selects clinical, Dataset274 nnUNet,
and Bosma22b. Plot labels are **Manual ellipsoid**, **External (public)** for
Dataset274 nnUNet trained on public data external to this evaluation dataset,
and **Internal (institutional)** for Bosma22b from the same institution.
Clinical volume comes from the marksheet. HeviAI23 and Yuan23 whole gland is the union of labels 1 and 2.
All comparisons retain the same 190-case cohort; masks are measured on native
grids without resampling. This evaluates volume accuracy, not spatial overlap.

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

The density audit retains all 190 volume-analysis cases. Densities are calculated
where PSA is finite and nonnegative and the denominator is finite and positive.
Comparisons use one shared subset with positive PSA and valid volumes for the
reference and all selected methods. Zero PSA is excluded from comparisons because
log axes and relative errors require positive reference density. Missing PSA or
invalid/empty-volume predictions are audited and excluded from every method in
this density subset. The original volume analysis retains its cohort. Current
default inputs yield **189 density cases**, with one missing PSA.

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

New files under `results/prostatex_linked_volume/`:

- `psa_density_cohort_audit.csv`, `psa_density_per_case.csv`, and `psa_density_summary.csv`.
- `psa_density_agreement.png` / `.pdf` and `psa_density_confusion.png` / `.pdf`.
- `psa_density_confusion.csv`, `psa_density_classification_summary.csv`,
  `psa_density_per_stratum.csv`, and `psa_density_threshold_statistics.csv`.
- `psa_density_pairwise_per_case.csv`, `psa_density_pairwise_counts.csv`, and
  `psa_density_pairwise_statistics.csv`. Single-method runs write empty pairwise tables.

## Reported marksheet PSA density

The final cells compare reported `psad` against **PSA / clinical volume** and
**PSA / manual-mask volume**. They also compare recalculated clinical density with
manual-mask density on the same complete cases. This subset is independent of
prediction-method selection; current inputs provide **181 complete reported-density
cases**. The audit retains missing/invalid reports and exclusions. Nonnegative
reported density is valid; a zero report has undefined implied volume and is
omitted from log-density display, but remains in statistics and strata.

Per-case residuals decompose total disagreement:

`reported − manual-mask = (reported − recalculated clinical) + (recalculated clinical − manual-mask)`.

The first term measures reporting inconsistency; the second measures the density
difference from using clinical volume with the same PSA. An absolute reporting
residual above **0.005 ng/mL²** is flagged for inspection. This configurable band
assumes density alone was rounded to two decimal places; source precision varies,
and rounded PSA/volume or different source measurements can also explain residuals.
A flag is not confirmation of a reporting error.

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
