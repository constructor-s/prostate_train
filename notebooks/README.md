# Scientific notebooks

These are Python cell notebooks: run `# %%` cells in order using the existing
project environment. Paths resolve from the repository root. Generated results
and private case-level audits stay local. See the [project README](../README.md)
for setup, training, and inference commands.

## Notebook index

### Cohort and input preparation

| Notebook | Purpose and required inputs | Outputs |
| --- | --- | --- |
| [_hlab20260928_clinical_image_listing.py](_hlab20260928_clinical_image_listing.py) | Prepare the Toronto T2/clinical listing from the existing image listing, clinical workbook, and reference-volume eligibility rules. | `data_internal/hlab20260928/hlab20260928_t2_image_listing_full_clinical.csv` |
| [_picai_train_rumc_incomplete_not_prostatex_test_prostatex.py](_picai_train_rumc_incomplete_not_prostatex_test_prostatex.py) | Select clinically incomplete RUMC training patients excluding PROSTATEx-mapped patients; construct manual-mask PROSTATEx testing folds from local marksheet, mappings, image indexes, and masks. | Candidate `configs/picai_rumc_incomplete_not_prostatex_folds.json`; audits in `results/picai_rumc_incomplete_not_prostatex/` |
| [_picai_test_clinical.py](_picai_test_clinical.py) | Preserve the fixed Dataset236 training cohort and partition local PI-CAI visits using clinical completeness, reviewed Bosma22b exclusions, boundary contact, and visit ranking. | `configs/picai_test_clinical_folds.json`; `results/picai_test_clinical/` |

The RUMC/PROSTATEx generator writes a candidate manifest; it does not replace
`configs/rumcnotprostatexnoclinical_folds.json`. Do not regenerate a training cohort
as an incidental step when analyzing existing models.

### Scientific analyses

| Notebook | Required upstream inputs and purpose | Output directory |
| --- | --- | --- |
| [paper_new_results.py](paper_new_results.py) | Completed performance, clinical, segmentation, and three site-specific disagreement exports; derive disjoint clinical cohorts, assemble eight testing cohorts, six-cohort closeness tests, a 12-panel Bland–Altman figure, and every PI-RADS 3 intersection outlier for `paper_new.md`. | `results/paper_new_results/` and marked manuscript tables |
| [model_study_performance.py](model_study_performance.py) | Model configurations, manifests, exported bundles, and saved predictions; evaluate segmentation and volume error by dataset/split. It can offer inference commands for explicit execution. | `results/20261105_tile0.8_nomirroring/tables/` |
| [prostatex_linked_clinical_volume.py](prostatex_linked_clinical_volume.py) | Active RUMC/PROSTATEx manifest, marksheet linkage, manual reference masks, and study predictions; volume, PSAD, biopsy, and agreement analyses. | `results/prostatex_linked_volume/` |
| [picai_test_clinical_volume.py](picai_test_clinical_volume.py) | Reviewed PI-CAI manifest, clinical marksheet, Bosma22b references, and verified study predictions; analyze one institution's volume, PSAD, biopsy, and whole-gland DSC results. | `results/picai_test_clinical_volume/<SITE>/` |
| [toronto_jdmi_clinical_volume.py](toronto_jdmi_clinical_volume.py) | Prepared Toronto listing, clinical workbook, accession exclusions, manual-reader planimetry, and saved model masks; analyze the 383-patient manual-reference cohort. | `results/toronto_jdmi_clinical_volume/` |
| [toronto_jdmi_4111_clinical_volume.py](toronto_jdmi_4111_clinical_volume.py) | Toronto listing/workbook, accession exclusions, nnUNet702 surrogate masks, and comparator predictions; expanded-cohort analyses. The filename is historical; current membership is 4,105 patients. | `results/toronto_jdmi_4111_clinical_volume/` |
| [volume_disagreement_quality_improvement.py](volume_disagreement_quality_improvement.py) | Completed clinical-analysis exports for PROSTATEx, Toronto manual, expanded Toronto, and one PI-CAI site; estimate disagreement screens and evaluate flagged volume, PSAD, and biopsy results. | `results/volume_disagreement_quality_improvement/<SITE>/`; optional figure in `results/paper_picai_sites/` |

Clinical analyses read existing predictions; they do not launch inference.
PI-CAI preflight checks five ensemble folds, tile step 0.8, Gaussian weighting,
disabled mirroring, final checkpoint, and model metadata. Missing or unverified
predictions stop execution and produce availability audits and reviewable commands.

### Optional diagnostics

| Notebook | Required inputs and purpose | Output directory |
| --- | --- | --- |
| [toronto_jdmi_ellipsoid_tolerance.py](toronto_jdmi_ellipsoid_tolerance.py) | Raw Toronto listing, clinical workbook, accession exclusions, and saved masks; compare reported volumes with dimension-derived tolerances and review flagged cases. Proposed exclusions do not modify the shared exclusion file. | `results/toronto_jdmi_ellipsoid_tolerance/` |
| [toronto_jdmi_regex_check.py](toronto_jdmi_regex_check.py) | Toronto listing, clinical workbook, and accession exclusions; audit PSA regex matches and selected values. | `results/toronto_jdmi_regex_check/` |

## Execution order

1. Prepare the relevant local listing or cohort manifest when intentionally
   revising cohort membership. Existing exported models retain their original
   training cohorts.
2. Run the clinical analyses needed for the study. For PI-CAI, run
   `picai_test_clinical_volume.py` separately for PCNN, RUMC, and ZGT, including
   its supplementary segmentation cells.
3. Run `volume_disagreement_quality_improvement.py` separately for the same three
   sites after the PROSTATEx and both Toronto analyses are available.
4. After all three quality-improvement exports exist, enable the optional
   all-site figure section described below.

For ordinary script execution from the repository root:

```bash
MPLBACKEND=Agg .venv/bin/python notebooks/picai_test_clinical_volume.py
MPLBACKEND=Agg .venv/bin/python notebooks/volume_disagreement_quality_improvement.py
```

Both notebooks default to PCNN. Set the `SITE` parameter cell to `PCNN`, `RUMC`,
or `ZGT` before executing cells. This is a notebook parameter, not a CLI option.

## PI-CAI site selection and figures

| SITE | Manifest fold | Patients/exams |
| --- | --- | ---: |
| PCNN | `testing_PCNN` | 315 |
| RUMC | `testing_RUMC` | 391 |
| ZGT | `testing_ZGT` | 168 |

The sites contain 874 distinct patients, one exam each. Cohort preparation
preserves 236 training exams and reserves other exams from those training
patients. Six manually reviewed poor Bosma22b segmentations have their own
exclusion fold. Superior/inferior mask-face contact is excluded before selecting
one remaining visit per patient: complete clinicals first, then earliest MRI,
then marksheet row order. Incomplete and duplicate visits remain in separate
folds. Boundary contact suggests possible incomplete coverage; it is not proof
of truncation. Neither model's PI-CAI training patients overlap testing.

Each site's analyses use only its manifest fold and write to its own directory.
Thresholds, statistical correction families, and flags are calculated within
that site. PROSTATEx and PI-CAI RUMC share 116 patients; PCNN and ZGT share none.
The overlapping analyses are not independent replications.

The quality-improvement notebook has `BUILD_ALL_SITE_FIGURE = False` by default.
After all sites have completed, set it to `True` and run the final optional
section. It can also run independently after establishing `Path`, `REPO_ROOT`,
and the flag. It validates saved exports and generates the six-panel
`results/paper_picai_sites/paired_volume_errors.png` and `.pdf`, with provenance
in `run_metadata.json`. Panels are PROSTATEx, Toronto manual, expanded Toronto,
PCNN, RUMC, and ZGT. This plotting section does not repeat statistical resampling.

Historical pooled exports and retired diagnostic results are retained under
`results/`; current analyses do not use those pooled PI-CAI exports. Historical
metadata records the code and inputs used for those original runs.

## Scientific conventions

- **References:** PROSTATEx uses manual whole-gland masks; Toronto manual uses
  median reader planimetry. Bosma22b PI-CAI references and expanded Toronto
  nnUNet702 references are automated surrogates, not independent manual truth.
- **Labels and units:** verify each model's `models/dataset.json`. Bosma22b
  whole gland is label 1; expanded Toronto nnUNet702 whole gland is the union
  of labels 1, 2, and 3. Volume is foreground voxels × voxel volume in mm³,
  divided by 1,000 for mL. Spatial comparisons require matching grids and use mm.
- **PSAD:** PSA (ng/mL) / volume (mL), in ng/mL². Strata use unrounded cutoffs
  0.10, 0.15, and 0.20; equality enters the higher stratum. Biopsy simulation
  uses PSAD ≥0.15 and the notebook's documented PI-RADS/outcome eligibility.
- **Statistics:** current paired analyses use 100,000 resamples and seeds
  20261102/20261103. Preserve notebook-defined Holm families, paired resampling,
  zero-denominator handling, and fixed cohort/threshold assumptions.
- **Interpretation:** training-pool ensemble performance is not out-of-fold
  validation. Reporting and disagreement flags identify cases for review, not
  adjudicated errors. Largest-component filtering and boundary-based exclusions
  have documented post hoc limitations. Clinical flags do not silently change
  reviewed cohort membership.

Keep new analyses as readable `.py` cell notebooks and resolve inputs from the
repository root. Leave [notebooks_legacy](../notebooks_legacy/) unchanged.

## Revised manuscript synthesis

After the upstream clinical analyses and all three site-specific disagreement
runs are complete, execute:

```bash
.venv/bin/python notebooks/paper_new_results.py
```

This notebook reads saved measurements and segmentation metrics, reconciles
repeated ProstateX/Toronto exports, and generates Tables 1–6 in `paper_new.md`.
Only marked table blocks are updated; surrounding prose is preserved. Set
`UPDATE_MANUSCRIPT = False` in the parameter cell to generate outputs alone.
All generated CSV/Markdown tables, case audits, and hashed provenance go to
`results/paper_new_results/`. Case identifiers and recorded outcomes are private
local audit material; manuscript Table 6 contains those identifiers.

The synthesis validates the archived expanded Toronto parent cohort of
4,105 patients, then subtracts the 383 manual-reference patients to obtain
Toronto JDMI Additional (3,722 patients). It also removes 116 ProstateX patients
from PI-CAI RUMC, retaining 275 patients. Both subtractions use patient identifiers.
Selected/removed membership audits and parent/subset hashes are exported, and all
six clinical analysis cohorts are checked for patient-level disjointness. The current exclusion file would select 4,109 with six additions
and two removals. The notebook verifies and exports this known discrepancy and
the archived exclusions; parent configs and source exports remain unchanged. It stops on
unreviewed input changes. The 383-patient manual cohort has unchanged membership.
Volume and ordinal PSAD closeness use two-sided exact binomial tests among
non-ties, with separate six-cohort Holm families. These differ from the older
quality-improvement notebook's statistical correction families.

The synthesis uses the exact fixed ProstateX RUMC criteria for every disjoint
cohort: −14.28 to +25.74 mL and −22.70 to +39.60%, with strict exceedance of
both intervals. Cohort-specific means and LoA remain descriptive figure values.
Volume, PSAD, and biopsy summaries all use this common intersection selection. `bland_altman_clinical_nnunet274.png` (300 dpi), `.pdf`, and
`bland_altman_summary.csv` contain the 6×2 absolute/relative agreement figure and its values.
`BA_XLIM`, `BA_YLIM`, and `BA_REL_YLIM` default to `None`, showing all points.
Set increasing tuples for the common x-axis (mL), absolute y-axis (mL), and
relative y-axis (%), respectively, then rerun the dedicated plotting cell.
Each dataset occupies one row: absolute differences left, relative differences
right, using 100 × difference / pair mean. Y-limits are shared within columns.
The compact 8.5 × 9.75 inch portrait export uses constrained layout with no
reserved caption area and x-axis tick labels on every panel. Black points have
alpha 0.15; dataset means are gray solid lines and dataset LoA are gray dotted
lines. Black dashed lines overlay the exact fixed ProstateX criteria
(−14.28 to +25.74 mL; −22.70 to +39.60%). These define both analytical selection and figure highlighting. Nonzero hidden counts
are annotated; zero counts are omitted. The summary CSV and provenance record
the common plot criteria.
The summary CSV has 12 rows keyed by cohort and scale, with explicit units. Every panel reports points outside either
axis range once; cropping never removes observations from statistics. The sign
is clinical ellipsoid minus nnUNet274. Table 4 now reports common-criteria
volume accuracy, Table 5 PSAD closeness, and Table 6 all selected PI-RADS 3 cases. Plot-only reruns update
the figure and summary; rerun the final provenance cell to refresh run metadata.

All Bland–Altman x-axes use a logarithmic display with labeled major ticks at
10, 30, 100, and 300 mL on every panel; minor tick labels are suppressed.
`BA_XLIM` remains in mL and requires strictly positive bounds. Pair means and
differences used in all calculations remain in their original units.

In both plot columns, cases strictly outside both common intervals are orange
(`#D55E00`, alpha 1.0, size 4), drawn above the black points (alpha 0.15, size 4).
The legend and `common_both_n` in the plot summary report the full-cohort count,
independent of display cropping. This highlighting exactly matches analytical intersection membership.
Prior derived outputs are archived in `before_common_criteria/`; parent exports
and configs remain unchanged. Tables 1–3 are unaffected. The volume ties column
is shown only if a nonzero exact tie count occurs; PSAD equal-distance counts
are always retained.
