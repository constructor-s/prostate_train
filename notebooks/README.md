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
discrepancies are recorded without resampling or exclusion. The existing
`clinical_vs_gt_summary.csv` and `clinical_vs_gt.png` filenames now include both
comparators; the figure retains the 1×3 layout.

## Selectable volume comparators

At the top of `prostatex_linked_clinical_volume.py`, set `SELECTED_COMPARATORS`
to any nonempty subset of `clinical`, `nnunet274`, `bosma22b`, `guerbet23`,
`heviai23`, and `yuan23`. The default includes all six. Clinical means reported
marksheet volume. HeviAI23 and Yuan23 whole gland is the union of labels 1 and 2.
All comparisons retain the same 190-case cohort; masks are measured on native
grids without resampling. This evaluates volume accuracy, not spatial overlap.

The generic outputs replace the former hardcoded comparison outputs:
`comparator_volumes.csv`, `comparator_summary.csv`, `comparator_agreement.png`,
and (for multiple methods) `comparator_pairwise_per_case.csv` and
`comparator_pairwise_counts.csv`, under `results/prostatex_linked_volume/`.
Older output files may still exist and are not refreshed by this notebook.
