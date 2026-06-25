# Manuscript Outline

## Introduction
- One application of prostate segmentation is automated volume prediction.
- Useful in PI-RADS in calculation of prostate density.
- Manual measurement is limited to ellipsoid formula, which is an “indirect” measure of volume.
- Existing machine learning literature focus on reporting DSC, but not physically meaningful volume.
- nnUNetv2, with correct preprocessing, augmentation, hyperparameters, and postprocessing, has been shown to achieve SOTA performance. (Cite nnUNetv2 revisited paper).



## Methods
- Open source data/models based on only T2-weighted image for practicality:
1. Derived from Prostate158 open weights MONAI model (“Prostate158-Adams22”) https://project-monai.github.io/model-zoo.html#/model/prostate_mri_anatomy 
2. Derived from Prostate158, trained 3D nnUNetv2 (“Prostate158-nnUNet”)
3. Derived from PROMISE12, trained 3D nnUNetv2 (“PROMISE12-nnUNet”)
- These models were used to predict the segmentation in the PI-CAI dataset.

- Closed source “silver standard” segmentation on PI-CAI dataset (Bosma22b, Guerbet23, HeviAI23, Yuan23)
- PI-CAI clinical information, containing prostate_volume, PSA

## Results:
- Inter-rater variability matrix

- x axis=average of silver standard volumes, y axis=deviation of each method from the average





- Confusion matrix for prostate volume and PSAD and PSAD ROC between manual and average of silver standard


- Confusion matrix for prostate volume and PSAD and PSAD ROC between “PROMISE12-3DnnUNet” and average of silver standard


- Confusion matrix for prostate volume and PSAD and PSAD ROC between “Prostate158-3DnnUNet” and average of silver standard





## Discussion
- Inter-rater variability in manual ellipsoid from previous studies is higher than inter-rater variability of deep learning algorithms.
- Therefore, deep learning algorithms provide a more reproducible measurement

# Coding Conventions

## Evaluation

Some models segment two prostate zones:

**Labels** (from `bundles/prostate158_2d_5fold/models/dataset.json`):
- Label 1 = transition zone (TZ)
- Label 2 = peripheral zone (PZ)
- Whole gland = union of labels 1 and 2

Some models only segment the whole gland (label 1).

**Inference**: 5-fold ensemble via `scripts/infer.py`.

**Metrics** computed per case × label (volumes in mm³):
| Metric | Description |
|--------|-------------|
| TP / FP / FN / TN vol | Voxel-count × voxel volume |
| DSC | Dice Similarity Coefficient |
| IoU | Jaccard index |
| Precision / Recall / Specificity | Standard binary classification metrics |
| Volume similarity | `|pred_vol − gt_vol| / gt_vol` |
| HD95 | 95th-percentile Hausdorff Distance (mm) |
| ASSD | Average Symmetric Surface Distance (mm) |
| inference_sec | Per-case wall time (estimated from output file mtimes) |

## Repository Structure

```
prostate_train/
├── .venv/                      # Virtual environment (not tracked)
├── bundles/                    # Trained model bundles (see "Bundle Formats")
│   ├── prostate158_2d_5fold/           # MONAI-style bundle (2D)
│   ├── prostate158_3d_fullres_5fold/   # MONAI-style bundle (3D)
│   ├── Dataset158_...__2d.zip          # nnUNet native output (2D)
│   └── Dataset158_...__3d_fullres.zip  # nnUNet native output (3D)
├── configs/
│   ├── prostate158_folds.json  # Train/test split (119 train, 20 test)
│   └── prostate158_input.yaml  # MONAI nnUNetV2Runner input config
├── data/                       # Downloaded datasets (not tracked)
│   ├── raw/                    # Persistent extracted data
│   └── tmp/                    # Temporary archives (auto-cleaned)
├── notebooks/
│   └── evaluate_2d.ipynb       # End-to-end evaluation notebook
├── src/
│   └── download_prostate158.py     # Main downloader script
├── tests/
│   └── test_download_prostate158.py
├── pyproject.toml              # Project metadata & deps
├── README.md                   # User-facing quick start
├── LICENSE                     # (present)
└── AGENTS.md                   # This file
```

## Reference:

- [nnUNetV2Runner](.venv/lib/python3.12/site-packages/monai/apps/nnunet/nnunetv2_runner.py)
- [mmUNetV2 console scripts](.venv/lib/python3.12/site-packages/nnunetv2-2.7.0.dist-info/entry_points.txt)
