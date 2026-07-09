# prostate_train

## Setup

```bash
uv sync
source .venv/bin/activate
python src/download_prostate158.py
```

By default, the dataset is extracted into:

```text
data/raw/zenodo/prostate158/record-6481141/prostate158_train
```

<!-- ## Data setup

```bash
# Example for PROMISE12. Adjust paths for prostate158.
python src/generate_cases.py \
  --training data/raw/zenodo/promise12/record-8026660/promise12/training_data.csv \
  --testing data/raw/zenodo/promise12/record-8026660/promise12/test_data.csv \
  --output configs/promise012_folds.json

python -m monai.apps.nnunet nnUNetV2Runner convert_dataset --input_config=configs/promise12_input.yaml --work_dir=data/raw/zenodo/promise12/record-8026660/promise12/training_data_work_dir
``` -->

## Pooled Dataset

The following data are not from Radboud UMC, and are pooled together for training:

- Prostate158 - 139 cases, 120 train (data/raw/zenodo/prostate158/record-6481141/prostate158_train/train.csv), 19 validation (data/raw/zenodo/prostate158/record-6481141/prostate158_train/valid.csv)
- QIN - 15 cases each repeated twice (data/raw/tcia/qin_prostate_repeatability/index.csv), split 10% about 2 cases for validation (stratified same 2 subjects' 4 scans)
- PROSTATE-DIAGNOSIS - 30 cases with segmentations (data/raw/tcia/prostate_diagnosis/index.csv), split 10% about 3 cases for validation
- PROMISE12 - only use the 75 non-runmc cases in (data/raw/zenodo/promise12/record-8026660/promise12/pooled_nonrunmc.csv), split 10% about 8 cases for validation

## Preprocess

```bash
python -m monai.apps.nnunet nnUNetV2Runner convert_dataset --input_config ./configs/prostate158_input.yaml
python -m monai.apps.nnunet nnUNetV2Runner plan_and_process --input_config ./configs/prostate158_input.yaml -pl nnUNetPlannerResEncM -gpu_memory_target 24
```

[Make sure to pass `-pl nnUNetPlannerResEncM` to use the new planner.](https://github.com/MIC-DKFZ/nnUNet/blob/c65537bebc5b50356df5dad352474bc3389e5e8b/documentation/resenc_presets.md) Verify in `work_dir/nnUNet_preprocessed/Dataset158_prostate158_train/nnUNetPlans.json`. Then you can ignore the warning about the old planner being used.

## Sanity-check (1-epoch)

Train a single fold first to verify the GPU setup:

```bash
python -m monai.apps.nnunet nnUNetV2Runner train_single_model \
  --input_config ./configs/prostate158_input.yaml \
  --config 3d_fullres \
  --fold 0 \
  --trainer_class_name "nnUNetTrainer_1epoch"
```

Check GPU specs with `nvidia-smi` and adjust batch size in `work_dir/nnUNet_preprocessed/Dataset158_prostate158_train/nnUNetPlans.json` to maximize GPU utilization without running out of memory.

## Full Training

Run all 5 folds:

```bash
nnUNet_wandb_enabled=1 \
nnUNet_wandb_project="prostate158" \
python -m monai.apps.nnunet nnUNetV2Runner train \
  --input_config "./configs/prostate158_input.yaml" \
  --configs='["3d_fullres","2d"]' \
  --trainer_class_name "nnUNetTrainer_100epochs" \
  --export_validation_probabilities True
```

Note: `--export_validation_probabilities True` (equivalent to `--npz`) is required for ensemble postprocessing.

## Work-dir layout

```python
os.environ["nnUNet_preprocessed"] = "./work_dir/nnUNet_preprocessed"
os.environ["nnUNet_raw_data_base"] = "./work_dir/nnUNet_raw_data_base"
os.environ["nnUNet_trained_models"] = "./work_dir/nnUNet_trained_models"
```

## Find Best Configuration & Predict

Test a single fold (debug):

```bash
python -m monai.apps.nnunet nnUNetV2Runner find_best_configuration --input_config ./configs/prostate158_input.yaml --trainer_class_name "nnUNetTrainer_100epochs" --configs "3d_fullres" --allow_ensembling=False --folds=0
```

Or run full ensemble:

```bash
python -m monai.apps.nnunet nnUNetV2Runner find_best_configuration --input_config ./configs/prostate158_input.yaml --trainer_class_name "nnUNetTrainer_100epochs" --configs "3d_fullres"
```

Then predict and postprocess:

```bash
python -m monai.apps.nnunet nnUNetV2Runner predict_ensemble_postprocessing --input_config ./configs/prostate158_input.yaml --trainer_class_name "nnUNetTrainer_100epochs" --use_mirroring=False --tile_step_size=0.8
```

## Export Bundle

Custmon MONAI bundle:

```bash
python src/generate_bundle.py \
    --model_dir work_dir/nnUNet_trained_models/Dataset158_prostate158_train/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres

python bundles/Dataset158_prostate158_train_3d_fullres_5fold/scripts/infer.py \
    --input data/raw/zenodo/promise12/record-8026660/promise12/training_data.csv \
    --output_folder results/Dataset158_prostate158_train_3d_fullres_5fold/promise12/training_data/ \
    --no-use_mirroring --tile_step_size=0.8
```

Or nnUNetV2 format bundle:

```bash
nnUNet_results="work_dir/nnUNet_trained_models" nnUNetv2_export_model_to_zip \
  -d Dataset158_prostate158_train -tr nnUNetTrainer_100epochs -p nnUNetPlans -c 3d_fullres \
  -o bundles/Dataset158_prostate158_train_nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres.zip
```
