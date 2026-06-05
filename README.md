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
  --configs "3d_fullres" \
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
python generate_bundle.py
```

Or nnUNetV2 format bundle:

```bash
nnUNet_results="work_dir/nnUNet_trained_models" nnUNetv2_export_model_to_zip \
  -d Dataset158_prostate158_train -tr nnUNetTrainer_100epochs -p nnUNetPlans -c 3d_fullres \
  -o bundles/Dataset158_prostate158_train_nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres.zip
```
