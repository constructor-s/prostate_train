# prostate_train

## Reproducible setup and Download Prostate158

1. Clone the repository on any machine:

```bash
git clone <repo-url> prostate_train
cd prostate_train
```

2. Install the downloader dependency and run the built-in dataset downloader:

```bash
uv pip install zenodo-get
python download_prostate158.py
```

3. Run the tests to confirm the repo and download logic are working:

```bash
python -m unittest discover -s tests -v
```

By default, the dataset is extracted into:

```text
data/raw/zenodo/prostate158/record-6481141/prostate158_train
```

Temporary archives are kept under `data/tmp/zenodo/...` while the download runs. This makes the setup reproducible and keeps raw data separate from generated outputs.

## GPU Server Transfer and Training

1. Copy the repo and dataset to the GPU server.

If the repo is not already on the server, use `rsync` from the local project root:

```bash
rsync -av --progress . user@gpu-server:/path/to/prostate_train \
  --exclude '.git' --exclude 'data/tmp'
```

If the dataset must be transferred separately, use:

```bash
rsync -av --progress data/raw/zenodo/prostate158/record-6481141 \
  user@gpu-server:/path/to/prostate_train/data/raw/zenodo/prostate158/
```

2. SSH into the GPU server and go to the project folder:

```bash
ssh user@gpu-server
cd /path/to/prostate_train
```

3. Activate the Python environment.

If the repo includes a `.venv` folder:

```bash
source .venv/bin/activate
```

Otherwise, use `uv` to create/activate the environment from the repo root:

```bash
uv sync
uv shell
```

4. Run the MONAI nnU-Net v2 pipeline with the committed config.

Start with conversion and preprocessing:

```bash
python -m monai.apps.nnunet nnUNetV2Runner convert_dataset --input_config ./configs/prostate158_input.yaml
python -m monai.apps.nnunet nnUNetV2Runner plan_and_process --input_config ./configs/prostate158_input.yaml -pl nnUNetPlannerResEncM
```

<!-- TODO: -gpu_memory_target 24 -->

Train a single fold first to verify the GPU setup:

```bash
python -m monai.apps.nnunet nnUNetV2Runner train_single_model \
  --input_config ./configs/prostate158_input.yaml \
  --config 3d_fullres \
  --fold 0 \
  --trainer_class_name "nnUNetTrainer_1epoch"
```

Check GPU specs with `nvidia-smi` and adjust batch size in `work_dir/nnUNet_preprocessed/Dataset158_prostate158_train/nnUNetPlans.json` to maximize GPU utilization without running out of memory.

If you have a GPU available and want a standard training run:

```bash
CUDA_VISIBLE_DEVICES=0 python -m monai.apps.nnunet nnUNetV2Runner train \
  --input_config ./configs/prostate158_input.yaml
```

```bash
nnUNet_wandb_enabled=1 \
nnUNet_wandb_project="prostate158" \
python -m monai.apps.nnunet nnUNetV2Runner train \
  --input_config "./configs/prostate158_input.yaml" \
  --configs "3d_fullres" \
  --trainer_class_name "nnUNetTrainer_100epochs" \
  --export_validation_probabilities True # required for later ensemble postprocessing, same as --npz
```

5. Check outputs.

- Preprocessed data: `./work_dir/nnUNet_preprocessed`
- Raw formatted data: `./work_dir/nnUNet_raw_data_base`
- Trained models: `./work_dir/nnUNet_trained_models`

```bash
ls -lh ./work_dir/nnUNet_trained_models
```

6. Run inference later:

```bash
python -m monai.apps.nnunet nnUNetV2Runner find_best_configuration --input_config ./configs/prostate158_input.yaml --trainer_class_name "nnUNetTrainer_100epochs" --configs "3d_fullres" --allow_ensembling=False --folds=0,
python -m monai.apps.nnunet nnUNetV2Runner find_best_configuration --input_config ./configs/prostate158_input.yaml --trainer_class_name "nnUNetTrainer_100epochs" --configs "3d_fullres"
python -m monai.apps.nnunet nnUNetV2Runner predict_ensemble_postprocessing --input_config ./configs/prostate158_input.yaml --trainer_class_name "nnUNetTrainer_100epochs" --use_mirroring=False --tile_step_size=0.8
```

7. Bundle:

```bash
python generate_bundle.py
# or
nnUNet_results="work_dir/nnUNet_trained_models" nnUNetv2_export_model_to_zip \
  -d Dataset158_prostate158_train -tr nnUNetTrainer_100epochs -p nnUNetPlans -c 3d_fullres \
  -o bundles/Dataset158_prostate158_train_nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres.zip
```
