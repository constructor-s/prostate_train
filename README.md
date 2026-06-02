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
python -m monai.apps.nnunet nnUNetV2Runner plan_and_process --input_config ./configs/prostate158_input.yaml
```

Train a single fold first to verify the GPU setup:

```bash
python -m monai.apps.nnunet nnUNetV2Runner train_single_model \
  --input_config ./configs/prostate158_input.yaml \
  --config 3d_fullres \
  --fold 0
```

If you have a GPU available and want a standard training run:

```bash
CUDA_VISIBLE_DEVICES=0 python -m monai.apps.nnunet nnUNetV2Runner train \
  --input_config ./configs/prostate158_input.yaml
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
python -m monai.apps.nnunet nnUNetV2Runner predict_ensemble_postprocessing --input_config ./configs/prostate158_input.yaml
```
