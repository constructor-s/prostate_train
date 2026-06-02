# AI Agent Guide for prostate_train

## Project Overview

**prostate_train** is a standalone Python script for downloading and managing the Prostate158 medical imaging dataset from Zenodo. It emphasizes reproducibility, clean architecture, and test-driven development.

**Current scope**: Data download stage only. Future stages will include dataset conversion and nnU-Net training.

## Quick Start Commands

```bash
# Install dependencies
uv pip install zenodo-get

# Run the downloader (idempotent—reuses existing data)
python download_prostate158.py

# Run the test suite
python -m unittest discover -s tests -v

# Check download status
ls -lh data/raw/zenodo/prostate158/record-6481141/prostate158_train/
```

## Key Conventions

### Data Layout (Reproducible & Extensible)

- **Persistent data**: `data/raw/<source>/<dataset>/record-<id>/...`
  - Example: `data/raw/zenodo/prostate158/record-6481141/prostate158_train/`
- **Temporary archives**: `data/tmp/<source>/<dataset>/record-<id>/...` (auto-cleaned after extraction)
- **Completion marker**: `.complete` sentinel file + `download_manifest.json` for detecting complete vs. partial downloads
- **Not tracked in git**: All data is in `.gitignore`; it is reproducible via the download script

### Download Safety

- Use `zenodo_get` CLI (not custom HTTP) for built-in checksum verification and retries
- Check for required paths (e.g., `train.csv`, `valid.csv`, `train/`) to detect incomplete downloads
- Fail fast if a partial download folder is detected (don't silently use corrupt data)
- Write manifest JSON with metadata (dataset name, source, record ID, required paths) for traceability

### Testing Approach

All changes to `download_prostate158.py` must pass unit tests:

```bash
python -m unittest discover -s tests -v
```

Tests use mocking to avoid real Zenodo downloads; they verify:
1. Fresh downloads create the correct folder structure and metadata
2. Complete datasets are reused without re-downloading
3. Partial datasets are rejected with clear errors

When adding features (e.g., new dataset sources), write tests first (`test_download_*.py`), then implement.

## Architecture Notes

**Single-file design**: The downloader is a standalone script (`download_prostate158.py`), not a package. This keeps it simple and easy to integrate into larger workflows.

**Extensibility**: The download logic is parameterized (`download_prostate158()` takes `data_root`, `record_id`), making it easy to add support for new datasets without refactoring.

**Idempotency**: Calling the script twice with the same arguments is safe—it skips re-download on the second run.

## Common Tasks

### Add a New Dataset Source

1. Create a new `download_<dataset>.py` script (or add to `download_prostate158.py` if sharing logic)
2. Follow the same pattern: `<data_root>/raw/<source>/<dataset>/record-<id>/...`
3. Write unit tests in `tests/test_download_<dataset>.py` before implementing
4. Update `README.md` with the new download command

### Modify the Download Logic

1. Write or update tests in `tests/test_download_prostate158.py` first
2. Run tests to confirm they fail: `python -m unittest discover -s tests -v`
3. Implement changes to `download_prostate158.py`
4. Run tests again to confirm they pass
5. Verify idempotency: run the script twice and confirm the second run skips download

### Debug a Download Issue

- Check if data folder exists but is incomplete: `ls data/raw/zenodo/prostate158/record-6481141/prostate158_train/`
- Look for the `.complete` sentinel: if missing, the download was interrupted
- Check the manifest: `cat data/raw/zenodo/prostate158/record-6481141/prostate158_train/download_manifest.json`
- Delete partial data to retry: `rm -rf data/raw/zenodo/prostate158/record-6481141/`
- Check `data/tmp/` for leftover archives (should be auto-cleaned, but check if interrupted)

## Dependencies & Environment

- **Python**: 3.11+ (specified in `pyproject.toml`)
- **Package manager**: `uv` (modern, fast Python package manager)
- **Core dependency**: `zenodo-get>=3.0.3` (handles Zenodo API, checksums, retries)
- **No heavy dependencies**: intentionally minimal to keep the script lightweight
 - **Dependency policy**: Keep the project's top-level dependency list minimal. Add heavy or optional packages (for experiments or extra utilities) only when needed and record them in `pyproject.toml` or an extras group. This helps keep installs fast, reduces CI load, and avoids version/compatibility surprises for users who only need the downloader.

## Future Directions

Per original requirements ("train an nnU-Net (2D or 3D) and evaluate"):

1. **Data conversion** (next): Add `convert_prostate158.py` to convert raw Prostate158 into nnU-Net v2 format
2. **Training** (next): Wire nnU-Net training via the nnunetv2 CLI or Python API
3. **Evaluation** (next): Implement evaluation on a held-out dataset

## Repository Structure

```
prostate_train/
├── .github/                    # (future) GitHub Actions workflows
├── .gitignore                  # Ignores data/ and uv.lock
├── .venv/                      # Virtual environment (not tracked)
├── data/                       # Downloaded datasets (not tracked)
│   ├── raw/                    # Persistent extracted data
│   └── tmp/                    # Temporary archives
├── tests/
│   └── test_download_prostate158.py
├── download_prostate158.py     # Main downloader script
├── pyproject.toml              # Project metadata & deps
├── README.md                   # User-facing quick start
├── LICENSE                     # (present)
└── AGENTS.md                   # This file
```

## Prostate158: MONAI + nnU-Net v2 Pipeline

This project aims to train a reproducible prostate segmentation pipeline using minimal custom code by leveraging MONAI's `nnUNetV2Runner`, which wraps nnU-Net v2 functionality. The notes below show recommended installation, configuration, and run steps tailored to the existing `prostate158` data layout.

**Dependencies**
- Python 3.11+
- `zenodo-get` (for dataset download) — already used by the repo
- `monai` (install with `pip install "monai[all]"` or follow MONAI docs)
- `nnUNet v2` (follow the official nnU-Net v2 install instructions; MONAI's runner integrates with it)

**Minimal `input.yaml` (example)**

Create `input.yaml` in the repo root pointing at the local Prostate158 extraction and a datalist JSON (MSD-style). Example:

```
modality: MRI
datalist: "./prostate158_folds.json"
dataroot: "./data/raw/zenodo/prostate158/record-6481141/prostate158_train"
dataset_name_or_id: Prostate158
nnunet_preprocessed: "./work_dir/nnUNet_preprocessed"
nnunet_raw: "./work_dir/nnUNet_raw_data_base"
nnunet_results: "./work_dir/nnUNet_trained_models"
```

**Datalist guidance**
- The datalist should follow MSD/nnU-Net expected format: a JSON listing training and validation cases and file paths for image(s) and label(s). Use the MONAI datalist examples as a template and point `dataroot` to the `prostate158_train` directory.
- For quick debugging create a small folds file that includes a handful of cases and one fold.

**Recommended run sequence (MONAI nnUNetV2Runner)**

Use the `nnUNetV2Runner` entrypoints provided by MONAI to run each pipeline step. Examples:

```
# Convert dataset to nnU-Net format
python -m monai.apps.nnunet nnUNetV2Runner convert_dataset --input_config "./input.yaml"

# Experiment planning and preprocessing
python -m monai.apps.nnunet nnUNetV2Runner plan_and_process --input_config "./input.yaml"

# Train all default models (uses available GPUs)
python -m monai.apps.nnunet nnUNetV2Runner train --input_config "./input.yaml"

# Train a single model config/fold (debug)
python -m monai.apps.nnunet nnUNetV2Runner train_single_model --input_config "./input.yaml" --config "3d_fullres" --fold 0

# Predict, ensemble, and postprocess
python -m monai.apps.nnunet nnUNetV2Runner predict_ensemble_postprocessing --input_config "./input.yaml"
```

For rapid iteration set `--trainer_class_name nnUNetTrainer_1epoch` (or another short trainer) to validate end-to-end connectivity without long training runs.

**Reproducibility & testing**
- Keep `input.yaml` and a small `prostate158_folds.json` under `tests/` (or `tests/fixtures/`) to enable CI smoke-tests that run `convert_dataset` and `plan_and_process` with a 1-epoch trainer. Do not commit large data; use mocks when necessary.
- The downloader unit tests remain valid; consider adding a lightweight integration test that runs the MONAI conversion + planning on a tiny subset.

**Notes & next steps**
- `nnUNetV2Runner` handles most preprocessing, augmentation, and training defaults—minimize custom transforms unless needed for experiments.
- To convert Prostate158 into an nnU-Net Decathlon layout, inspect `convert_msd_dataset` as a template.
- For multi-GPU scaling, use `--gpu_id_for_all` or DDP options documented in MONAI/nnU-Net.

If you'd like, I can generate a template `prostate158_folds.json`, add a CI smoke test that runs the MONAI conversion on a tiny subset, or create a ready-to-run `input.yaml` in the repo. Which would you prefer next?
