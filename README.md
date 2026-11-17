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

### Standalone RUMC excluding linked PROSTATEx patients

Run `python src/generate_pooled_folds.py --pooling rumcnotprostatex` to generate
`configs/rumcnotprostatex_folds.json` (paths relative to `data/raw`). Training
contains 609 RUMC studies from 602 patients; testing contains 190 linked PROSTATEx
studies plus the additional examination `11383_1001407` (191 studies, 190 patients).
Both splits use PI-CAI T2 images and Bosma22b AI whole-gland labels, including testing.
Study and patient IDs are disjoint. The generator reconstructs the notebook's
linked cohort from source metadata and validates counts and files before writing.
## Manual PI-CAI mask review

Run from the repository root using the existing project environment:

```bash
.venv/bin/python src/picai_review.py
```

Forward server port **5000** in your IDE and open its local URL. Alternatively,
run this on your own computer (replace `your-server` with your SSH host):

```bash
ssh -N -L 5000:127.0.0.1:5000 your-server
```

Then open **http://127.0.0.1:5000**. The app binds only to server localhost.
The same Python launch command runs Gunicorn with one worker, sixteen threads,
and HTTP keep-alive for 30 seconds of inactivity. One worker keeps the case
cache and review records shared. Use Ctrl+C in its terminal to stop the server.
Open **http://127.0.0.1:5000/case/1** to start at the first case, or `/case/11`
to jump directly to case 11 in the centre-sorted queue. These one-based URLs
override saved resume progress, and navigation updates the URL automatically.
Opening `/` uses saved resume progress. Deleting `review.csv` alone does not
clear `review.last_case` or records held by a running server; stop the server
before manually resetting its files.
If setting up a fresh environment, use the project's usual `uv sync` workflow.

The grid has axial/sagittal/coronal rows and Bosma22b/Multicenter/RUMC columns.
Each case starts at the Bosma22b centroid. Wheel over a panel to change its
plane's slice; click to move the shared cursor. **Left/right arrows** or the
buttons switch cases. Shortcuts are inactive in form controls. Images preserve
physical proportions and acquisition obliquity; direction labels describe the
reoriented voxel axes, rather than a resampled anatomical grid.

Rate **Bosma22b** and optionally add a note. Records save **only when navigating
away** using Next/Previous or arrow keys. On the last case, Next becomes
**Finish** to save it without moving onward. Closing or reloading before leaving
a case does not save its rating or visit; the browser warns about pending work.
Save errors retain the current edits; retry navigation. Each navigation uses
one request to save the departing record and return the next case's metadata
and nine image URLs, with no debounce. Images are fetched as binary `image/png`
directly from memory, without base64 or temporary files. Stable per-plane/slice
URLs let the browser reuse unchanged views. Image responses have private
one-day caching and ETags; URLs change on server restart. Metadata and review
responses remain uncached. Each of the 10 cached cases keeps nine starting PNGs
plus up to 27 recently requested non-centroid PNGs. The CSV
`results/picai_manual_review/review.csv` stores only explicitly set ratings
(acceptable, incorrect, unsure) and their notes. Visited cases left Unreviewed
are saved with their notes in `review_unreviewed.csv` when navigating away.
Changing a rating moves
the row between these files. Their combined unique case IDs track visited cases;
an automatic visit record means the case was opened, not that its segmentation
was explicitly judged acceptable. Unreviewed visits before this feature was
enabled were not stored and are not backfilled. Last-visited position
is stored separately in `review.last_case`.
Reopening resumes there. Use one browser tab and reviewer at a time.

The queue groups cases alphabetically by centre (PCNN, RUMC, ZGT), preserving
manifest order within each centre. Ratings and resume progress use case IDs.
The case list is exactly the 880 testing entries in
`configs/picai_test_clinical_folds.json`. Centre names and saved prediction
paths come from `results/picai_test_clinical_volume/comparator_volumes.csv`.
Missing files or incompatible geometry appear as case-level errors; navigation
remains available. The app does not modify source images or masks.

For a frozen viewer, restart the server and hard-refresh the browser to load
the latest JavaScript. Server logs now show request START/END, cache loads,
render timing, and exception tracebacks. To capture them:

```bash
.venv/bin/python -u src/picai_review.py 2>&1 | tee /tmp/picai_review.log
```

Open the browser developer console to see the existing wheel/click events,
request timings, PNG loading, and redraw logs. The page has no debug panel.
Requests time out after 30 seconds instead of keeping navigation busy forever.
Scrolling constructs image URLs locally, with one batch of PNG loads in flight,
and coalesces further events into the latest cursor position. No separate
metadata request is needed for scrolling. The JSON API is `/api/navigate`;
images use `/api/image/...`. The old start, case-metadata, save, and slice-URL
endpoints have been removed.
The server handles browser connections concurrently, locking only cache/CSV
updates, never PNG waits. The **10-case window** is the current case plus the
next nine; near the cohort end it includes earlier cases to fill the window.
Two background loaders decode volumes, and one render worker prepares only
the nine central views per case (axial/sagittal/coronal for each mask), not full
slice stacks. Navigation waits for volume metadata, never for PNG rendering.
PNG endpoints wait independently, and panels appear progressively in the browser.
Preloading never writes ratings or changes resume progress. Restarting clears
the memory cache. Image/mask files are assumed unchanged during a session.
Cache logs list case indices, viewer positions, IDs, and separate volume/PNG
readiness. Evictions remove volumes and starting views; other slices render on
demand. Waiting image requests do not hold the cache/CSV lock.
If the server stops logging midway through a request, `kill -USR1 <worker-PID>` prints
its Python stack to the server terminal without stopping it. Review-note text
is omitted from diagnostic logs.
Use the `Booting worker with pid` startup log to identify that worker; Gunicorn
also has a separate master process.
