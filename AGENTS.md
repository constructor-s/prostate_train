# Repository Guidance

This repository supports prostate imaging dataset downloading and management, model training, inference, and scientific analysis. See [README.md](README.md) for setup and training commands and `pyproject.toml` / `uv.lock` for dependencies.

## Repository Layout

| Location | Purpose |
|----------|---------|
| `src/` | Dataset download and management tools, training and evaluation code, and shared utilities |
| `configs/` | Dataset splits and training configurations |
| `bundle_template/` | Template for exported model bundles |
| `bundles/` | Exported models and bundle-specific inference scripts |
| `data/` | Downloaded datasets and metadata |
| `data_internal/` | Internal datasets |
| `work_dir*` | Training, preprocessing, and intermediate model artifacts |
| `notebooks/` | Active scientific analyses using Python cell notebooks |
| `notebooks_legacy/` | Historical exploratory analyses retained for reference |
| `results/` | Generated predictions, tables, figures, and analysis outputs |
| `tests/` | Automated checks |

## Scientific Notebooks

- Create new analyses in `notebooks/` as `.py` files with `# %%` code cells and `# %% [markdown]` explanatory cells. Use descriptive `snake_case` filenames; add numeric prefixes only when there is an intended workflow order.
- Write in the style of a scientific notebook: introduce the question and inputs, explain methods, show calculations and intermediate results, and present plots and interpretations in a readable sequence.
- Prioritize replication, readability, and self-contained analyses over code reuse. Direct cell-level code is preferred. Duplicated code across notebooks is acceptable; repetition alone does not justify extracting helpers.
- Extract helpers only when they provide a substantial, concrete benefit. Place genuinely shared utilities in `src/`. Avoid unnecessary functions, classes, CLI entrypoints, configuration layers, and script boilerplate in notebooks.
- Resolve paths from an explicitly established repository root rather than assuming the current working directory. Keep generated outputs in a descriptive subdirectory of `results/`.
- Make inputs, cohort selection, exclusions, units, parameters, and any random seeds clear enough to replicate the analysis. Keep the analysis executable in cell order.
- Leave legacy analyses untouched unless a task specifically requires changing them.

## Inference and Evaluation

- Use the relevant exported bundle's `scripts/infer.py` entrypoint under `bundles/`. Check that bundle's configuration for ensemble folds and inference settings.
- Verify label mappings in the model's `models/dataset.json`. Some models use label 1 for transition zone (TZ) and label 2 for peripheral zone (PZ), with whole gland defined as their union. Other models use label 1 for whole gland.
- Distinguish zone-specific measurements from whole-gland measurements. Record metrics per case and evaluated label or region.
- State volume units explicitly. Voxel count multiplied by voxel volume gives mm³; divide by 1,000 to obtain mL. Match spacing order to array axes and ensure masks share the same spatial geometry before comparison.
- See `src/seg_metrics.py` for existing segmentation metric definitions: DSC, IoU, precision, recall, specificity, confusion volumes, predicted/reference volumes, HD95, and ASSD. Surface distances use millimetres and account for voxel spacing.
- The existing `vol_sim` key is relative absolute volume error, `abs(pred_vol - gt_vol) / gt_vol`, despite its name. Lower is better; it is NaN when reference volume is zero.
- If reporting inference time, state how it was measured; file modification times provide an estimate rather than a direct runtime measurement.

## Development and Validation

- Use the existing project environment and dependency configuration. Consult `README.md` for established workflows.
- Run checks relevant to changed behavior using the existing tests. For analysis changes, check cell execution order, input paths, and affected calculations or outputs as appropriate.
- Distinguish lightweight validation from dataset downloads, preprocessing, model training, and full inference; report what was actually run.
- Documentation-only changes require checking referenced paths and reviewing the diff, not training or inference runs.
