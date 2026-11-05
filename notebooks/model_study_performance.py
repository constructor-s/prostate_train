# %% [markdown]
# # Dataset274 study performance
#
# Compare whole-gland segmentation on the pooled training/CV pool, same-source
# held-out test, and external manual PROSTATEx test. Counts are scans, not patients.
# The five-fold ensemble training-pool score is not out-of-fold validation:
# ensemble members have seen these cases during training. PROSTATEx original
# train (mapped) describes the source cohort; it is an external test for this model.
#
# Run cells in order. Pending commands are printed together; one Y approval runs
# them sequentially on the GPU. Declining still generates tables from saved masks.
# Progress counts exported prediction files; completion also waits for inference
# and postprocessing to exit. Interrupting stops the launched worker group.
# Metrics stay NA until every expected case evaluates successfully. Historical
# inference settings are not certified by this analysis.

# %%
import itertools
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from IPython.display import display
from tqdm.auto import tqdm

STUDY_ID = '20261105_tile0.8_nomirroring'
REPO_ROOT = Path(__file__).resolve().parents[1] if '__file__' in globals() else Path.cwd().resolve()
if REPO_ROOT.name == 'notebooks':
    REPO_ROOT = REPO_ROOT.parent
assert (REPO_ROOT / 'pyproject.toml').is_file(), 'Set REPO_ROOT to the repository root'
assert Path(STUDY_ID).name == STUDY_ID and STUDY_ID not in ('', '.', '..')
MODEL = 'Dataset274_raw_3d_fullres_5fold'
BUNDLE = REPO_ROOT / 'bundles' / MODEL
STUDY_ROOT = REPO_ROOT / 'results' / STUDY_ID
TABLE_DIR = STUDY_ROOT / 'tables'
TABLE_DIR.mkdir(parents=True, exist_ok=True)
GEOMETRY_TOLERANCE_MM = 0.25
COHORTS = [
    dict(key='pooled2_train', title='pooled2 train', config='pooled2', split='training'),
    dict(key='pooled2_test', title='pooled2 test', config='pooled2', split='testing'),
    dict(key='prostatex_original_train_mapped', title='PROSTATEx original train (mapped)', config='rumcnotprostatexnoclinical', split='testing'),
]

# %% [markdown]
# ## Cohort inputs and reproducible commands
#
# Read the original manifests and data roots from their input configurations.
# Inference accepts these JSON files directly with --split training or testing.

# %%
for cohort in COHORTS:
    config = yaml.safe_load((REPO_ROOT / 'configs' / f"{cohort['config']}_input.yaml").read_text())
    cohort['manifest'] = (REPO_ROOT / config['datalist']).resolve()
    cohort['data_root'] = (REPO_ROOT / config['dataroot']).resolve()
    cohort['manifest_data'] = json.loads(cohort['manifest'].read_text())
    cohort['rows'] = cohort['manifest_data'][cohort['split']]
    cohort['output'] = STUDY_ROOT / MODEL / cohort['key']
    invocation = [
        ['.venv/bin/python', str((BUNDLE / 'scripts/infer.py').relative_to(REPO_ROOT))],
        ['--input', str(cohort['manifest'].relative_to(REPO_ROOT))],
        ['--split', cohort['split']],
        ['--data_root', str(cohort['data_root'].relative_to(REPO_ROOT))],
        ['--output_folder', str(cohort['output'].relative_to(REPO_ROOT))],
        ['--tile_step_size', '0.8'], ['--no-use_mirroring'],
        ['--device', 'cuda'], ['--overwrite'],
    ]
    cohort['argv'] = [item for group in invocation for item in group]
    cohort['command'] = (' ' + chr(92) + '\n  ').join(shlex.join(group) for group in invocation)

# Dataset274's label 1 represents the whole gland; zone models need a different mapping.
labels = json.loads((BUNDLE / 'models/dataset.json').read_text())['labels']
assert labels.get('background') == 0 and sorted(labels.values()) == [0, 1], 'Expected binary whole-gland model'

# %% [markdown]
# ## Whole-gland measurements
#
# Compare corresponding voxel corners in physical millimetres (maximum Euclidean
# displacement). Equal shapes with displacement <= 0.25 mm are accepted as small
# header discrepancies and recorded. Larger differences are invalid, without
# resampling. Volumes use each mask's native affine determinant, converted from
# mm³ to mL. DSC is 2|prediction ∩ reference| / (|prediction| + |reference|).
# Empty predictions score zero; empty references are invalid.

# %%
def measure_masks(prediction_path, reference_path):
    """One directly testable calculation, shared by every cohort."""
    pred_img, ref_img = nib.load(prediction_path), nib.load(reference_path)
    if len(pred_img.shape) != 3 or pred_img.shape != ref_img.shape:
        raise ValueError('Mask shapes differ or are not 3D')
    for img in (pred_img, ref_img):
        if img.header.get_xyzt_units()[0] != 'mm':
            raise ValueError('Mask spatial units must be mm')
        if not np.isfinite(img.affine).all() or abs(np.linalg.det(img.affine[:3, :3])) <= 0:
            raise ValueError('Invalid mask affine')
    corners = np.array(list(itertools.product(*[(0, n - 1) for n in ref_img.shape])))
    displacement = np.linalg.norm(
        nib.affines.apply_affine(pred_img.affine, corners)
        - nib.affines.apply_affine(ref_img.affine, corners), axis=1,
    ).max()
    if displacement > GEOMETRY_TOLERANCE_MM:
        raise ValueError(f'Geometry displacement {displacement:.6g} mm exceeds tolerance')
    pred, ref = np.asanyarray(pred_img.dataobj), np.asanyarray(ref_img.dataobj)
    if not np.isin(pred, [0, 1]).all() or not np.isin(ref, [0, 1]).all():
        raise ValueError('Expected binary whole-gland labels 0 and 1')
    pred, ref = pred == 1, ref == 1
    pred_n, ref_n = int(pred.sum()), int(ref.sum())
    if ref_n == 0:
        raise ValueError('Empty reference mask')
    pred_ml = pred_n * abs(np.linalg.det(pred_img.affine[:3, :3])) / 1000
    ref_ml = ref_n * abs(np.linalg.det(ref_img.affine[:3, :3])) / 1000
    error_ml = pred_ml - ref_ml
    return dict(
        dsc=2 * int((pred & ref).sum()) / (pred_n + ref_n),
        predicted_volume_ml=pred_ml, reference_volume_ml=ref_ml,
        volume_error_ml=error_ml, squared_volume_error_ml2=error_ml ** 2,
        predicted_voxels=pred_n, reference_voxels=ref_n,
        max_corner_displacement_mm=float(displacement),
    )

# %% [markdown]
# ## Evaluate saved predictions and audit coverage
#
# Missing runs are expected before inference. Keep a row for every manifest case,
# report failures, and withhold aggregate performance until coverage is complete.
# Per-case calculations may be inspected during partial runs; the evaluated count
# includes only cases with valid inputs, labels, and mask geometry.

# %%
def evaluate_cohorts(cohorts, failures=None):
    """Evaluate before and after inference using the same cohort and mask checks."""
    failures = failures or {}
    image_sets = [{(c['data_root'] / row['image']).resolve() for row in c['rows']} for c in cohorts]
    overlap_errors = {c['key']: [] for c in cohorts}
    for i, j in itertools.combinations(range(len(cohorts)), 2):
        if image_sets[i] & image_sets[j]:
            for k in (i, j):
                overlap_errors[cohorts[k]['key']].append('Input images overlap another cohort')

    case_records, audit_records = [], []
    for cohort in cohorts:
        out = cohort['output']
        errors = overlap_errors[cohort['key']].copy()
        if cohort['key'] in failures:
            errors.append(failures[cohort['key']])
        ids = [row['id'] for row in cohort['rows']]
        images = [(cohort['data_root'] / row['image']).resolve() for row in cohort['rows']]
        if not ids or len(ids) != len(set(ids)) or len(images) != len(set(images)):
            errors.append('Empty cohort, duplicate case IDs, or duplicate images')
        if any(not isinstance(case_id, str) or Path(case_id).name != case_id
               or case_id in ('', '.', '..') for case_id in ids):
            errors.append('Invalid case IDs')
        expected_names = {f"{row['id']}.nii.gz" for row in cohort['rows']}
        actual_names = {p.name for p in out.glob('*.nii*') if p.is_file()}
        unexpected = sorted(actual_names - expected_names)
        if unexpected:
            errors.append('Unexpected prediction filenames')
        cohort_cases = []
        with tqdm(total=len(cohort['rows']), desc=f"Evaluating {cohort['title']}", unit='cases') as progress:
            for row in cohort['rows']:
                pred = out / f"{row['id']}.nii.gz"
                image, ref = [cohort['data_root'] / row[key] for key in ('image', 'label')]
                record = dict(model=MODEL, cohort=cohort['key'], case_id=row['id'],
                              image_path=str(image), reference_path=str(ref), prediction_path=str(pred),
                              prediction_present=pred.is_file(), status='missing_prediction', error='')
                if errors:
                    record.update(status='invalid_cohort', error='; '.join(errors))
                elif not image.is_file() or not ref.is_file():
                    record.update(status='invalid_input', error='Missing image or reference')
                elif pred.is_file():
                    try:
                        record.update(measure_masks(pred, ref))
                        record['status'] = 'evaluated'
                    except Exception as exc:
                        record.update(status='invalid_mask', error=f'{type(exc).__name__}: {exc}')
                cohort_cases.append(record)
                progress.update(1)
        case_records.extend(cohort_cases)
        evaluated = sum(r['status'] == 'evaluated' for r in cohort_cases)
        audit_records.append(dict(
            model=MODEL, cohort=cohort['key'], expected=len(cohort_cases),
            present=sum(r['prediction_present'] for r in cohort_cases),
            missing=sum(not r['prediction_present'] for r in cohort_cases),
            invalid=sum(r['status'].startswith('invalid') for r in cohort_cases), evaluated=evaluated,
            unexpected=len(unexpected), unexpected_filenames='; '.join(unexpected),
            complete=not errors and evaluated == len(cohort_cases), cohort_errors='; '.join(errors),
        ))

    metric_columns = ['dsc', 'predicted_volume_ml', 'reference_volume_ml', 'volume_error_ml',
                      'squared_volume_error_ml2', 'predicted_voxels', 'reference_voxels',
                      'max_corner_displacement_mm']
    per_case = pd.DataFrame(case_records).reindex(columns=[
        'model', 'cohort', 'case_id', 'image_path', 'reference_path', 'prediction_path',
        'prediction_present', 'status', 'error', *metric_columns,
    ])
    cohort_audit = pd.DataFrame(audit_records)
    return per_case, cohort_audit


# %% [markdown]
# ## Approve and run pending GPU inference
#
# The single approval covers all printed commands. Logs go to each prediction
# directory's inference.log. Existing masks are overwritten, so progress counts
# newly created or changed expected files rather than all files in the folder.
# File counts are approximate export progress, not proof of postprocessing success.

# %%
def prediction_signatures(cohort):
    signatures = {}
    for row in cohort['rows']:
        path = cohort['output'] / f"{row['id']}.nii.gz"
        try:
            stat = path.stat()
            if path.is_file():
                signatures[path.name] = (stat.st_mtime_ns, stat.st_size)
        except FileNotFoundError:
            pass
    return signatures


def stop_inference_group(process):
    """Clean up nnUNet multiprocessing children as well as the launcher."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # Also catch children that outlive a launcher which has already exited.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_pending(cohorts, audit):
    """Return failure records and whether inference execution was attempted."""
    pending = [c for c in cohorts if not audit.loc[audit.cohort.eq(c['key']), 'complete'].iloc[0]]
    failures = {}
    if not pending:
        print('All cohorts are complete; no inference needed.')
        return failures, False
    for cohort in pending:
        row = audit.loc[audit.cohort.eq(cohort['key'])].iloc[0]
        print(f"{cohort['title']}: {row.evaluated}/{row.expected} evaluated "
              f"({row.missing} missing, {row.invalid} invalid)")
        print(cohort['command'] + '\n')
    try:
        approved = input('Run all commands above? [y/N] ').strip().lower() in ('y', 'yes')
    except (EOFError, OSError, NotImplementedError, KeyboardInterrupt):
        approved = False
    if not approved:
        print('Inference skipped; generating tables from available predictions.')
        return failures, False

    for cohort in pending:
        process = None
        log_path = cohort['output'] / 'inference.log'
        try:
            cohort['output'].mkdir(parents=True, exist_ok=True)
            print(f"{cohort['title']} log: {log_path.relative_to(REPO_ROOT)}")
            with log_path.open('w') as log:
                baseline = prediction_signatures(cohort)
                environment = dict(os.environ, PYTHONUNBUFFERED='1')
                process = subprocess.Popen(cohort['argv'], cwd=REPO_ROOT, env=environment,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                with tqdm(total=len(cohort['rows']), desc=cohort['title'], unit='files') as progress:
                    while True:
                        returncode = process.poll()
                        exported = sum(signature != baseline.get(name)
                                       for name, signature in prediction_signatures(cohort).items())
                        progress.n = exported
                        progress.set_postfix_str('waiting for inference/postprocessing'
                                                 if exported == progress.total else 'exported prediction files')
                        progress.refresh()
                        if returncode is not None:
                            break
                        time.sleep(1)
            if returncode != 0:
                stop_inference_group(process)
                failures[cohort['key']] = f'Inference failed with exit code {returncode}'
                print(f"{failures[cohort['key']]}; see {log_path.relative_to(REPO_ROOT)}. Queue stopped.")
                break
            print(f"{cohort['title']}: inference and postprocessing finished.")
        except KeyboardInterrupt:
            if process is not None:
                stop_inference_group(process)
            failures[cohort['key']] = 'Inference interrupted'
            print('Inference interrupted; worker group stopped. Queue stopped.')
            break
        except OSError as exc:
            if process is not None:
                stop_inference_group(process)
            failures[cohort['key']] = f'Inference could not run: {exc}'
            print(f"{failures[cohort['key']]}; see {log_path.relative_to(REPO_ROOT)}. Queue stopped.")
            break
    return failures, True


# %%
per_case, cohort_audit = evaluate_cohorts(COHORTS)
# The first pass identifies pending cohorts; refresh only if execution was attempted.
run_failures, execution_attempted = run_pending(COHORTS, cohort_audit)
if execution_attempted:
    per_case, cohort_audit = evaluate_cohorts(COHORTS, run_failures)
display(cohort_audit)

# %% [markdown]
# ## Performance and case-count tables
#
# Each scan has equal weight in mean DSC. RMSE = sqrt(mean((predicted mL −
# reference mL)²)), not the mean absolute error or the existing `vol_sim` metric.
# All three tables use the same model/cohort layout. Counts show evaluated/expected.

# %%
dsc_table = pd.DataFrame(index=pd.Index([MODEL], name='Model'), dtype=float)
rmse_table = dsc_table.copy()
counts_table = pd.DataFrame(index=dsc_table.index)
for cohort in COHORTS:
    audit = cohort_audit.loc[cohort_audit.cohort.eq(cohort['key'])].iloc[0]
    cases = per_case.loc[per_case.cohort.eq(cohort['key'])]
    dsc_table[cohort['title']] = cases.dsc.mean() if audit.complete else np.nan
    rmse_table[cohort['title']] = np.sqrt(cases.squared_volume_error_ml2.mean()) if audit.complete else np.nan
    counts_table[cohort['title']] = f'{audit.evaluated}/{audit.expected}'

# Avoid an optional tabulate dependency for these small, rectangular tables.
for name, title, table in [
    ('dsc', 'Mean whole-gland DSC (0–1)', dsc_table),
    ('volume_rmse_ml', 'Whole-gland volume RMSE (mL)', rmse_table),
    ('case_counts', 'Evaluated / expected scans', counts_table),
]:
    print(title)
    display(table.fillna('NA'))
    table.to_csv(TABLE_DIR / f'{name}.csv', na_rep='NA')
    formatted = table.map(lambda value: 'NA' if pd.isna(value) else
                          f'{value:.6f}' if isinstance(value, (float, np.floating)) else str(value))
    lines = ['| Model | ' + ' | '.join(formatted.columns) + ' |',
             '| --- | ' + ' | '.join(['---'] * len(formatted.columns)) + ' |']
    lines += ['| ' + str(index) + ' | ' + ' | '.join(row) + ' |' for index, row in formatted.iterrows()]
    (TABLE_DIR / f'{name}.md').write_text(f'# {title}\n\n' + '\n'.join(lines) + '\n')
per_case.to_csv(TABLE_DIR / 'per_case_metrics.csv', index=False, na_rep='NA')
cohort_audit.to_csv(TABLE_DIR / 'cohort_audit.csv', index=False, na_rep='NA')
print(f'Tables and audit: {TABLE_DIR.relative_to(REPO_ROOT)}')
