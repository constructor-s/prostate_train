# %% [markdown]
# # Model study performance by dataset
#
# Compare Dataset274 (pooled2) and Dataset236 (RUMC excluding PROSTATEx) on
# their requested training/CV and test cohorts. Counts are scans, not patients.
# Five-fold ensemble training-pool scores are not out-of-fold validation:
# ensemble members have seen these cases during training.
# RUMC testing is the mapped PROSTATEx original training cohort (manual masks),
# held out from both models. RUMC training uses Bosma22b automated surrogate
# reference masks; its scores measure agreement with that automated reference.
# pooled2 testing is external to Dataset236. The separate endorectal_coil split
# is evaluated on both models, immediately below pooled2 testing in tables.
# Tables group rows by dataset/split;
# Dataset274 on RUMC training is not requested and remains NA. The clinical
# cohort compares volumes with median manual-reader planimetry; its DSC is NA.
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
from tqdm.contrib.concurrent import thread_map

STUDY_ID = '20261105_tile0.8_nomirroring'
REPO_ROOT = Path(__file__).resolve().parents[1] if '__file__' in globals() else Path.cwd().resolve()
if REPO_ROOT.name == 'notebooks':
    REPO_ROOT = REPO_ROOT.parent
assert (REPO_ROOT / 'pyproject.toml').is_file(), 'Set REPO_ROOT to the repository root'
assert Path(STUDY_ID).name == STUDY_ID and STUDY_ID not in ('', '.', '..')
STUDY_ROOT = REPO_ROOT / 'results' / STUDY_ID
TABLE_DIR = STUDY_ROOT / 'tables'
TABLE_DIR.mkdir(parents=True, exist_ok=True)
GEOMETRY_TOLERANCE_MM = 0.5
GEOMETRY_WARNING_THRESHOLD_MM = 1e-3
# Keep Dataset274's existing output keys so its saved masks remain usable.
DATASETS = [
    dict(key='pooled2_train', dataset='pooled2', split='training'),
    dict(key='pooled2_test', dataset='pooled2', split='testing'),
    dict(key='pooled2_endorectal_coil', dataset='pooled2', split='endorectal_coil'),
    dict(key='rumc_train', dataset='rumcnotprostatexnoclinical', split='training'),
    dict(key='prostatex_original_train_mapped', dataset='rumcnotprostatexnoclinical', split='testing'),
]
MODELS = [
    dict(name='Dataset274_raw_3d_fullres_5fold',
         cohort_keys=['pooled2_train', 'pooled2_test', 'pooled2_endorectal_coil', 'prostatex_original_train_mapped']),
    dict(name='Dataset236_raw_3d_fullres_5fold',
         cohort_keys=['rumc_train', 'prostatex_original_train_mapped', 'pooled2_test', 'pooled2_endorectal_coil']),
]

# %% [markdown]
# ## Cohort inputs and reproducible commands
#
# Read the original manifests and data roots from their input configurations.
# Inference accepts these JSON files directly with --split training, testing,
# or endorectal_coil.

# %%
COHORTS = []
for model in MODELS:
    bundle = REPO_ROOT / 'bundles' / model['name']
    labels = json.loads((bundle / 'models/dataset.json').read_text())['labels']
    assert labels.get('background') == 0 and sorted(labels.values()) == [0, 1], (
        f"{model['name']}: expected binary whole-gland model"
    )
    for key in model['cohort_keys']:
        cohort = next(d.copy() for d in DATASETS if d['key'] == key)
        cohort['reference_type'] = 'segmentation_mask'
        cohort['model'] = model['name']
        # Run IDs distinguish shared evaluation datasets across models.
        cohort['run_id'] = (cohort['model'], cohort['key'])
        cohort['title'] = f"{model['name']}: {cohort['dataset']} {cohort['split']}"
        config = yaml.safe_load((REPO_ROOT / 'configs' / f"{cohort['dataset']}_input.yaml").read_text())
        cohort['manifest'] = (REPO_ROOT / config['datalist']).resolve()
        cohort['data_root'] = (REPO_ROOT / config['dataroot']).resolve()
        cohort['manifest_data'] = json.loads(cohort['manifest'].read_text())
        cohort['rows'] = cohort['manifest_data'][cohort['split']]
        cohort['output'] = STUDY_ROOT / cohort['model'] / cohort['key']
        invocation = [
            ['.venv/bin/python', str((bundle / 'scripts/infer.py').relative_to(REPO_ROOT))],
            ['--input', str(cohort['manifest'].relative_to(REPO_ROOT))],
            ['--split', cohort['split']],
            ['--data_root', str(cohort['data_root'].relative_to(REPO_ROOT))],
            ['--output_folder', str(cohort['output'].relative_to(REPO_ROOT))],
            ['--tile_step_size', '0.8'], ['--no-use_mirroring'],
            ['--device', 'cuda'], ['--overwrite'],
        ]
        cohort['argv'] = [item for group in invocation for item in group]
        cohort['command'] = (' ' + chr(92) + '\n  ').join(shlex.join(group) for group in invocation)
        COHORTS.append(cohort)

# %% [markdown]
# Clinical references are median manual-reader planimetry in mL. Select finite,
# positive volumes and exclude every accession listed in data_internal/hlab20260928/exclude.json,
# regardless of reason. Match aano to the final underscore-delimited ID component.
# Record all causes in tables/clinical_exclusions.csv; give retained scans equal weight.
# Only missing predictions
# join the GPU queue; existing clinical masks are preserved.

# %%
CLINICAL_CSV = REPO_ROOT / 'data_internal/hlab20260928/hlab20260928_t2_image_listing_full_clinical.csv'
clinical_listing = pd.read_csv(CLINICAL_CSV, dtype={'ID': 'string', 't2': 'string'})
if clinical_listing.ID.isna().any() or clinical_listing.ID.duplicated().any():
    raise ValueError('Clinical listing requires nonmissing, unique scan IDs')
if any(Path(case_id).name != case_id or case_id in ('', '.', '..') for case_id in clinical_listing.ID):
    raise ValueError('Invalid clinical scan IDs')
CLINICAL_EXCLUDE_JSON = REPO_ROOT / 'data_internal/hlab20260928/exclude.json'
clinical_exclude_entries = json.loads(CLINICAL_EXCLUDE_JSON.read_text())
if not isinstance(clinical_exclude_entries, list) or any(
    not isinstance(entry, dict) or not isinstance(entry.get('aano'), str)
    or not entry['aano'].strip() for entry in clinical_exclude_entries
):
    raise ValueError('Clinical exclusion JSON requires a list of entries with nonempty aano strings')
clinical_exclude_reasons = {}
for entry in clinical_exclude_entries:
    details = '; '.join(str(entry[key]) for key in ('reason', 'description') if entry.get(key))
    reason = 'Listed in exclude.json' + (f': {details}' if details else '')
    clinical_exclude_reasons.setdefault(entry['aano'], []).append(reason)
clinical_accession = clinical_listing.ID.str.rsplit('_', n=1).str[-1]
clinical_manual_excluded = clinical_accession.isin(clinical_exclude_reasons)
clinical_volume = pd.to_numeric(clinical_listing.ground_truth_volume_ml, errors='coerce')
clinical_valid_volume = np.isfinite(clinical_volume) & clinical_volume.gt(0)
clinical_eligible = clinical_valid_volume & ~clinical_manual_excluded
clinical_exclusion_reason = clinical_accession.map(
    {accession: '; '.join(reasons) for accession, reasons in clinical_exclude_reasons.items()}
).fillna('')
invalid_volume_reason = 'Missing, nonnumeric, nonfinite, or nonpositive clinical volume'
clinical_exclusion_reason.loc[~clinical_valid_volume] = clinical_exclusion_reason.loc[
    ~clinical_valid_volume
].map(lambda reason: f'{reason}; {invalid_volume_reason}' if reason else invalid_volume_reason)
clinical_exclusions = clinical_listing.loc[~clinical_eligible].copy()
clinical_exclusions['exclusion_reason'] = clinical_exclusion_reason.loc[~clinical_eligible]
clinical_cases = clinical_listing.loc[clinical_eligible].copy()
clinical_cases['ground_truth_volume_ml'] = clinical_volume.loc[clinical_eligible]
if clinical_cases.empty:
    raise ValueError('No eligible clinical reference volumes')
if clinical_cases.t2.isna().any() or clinical_cases.t2.str.strip().eq('').any():
    raise ValueError('Eligible clinical scans require T2 paths')
clinical_cases['image_path'] = clinical_cases.t2.map(lambda p: str((CLINICAL_CSV.parent / p).resolve()))
if clinical_cases.image_path.duplicated().any():
    raise ValueError('Duplicate eligible clinical image paths')
print(f'Clinical scans: {len(clinical_cases)} eligible / {len(clinical_listing)} listed; '
      f'{len(clinical_exclusions)} excluded total; '
      f'{clinical_manual_excluded.sum()} listed in exclude.json; '
      f'{(~clinical_valid_volume).sum()} invalid reference volumes (causes may overlap)')
clinical_exclusions.to_csv(TABLE_DIR / 'clinical_exclusions.csv', index=False, na_rep='NA')


DATASETS.append(dict(key='clinical', dataset='hlab20260928_manual_planimetry', split='clinical'))
for model in MODELS:
    name = model['name']
    COHORTS.append(dict(
        key='clinical', dataset='hlab20260928_manual_planimetry', split='clinical',
        reference_type='clinical_volume', model=name, run_id=(name, 'clinical'),
        title=f'{name}: clinical manual planimetry', data_root=CLINICAL_CSV.parent,
        rows=[dict(id=row.ID, image=row.image_path, reference_volume_ml=row.ground_truth_volume_ml)
              for row in clinical_cases.itertuples(index=False)],
        output=CLINICAL_CSV.parent / f'seg_out_{name}',
    ))

# Saved nnUNet-702 masks use the union of labels 1, 2, and 3 as whole prostate.
# This comparator shares clinical eligibility and is never queued for inference.
COHORTS.append(dict(
    key='clinical', dataset='hlab20260928_manual_planimetry', split='clinical',
    reference_type='clinical_volume', model='nnunet-702', run_id=('nnunet-702', 'clinical'),
    title='nnunet-702: clinical manual planimetry', data_root=CLINICAL_CSV.parent,
    saved_mask_only=True, foreground_labels=(1, 2, 3),
    rows=[dict(id=row.ID, image=row.image_path, reference_volume_ml=row.ground_truth_volume_ml,
               prediction_path=str(CLINICAL_CSV.parent / 'PROSATEPSAD' / row.ID
                                   / 'wp-pz-tz_axt2_nnunet-702.nii.gz'))
          for row in clinical_cases.itertuples(index=False)],
    output=CLINICAL_CSV.parent / 'PROSATEPSAD',
))

# %% [markdown]
# ## Clinical manual ellipsoid comparator
#
# Reported volumes are already in mL. PROSTATEx uses the current testing
# manifest and an unambiguous patient/study marksheet match; HLAB uses the same
# eligible manual-planimetry scans as the models. Neither comparator runs inference.

# %%
ELLIPSOID_NAME = 'Clinical manual ellipsoid'
MARKSHEET_PATH = REPO_ROOT / 'data/raw/zenodo/picai/picai_labels/clinical_information/marksheet.csv'
MAPPING_PATH = REPO_ROOT / 'data/raw/zenodo/picai/picai_labels/additional_resources/ProstateX-mapping.json'
marksheet = pd.read_csv(MARKSHEET_PATH)
mapping = json.loads(MAPPING_PATH.read_text())
prostatex_source = next(c for c in COHORTS if c['key'] == 'prostatex_original_train_mapped')
links = pd.DataFrame(
    [(key.split('_')[0], value, *value.split('_')) for key, value in mapping.items()],
    columns=['id', 'mapped_exam', 'patient_id', 'study_id'],
)
for column in ('patient_id', 'study_id'):
    links[column] = pd.to_numeric(links[column], errors='raise')
if marksheet.duplicated(['patient_id', 'study_id']).any():
    raise ValueError('Duplicate marksheet patient/study keys')
links = links.merge(marksheet[['patient_id', 'study_id', 'prostate_volume']],
                    on=['patient_id', 'study_id'], how='left', validate='many_to_one')
ellipsoid_rows = []
for row in prostatex_source['rows']:
    matches = links.loc[links.id.eq(row['id'])]
    unambiguous = len(matches) == 1
    ellipsoid_rows.append(dict(
        row, predicted_volume_ml=pd.to_numeric(matches.iloc[0].prostate_volume, errors='coerce')
        if unambiguous else np.nan,
        source_file=str(MARKSHEET_PATH), source_column='prostate_volume',
        mapped_exam=','.join(matches.mapped_exam),
        mapped_patient_id=matches.iloc[0].patient_id if unambiguous else np.nan,
        mapped_study_id=matches.iloc[0].study_id if unambiguous else np.nan,
        source_error='' if unambiguous else 'Missing or ambiguous PROSTATEx mapping',
    ))
COHORTS.append(dict(
    key=prostatex_source['key'], dataset=prostatex_source['dataset'], split=prostatex_source['split'],
    reference_type='segmentation_mask', prediction_type='reported_volume', model=ELLIPSOID_NAME,
    run_id=(ELLIPSOID_NAME, prostatex_source['key']), title='Clinical ellipsoid: PROSTATEx',
    data_root=prostatex_source['data_root'], rows=ellipsoid_rows,
))
COHORTS.append(dict(
    key='clinical', dataset='hlab20260928_manual_planimetry', split='clinical',
    reference_type='clinical_volume', prediction_type='reported_volume', model=ELLIPSOID_NAME,
    run_id=(ELLIPSOID_NAME, 'clinical'), title='Clinical ellipsoid: HLAB', data_root=CLINICAL_CSV.parent,
    rows=[dict(id=row.ID, image=row.image_path, reference_volume_ml=row.ground_truth_volume_ml,
               predicted_volume_ml=pd.to_numeric(row.vol_report_ref, errors='coerce'),
               source_file=str(CLINICAL_CSV), source_column='vol_report_ref')
          for row in clinical_cases.itertuples(index=False)],
))

# %% [markdown]
# ## Whole-gland measurements
#
# Compare corresponding voxel corners in physical millimetres (maximum Euclidean
# displacement). Equal shapes with displacement <= 0.5 mm are accepted as small
# header discrepancies and recorded. Larger differences are invalid, without
# resampling. Volumes use each mask's native affine determinant, converted from
# mm³ to mL. DSC is 2|prediction ∩ reference| / (|prediction| + |reference|).
# Empty predictions have zero volume; empty segmentation references are invalid.
# Clinical geometry uses the T2 image; clinical reference volumes have no DSC.
# Evaluated cases with displacement > 1e-3 mm are reported; smaller differences
# remain recorded in the per-case CSV. Unrelated header metadata does not affect
# this comparison.

# %%
def measure_reported_volume(predicted_ml, reference_path=None, reference_ml=None):
    """Compare a reported ellipsoid volume with a mask or scalar reference."""
    if not np.isfinite(predicted_ml) or predicted_ml <= 0:
        raise ValueError('Reported volume must be finite and positive')
    if reference_path is not None:
        img = nib.load(reference_path)
        if len(img.shape) != 3 or img.header.get_xyzt_units()[0] != 'mm':
            raise ValueError('Reference mask must be 3D with spatial units mm')
        if not np.isfinite(img.affine).all() or abs(np.linalg.det(img.affine[:3, :3])) <= 0:
            raise ValueError('Invalid reference affine')
        data = np.asanyarray(img.dataobj)
        if not np.isin(data, [0, 1]).all() or not (data == 1).any():
            raise ValueError('Expected nonempty binary whole-gland reference')
        reference_ml = int((data == 1).sum()) * abs(np.linalg.det(img.affine[:3, :3])) / 1000
    if reference_ml is None or not np.isfinite(reference_ml) or reference_ml <= 0:
        raise ValueError('Reference volume must be finite and positive')
    error = predicted_ml - reference_ml
    return dict(dsc=np.nan, predicted_volume_ml=predicted_ml, reference_volume_ml=reference_ml,
                volume_error_ml=error, squared_volume_error_ml2=error ** 2)


def measure_masks(prediction_path, reference_path, clinical_volume_ml=None, foreground_labels=(1,)):
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
    pred = np.asanyarray(pred_img.dataobj)
    if not np.isin(pred, [0, *foreground_labels]).all():
        if tuple(foreground_labels) == (1,):
            raise ValueError('Expected binary whole-gland labels 0 and 1')
        raise ValueError(f'Expected whole-gland labels 0 and {foreground_labels}')
    pred = np.isin(pred, foreground_labels)
    pred_n = int(pred.sum())
    pred_ml = pred_n * abs(np.linalg.det(pred_img.affine[:3, :3])) / 1000
    if clinical_volume_ml is not None:
        if not np.isfinite(clinical_volume_ml) or clinical_volume_ml <= 0:
            raise ValueError('Clinical reference volume must be finite and positive')
        error_ml = pred_ml - clinical_volume_ml
        return dict(dsc=np.nan, predicted_volume_ml=pred_ml, reference_volume_ml=clinical_volume_ml,
                    volume_error_ml=error_ml, squared_volume_error_ml2=error_ml ** 2,
                    predicted_voxels=pred_n, reference_voxels=np.nan,
                    max_corner_displacement_mm=float(displacement))
    ref = np.asanyarray(ref_img.dataobj)
    if not np.isin(ref, [0, 1]).all():
        raise ValueError('Expected binary whole-gland labels 0 and 1')
    ref = ref == 1
    ref_n = int(ref.sum())
    if ref_n == 0:
        raise ValueError('Empty reference mask')
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
# Only manifest-listed cases are evaluated; extra saved predictions are audited
# and reported as warnings without affecting cohort completeness.

# %%
def evaluate_cohorts(cohorts, failures=None):
    """Evaluate before and after inference using the same cohort and mask checks."""
    failures = failures or {}
    image_sets = [{(c['data_root'] / row['image']).resolve() for row in c['rows']} for c in cohorts]
    overlap_errors = {c['run_id']: [] for c in cohorts}
    for i, j in itertools.combinations(range(len(cohorts)), 2):
        if cohorts[i]['model'] == cohorts[j]['model'] and image_sets[i] & image_sets[j]:
            for k in (i, j):
                overlap_errors[cohorts[k]['run_id']].append('Input images overlap another cohort')

    case_records, audit_records = [], []
    for cohort in cohorts:
        scalar = cohort.get('prediction_type') == 'reported_volume'
        out = cohort.get('output')
        errors = overlap_errors[cohort['run_id']].copy()
        if cohort['run_id'] in failures:
            errors.append(failures[cohort['run_id']])
        ids = [row['id'] for row in cohort['rows']]
        images = [(cohort['data_root'] / row['image']).resolve() for row in cohort['rows']]
        if not ids or len(ids) != len(set(ids)) or len(images) != len(set(images)):
            errors.append('Empty cohort, duplicate case IDs, or duplicate images')
        if any(not isinstance(case_id, str) or Path(case_id).name != case_id
               or case_id in ('', '.', '..') for case_id in ids):
            errors.append('Invalid case IDs')
        expected_names = {f"{row['id']}.nii.gz" for row in cohort['rows']}
        actual_names = set() if scalar else {p.name for p in out.glob('*.nii*') if p.is_file()}
        unexpected = sorted(actual_names - expected_names)
        clinical = cohort.get('reference_type') == 'clinical_volume'
        def evaluate_case(row):
            pred = None if scalar else (Path(row['prediction_path']) if 'prediction_path' in row else out / f"{row['id']}.nii.gz")
            image = cohort['data_root'] / row['image']
            ref = None if clinical else cohort['data_root'] / row['label']
            record = dict(model=cohort['model'], dataset=cohort['dataset'], split=cohort['split'],
                          cohort=cohort['key'], region='whole_gland', case_id=row['id'],
                          reference_type=cohort.get('reference_type', 'segmentation_mask'),
                          prediction_type=cohort.get('prediction_type', 'segmentation_mask'),
                          source_file=row.get('source_file', ''), source_column=row.get('source_column', ''),
                          mapped_exam=row.get('mapped_exam', ''),
                          mapped_patient_id=row.get('mapped_patient_id', np.nan),
                          mapped_study_id=row.get('mapped_study_id', np.nan),
                          image_path=str(image), reference_path=str(ref) if ref else '', prediction_path=str(pred) if pred else '',
                          reference_volume_ml=row.get('reference_volume_ml', np.nan),
                          prediction_present=bool(np.isfinite(row['predicted_volume_ml'])) if scalar else pred.is_file(),
                          status='missing_prediction', error='')
            if errors:
                record.update(status='invalid_cohort', error='; '.join(errors))
            elif not image.is_file() or (ref is not None and not ref.is_file()):
                record.update(status='invalid_input', error='Missing image or reference')
            elif scalar:
                try:
                    if row.get('source_error'):
                        raise ValueError(row['source_error'])
                    record.update(measure_reported_volume(
                        row['predicted_volume_ml'], reference_path=ref,
                        reference_ml=row.get('reference_volume_ml')))
                    record['status'] = 'evaluated'
                except Exception as exc:
                    record.update(status='invalid_volume', error=f'{type(exc).__name__}: {exc}')
            elif pred.is_file():
                try:
                    measurement_options = (
                        dict(foreground_labels=cohort['foreground_labels'])
                        if 'foreground_labels' in cohort else {}
                    )
                    record.update(measure_masks(pred, image, row['reference_volume_ml'], **measurement_options)
                                  if clinical else measure_masks(pred, ref, **measurement_options))
                    record['status'] = 'evaluated'
                except Exception as exc:
                    record.update(status='invalid_mask', error=f'{type(exc).__name__}: {exc}')
            return record

        cohort_cases = thread_map(
            evaluate_case, cohort['rows'],
            desc=f"Evaluating {cohort['title']}", unit='cases',
        )
        case_records.extend(cohort_cases)
        evaluated = sum(r['status'] == 'evaluated' for r in cohort_cases)
        provenance_path = None if scalar else out / 'inference_command.txt'
        provenance = provenance_path.read_text().strip() if provenance_path and provenance_path.is_file() else ''
        audit_records.append(dict(
            model=cohort['model'], dataset=cohort['dataset'], split=cohort['split'],
            cohort=cohort['key'], reference_type=cohort.get('reference_type', 'segmentation_mask'),
            prediction_type=cohort.get('prediction_type', 'segmentation_mask'),
            inference_command=provenance, provenance_available=bool(provenance),
            expected=len(cohort_cases),
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
        'model', 'dataset', 'split', 'cohort', 'reference_type', 'region', 'case_id', 'image_path', 'reference_path', 'prediction_path',
        'prediction_type', 'source_file', 'source_column', 'mapped_exam', 'mapped_patient_id', 'mapped_study_id',
        'prediction_present', 'status', 'error', *metric_columns,
    ])
    per_case['absolute_volume_error_ml'] = per_case.volume_error_ml.abs()
    per_case['relative_volume_error'] = per_case.volume_error_ml / per_case.reference_volume_ml
    cohort_audit = pd.DataFrame(audit_records)
    return per_case, cohort_audit


# %% [markdown]
# ## Approve and run pending GPU inference
#
# The single approval covers all printed commands. Logs go to each prediction
# directory's inference.log. Segmentation runs overwrite masks; clinical runs
# process only missing predictions. Progress counts
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
    pending = []
    for cohort in cohorts:
        if cohort.get('saved_mask_only', False) or cohort.get('prediction_type') == 'reported_volume':
            continue
        row = audit.loc[audit.model.eq(cohort['model']) & audit.cohort.eq(cohort['key'])].iloc[0]
        if row.complete:
            continue
        if cohort.get('reference_type') != 'clinical_volume':
            pending.append(cohort)
            continue
        missing = [r for r in cohort['rows'] if not (cohort['output'] / f"{r['id']}.nii.gz").is_file()]
        if row.invalid:
            print(f"{cohort['title']}: invalid existing cases require review; masks will be preserved.")
        if not missing:
            continue
        manifest = TABLE_DIR / f"clinical_missing_{cohort['model']}.json"
        manifest.write_text(json.dumps({'testing': [dict(id=r['id'], image=str(
            (cohort['data_root'] / r['image']).resolve())) for r in missing]}, indent=2) + '\n')
        argv = [str(REPO_ROOT / '.venv/bin/python'),
                str(REPO_ROOT / 'bundles' / cohort['model'] / 'scripts/infer.py'),
                '--input', str(manifest.resolve()), '--split', 'testing',
                '--output_folder', str(cohort['output'].resolve()),
                '--tile_step_size', '0.8', '--no-use_mirroring', '--device', 'cuda']
        pending.append(dict(cohort, rows=missing, argv=argv, command=shlex.join(argv)))
    failures = {}
    if not pending:
        print('No runnable pending inference; inspect coverage for invalid cases.')
        return failures, False
    for cohort in pending:
        row = audit.loc[audit.model.eq(cohort['model']) & audit.cohort.eq(cohort['key'])].iloc[0]
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
                failures[cohort['run_id']] = f'Inference failed with exit code {returncode}'
                print(f"{failures[cohort['run_id']]}; see {log_path.relative_to(REPO_ROOT)}. Queue stopped.")
                break
            print(f"{cohort['title']}: inference and postprocessing finished.")
        except KeyboardInterrupt:
            if process is not None:
                stop_inference_group(process)
            failures[cohort['run_id']] = 'Inference interrupted'
            print('Inference interrupted; worker group stopped. Queue stopped.')
            break
        except OSError as exc:
            if process is not None:
                stop_inference_group(process)
            failures[cohort['run_id']] = f'Inference could not run: {exc}'
            print(f"{failures[cohort['run_id']]}; see {log_path.relative_to(REPO_ROOT)}. Queue stopped.")
            break
    return failures, True


# %%
per_case, cohort_audit = evaluate_cohorts(COHORTS)
# The first pass identifies pending cohorts; refresh only if execution was attempted.
run_failures, execution_attempted = run_pending(COHORTS, cohort_audit)
if execution_attempted:
    per_case, cohort_audit = evaluate_cohorts(COHORTS, run_failures)
display(cohort_audit)
for row in cohort_audit.loc[cohort_audit.unexpected.gt(0)].itertuples(index=False):
    print(f'WARNING: {row.model} / {row.cohort}: ignoring {row.unexpected} '
          f'prediction files outside the manifest: {row.unexpected_filenames}')
for row in cohort_audit.itertuples(index=False):
    if row.prediction_type == 'reported_volume':
        continue
    print(f'{row.model} / {row.cohort} saved command: '
          f'{row.inference_command or "Unavailable; historical settings unverified"}')

# Report once after the final pass, preserving cohort and manifest case order.
geometry_mismatches = per_case.loc[
    per_case.status.eq('evaluated')
    & per_case.max_corner_displacement_mm.gt(GEOMETRY_WARNING_THRESHOLD_MM),
    ['model', 'cohort', 'case_id', 'max_corner_displacement_mm'],
]
if not geometry_mismatches.empty:
    print('WARNING: Evaluated cases with prediction/reference geometry displacement '
          f'> {GEOMETRY_WARNING_THRESHOLD_MM:g} mm '
          f'(accepted tolerance: {GEOMETRY_TOLERANCE_MM:g} mm):')
    print(geometry_mismatches.to_string(
        index=False, formatters={'max_corner_displacement_mm': lambda value: f'{value:.6g}'},
    ))
invalid_cases = per_case.loc[
    per_case.status.astype('string').str.startswith('invalid', na=False),
    ['model', 'cohort', 'case_id', 'error'],
]
if not invalid_cases.empty:
    print('WARNING: Invalid cases excluded from evaluation:')
    print(invalid_cases.to_string(index=False))

# %% [markdown]
# ## Performance and case-count tables
#
# Each scan has equal weight in mean DSC and volume metrics.
# MAE (mL) = mean(abs(predicted mL − reference mL)).
# RMSE (mL) = sqrt(mean((predicted mL − reference mL)²)).
# MAPE (%) = 100 × mean(abs((predicted mL − reference mL) / reference mL)).
# RMSPE (%) = 100 × sqrt(mean(((predicted mL − reference mL) / reference mL)²)).
# Each scan contributes equally after normalizing by its own reference volume.
# Per-case relative_volume_error is a signed fraction; MAPE and RMSPE are percentages.
# Lower MAE, RMSE, MAPE, and RMSPE are better; zero indicates perfect volume agreement.
# Tables group rows by dataset and split, with models in columns. The clinical
# row uses manual-reader planimetry as its volume reference; DSC remains NA.
# nnunet-702 uses saved multilabel masks and appears only in the clinical row.
# Clinical manual ellipsoid compares reported mL with manual-mask volumes in
# PROSTATEx and manual planimetry in HLAB. It has no DSC or inference command.
# Counts show evaluated/expected; NA means that model/cohort was not requested.

# %%
table_index = pd.MultiIndex.from_tuples(
    [(d['dataset'], d['split']) for d in DATASETS], names=['Dataset', 'Split'],
)
model_names = pd.Index(list(dict.fromkeys(
    [m['name'] for m in MODELS] + [c['model'] for c in COHORTS]
)), name='Model')
dsc_table = pd.DataFrame(index=table_index, columns=model_names, dtype=float)
mae_table = dsc_table.copy()
rmse_table = dsc_table.copy()
mape_table = dsc_table.copy()
rmspe_table = dsc_table.copy()
counts_table = pd.DataFrame('NA', index=table_index, columns=model_names)
for cohort in COHORTS:
    selected = cohort_audit.model.eq(cohort['model']) & cohort_audit.cohort.eq(cohort['key'])
    audit = cohort_audit.loc[selected].iloc[0]
    cases = per_case.loc[per_case.model.eq(cohort['model']) & per_case.cohort.eq(cohort['key'])]
    row_key, model_name = (cohort['dataset'], cohort['split']), cohort['model']
    dsc_table.loc[row_key, model_name] = cases.dsc.mean() if audit.complete else np.nan
    mae_table.loc[row_key, model_name] = cases.absolute_volume_error_ml.mean() if audit.complete else np.nan
    rmse_table.loc[row_key, model_name] = np.sqrt(cases.squared_volume_error_ml2.mean()) if audit.complete else np.nan
    mape_table.loc[row_key, model_name] = 100 * cases.relative_volume_error.abs().mean() if audit.complete else np.nan
    rmspe_table.loc[row_key, model_name] = 100 * np.sqrt(cases.relative_volume_error.pow(2).mean()) if audit.complete else np.nan
    counts_table.loc[row_key, model_name] = f'{audit.evaluated}/{audit.expected}'

# Avoid an optional tabulate dependency for these small, rectangular tables.
for name, title, table in [
    ('dsc', 'Mean whole-gland DSC (0–1)', dsc_table),
    ('volume_mae_ml', 'Whole-gland volume MAE (mL)', mae_table),
    ('volume_rmse_ml', 'Whole-gland volume RMSE (mL)', rmse_table),
    ('volume_mape_pct', 'Whole-gland volume MAPE (%)', mape_table),
    ('volume_rmspe_pct', 'Whole-gland volume RMSPE (%)', rmspe_table),
    ('case_counts', 'Evaluated / expected scans', counts_table),
]:
    print(title)
    display(table.fillna('NA'))
    table.to_csv(TABLE_DIR / f'{name}.csv', na_rep='NA')
    formatted = table.map(lambda value: 'NA' if pd.isna(value) else
                          f'{value:.6f}' if isinstance(value, (float, np.floating)) else str(value))
    columns = ['Dataset', 'Split', *formatted.columns]
    lines = ['| ' + ' | '.join(columns) + ' |',
             '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    lines += ['| ' + ' | '.join([*index, *row]) + ' |' for index, row in formatted.iterrows()]
    (TABLE_DIR / f'{name}.md').write_text(f'# {title}\n\n' + '\n'.join(lines) + '\n')
per_case.to_csv(TABLE_DIR / 'per_case_metrics.csv', index=False, na_rep='NA')
cohort_audit.to_csv(TABLE_DIR / 'cohort_audit.csv', index=False, na_rep='NA')
print(f'Tables and audit: {TABLE_DIR.relative_to(REPO_ROOT)}')
