# %% [markdown]
# # Whole-gland volume, PSA density, and biopsy simulation in 4,105 Toronto JDMI patients
#
# Compare clinical manual ellipsoid estimates, Multicenter Dataset274, and RUMC
# Dataset236 against nnUNet702 automated surrogate reference volumes, in mL.
# The full clinical listing supplies one examination per patient; accession
# exclusions leave an expected **4,105 patients**. Missing manual-reader volumes
# do not exclude patients. Results measure agreement with an automated reference,
# not accuracy against independent manual annotations.
# Patient selection is performed upstream in `_hlab20260928_clinical_image_listing.py`.
# Do not select another visit here, including when a reference is unavailable.
#
# Run cells in order. This notebook reads existing internal masks directly and
# performs no downloads or inference. No manual reference masks are available
# here, so these are volume comparisons; DSC is not calculated. Link workbook
# PSA and outcome fields and derive PSA density using the same PSA for all methods.
# Include PSAD agreement and PI-RADS 3 biopsy threshold simulation. All outputs remain local under the
# Git-ignored `results/toronto_jdmi_4111_clinical_volume/` directory. Display only
# aggregates, local plot-outlier tables with case identifiers, and the final
# flagged-case review table without patient identifiers.

# %%
from concurrent.futures import ThreadPoolExecutor
import hashlib
from getpass import getpass
from io import BytesIO
from importlib.metadata import version
import itertools
from itertools import combinations
import json
from pathlib import Path
import platform
from textwrap import fill

import matplotlib.pyplot as plt
import msoffcrypto
import nibabel as nib
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.ticker import FuncFormatter, NullFormatter
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1] if '__file__' in globals() else Path.cwd().resolve()
if REPO_ROOT.name == 'notebooks':
    REPO_ROOT = REPO_ROOT.parent
assert (REPO_ROOT / 'pyproject.toml').is_file(), 'Set REPO_ROOT to the repository root'
DATA_ROOT = REPO_ROOT / 'data_internal/hlab20260928'
CLINICAL_CSV = DATA_ROOT / 'hlab20260928_t2_image_listing_full_clinical.csv'
WORKBOOK = DATA_ROOT / 'PROSATELLMVOL20260703-12-WorkingBillShi.xlsx'
# Optional archived exclusions preserve a previously reviewed cohort.
EXCLUDE_JSON = Path(globals().get('EXCLUDE_JSON', DATA_ROOT / 'exclude.json')).resolve()
REFERENCE_ROOT = DATA_ROOT / 'PROSATEPSAD'
REFERENCE_FILENAME = 'wp-pz-tz_axt2_nnunet-702.nii.gz'
REFERENCE_LABELS = (1, 2, 3)
REFERENCE_NAME = 'nnUNet702 automated surrogate'
OUTPUT_DIR = REPO_ROOT / 'results/toronto_jdmi_4111_clinical_volume'
EXPECTED_CASES = 4105
GEOMETRY_TOLERANCE_MM = 0.5
GEOMETRY_WARNING_THRESHOLD_MM = 1e-3
UNIT_ANCHOR_TOLERANCE_MM = 1e-2
SELECTED_COMPARATORS = ['clinical', 'nnunet274', 'rumcnotprostatexnoclinical236']
COMPARATORS = {
    'clinical': ('Manual ellipsoid', None),
    'nnunet274': ('Multicenter', 'Dataset274_raw_3d_fullres_5fold'),
    'rumcnotprostatexnoclinical236': ('RUMC', 'Dataset236_raw_3d_fullres_5fold'),
}
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Cohort selection and local audit
#
# Preserve listing order and all exclusion causes. Exclude every accession in
# `exclude.json`, regardless of its reason. Preserve available manual-reader
# volumes for provenance only. Ellipsoid validity is a measurement check, not an
# additional selection criterion. IDs have the structure `AIPR_{amrn}_{date}_{aano}`.
# Require one scan per patient throughout the cleaned listing. Write the audit
# before rejecting a changed count or duplicate patient.
#
# A previous successful run's metadata pins the selected membership by SHA-256.
# A changed membership stops even when its count is unchanged. To intentionally
# establish a revised cohort, review the audit and archive the previous output
# directory before rerunning with the corresponding expected count.

# %%
listing = pd.read_csv(CLINICAL_CSV, dtype={'ID': 'string', 't2': 'string'})
required = {'ID', 't2', 'ground_truth_volume_ml', 'vol_report_ref'}
assert required <= set(listing.columns), 'Clinical listing lacks required columns'
parts = listing.ID.str.extract(r'^AIPR_([^_]+)_([^_]+)_([^_]+)$')
listing['amrn'] = parts[0]
listing['exam_date'] = parts[1]
listing['aano'] = parts[2]
listing['invalid_id'] = parts.isna().any(axis=1) | listing.ID.str.contains(r'[/\\]', na=True)
listing['duplicate_scan_id'] = listing.ID.duplicated(keep=False)
listing['duplicate_amrn'] = listing.amrn.duplicated(keep=False)
exclusions = json.loads(EXCLUDE_JSON.read_text())
assert isinstance(exclusions, list) and all(
    isinstance(entry, dict) and isinstance(entry.get('aano'), str)
    and entry['aano'].strip() for entry in exclusions
), 'Exclusion JSON requires a list of entries with nonempty aano strings'
exclusion_reasons = {}
for entry in exclusions:
    detail = '; '.join(str(entry[k]) for k in ('reason', 'description') if entry.get(k))
    exclusion_reasons.setdefault(entry['aano'], []).append(detail or 'Listed in exclude.json')
listing['listing_manual_reference_ml'] = pd.to_numeric(listing.ground_truth_volume_ml, errors='coerce')
listing['has_manual_reference'] = (
    np.isfinite(listing.listing_manual_reference_ml) & listing.listing_manual_reference_ml.gt(0)
)
listing['aano_excluded'] = listing.aano.isin(exclusion_reasons)
listing['exclusion_reason'] = listing.aano.map(
    {key: '; '.join(value) for key, value in exclusion_reasons.items()}
).fillna('')
listing['included'] = ~listing.aano_excluded
listing.to_csv(OUTPUT_DIR / 'cohort_audit.csv', index=False)
assert not listing[['invalid_id', 'duplicate_scan_id', 'duplicate_amrn']].any().any(), (
    'Invalid or repeated scan/patient IDs; inspect local cohort_audit.csv'
)
cases = listing.loc[listing.included].copy().reset_index(drop=True)
flow = pd.DataFrame([
    dict(stage='Cleaned clinical listing', n=len(listing)),
    dict(stage='After accession exclusions', n=int((~listing.aano_excluded).sum())),
])
flow['excluded_since_previous'] = flow.n.shift() - flow.n
flow.to_csv(OUTPUT_DIR / 'cohort_flow.csv', index=False)
display(flow)
assert len(cases) == EXPECTED_CASES, 'Cohort count changed; inspect local cohort_audit.csv'
cohort_sha256 = hashlib.sha256('\n'.join(sorted(cases.ID)).encode()).hexdigest()
metadata_path = OUTPUT_DIR / 'run_metadata.json'
if metadata_path.is_file():
    previous = json.loads(metadata_path.read_text())
    assert previous['cohort_sha256'] == cohort_sha256, (
        'Cohort membership changed; inspect local cohort_audit.csv'
    )
cases['clinical_volume_ml'] = pd.to_numeric(cases.vol_report_ref, errors='coerce')
assert (np.isfinite(cases.clinical_volume_ml) & cases.clinical_volume_ml.gt(0)).all(), (
    'Invalid reported clinical volumes; inspect local cohort_audit.csv'
)
assert cases.t2.notna().all() and cases.t2.str.strip().ne('').all(), (
    'Missing T2 paths; inspect local cohort_audit.csv'
)
cases['image_path'] = cases.t2.map(lambda p: str((DATA_ROOT / p).resolve()))
assert cases.image_path.is_unique, 'Repeated eligible T2 paths; inspect local cohort_audit.csv'

# %% [markdown]
# ## Workbook loading at runtime
#
# Supply the password at the prompt. Decrypt only in memory and preserve raw
# cells (including blank strings), plus original Excel row numbers starting at 2.
# The workbook enriches the selected scans; it never selects a different visit.

# %%
# For a mask-only refresh, an explicitly supplied raw-field audit can replace
# workbook decryption only when the workbook and listing match the prior run.
WORKBOOK_AUDIT_PATH = globals().get('WORKBOOK_AUDIT_PATH')
workbook_audit_provenance = None
if WORKBOOK_AUDIT_PATH is not None:
    audit_path = Path(WORKBOOK_AUDIT_PATH).resolve()
    prior_metadata = json.loads((audit_path.parent / 'run_metadata.json').read_text())
    for unchanged_input in (WORKBOOK, CLINICAL_CSV):
        relative = str(unchanged_input.relative_to(REPO_ROOT))
        assert hashlib.sha256(unchanged_input.read_bytes()).hexdigest() == prior_metadata['input_sha256'][relative], relative
    workbook = pd.read_csv(audit_path, dtype=object, keep_default_na=False)
    assert np.array_equal(pd.to_numeric(workbook.source_excel_row), np.arange(2, len(workbook) + 2))
    workbook['aano'] = workbook.workbook_aano_raw
    workbook['amrn'] = workbook.workbook_amrn_raw
    workbook = workbook.drop(columns=['source_excel_row', 'workbook_aano_raw', 'workbook_amrn_raw', 'invalid_aano', 'duplicate_aano'])
    workbook_audit_provenance = dict(path=str(audit_path.relative_to(REPO_ROOT)),
        sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        prior_metadata_sha256=hashlib.sha256((audit_path.parent / 'run_metadata.json').read_bytes()).hexdigest())
else:
    with WORKBOOK.open('rb') as stream:
        office = msoffcrypto.OfficeFile(stream)
        try:
            if office.is_encrypted():
                password = getpass('Excel password: ')
                try:
                    office.load_key(password=password, verify_password=True)
                finally:
                    del password
                with BytesIO() as decrypted:
                    office.decrypt(decrypted)
                    decrypted.seek(0)
                    workbook = pd.read_excel(decrypted, engine='openpyxl', dtype=object, keep_default_na=False)
            else:
                workbook = pd.read_excel(WORKBOOK, engine='openpyxl', dtype=object, keep_default_na=False)
        finally:
            del office

# %% [markdown]
# ## Link clinical fields to selected examinations
#
# AANO is the examination/accession key; AMRN is the patient key. Normalize workbook
# AMRN by trimming whitespace and removing one leading literal `AIPR_` to match
# the scan's extracted AMRN. Keep strings and leading zeros; preserve raw values.
# Preserve listing order, verify patient identity and available manual-reader
# values, and audit before stopping on inconsistent inputs. Numeric parsing
# follows the Haider analysis: decimal
# commas become decimal points and `>` is removed, with censoring flagged.
# Apply its documented PSA correction only to an analytical copy. Prefer positive
# finite reported PSA, falling back to positive finite corrected alternative PSA.
# Outcomes do not affect density eligibility. Dimensions are in cm, volumes mL.
# Manual clinical ellipsoid volume uses the listing's vol_report_ref in mL.
# Parsed workbook dimensions provide a separate π/6 ellipsoid audit diagnostic.

# %%
READER_FIELDS = [
    'p_axt2_ch_vol', 'p_axt2_ah_vol', 'p_axt2_dd_vol', 'p_axt2_ao_vol',
    'p_axt2_chepg_vol', 'p_axt2_epgch_vol', 'p_axt2_epg_vol',
]
DIMENSION_FIELDS = ['dim1_report_ref', 'dim2_report_ref', 'dim3_report_ref']
WORKBOOK_FIELDS = ['aano', 'amrn', 'psa_report_ref', 'psa', 'reportmaxpirads_ref',
                   'CSCANCER', *READER_FIELDS, *DIMENSION_FIELDS]
assert workbook.columns.is_unique and set(WORKBOOK_FIELDS) <= set(workbook.columns), (
    'Workbook requires unique columns and all clinical input fields'
)
clinical_fields = workbook[WORKBOOK_FIELDS].copy()
clinical_fields['source_excel_row'] = np.arange(2, len(workbook) + 2)
clinical_fields['workbook_aano_raw'] = clinical_fields.aano
clinical_fields['workbook_amrn_raw'] = clinical_fields.amrn
clinical_fields['aano'] = clinical_fields.aano.astype('string').str.strip()
clinical_fields['amrn'] = clinical_fields.amrn.astype('string').str.strip().str.removeprefix('AIPR_')
clinical_fields['invalid_aano'] = clinical_fields.aano.isna() | clinical_fields.aano.eq('')
clinical_fields['duplicate_aano'] = clinical_fields.aano.duplicated(keep=False)
clinical_fields.to_csv(OUTPUT_DIR / 'workbook_key_audit.csv', index=False)
assert not clinical_fields[['invalid_aano', 'duplicate_aano']].any().any(), (
    'Invalid or duplicate workbook aano; inspect workbook_key_audit.csv'
)
clinical_fields = clinical_fields.rename(columns={'amrn': 'workbook_amrn'})
clinical_audit = cases.merge(clinical_fields, on='aano', how='left', sort=False,
                             validate='one_to_one', indicator='workbook_match')
assert clinical_audit.ID.tolist() == cases.ID.tolist(), 'Clinical linkage changed scan order'
clinical_audit['amrn_matches'] = clinical_audit.amrn.eq(clinical_audit.workbook_amrn).fillna(False)

# Direct cell-level parsing keeps raw source columns intact.
for field in ['psa_report_ref', 'psa', 'reportmaxpirads_ref', *READER_FIELDS, *DIMENSION_FIELDS]:
    raw = clinical_audit[field].astype('string')
    clinical_audit[f'{field}_censored'] = raw.str.contains('>', regex=False, na=False)
    clinical_audit[f'{field}_numeric'] = pd.to_numeric(
        raw.str.replace(',', '.', regex=False).str.replace('>', '', regex=False), errors='coerce',
    ).astype(float)
clinical_audit['psa_correction_applied'] = clinical_audit.aano.eq('333abfe652a8889')
clinical_audit['psa_corrected_numeric'] = clinical_audit.psa_numeric.mask(
    clinical_audit.psa_correction_applied, 28.9,
)
report_psa = clinical_audit.psa_report_ref_numeric
alternative_psa = clinical_audit.psa_corrected_numeric
valid_report = np.isfinite(report_psa) & report_psa.gt(0)
valid_alternative = np.isfinite(alternative_psa) & alternative_psa.gt(0)
clinical_audit['psa_consolidated'] = report_psa.where(valid_report, alternative_psa.where(valid_alternative))
clinical_audit['psa_source'] = np.select(
    [valid_report, valid_alternative], ['psa_report_ref', 'psa'], default='unavailable',
)
clinical_audit['psa_selected_censored'] = np.where(
    valid_report, clinical_audit.psa_report_ref_censored,
    clinical_audit.psa_censored & ~clinical_audit.psa_correction_applied & valid_alternative,
)
pirads = clinical_audit.reportmaxpirads_ref_numeric
clinical_audit['pirads_valid'] = np.isfinite(pirads) & pirads.between(1, 5) & pirads.mod(1).eq(0)
clinical_audit['pirads'] = pirads.where(clinical_audit.pirads_valid)
status = clinical_audit.CSCANCER.astype('string').str.strip().str.upper()
missing_outcome = status.isin([
    '', 'NA', 'N/A', '#N/A', '#NA', 'NAN', 'NONE', 'NULL', '<NA>', 'UNKNOWN',
    'NOT AVAILABLE', 'NOT KNOWN', 'NOT BIOPSIED', 'NO BIOPSY',
])
initial = status.mask(missing_outcome).str[0]
clinical_audit['cscancer'] = initial.where(initial.isin(['Y', 'N']), 'unknown')
manual = clinical_audit[[f'{field}_numeric' for field in READER_FIELDS]].copy()
manual.loc[manual.p_axt2_ch_vol_numeric.eq(0), 'p_axt2_ch_vol_numeric'] = np.nan
clinical_audit['invalid_reader_volume'] = (manual.notna() & (~np.isfinite(manual) | manual.le(0))).any(axis=1)
clinical_audit['workbook_reference_volume_ml'] = manual.median(axis=1)
# Missing manual values are allowed; these are a source-consistency check only.
clinical_audit['manual_reference_matches'] = np.isclose(
    clinical_audit.workbook_reference_volume_ml, clinical_audit.listing_manual_reference_ml,
    rtol=1e-9, atol=1e-9, equal_nan=True,
)
dimensions = clinical_audit[[f'{field}_numeric' for field in DIMENSION_FIELDS]]
valid_dimensions = np.isfinite(dimensions).all(axis=1) & dimensions.gt(0).all(axis=1)
with np.errstate(over='ignore', invalid='ignore'):
    ellipsoid = np.pi / 6 * dimensions.prod(axis=1, skipna=False)
clinical_audit['dimension_ellipsoid_volume_ml'] = ellipsoid.where(
    valid_dimensions & np.isfinite(ellipsoid) & ellipsoid.gt(0),
)
clinical_audit.to_csv(OUTPUT_DIR / 'clinical_fields_audit.csv', index=False)
assert clinical_audit.workbook_match.eq('both').all() and clinical_audit.amrn_matches.all(), (
    'Workbook linkage or amrn mismatch; inspect clinical_fields_audit.csv'
)
assert not clinical_audit.invalid_reader_volume.any() and clinical_audit.manual_reference_matches.all(), (
    'Workbook manual-reference mismatch or invalid reader volumes; inspect clinical_fields_audit.csv'
)
assert (np.isfinite(clinical_audit.psa_consolidated) & clinical_audit.psa_consolidated.gt(0)).all(), (
    'No positive finite PSA for selected cases; inspect clinical_fields_audit.csv'
)
cases = clinical_audit.copy()
display(pd.Series({
    'linked_examinations': len(cases), 'reported_psa_used': int(valid_report.sum()),
    'alternative_psa_used': int((~valid_report & valid_alternative).sum()),
    'psa_corrections': int(cases.psa_correction_applied.sum()),
    'valid_pirads': int(cases.pirads_valid.sum()),
    'known_cancer_outcome': int(cases.cscancer.ne('unknown').sum()),
}, name='n').to_frame())

# %% [markdown]
# ## Direct measurements on native prediction grids
#
# The reference filename identifies nnUNet702, an automated surrogate. Its labels
# must be finite integers drawn from 0/1/2/3. Whole gland is the union of labels
# 1, 2, 3, matching `model_study_performance.py`; count foreground voxels rather
# than summing numeric label values. A mask may lack a zone label, but its union
# must be nonempty. Observed labels and per-label voxel counts are audited.
# Reference model training and historical inference settings are unknown; no
# independence or external validation is inferred from the filename.
#
# Comparator metadata must identify the expected dataset and whole-gland labels 0/1.
# Both masks and T2 images must be 3D with finite, invertible affines. A missing
# spatial-unit code is accepted only with a matching known-mm input as an anchor:
# use T2 when marked mm, otherwise a selected comparator mask on the same grid.
# Unknown-unit grids must match that anchor within 0.01 mm at every voxel corner.
# This allows the observed <=0.003 mm coordinate precision differences and is
# stricter than the general 0.5 mm geometry tolerance.
# Audit the assumption and anchor path; reject explicit non-mm units or absent
# anchors. Current inputs have 1,554 reference masks and 183 T2s with unknown units.
# Equal shapes and maximum corresponding voxel-corner
# displacement <= 0.5 mm are accepted, matching `model_study_performance.py`.
# Displacements > 0.001 mm are counted below; every displacement is audited.
# No resampling occurs. Whole-gland mL = foreground voxels × |det(affine)| / 1,000.
# Empty comparator predictions remain zero-volume failures in volume statistics.
# Missing or invalid reference/prediction measurements stop after a local audit.
#
# Hash each input, including compressed masks and T2 images, for local provenance.
# Saved inference metadata is recorded when available, without assuming that
# historical settings were identical across models or independently certified.

# %%
input_hashes = {}
for path in (CLINICAL_CSV, EXCLUDE_JSON, WORKBOOK):
    with path.open('rb') as stream:
        input_hashes[str(path.relative_to(REPO_ROOT))] = hashlib.file_digest(stream, 'sha256').hexdigest()
if workbook_audit_provenance is not None:
    input_hashes[workbook_audit_provenance['path']] = workbook_audit_provenance['sha256']
model_metadata = {}
model_roots = {}
for key in SELECTED_COMPARATORS:
    model = COMPARATORS[key][1]
    if model is None:
        continue
    model_root = DATA_ROOT / f'seg_out_{model}'
    model_roots[key] = model_root
    saved_metadata = {}
    for filename in ('plans.json', 'dataset.json', 'inference_config.json', 'predict_from_raw_data_args.json', 'manual_postprocessing.json'):
        path = model_root / filename
        if path.is_file():
            saved_metadata[filename] = json.loads(path.read_text())
            with path.open('rb') as stream:
                input_hashes[str(path.relative_to(REPO_ROOT))] = hashlib.file_digest(stream, 'sha256').hexdigest()
    expected_dataset = model.removesuffix('_3d_fullres_5fold')
    if saved_metadata.get('plans.json', {}).get('dataset_name') != expected_dataset:
        raise ValueError(f'{key}: unexpected or missing dataset identity')
    if saved_metadata.get('dataset.json', {}).get('labels') != {'background': 0, 'class1': 1}:
        raise ValueError(f'{key}: expected binary whole-gland labels')
    model_metadata[key] = saved_metadata

def measure_case(case):
    case_rows = []
    case_hashes = {}
    mask_paths = {'gt': REFERENCE_ROOT / case.ID / REFERENCE_FILENAME}
    mask_paths.update({key: root / f'{case.ID}.nii.gz' for key, root in model_roots.items()})
    for key, path in mask_paths.items():
        foreground_labels = REFERENCE_LABELS if key == 'gt' else (1,)
        record = dict(ID=case.ID, method=key, prediction_path=str(path.relative_to(REPO_ROOT)))
        try:
            if not path.is_file():
                raise FileNotFoundError('Required mask does not exist')
            pred_img, image = nib.load(path), nib.load(case.image_path)
            if len(pred_img.shape) != 3 or pred_img.shape != image.shape:
                raise ValueError('Mask and T2 shapes differ or are not 3D')
            mask_units = pred_img.header.get_xyzt_units()[0]
            image_units = image.header.get_xyzt_units()[0]
            record.update(mask_spatial_units=mask_units, image_spatial_units=image_units,
                          units_assumed_mm=mask_units == 'unknown' or image_units == 'unknown',
                          unit_anchor_path='', unit_anchor_max_displacement_mm=0.)
            for img in (pred_img, image):
                determinant = float(np.linalg.det(img.affine[:3, :3]))
                if img.header.get_xyzt_units()[0] not in ('mm', 'unknown'):
                    raise ValueError('Spatial units must be mm or unknown with a known-mm grid anchor')
                if not np.isfinite(img.affine).all() or not np.isfinite(determinant) or determinant == 0:
                    raise ValueError('Invalid spatial affine')
            corners = np.array(list(itertools.product(*[(0, n - 1) for n in image.shape])))
            if record['units_assumed_mm']:
                anchor_img = image if image_units == 'mm' else None
                anchor_path = Path(case.image_path) if anchor_img is not None else None
                if anchor_img is None:
                    for candidate_path in (mask_paths[k] for k in model_roots):
                        try:
                            candidate = nib.load(candidate_path)
                        except (OSError, ValueError, nib.filebasedimages.ImageFileError):
                            continue  # Its measurement row will independently report this error.
                        if candidate.header.get_xyzt_units()[0] != 'mm' or candidate.shape != image.shape:
                            continue
                        if not np.isfinite(candidate.affine).all():
                            continue
                        candidate_displacement = float(np.linalg.norm(
                            nib.affines.apply_affine(candidate.affine, corners)
                            - nib.affines.apply_affine(image.affine, corners), axis=1,
                        ).max())
                        candidate_determinant = float(np.linalg.det(candidate.affine[:3, :3]))
                        if (np.isfinite(candidate_displacement) and candidate_displacement <= UNIT_ANCHOR_TOLERANCE_MM
                                and np.isfinite(candidate_determinant) and candidate_determinant != 0):
                            anchor_img, anchor_path = candidate, candidate_path
                            break
                if anchor_img is None:
                    raise ValueError('Unknown spatial units require a matching known-mm grid anchor')
                record['unit_anchor_path'] = str(anchor_path.relative_to(REPO_ROOT))
                anchor_displacements = [float(np.linalg.norm(
                    nib.affines.apply_affine(img.affine, corners)
                    - nib.affines.apply_affine(anchor_img.affine, corners), axis=1,
                ).max()) for img in (pred_img, image) if img.header.get_xyzt_units()[0] == 'unknown']
                record['unit_anchor_max_displacement_mm'] = max(anchor_displacements)
                if (not np.isfinite(anchor_displacements).all()
                        or max(anchor_displacements) > UNIT_ANCHOR_TOLERANCE_MM):
                    raise ValueError('Unknown-unit grid differs from known-mm anchor')
            displacement = float(np.linalg.norm(
                nib.affines.apply_affine(pred_img.affine, corners)
                - nib.affines.apply_affine(image.affine, corners), axis=1,
            ).max())
            record['max_corner_displacement_mm'] = displacement
            if not np.isfinite(displacement) or displacement > GEOMETRY_TOLERANCE_MM:
                raise ValueError('Geometry displacement exceeds tolerance')
            mask = np.asanyarray(pred_img.dataobj)
            if not np.isin(mask, [0, *foreground_labels]).all():
                raise ValueError(f'Expected finite integer labels drawn from {[0, *foreground_labels]}')
            labels, counts = np.unique(mask, return_counts=True)
            record['observed_labels'] = json.dumps([int(label) for label in labels])
            for label in (0, 1, 2, 3):
                record[f'label_{label}_voxel_count'] = int(counts[labels == label].sum())
            voxel_count = int(np.count_nonzero(np.isin(mask, foreground_labels)))
            voxel_volume_mm3 = float(abs(np.linalg.det(pred_img.affine[:3, :3])))
            volume_ml = voxel_count * voxel_volume_mm3 / 1000
            record.update(voxel_count=voxel_count, voxel_volume_mm3=voxel_volume_mm3, volume_ml=volume_ml)
            if not np.isfinite(volume_ml) or (key == 'gt' and volume_ml <= 0):
                raise ValueError('Reference volume must be positive and all measured volumes finite')
            for input_path in (path, Path(case.image_path)):
                relative_path = str(input_path.relative_to(REPO_ROOT))
                if relative_path not in case_hashes:
                    with input_path.open('rb') as stream:
                        case_hashes[relative_path] = hashlib.file_digest(stream, 'sha256').hexdigest()
            record.update(status='measured', error='')
        except (OSError, ValueError, nib.filebasedimages.ImageFileError) as error:
            # Details may contain identifiers; keep them in the local audit only.
            record.update(status='invalid', error=str(error))
        case_rows.append(record)
    return case_rows, case_hashes


measurement_rows = []
with ThreadPoolExecutor() as executor:
    measurements = executor.map(measure_case, cases.itertuples(index=False))
    for case_rows, case_hashes in tqdm(measurements, total=len(cases), desc='Measuring Toronto volumes'):
        measurement_rows.extend(case_rows)
        input_hashes.update(case_hashes)
measurement_audit = pd.DataFrame(measurement_rows)
measurement_audit.to_csv(OUTPUT_DIR / 'measurement_audit.csv', index=False)
if measurement_audit.status.ne('measured').any():
    raise ValueError('Incomplete or invalid measurements; inspect local measurement_audit.csv')
volumes = cases.copy()
for key in ['gt', *model_roots]:
    measured = measurement_audit.loc[measurement_audit.method.eq(key)]
    measured = measured.drop(columns=['method', 'status', 'error']).rename(
        columns={column: f'{key}_{column}' for column in measured.columns if column != 'ID'}
    )
    volumes = volumes.merge(measured, on='ID', validate='one_to_one', how='left', sort=False)
assert volumes.ID.tolist() == cases.ID.tolist(), 'Measurements changed cohort order'
cases = volumes.copy()  # Report-quality cells use the same measured reference.
flow = pd.concat([flow, pd.DataFrame([
    dict(stage='Positive finite nnUNet702 reference and valid comparator measurements', n=len(volumes)),
])], ignore_index=True)
flow['excluded_since_previous'] = flow.n.shift() - flow.n
flow.to_csv(OUTPUT_DIR / 'cohort_flow.csv', index=False)
display(measurement_audit.groupby('method').agg(
    measured=('status', 'size'), empty_predictions=('voxel_count', lambda s: int(s.eq(0).sum())),
    geometry_warnings=('max_corner_displacement_mm', lambda s: int(s.gt(GEOMETRY_WARNING_THRESHOLD_MM).sum())),
    unit_assumptions=('units_assumed_mm', 'sum'),
))

# %% [markdown]
# ## Volume agreement with the nnUNet702 automated surrogate
#
# All three methods use the same complete cohort. Positive signed differences
# indicate overestimation. Relative differences are 100 × (measured − reference)
# / reference, in percent. MAPE and RMSPE are also in percent. Limits of agreement
# are mean difference ± 1.96 sample SD, not confidence intervals. Correlation uses
# original volumes before logarithmic plotting. Ranking by MAE describes this
# selected cohort and does not establish segmentation overlap performance.
# Pearson correlation is undefined (NaN) if either volume series is constant.

# %%
gt = volumes['gt_volume_ml']
volume_columns = [f'{key}_volume_ml' for key in SELECTED_COMPARATORS]
measured_volumes = volumes[volume_columns].set_axis(SELECTED_COMPARATORS, axis='columns')
differences_ml = measured_volumes.sub(gt, axis='index')
differences_percent = (100 * differences_ml).div(gt, axis='index')

summary_rows = []
for key in SELECTED_COMPARATORS:
    name = COMPARATORS[key][0]
    value = measured_volumes[key]
    error = differences_ml[key]
    relative = differences_percent[key]
    bias_ml, sd_ml = error.mean(), error.std()
    relative_bias, relative_sd = relative.mean(), relative.std()
    volumes[f'{key}_minus_gt_ml'] = error
    volumes[f'{key}_absolute_error_ml'] = error.abs()
    volumes[f'{key}_absolute_percentage_error'] = relative.abs()
    summary_rows.append(dict(
        method=key, name=name, n=len(value),
        bias_ml=bias_ml,
        mae_ml=error.abs().mean(),
        rmse_ml=np.sqrt((error**2).mean()),
        mean_absolute_percentage_error=relative.abs().mean(),
        rmspe_percent=np.sqrt((relative**2).mean()),
        pearson_r=value.corr(gt) if value.nunique() > 1 and gt.nunique() > 1 else np.nan,
        loa_lower_ml=bias_ml - 1.96 * sd_ml,
        loa_upper_ml=bias_ml + 1.96 * sd_ml,
        relative_bias_percent=relative_bias,
        relative_loa_lower_percent=relative_bias - 1.96 * relative_sd,
        relative_loa_upper_percent=relative_bias + 1.96 * relative_sd,
    ))

summary = pd.DataFrame(summary_rows).sort_values('mae_ml')
display(summary)
volumes.to_csv(OUTPUT_DIR / 'comparator_volumes.csv', index=False)
summary.to_csv(OUTPUT_DIR / 'comparator_summary.csv', index=False)

# %% [markdown]
# ## Agreement plots and empirical difference distributions
#
# Columns separate methods; rows share y-axes within each measurement type.
# The rightmost column shows horizontal count histograms without smoothing.
# Each row uses common equal-width bins based on the pooled selected methods:
# halve the Freedman–Diaconis bin count (rounded up) for wider, readable bins.
# All observations enter the bins and statistics. The displayed relative range
# is fixed below; values outside it are clipped only in the figure.

# %%
FIGURE_SIZE = (9.5, 7.5)
VOLUME_TICKS = [10, 30, 100, 300]
RELATIVE_DIFFERENCE_LIMITS = (-100, 480)

# Keep each method's color, marker, and histogram line style together.
PLOT_STYLES = {
    'clinical': dict(color='#4C6A85', marker='o', linestyle='-'),
    'nnunet274': dict(color='#B88254', marker='s', linestyle='--'),
    'rumcnotprostatexnoclinical236': dict(color='#91A78C', marker='^', linestyle=':'),
}
scatter_options = dict(s=16, edgecolors='none', linewidths=0, alpha=.6)
legend_options = dict(
    fontsize=7, frameon=True, framealpha=.85, facecolor='white',
    edgecolor='0.6', loc='best', handlelength=1, handletextpad=.4,
)

fig, axes = plt.subplots(
    3, len(SELECTED_COMPARATORS) + 1, figsize=FIGURE_SIZE,
    sharey='row', squeeze=False, constrained_layout=True,
    gridspec_kw={'width_ratios': [1] * len(SELECTED_COMPARATORS) + [.85]},
)
fig.get_layout_engine().set(w_pad=.015, h_pad=.015, wspace=.02, hspace=.02)
fig.suptitle(f'Toronto JDMI (n = {len(volumes)} patients; nnUNet702 reference)', fontsize=10)
axes[0, -1].set_axis_off()

volume_values = volumes[['gt_volume_ml', *volume_columns]].to_numpy()
lower = 10 # min(volume_values[volume_values > 0].min() * 0.95, 10)
upper = 500 # max(volume_values.max() * 1.05, 300)
summary_by_method = summary.set_index('method')

for column, key in enumerate(SELECTED_COMPARATORS):
    name = COMPARATORS[key][0]
    panel_name = fill(name, width=16, break_on_hyphens=False)
    style = PLOT_STYLES[key]
    metrics = summary_by_method.loc[key]

    ax = axes[0, column]
    ax.scatter(
        gt, measured_volumes[key], color=style['color'], marker=style['marker'],
        label=f'{panel_name}\nPearson r = {metrics.pearson_r:.2f}', **scatter_options,
    )
    ax.plot([lower, upper], [lower, upper], 'k--')
    # Zero-volume failures remain in the statistics but cannot appear on a log axis.
    ax.set_yscale('log', nonpositive='mask')
    ax.set_yticks(VOLUME_TICKS, labels=[str(tick) for tick in VOLUME_TICKS])
    ax.set_ylim(lower, upper)
    ax.yaxis.set_minor_formatter(NullFormatter())

    agreement_rows = [
        (differences_ml[key], 'mL', ['bias_ml', 'loa_lower_ml', 'loa_upper_ml']),
        (differences_percent[key], '%', [
            'relative_bias_percent', 'relative_loa_lower_percent', 'relative_loa_upper_percent',
        ]),
    ]
    for row, (differences, unit, summary_columns) in enumerate(agreement_rows, start=1):
        ax = axes[row, column]
        bias, loa_lower, loa_upper = metrics[summary_columns]
        legend_label = (
            f'{panel_name}\nMean: {bias:.1f} {unit}\n'
            f'95% LoA:\n[{loa_lower:.1f}, {loa_upper:.1f}]'
        )
        ax.scatter(
            gt, differences, color=style['color'], marker=style['marker'],
            label=legend_label, **scatter_options,
        )
        for level, linestyle in [(bias, '-'), (loa_lower, '--'), (loa_upper, '--')]:
            ax.axhline(level, color=style['color'], linestyle=linestyle, linewidth=.8)

    ylabels = [
        f'{name}\nvolume (mL; log scale)',
        f'{name}\n− nnUNet702 reference (mL)',
        'Relative difference (%)',
    ]
    for row, ylabel in enumerate(ylabels):
        ax = axes[row, column]
        if (row, column) != (0, 0):
            ax.sharex(axes[0, 0])
        ax.set_xscale('log')
        ax.set_xticks(VOLUME_TICKS, labels=[str(tick) for tick in VOLUME_TICKS])
        ax.set_xlim(lower, upper)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(
            'nnUNet702 reference volume\n(mL; log scale)',
            fontsize=8, labelpad=1, linespacing=1.05,
        )
        ax.set_ylabel(ylabel, fontsize=8, labelpad=1, linespacing=1.05)

for row, difference_table in [(1, differences_ml), (2, differences_percent)]:
    ax = axes[row, -1]
    fd_edges = np.histogram_bin_edges(difference_table.to_numpy().ravel(), bins='fd')
    histogram_bin_count = max(1, int(np.ceil((len(fd_edges) - 1) / 2)))
    histogram_edges = np.linspace(fd_edges[0], fd_edges[-1], histogram_bin_count + 1)
    # Include the complete bins even when every difference is identical.
    difference_limits = axes[row, 0].get_ylim()
    ax.set_ylim(min(difference_limits[0], histogram_edges[0]),
                max(difference_limits[1], histogram_edges[-1]))
    for key in SELECTED_COMPARATORS:
        counts, _ = np.histogram(difference_table[key], bins=histogram_edges)
        style = PLOT_STYLES[key]
        ax.stairs(
            counts, histogram_edges, orientation='horizontal', color=style['color'],
            linestyle=style['linestyle'], linewidth=1.2,
            label=fill(COMPARATORS[key][0], width=16, break_on_hyphens=False),
        )
    ax.set_xlim(left=0)
    ax.locator_params(axis='x', nbins=3, integer=True)

axes[2, 0].set_ylim(*RELATIVE_DIFFERENCE_LIMITS)
axes[1, 0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:+g}' if value else '0'))
axes[2, 0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:+.0f}%' if value else '0%'))
for ax in axes.flat:
    if ax.axison:
        ax.tick_params(axis='both', labelsize=8, pad=1.5, labelbottom=True, labelleft=True)
        ax.legend(**legend_options)

# Collect outliers from the original data against the final shared volume axes.
volume_plot_outliers = []
for column, key in enumerate(SELECTED_COMPARATORS):
    for row, (panel_name, y_values) in enumerate([
        ('volume', measured_volumes[key]),
        ('difference', differences_ml[key]),
        ('relative difference', differences_percent[key]),
    ]):
        panel = axes[row, column]
        x_values = gt.to_numpy(dtype=float)
        y_values = y_values.to_numpy(dtype=float)
        x_limits = sorted(panel.get_xlim())
        y_limits = sorted(panel.get_ylim())
        visible = np.isfinite(x_values) & np.isfinite(y_values)
        if panel.get_xscale() == 'log':
            visible &= x_values > 0
        if panel.get_yscale() == 'log':
            visible &= y_values > 0
        visible &= (x_values >= x_limits[0]) & (x_values <= x_limits[1])
        visible &= (y_values >= y_limits[0]) & (y_values <= y_limits[1])
        if not visible.all():
            outliers = volumes.loc[~visible, ['ID', 'aano']].copy()
            outliers['x'] = x_values[~visible]
            outliers['y'] = y_values[~visible]
            volume_plot_outliers.append((
                f'Volume agreement / {key} / {panel_name}: '
                f'{len(outliers)} / {len(visible)} hidden or invalid points; '
                f'x limits={x_limits}, y limits={y_limits}', outliers,
            ))

fig.savefig(OUTPUT_DIR / 'comparator_agreement.png', dpi=300)
fig.savefig(OUTPUT_DIR / 'comparator_agreement.pdf')
plt.show()
for message, outliers in volume_plot_outliers:
    print(message)
    print(outliers.to_string(index=False))

# %% [markdown]
# ## Paired absolute-error comparisons
#
# Positive improvement means method A is closer than method B. Exactly equal
# absolute errors are ties. All methods use the complete volume-analysis cohort.

# %%
pair_tables = []
count_rows = []
for a, b in combinations(SELECTED_COMPARATORS, 2):
    pair = volumes[['ID', 'amrn', 'aano', 'exam_date']].copy()
    pair['method_a'], pair['method_b'] = a, b
    pair['absolute_error_improvement_ml'] = (
        volumes[f'{b}_absolute_error_ml'] - volumes[f'{a}_absolute_error_ml']
    )
    delta = pair['absolute_error_improvement_ml']
    pair['a_vs_b'] = np.select([delta > 0, delta < 0], ['Closer', 'Farther'], default='Tie')
    pair_tables.append(pair)
    for outcome in ['Closer', 'Farther', 'Tie']:
        n = int(pair['a_vs_b'].eq(outcome).sum())
        count_rows.append(dict(
            method_a=a, method_b=b, outcome=outcome, n=n, percent=100 * n / len(pair),
        ))
if pair_tables:
    pairwise = pd.concat(pair_tables, ignore_index=True)
    pairwise_counts = pd.DataFrame(count_rows)
    display(pairwise_counts)
    pairwise.to_csv(OUTPUT_DIR / 'comparator_pairwise_per_case.csv', index=False)
    pairwise_counts.to_csv(OUTPUT_DIR / 'comparator_pairwise_counts.csv', index=False)

# %% [markdown]
# ## Paired post hoc inference: RMSE, RMSPE, and probability of being closer
#
# These are two different questions: a method can have lower population RMSE
# without being closer on most patients. Positive RMSE improvement = RMSE(B) −
# RMSE(A), in mL, favors A. Equality of population RMSE is equivalent to zero mean
# paired squared-error difference, d = error(B)² − error(A)² (mL²).
#
# RMSPE (%) = sqrt(mean((100 × (prediction − reference) / reference)²)).
# Each patient is normalized by their own positive reference. The same inference
# below is applied to percentage errors for RMSPE; squared-error differences are
# then in percent². RMSPE(B) − RMSPE(A) and its interval are in percentage points;
# positive favors A. BCa intervals remain pointwise and unadjusted.
# RMSPE bootstrap tests form a separate Holm family across method pairs in this
# analysis; RMSPE t-test sensitivity p-values have their own analogous family.
# Existing RMSE/closer families are unchanged. Zero predictions remain eligible
# where allowed by the existing cohort rules.
#
# Primary RMSE inference uses a two-sided, null-centered bootstrap-t test of
# mean(d)=0. Its null samples resample d−mean(d), not the original uncentered
# differences. Pointwise 95% BCa intervals resample both methods together and
# recompute RMSE(B)−RMSE(A). The paired t-test on d is a sensitivity analysis;
# sample size alone does not remove concern about skewed squared errors.
#
# Independently, an exact two-sided binomial sign test assesses
# P(A closer | no tie)=0.5. The exact probability interval excludes
# ties; all-patient win/loss/tie proportions remain visible. An all-tie comparison
# has undefined conditional probability and p=1, providing no evidence of a winner.
#
# Holm correction jointly covers both primary endpoints across every selected
# pair (six tests for three methods), without an omnibus gate. Sensitivity Holm
# correction replaces bootstrap p-values with t-test p-values in the same family.
# Undefined t-tests occupy their family slot as p=1 but remain undefined in the
# output. All intervals are pointwise and unadjusted; conclusions use adjusted
# p-values. Bootstrap p-values have finite Monte Carlo resolution, not exact zeros.
#
# Resampling treats predictions as fixed and patients as independent observations
# representative of the population of interest. This selected Toronto cohort is
# not a random population sample, so generalization requires that assumption;
# results concern volume accuracy, not spatial segmentation quality or training
# variability. See [Ludbrook, 1995](https://pubmed.ncbi.nlm.nih.gov/7487733/) and
# [SciPy paired BCa bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).

# %%
import warnings

from scipy import stats
from statsmodels.stats.multitest import multipletests

N_BOOTSTRAP = 100_000
RANDOM_SEED = 20261102
BOOTSTRAP_BATCH = 1_000
ALPHA = 0.05
CONFIDENCE_LEVEL = 0.95

assert N_BOOTSTRAP > 0 and BOOTSTRAP_BATCH > 0
assert len(volumes) >= 2, 'Paired inference requires at least two patients'
assert volumes['amrn'].is_unique, 'Patient-level resampling requires unique patients'


# A vectorized statistic is needed by SciPy for paired BCa resampling/jackknifing.
def rmse_improvement(error_a, error_b, axis=-1):
    return np.sqrt(np.mean(error_b**2, axis=axis)) - np.sqrt(np.mean(error_a**2, axis=axis))


# %%
statistics_rows = []
for a, b in combinations(SELECTED_COMPARATORS, 2):
    error_a = (volumes[f'{a}_volume_ml'] - volumes['gt_volume_ml']).to_numpy(dtype=float)
    error_b = (volumes[f'{b}_volume_ml'] - volumes['gt_volume_ml']).to_numpy(dtype=float)
    assert np.isfinite(error_a).all() and np.isfinite(error_b).all(), (a, b)
    n = len(error_a)
    reference = volumes['gt_volume_ml'].to_numpy(dtype=float)
    assert np.isfinite(reference).all() and (reference > 0).all(), 'RMSPE requires positive finite references'
    raw_error_a, raw_error_b = error_a, error_b
    endpoint_statistics = {}
    for endpoint in ['rmse', 'rmspe']:
        error_a = raw_error_a if endpoint == 'rmse' else 100 * raw_error_a / reference
        error_b = raw_error_b if endpoint == 'rmse' else 100 * raw_error_b / reference
        assert np.isfinite(error_a).all() and np.isfinite(error_b).all(), (a, b, endpoint)
        # Reset for each pair so reordering methods preserves the sampled patients.
        ci_rng = np.random.default_rng(RANDOM_SEED)
        test_rng = np.random.default_rng(RANDOM_SEED + 1)
        improvement = float(rmse_improvement(error_a, error_b))
        d = error_b**2 - error_a**2
        d_mean = float(d.mean())
        constant_d = np.ptp(d) == 0

        # Constant per-patient squared errors give a deterministic RMSE difference.
        # Identical squared errors also give exactly zero difference on every sample.
        if np.all(d == 0) or (np.ptp(error_a**2) == 0 and np.ptp(error_b**2) == 0):
            ci_lower = ci_upper = improvement
            ci_method = 'deterministic'
        else:
            with warnings.catch_warnings(), np.errstate(invalid='ignore', divide='ignore'):
                warnings.simplefilter('ignore', stats.DegenerateDataWarning)
                bootstrap_ci = stats.bootstrap(
                    (error_a, error_b), rmse_improvement, paired=True, vectorized=True,
                    n_resamples=N_BOOTSTRAP, batch=BOOTSTRAP_BATCH, rng=ci_rng,
                    confidence_level=CONFIDENCE_LEVEL, method='BCa',
                )
            ci_lower, ci_upper = bootstrap_ci.confidence_interval
            ci_method = 'paired_BCa'
            if not np.isfinite([ci_lower, ci_upper]).all():
                # BCa acceleration can be undefined for a degenerate jackknife.
                ci_lower, ci_upper = np.quantile(
                    bootstrap_ci.bootstrap_distribution,
                    [(1 - CONFIDENCE_LEVEL) / 2, (1 + CONFIDENCE_LEVEL) / 2],
                )
                ci_method = 'paired_percentile_degenerate_BCa'

        centered_d = d - d_mean
        if constant_d:
            # Center analytically to avoid floating-point residuals for constant d.
            centered_d = np.zeros_like(d)
            observed_statistic = abs(d_mean)
            test_method = 'null_centered_bootstrap_mean_zero_variance'
            t_pvalue = np.nan
            t_status = 'undefined_zero_variance'
        else:
            observed_statistic = abs(d_mean) / (d.std(ddof=1) / np.sqrt(n))
            test_method = 'null_centered_bootstrap_t'
            t_pvalue = float(stats.ttest_1samp(d, popmean=0).pvalue)
            t_status = 'defined'

        extreme = 0
        for start in range(0, N_BOOTSTRAP, BOOTSTRAP_BATCH):
            batch_size = min(BOOTSTRAP_BATCH, N_BOOTSTRAP - start)
            indices = test_rng.integers(0, n, size=(batch_size, n))
            null_samples = centered_d[indices]
            null_means = np.abs(null_samples.mean(axis=1))
            if constant_d:
                null_statistics = null_means
            else:
                null_se = null_samples.std(axis=1, ddof=1) / np.sqrt(n)
                # A zero-SE draw with nonzero mean has infinite |t|; 0/0 is zero.
                null_statistics = np.divide(
                    null_means, null_se, out=np.full(batch_size, np.inf), where=null_se > 0,
                )
                null_statistics[(null_se == 0) & (null_means == 0)] = 0
            extreme += int(np.count_nonzero(null_statistics >= observed_statistic))
        bootstrap_pvalue = (extreme + 1) / (N_BOOTSTRAP + 1)

        if endpoint == 'rmse':
            endpoint_statistics.update(
                rmse_a_ml=float(np.sqrt(np.mean(error_a**2))),
                rmse_b_ml=float(np.sqrt(np.mean(error_b**2))), rmse_improvement_ml=improvement,
                rmse_improvement_ci_lower_ml=float(ci_lower), rmse_improvement_ci_upper_ml=float(ci_upper),
                rmse_ci_method=ci_method, mean_squared_error_improvement_ml2=d_mean,
                rmse_test_method=test_method, rmse_bootstrap_p=bootstrap_pvalue,
                rmse_t_sensitivity_p=t_pvalue, rmse_t_sensitivity_status=t_status,
            )
        elif endpoint == 'rmspe':
            endpoint_statistics.update(
                rmspe_a_percent=float(np.sqrt(np.mean(error_a**2))),
                rmspe_b_percent=float(np.sqrt(np.mean(error_b**2))), rmspe_improvement_pp=improvement,
                rmspe_improvement_ci_lower_pp=float(ci_lower), rmspe_improvement_ci_upper_pp=float(ci_upper),
                rmspe_ci_method=ci_method, mean_squared_error_improvement_percent2=d_mean,
                rmspe_test_method=test_method, rmspe_bootstrap_p=bootstrap_pvalue,
                rmspe_t_sensitivity_p=t_pvalue, rmspe_t_sensitivity_status=t_status,
            )
    error_a, error_b = raw_error_a, raw_error_b

    absolute_improvement = np.abs(error_b) - np.abs(error_a)
    wins = int(np.count_nonzero(absolute_improvement > 0))
    losses = int(np.count_nonzero(absolute_improvement < 0))
    ties = n - wins - losses
    non_ties = wins + losses
    if non_ties:
        sign_test = stats.binomtest(wins, non_ties, p=0.5, alternative='two-sided')
        probability = wins / non_ties
        probability_lower, probability_upper = sign_test.proportion_ci(
            confidence_level=CONFIDENCE_LEVEL, method='exact',
        )
        sign_pvalue = float(sign_test.pvalue)
    else:
        probability = probability_lower = probability_upper = np.nan
        sign_pvalue = 1.0

    statistics_rows.append(dict(
        method_a=a, name_a=COMPARATORS[a][0], method_b=b, name_b=COMPARATORS[b][0],
        n=n, **endpoint_statistics,
        wins_a=wins, losses_a=losses, ties=ties, n_non_ties=non_ties,
        win_proportion_all=wins / n, loss_proportion_all=losses / n, tie_proportion_all=ties / n,
        probability_a_closer_given_no_tie=probability,
        probability_ci_lower=probability_lower, probability_ci_upper=probability_upper,
        closer_exact_p=sign_pvalue, n_bootstrap=N_BOOTSTRAP, random_seed=RANDOM_SEED,
        bootstrap_test_seed=RANDOM_SEED + 1, bootstrap_batch=BOOTSTRAP_BATCH,
        monte_carlo_p_resolution=1 / (N_BOOTSTRAP + 1), confidence_level=CONFIDENCE_LEVEL,
        intervals_multiplicity_adjusted=False, alpha=ALPHA,
    ))

# %%
pairwise_statistics = pd.DataFrame(statistics_rows)
if not pairwise_statistics.empty:
    primary_p = pairwise_statistics[['rmse_bootstrap_p', 'closer_exact_p']].to_numpy()
    sensitivity_p = pairwise_statistics[['rmse_t_sensitivity_p', 'closer_exact_p']].to_numpy()
    primary_adjusted = multipletests(primary_p.ravel(), method='holm')[1].reshape(-1, 2)
    sensitivity_adjusted = multipletests(
        np.nan_to_num(sensitivity_p, nan=1.0).ravel(), method='holm',
    )[1].reshape(-1, 2)
    pairwise_statistics['holm_family_size'] = primary_p.size
    pairwise_statistics['rmse_bootstrap_p_holm'] = primary_adjusted[:, 0]
    pairwise_statistics['closer_exact_p_holm'] = primary_adjusted[:, 1]
    pairwise_statistics['rmse_t_sensitivity_p_holm'] = np.where(
        np.isfinite(sensitivity_p[:, 0]), sensitivity_adjusted[:, 0], np.nan,
    )
    pairwise_statistics['closer_exact_p_holm_sensitivity'] = sensitivity_adjusted[:, 1]
    # RMSPE is its own family across method pairs, separate from RMSE/closer.
    pairwise_statistics['rmspe_holm_family_size'] = len(pairwise_statistics)
    pairwise_statistics['rmspe_bootstrap_p_holm'] = multipletests(
        pairwise_statistics['rmspe_bootstrap_p'], method='holm',
    )[1]
    rmspe_t_p = pairwise_statistics['rmspe_t_sensitivity_p'].to_numpy()
    rmspe_t_adjusted = multipletests(np.nan_to_num(rmspe_t_p, nan=1.0), method='holm')[1]
    pairwise_statistics['rmspe_t_sensitivity_p_holm'] = np.where(
        np.isfinite(rmspe_t_p), rmspe_t_adjusted, np.nan,
    )
    for endpoint, p_column, effect_column in [
        ('rmspe', 'rmspe_bootstrap_p_holm', 'rmspe_improvement_pp'),
        ('rmspe_t_sensitivity', 'rmspe_t_sensitivity_p_holm', 'rmspe_improvement_pp'),
        ('rmse', 'rmse_bootstrap_p_holm', 'rmse_improvement_ml'),
        ('closer', 'closer_exact_p_holm', 'probability_a_closer_given_no_tie'),
        ('rmse_t_sensitivity', 'rmse_t_sensitivity_p_holm', 'rmse_improvement_ml'),
    ]:
        p = pairwise_statistics[p_column]
        effect = pairwise_statistics[effect_column] - (0.5 if endpoint == 'closer' else 0)
        pairwise_statistics[f'{endpoint}_conclusion'] = np.select(
            [p.isna(), (p <= ALPHA) & effect.gt(0), (p <= ALPHA) & effect.lt(0)],
            ['undefined', 'A better', 'B better'], default='no significant difference',
        )
    pairwise_statistics.loc[
        pairwise_statistics['n_non_ties'].eq(0), 'closer_conclusion'
    ] = 'uninformative_all_ties'
    print('RMSE improvement: positive favors A. Intervals are pointwise and unadjusted.')
    with pd.option_context('display.max_columns', None):
        display(pairwise_statistics[[
            'method_a', 'method_b', 'n', 'rmse_improvement_ml',
            'rmse_improvement_ci_lower_ml', 'rmse_improvement_ci_upper_ml',
            'rmse_bootstrap_p_holm', 'rmse_t_sensitivity_p_holm', 'rmse_conclusion',
        ]])
        print('RMSPE improvement in percentage points: positive favors A. Separate Holm family.')
        display(pairwise_statistics[[
            'method_a', 'method_b', 'n', 'rmspe_a_percent', 'rmspe_b_percent',
            'rmspe_improvement_pp', 'rmspe_improvement_ci_lower_pp', 'rmspe_improvement_ci_upper_pp',
            'rmspe_bootstrap_p_holm', 'rmspe_t_sensitivity_p_holm', 'rmspe_conclusion',
        ]])
        print('Probability A is closer, conditional on no tie. Exact intervals are unadjusted.')
        display(pairwise_statistics[[
            'method_a', 'method_b', 'wins_a', 'losses_a', 'ties',
            'probability_a_closer_given_no_tie', 'probability_ci_lower', 'probability_ci_upper',
            'closer_exact_p_holm', 'closer_conclusion',
        ]])
else:
    # Write an explicit empty result so a previous multi-method run is not stale.
    pairwise_statistics = pd.DataFrame(columns=['method_a', 'method_b', 'n'])
    print('One comparator selected: no pairwise statistical tests.')
pairwise_statistics.to_csv(OUTPUT_DIR / 'comparator_pairwise_statistics.csv', index=False)

# %% [markdown]
# ## PSA density inputs and derived measurements
#
# PSA density = consolidated PSA (ng/mL) / whole-gland volume (mL), in ng/mL².
# Use the same PSA for nnUNet702 reference and every selected method. This is a
# derived reference, not an independent PSA measurement. No reported PSAD is used.
# Empty predictions remain in the volume results, but cannot provide a finite
# density; audit them and stop this section rather than silently exclude patients.
# All methods share the same patients. Signed errors are estimate minus reference;
# relative errors, MAPE and RMSPE are percentages. Limits of agreement use mean
# difference ± 1.96 sample SD, not confidence intervals. Pearson correlation is
# computed before logarithmic display and is NaN for constant input series.

# %%
PSAD_UNIT = 'ng/mL²'
psad_audit = volumes.copy()
psad_audit['valid_psa'] = np.isfinite(psad_audit.psa_consolidated) & psad_audit.psa_consolidated.gt(0)
for key in ['gt', *SELECTED_COMPARATORS]:
    value = pd.to_numeric(psad_audit[f'{key}_volume_ml'], errors='coerce')
    valid_volume = np.isfinite(value) & value.gt(0)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        density = psad_audit.psa_consolidated.where(psad_audit.valid_psa) / value.where(valid_volume)
    psad_audit[f'{key}_psad'] = density.where(np.isfinite(density))
    psad_audit[f'{key}_valid_psad'] = valid_volume & np.isfinite(density)
psad_valid_fields = ['valid_psa', *[f'{key}_valid_psad' for key in ['gt', *SELECTED_COMPARATORS]]]
psad_audit['psad_evaluable'] = psad_audit[psad_valid_fields].all(axis=1)
psad_audit.to_csv(OUTPUT_DIR / 'psa_density_cohort_audit.csv', index=False)
assert psad_audit.psad_evaluable.all(), 'Invalid PSA-density inputs; inspect psa_density_cohort_audit.csv'
psad_cases = psad_audit.copy()
assert len(psad_cases) >= 2, 'Paired analysis requires at least two patients'
assert psad_cases['amrn'].is_unique, 'Patient-level resampling requires unique patients'
PSAD_BINS = [0, 0.10, 0.15, 0.20, np.inf]
PSAD_STRATA = ['0–<0.10', '0.10–<0.15', '0.15–<0.20', '≥0.20']
psad_gt = psad_cases['gt_psad']
psad_values = psad_cases[[f'{key}_psad' for key in SELECTED_COMPARATORS]].set_axis(
    SELECTED_COMPARATORS, axis='columns',
)
psad_differences = psad_values.sub(psad_gt, axis='index')
psad_relative_differences = 100 * psad_differences.div(psad_gt, axis='index')

psad_summary_rows = []
for key in SELECTED_COMPARATORS:
    error = psad_differences[key]
    relative = psad_relative_differences[key]
    bias, sd = error.mean(), error.std(ddof=1)
    relative_bias, relative_sd = relative.mean(), relative.std(ddof=1)
    for suffix, values in [
        ('minus_gt_psad', error), ('absolute_error_psad', error.abs()),
        ('absolute_percentage_error_psad', relative.abs()),
    ]:
        psad_cases[f'{key}_{suffix}'] = values
    psad_summary_rows.append(dict(
        method=key, name=COMPARATORS[key][0], n=len(psad_cases), density_unit=PSAD_UNIT,
        bias_psad=bias, mae_psad=error.abs().mean(), rmse_psad=np.sqrt((error**2).mean()),
        rmspe_percent=np.sqrt((relative**2).mean()),
        mean_absolute_percentage_error=relative.abs().mean(),
        pearson_r=psad_values[key].corr(psad_gt)
        if psad_values[key].nunique() > 1 and psad_gt.nunique() > 1 else np.nan,
        loa_lower_psad=bias - 1.96 * sd, loa_upper_psad=bias + 1.96 * sd,
        relative_bias_percent=relative_bias,
        relative_loa_lower_percent=relative_bias - 1.96 * relative_sd,
        relative_loa_upper_percent=relative_bias + 1.96 * relative_sd,
    ))
psad_summary = pd.DataFrame(psad_summary_rows).sort_values('mae_psad')
display(psad_summary)
psad_summary.to_csv(OUTPUT_DIR / 'psa_density_summary.csv', index=False)

# %% [markdown]
# ## PSA density: agreement plots
#
# Match the volume figure's layout and styles. Pearson correlation is calculated
# before logarithmic display. Differences are estimate minus reference: positive
# means higher density. Both difference rows show the complete range; histograms
# include every eligible case and use half the pooled Freedman–Diaconis bin count.
# The error/reference plots follow the existing volume figure, rather than using
# the average of the two measurements on the horizontal axis.

# %%
PSAD_FIGURE_SIZE = (9.5, 8.5)
PSAD_TICKS = [0.01, 0.03, 0.1, 0.3, 1, 3]
psad_all_values = np.column_stack([psad_gt, psad_values.to_numpy()])
psad_lower = psad_all_values.min() * 0.85
psad_upper = psad_all_values.max() * 1.15
psad_ticks = [tick for tick in PSAD_TICKS if psad_lower <= tick <= psad_upper]
psad_summary_by_method = psad_summary.set_index('method')
fig, axes = plt.subplots(
    3, len(SELECTED_COMPARATORS) + 1, figsize=PSAD_FIGURE_SIZE,
    sharey='row', squeeze=False, constrained_layout=True,
    gridspec_kw={'width_ratios': [1] * len(SELECTED_COMPARATORS) + [.85]},
)
fig.get_layout_engine().set(w_pad=.015, h_pad=.015, wspace=.02, hspace=.02)
axes[0, -1].set_axis_off()
for column, key in enumerate(SELECTED_COMPARATORS):
    name = COMPARATORS[key][0]
    panel_name = fill(name, width=16, break_on_hyphens=False)
    style = PLOT_STYLES[key]
    metrics = psad_summary_by_method.loc[key]
    ax = axes[0, column]
    ax.scatter(
        psad_gt, psad_values[key], color=style['color'], marker=style['marker'],
        label=f'{panel_name}\nPearson r = {metrics.pearson_r:.2f}', **scatter_options,
    )
    ax.plot([psad_lower, psad_upper], [psad_lower, psad_upper], 'k--', linewidth=.8)
    ax.set_yscale('log')
    ax.set_ylim(psad_lower, psad_upper)
    ax.set_yticks(psad_ticks, labels=[f'{tick:g}' for tick in psad_ticks])
    ax.yaxis.set_minor_formatter(NullFormatter())
    for row, (differences, unit, fields) in enumerate([
        (psad_differences[key], PSAD_UNIT, ['bias_psad', 'loa_lower_psad', 'loa_upper_psad']),
        (psad_relative_differences[key], '%', [
            'relative_bias_percent', 'relative_loa_lower_percent', 'relative_loa_upper_percent',
        ]),
    ], start=1):
        bias, loa_lower, loa_upper = metrics[fields]
        ax = axes[row, column]
        label = (
            f'{panel_name}\nMean: {bias:.3g} {unit}\n'
            f'95% LoA:\n[{loa_lower:.3g}, {loa_upper:.3g}]'
        )
        ax.scatter(
            psad_gt, differences, color=style['color'], marker=style['marker'],
            label=label, **scatter_options,
        )
        for level, linestyle in [(bias, '-'), (loa_lower, '--'), (loa_upper, '--')]:
            ax.axhline(level, color=style['color'], linestyle=linestyle, linewidth=.8)
    for row, ylabel in enumerate([
        f'{name}\nPSA density ({PSAD_UNIT}; log scale)',
        f'{name} − reference\nPSA density ({PSAD_UNIT})', 'Relative difference (%)',
    ]):
        ax = axes[row, column]
        if (row, column) != (0, 0):
            ax.sharex(axes[0, 0])
        ax.set_xscale('log')
        ax.set_xlim(psad_lower, psad_upper)
        ax.set_xticks(psad_ticks, labels=[f'{tick:g}' for tick in psad_ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(f'nnUNet702 reference PSA density\n({PSAD_UNIT}; log scale)', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
for row, table in [(1, psad_differences), (2, psad_relative_differences)]:
    ax = axes[row, -1]
    fd_edges = np.histogram_bin_edges(table.to_numpy().ravel(), bins='fd')
    bin_count = max(1, int(np.ceil((len(fd_edges) - 1) / 2)))
    edges = np.linspace(fd_edges[0], fd_edges[-1], bin_count + 1)
    limits = axes[row, 0].get_ylim()
    ax.set_ylim(min(limits[0], edges[0]), max(limits[1], edges[-1]))
    for key in SELECTED_COMPARATORS:
        counts, _ = np.histogram(table[key], bins=edges)
        style = PLOT_STYLES[key]
        ax.stairs(
            counts, edges, orientation='horizontal', color=style['color'],
            linestyle=style['linestyle'], linewidth=1.2,
            label=fill(COMPARATORS[key][0], width=16, break_on_hyphens=False),
        )
    ax.set_xlim(left=0)
    ax.locator_params(axis='x', nbins=3, integer=True)
axes[1, 0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:+g}' if value else '0'))
axes[2, 0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:+g}%' if value else '0%'))
for ax in axes.flat:
    if ax.axison:
        ax.tick_params(axis='both', labelsize=7, pad=1.5, labelbottom=True, labelleft=True)
        ax.legend(**legend_options)
# Collect outliers from the original data against the final shared PSA-density axes.
psad_plot_outliers = []
for column, key in enumerate(SELECTED_COMPARATORS):
    for row, (panel_name, y_values) in enumerate([
        ('PSA density', psad_values[key]),
        ('difference', psad_differences[key]),
        ('relative difference', psad_relative_differences[key]),
    ]):
        panel = axes[row, column]
        x_values = psad_gt.to_numpy(dtype=float)
        y_values = y_values.to_numpy(dtype=float)
        x_limits = sorted(panel.get_xlim())
        y_limits = sorted(panel.get_ylim())
        visible = np.isfinite(x_values) & np.isfinite(y_values)
        if panel.get_xscale() == 'log':
            visible &= x_values > 0
        if panel.get_yscale() == 'log':
            visible &= y_values > 0
        visible &= (x_values >= x_limits[0]) & (x_values <= x_limits[1])
        visible &= (y_values >= y_limits[0]) & (y_values <= y_limits[1])
        if not visible.all():
            outliers = psad_cases.loc[~visible, ['ID', 'aano']].copy()
            outliers['x'] = x_values[~visible]
            outliers['y'] = y_values[~visible]
            psad_plot_outliers.append((
                f'PSA-density agreement / {key} / {panel_name}: '
                f'{len(outliers)} / {len(visible)} hidden or invalid points; '
                f'x limits={x_limits}, y limits={y_limits}', outliers,
            ))

fig.savefig(OUTPUT_DIR / 'psa_density_agreement.png', dpi=300)
fig.savefig(OUTPUT_DIR / 'psa_density_agreement.pdf')
plt.show()
for message, outliers in psad_plot_outliers:
    print(message)
    print(outliers.to_string(index=False))

# %% [markdown]
# ## PSA density: strata and threshold statistics
#
# Use unrounded densities and left-closed strata: **[0, 0.10), [0.10, 0.15),
# [0.15, 0.20), [0.20, ∞)**. Confusion-matrix rows are nnUNet702 automated reference
# strata; columns are estimated strata. These evaluate density-stratum agreement,
# not cancer detection. The cutoff analyses define positive as density ≥ cutoff.
#
# Balanced accuracy is mean recall across reference strata with support. Macro F1
# averages defined per-stratum F1 scores. Quadratic kappa uses squared ordinal
# stratum distances. A zero denominator gives NaN, not perfect performance or zero;
# undefined components are omitted from macro means. Empty strata remain visible.

# %%
for key in ['gt', *SELECTED_COMPARATORS]:
    psad_cases[f'{key}_psad_stratum'] = pd.cut(
        psad_cases[f'{key}_psad'], PSAD_BINS, labels=PSAD_STRATA, right=False,
    )
psad_cases.to_csv(OUTPUT_DIR / 'psa_density_per_case.csv', index=False)
psad_confusion_rows, psad_class_rows, psad_classification_rows, psad_threshold_rows = [], [], [], []
psad_confusion_matrices = {}
for key in SELECTED_COMPARATORS:
    matrix = pd.crosstab(
        psad_cases['gt_psad_stratum'], psad_cases[f'{key}_psad_stratum'], dropna=False,
    ).reindex(index=PSAD_STRATA, columns=PSAD_STRATA, fill_value=0).to_numpy(dtype=int)
    psad_confusion_matrices[key] = matrix
    n = int(matrix.sum())
    assert n == len(psad_cases), 'Every eligible case must belong to one stratum'
    reference_support, estimated_support = matrix.sum(axis=1), matrix.sum(axis=0)
    recalls, f1_scores = [], []
    for i, reference_stratum in enumerate(PSAD_STRATA):
        for j, estimated_stratum in enumerate(PSAD_STRATA):
            psad_confusion_rows.append(dict(
                method=key, reference_stratum=reference_stratum, estimated_stratum=estimated_stratum,
                n=int(matrix[i, j]), reference_support=int(reference_support[i]),
                row_percent=100 * matrix[i, j] / reference_support[i] if reference_support[i] else np.nan,
            ))
        tp = int(matrix[i, i])
        fn, fp = int(reference_support[i] - tp), int(estimated_support[i] - tp)
        tn = n - tp - fn - fp
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        specificity = tn / (tn + fp) if tn + fp else np.nan
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
        recalls.append(recall)
        f1_scores.append(f1)
        psad_class_rows.append(dict(
            method=key, stratum=reference_stratum, support=int(reference_support[i]),
            predicted_support=int(estimated_support[i]), tp=tp, fp=fp, fn=fn, tn=tn,
            precision=precision, recall=recall, specificity=specificity, f1=f1,
        ))
    weights = (np.arange(4)[:, None] - np.arange(4)[None, :])**2
    expected = np.outer(reference_support, estimated_support) / n
    expected_disagreement = float((weights * expected).sum())
    kappa = 1 - (weights * matrix).sum() / expected_disagreement if expected_disagreement else np.nan
    lower, equal, higher = int(np.tril(matrix, -1).sum()), int(np.trace(matrix)), int(np.triu(matrix, 1).sum())
    psad_classification_rows.append(dict(
        method=key, name=COMPARATORS[key][0], n=n, accuracy=equal / n,
        balanced_accuracy=np.nanmean(recalls), macro_f1=np.nanmean(f1_scores),
        quadratic_weighted_kappa=kappa, lower_stratum_n=lower, equal_stratum_n=equal,
        higher_stratum_n=higher, lower_stratum_proportion=lower / n,
        equal_stratum_proportion=equal / n, higher_stratum_proportion=higher / n,
    ))
    for cutoff in PSAD_BINS[1:-1]:
        reference_positive = psad_gt.ge(cutoff)
        estimated_positive = psad_cases[f'{key}_psad'].ge(cutoff)
        tp = int((reference_positive & estimated_positive).sum())
        fn = int((reference_positive & ~estimated_positive).sum())
        fp = int((~reference_positive & estimated_positive).sum())
        tn = int((~reference_positive & ~estimated_positive).sum())
        psad_threshold_rows.append(dict(
            method=key, cutoff_psad=cutoff, density_unit=PSAD_UNIT, n=n,
            tp=tp, fp=fp, fn=fn, tn=tn,
            sensitivity=tp / (tp + fn) if tp + fn else np.nan,
            specificity=tn / (tn + fp) if tn + fp else np.nan,
            ppv=tp / (tp + fp) if tp + fp else np.nan,
            npv=tn / (tn + fn) if tn + fn else np.nan,
            accuracy=(tp + tn) / n,
        ))
psad_confusion = pd.DataFrame(psad_confusion_rows)
psad_per_stratum = pd.DataFrame(psad_class_rows)
psad_classification = pd.DataFrame(psad_classification_rows)
psad_thresholds = pd.DataFrame(psad_threshold_rows)
display(psad_classification)
display(psad_per_stratum)
display(psad_thresholds)
psad_confusion.to_csv(OUTPUT_DIR / 'psa_density_confusion.csv', index=False)
psad_per_stratum.to_csv(OUTPUT_DIR / 'psa_density_per_stratum.csv', index=False)
psad_classification.to_csv(OUTPUT_DIR / 'psa_density_classification_summary.csv', index=False)
psad_thresholds.to_csv(OUTPUT_DIR / 'psa_density_threshold_statistics.csv', index=False)

# %%
fig, axes = plt.subplots(
    1, len(SELECTED_COMPARATORS), figsize=(3.4 * len(SELECTED_COMPARATORS), 3.6),
    squeeze=False, constrained_layout=True,
)
for ax, key in zip(axes.flat, SELECTED_COMPARATORS):
    matrix = psad_confusion_matrices[key]
    support = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, support, out=np.full(matrix.shape, np.nan), where=support > 0)
    heatmap = ax.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
    for i, j in np.ndindex(matrix.shape):
        percent = f'{normalized[i, j]:.1%}' if support[i, 0] else '—'
        ax.text(j, i, f'{matrix[i, j]}\n{percent}', ha='center', va='center', fontsize=8,
                color='white' if normalized[i, j] > .5 else 'black')
    ax.set_xticks(range(4), labels=PSAD_STRATA, rotation=35, ha='right')
    ax.set_yticks(range(4), labels=PSAD_STRATA)
    ax.set_xlabel('Estimated PSA-density stratum')
    ax.set_ylabel('nnUNet702 reference stratum')
    ax.set_title(f'{COMPARATORS[key][0]}\nn = {len(psad_cases)}', fontsize=10)
fig.colorbar(heatmap, ax=list(axes.flat), label='Proportion within reference stratum', shrink=.8)
fig.savefig(OUTPUT_DIR / 'psa_density_confusion.png', dpi=300)
fig.savefig(OUTPUT_DIR / 'psa_density_confusion.pdf')
plt.show()

# %% [markdown]
# ## PSA density: paired stratum-accuracy comparisons
#
# Which method assigns more patients to the same stratum as the nnUNet702 automated
# reference? For each selected pair, compare correct/incorrect assignments on
# the same eligible patients using a two-sided exact McNemar test. The paired
# 2×2 table has rows A correct/incorrect and columns B correct/incorrect; it is
# distinct from the displayed 4×4 reference-versus-estimate matrices.
#
# Accuracy difference = 100 × (accuracy A − accuracy B), in percentage points;
# positive favors A. Holm correction covers all selected pairs as a separate
# family from continuous-error tests, without an omnibus gate. Conclusions use
# adjusted p-values at the existing ALPHA (0.05). No discordant correctness
# assignments gives p=1; no significant difference does not establish equivalence.
# Quadratic weighted kappa remains descriptive. These tests assess exact stratum
# agreement, not ordinal error severity or cancer detection. Patients are treated
# as independent; predictions are fixed, and cohort selection limits generalization.

# %%
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

psad_stratum_columns = ['gt_psad_stratum', *[
    f'{key}_psad_stratum' for key in SELECTED_COMPARATORS
]]
assert psad_cases[psad_stratum_columns].notna().all().all(), 'Missing PSA-density stratum'
psad_stratum_statistics_rows = []
for a, b in combinations(SELECTED_COMPARATORS, 2):
    correct_a = psad_cases[f'{a}_psad_stratum'].eq(psad_cases['gt_psad_stratum'])
    correct_b = psad_cases[f'{b}_psad_stratum'].eq(psad_cases['gt_psad_stratum'])
    both_correct = int((correct_a & correct_b).sum())
    a_only_correct = int((correct_a & ~correct_b).sum())
    b_only_correct = int((~correct_a & correct_b).sum())
    both_incorrect = int((~correct_a & ~correct_b).sum())
    n = len(psad_cases)
    assert both_correct + a_only_correct + b_only_correct + both_incorrect == n
    assert both_correct + a_only_correct == np.trace(psad_confusion_matrices[a])
    assert both_correct + b_only_correct == np.trace(psad_confusion_matrices[b])
    correctness_table = [[both_correct, a_only_correct], [b_only_correct, both_incorrect]]
    pvalue = (
        mcnemar(correctness_table, exact=True).pvalue
        if a_only_correct + b_only_correct else 1.0
    )
    accuracy_a, accuracy_b = correct_a.mean(), correct_b.mean()
    psad_stratum_statistics_rows.append(dict(
        method_a=a, name_a=COMPARATORS[a][0], method_b=b, name_b=COMPARATORS[b][0],
        n=n, both_correct=both_correct, a_only_correct=a_only_correct,
        b_only_correct=b_only_correct, both_incorrect=both_incorrect,
        accuracy_a=accuracy_a, accuracy_b=accuracy_b,
        accuracy_difference_pp=100 * (accuracy_a - accuracy_b), mcnemar_exact_p=pvalue,
    ))

psad_stratum_pairwise_statistics = pd.DataFrame(psad_stratum_statistics_rows, columns=[
    'method_a', 'name_a', 'method_b', 'name_b', 'n', 'both_correct',
    'a_only_correct', 'b_only_correct', 'both_incorrect', 'accuracy_a', 'accuracy_b',
    'accuracy_difference_pp', 'mcnemar_exact_p',
])
if not psad_stratum_pairwise_statistics.empty:
    psad_stratum_pairwise_statistics['mcnemar_exact_p_holm'] = multipletests(
        psad_stratum_pairwise_statistics['mcnemar_exact_p'], method='holm',
    )[1]
    p = psad_stratum_pairwise_statistics['mcnemar_exact_p_holm']
    effect = psad_stratum_pairwise_statistics['accuracy_difference_pp']
    psad_stratum_pairwise_statistics['conclusion'] = np.select(
        [(p <= ALPHA) & effect.gt(0), (p <= ALPHA) & effect.lt(0)],
        ['A better', 'B better'], default='no significant difference',
    )
else:
    psad_stratum_pairwise_statistics['mcnemar_exact_p_holm'] = pd.Series(dtype=float)
    psad_stratum_pairwise_statistics['conclusion'] = pd.Series(dtype=str)
    print('One comparator selected: no pairwise stratum-accuracy tests.')
psad_stratum_pairwise_statistics['holm_family_size'] = len(psad_stratum_statistics_rows)
psad_stratum_pairwise_statistics['alpha'] = ALPHA
print('Accuracy difference in percentage points: positive favors A. P-values are Holm-adjusted.')
display(psad_stratum_pairwise_statistics[[
    'name_a', 'name_b', 'n', 'accuracy_a', 'accuracy_b', 'accuracy_difference_pp',
    'mcnemar_exact_p_holm', 'conclusion',
]])
psad_stratum_pairwise_statistics.to_csv(
    OUTPUT_DIR / 'psa_density_stratum_pairwise_statistics.csv', index=False,
)

# %% [markdown]
# ## PSA density: paired absolute-error comparisons
#
# Positive improvement means method A is closer than method B. Exactly equal
# absolute errors are ties. All methods use the same eligible PSA-density cases.

# %%
psad_pair_tables = []
psad_count_rows = []
for a, b in combinations(SELECTED_COMPARATORS, 2):
    pair = psad_cases[['ID', 'amrn', 'aano']].copy()
    pair['method_a'], pair['method_b'] = a, b
    pair['absolute_error_improvement_psad'] = (
        psad_cases[f'{b}_absolute_error_psad'] - psad_cases[f'{a}_absolute_error_psad']
    )
    delta = pair['absolute_error_improvement_psad']
    pair['a_vs_b'] = np.select([delta > 0, delta < 0], ['Closer', 'Farther'], default='Tie')
    psad_pair_tables.append(pair)
    for outcome in ['Closer', 'Farther', 'Tie']:
        n = int(pair['a_vs_b'].eq(outcome).sum())
        psad_count_rows.append(dict(
            method_a=a, method_b=b, outcome=outcome, n=n, percent=100 * n / len(pair),
        ))
if psad_pair_tables:
    psad_pairwise = pd.concat(psad_pair_tables, ignore_index=True)
    psad_pairwise_counts = pd.DataFrame(psad_count_rows)
    display(psad_pairwise_counts)
    psad_pairwise.to_csv(OUTPUT_DIR / 'psa_density_pairwise_per_case.csv', index=False)
    psad_pairwise_counts.to_csv(OUTPUT_DIR / 'psa_density_pairwise_counts.csv', index=False)
else:
    pd.DataFrame(columns=['ID', 'amrn', 'aano', 'method_a', 'method_b',
                          'absolute_error_improvement_psad', 'a_vs_b']).to_csv(
        OUTPUT_DIR / 'psa_density_pairwise_per_case.csv', index=False,
    )
    pd.DataFrame(columns=['method_a', 'method_b', 'outcome', 'n', 'percent']).to_csv(
        OUTPUT_DIR / 'psa_density_pairwise_counts.csv', index=False,
    )

# %% [markdown]
# ## PSA density: paired post hoc inference: RMSE, RMSPE, and probability of being closer
#
# These are two different questions: a method can have lower population RMSE
# without being closer on most patients. Positive RMSE improvement = RMSE(B) −
# RMSE(A), in ng/mL², favors A. Equality of population RMSE is equivalent to zero mean
# paired squared-error difference, d = error(B)² − error(A)² ((ng/mL²)²).
#
# RMSPE (%) = sqrt(mean((100 × (prediction − reference) / reference)²)).
# Each patient is normalized by their own positive reference. The same inference
# below is applied to percentage errors for RMSPE; squared-error differences are
# then in percent². RMSPE(B) − RMSPE(A) and its interval are in percentage points;
# positive favors A. BCa intervals remain pointwise and unadjusted.
# RMSPE bootstrap tests form a separate Holm family across method pairs in this
# analysis; RMSPE t-test sensitivity p-values have their own analogous family.
# Existing RMSE/closer families are unchanged. Zero predictions remain eligible
# where allowed by the existing cohort rules.
#
# Primary RMSE inference uses a two-sided, null-centered bootstrap-t test of
# mean(d)=0. Its null samples resample d−mean(d), not the original uncentered
# differences. Pointwise 95% BCa intervals resample both methods together and
# recompute RMSE(B)−RMSE(A). The paired t-test on d is a sensitivity analysis;
# sample size alone does not remove concern about skewed squared errors.
#
# Independently, an exact two-sided binomial sign test assesses
# P(A closer | no tie)=0.5. The exact probability interval excludes
# ties; all-patient win/loss/tie proportions remain visible. An all-tie comparison
# has undefined conditional probability and p=1, providing no evidence of a winner.
#
# Holm correction jointly covers both primary endpoints across every selected
# pair (six tests for three methods), without an omnibus gate. Sensitivity Holm
# correction replaces bootstrap p-values with t-test p-values in the same family.
# PSA-density tests form a separate family from the earlier volume tests; no
# correction across the two measurement types is applied.
# Undefined t-tests occupy their family slot as p=1 but remain undefined in the
# output. All intervals are pointwise and unadjusted; conclusions use adjusted
# p-values. Bootstrap p-values have finite Monte Carlo resolution, not exact zeros.
#
# Resampling treats predictions as fixed and patients as independent observations
# representative of the population of interest. This selected linked cohort is
# not a random population sample, so generalization requires that assumption;
# results concern PSA-density accuracy, not spatial segmentation quality or training
# variability. See [Ludbrook, 1995](https://pubmed.ncbi.nlm.nih.gov/7487733/) and
# [SciPy paired BCa bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).

# %%
import warnings

from scipy import stats
from statsmodels.stats.multitest import multipletests

N_BOOTSTRAP = 100_000
RANDOM_SEED = 20261102
BOOTSTRAP_BATCH = 1_000
ALPHA = 0.05
CONFIDENCE_LEVEL = 0.95

assert N_BOOTSTRAP > 0 and BOOTSTRAP_BATCH > 0
assert len(psad_cases) >= 2, 'Paired inference requires at least two patients'
assert psad_cases['amrn'].is_unique, 'Patient-level resampling requires unique patients'


# A vectorized statistic is needed by SciPy for paired BCa resampling/jackknifing.
def psad_rmse_improvement(error_a, error_b, axis=-1):
    return np.sqrt(np.mean(error_b**2, axis=axis)) - np.sqrt(np.mean(error_a**2, axis=axis))


# %%
psad_statistics_rows = []
for a, b in combinations(SELECTED_COMPARATORS, 2):
    error_a = (psad_cases[f'{a}_psad'] - psad_cases['gt_psad']).to_numpy(dtype=float)
    error_b = (psad_cases[f'{b}_psad'] - psad_cases['gt_psad']).to_numpy(dtype=float)
    assert np.isfinite(error_a).all() and np.isfinite(error_b).all(), (a, b)
    n = len(error_a)
    reference = psad_cases['gt_psad'].to_numpy(dtype=float)
    assert np.isfinite(reference).all() and (reference > 0).all(), 'RMSPE requires positive finite references'
    raw_error_a, raw_error_b = error_a, error_b
    endpoint_statistics = {}
    for endpoint in ['rmse', 'rmspe']:
        error_a = raw_error_a if endpoint == 'rmse' else 100 * raw_error_a / reference
        error_b = raw_error_b if endpoint == 'rmse' else 100 * raw_error_b / reference
        assert np.isfinite(error_a).all() and np.isfinite(error_b).all(), (a, b, endpoint)
        # Reset for each pair so reordering methods preserves the sampled patients.
        ci_rng = np.random.default_rng(RANDOM_SEED)
        test_rng = np.random.default_rng(RANDOM_SEED + 1)
        improvement = float(psad_rmse_improvement(error_a, error_b))
        d = error_b**2 - error_a**2
        d_mean = float(d.mean())
        constant_d = np.ptp(d) == 0

        # Constant per-patient squared errors give a deterministic RMSE difference.
        # Identical squared errors also give exactly zero difference on every sample.
        if np.all(d == 0) or (np.ptp(error_a**2) == 0 and np.ptp(error_b**2) == 0):
            ci_lower = ci_upper = improvement
            ci_method = 'deterministic'
        else:
            with warnings.catch_warnings(), np.errstate(invalid='ignore', divide='ignore'):
                warnings.simplefilter('ignore', stats.DegenerateDataWarning)
                bootstrap_ci = stats.bootstrap(
                    (error_a, error_b), psad_rmse_improvement, paired=True, vectorized=True,
                    n_resamples=N_BOOTSTRAP, batch=BOOTSTRAP_BATCH, rng=ci_rng,
                    confidence_level=CONFIDENCE_LEVEL, method='BCa',
                )
            ci_lower, ci_upper = bootstrap_ci.confidence_interval
            ci_method = 'paired_BCa'
            if not np.isfinite([ci_lower, ci_upper]).all():
                # BCa acceleration can be undefined for a degenerate jackknife.
                ci_lower, ci_upper = np.quantile(
                    bootstrap_ci.bootstrap_distribution,
                    [(1 - CONFIDENCE_LEVEL) / 2, (1 + CONFIDENCE_LEVEL) / 2],
                )
                ci_method = 'paired_percentile_degenerate_BCa'

        centered_d = d - d_mean
        if constant_d:
            # Center analytically to avoid floating-point residuals for constant d.
            centered_d = np.zeros_like(d)
            observed_statistic = abs(d_mean)
            test_method = 'null_centered_bootstrap_mean_zero_variance'
            t_pvalue = np.nan
            t_status = 'undefined_zero_variance'
        else:
            observed_statistic = abs(d_mean) / (d.std(ddof=1) / np.sqrt(n))
            test_method = 'null_centered_bootstrap_t'
            t_pvalue = float(stats.ttest_1samp(d, popmean=0).pvalue)
            t_status = 'defined'

        extreme = 0
        for start in range(0, N_BOOTSTRAP, BOOTSTRAP_BATCH):
            batch_size = min(BOOTSTRAP_BATCH, N_BOOTSTRAP - start)
            indices = test_rng.integers(0, n, size=(batch_size, n))
            null_samples = centered_d[indices]
            null_means = np.abs(null_samples.mean(axis=1))
            if constant_d:
                null_statistics = null_means
            else:
                null_se = null_samples.std(axis=1, ddof=1) / np.sqrt(n)
                # A zero-SE draw with nonzero mean has infinite |t|; 0/0 is zero.
                null_statistics = np.divide(
                    null_means, null_se, out=np.full(batch_size, np.inf), where=null_se > 0,
                )
                null_statistics[(null_se == 0) & (null_means == 0)] = 0
            extreme += int(np.count_nonzero(null_statistics >= observed_statistic))
        bootstrap_pvalue = (extreme + 1) / (N_BOOTSTRAP + 1)

        if endpoint == 'rmse':
            endpoint_statistics.update(
                rmse_a_psad=float(np.sqrt(np.mean(error_a**2))),
                rmse_b_psad=float(np.sqrt(np.mean(error_b**2))), rmse_improvement_psad=improvement,
                rmse_improvement_ci_lower_psad=float(ci_lower), rmse_improvement_ci_upper_psad=float(ci_upper),
                rmse_ci_method=ci_method, mean_squared_error_improvement_psad2=d_mean,
                rmse_test_method=test_method, rmse_bootstrap_p=bootstrap_pvalue,
                rmse_t_sensitivity_p=t_pvalue, rmse_t_sensitivity_status=t_status,
            )
        elif endpoint == 'rmspe':
            endpoint_statistics.update(
                rmspe_a_percent=float(np.sqrt(np.mean(error_a**2))),
                rmspe_b_percent=float(np.sqrt(np.mean(error_b**2))), rmspe_improvement_pp=improvement,
                rmspe_improvement_ci_lower_pp=float(ci_lower), rmspe_improvement_ci_upper_pp=float(ci_upper),
                rmspe_ci_method=ci_method, mean_squared_error_improvement_percent2=d_mean,
                rmspe_test_method=test_method, rmspe_bootstrap_p=bootstrap_pvalue,
                rmspe_t_sensitivity_p=t_pvalue, rmspe_t_sensitivity_status=t_status,
            )
    error_a, error_b = raw_error_a, raw_error_b

    absolute_improvement = np.abs(error_b) - np.abs(error_a)
    wins = int(np.count_nonzero(absolute_improvement > 0))
    losses = int(np.count_nonzero(absolute_improvement < 0))
    ties = n - wins - losses
    non_ties = wins + losses
    if non_ties:
        sign_test = stats.binomtest(wins, non_ties, p=0.5, alternative='two-sided')
        probability = wins / non_ties
        probability_lower, probability_upper = sign_test.proportion_ci(
            confidence_level=CONFIDENCE_LEVEL, method='exact',
        )
        sign_pvalue = float(sign_test.pvalue)
    else:
        probability = probability_lower = probability_upper = np.nan
        sign_pvalue = 1.0

    psad_statistics_rows.append(dict(
        method_a=a, name_a=COMPARATORS[a][0], method_b=b, name_b=COMPARATORS[b][0],
        n=n, **endpoint_statistics,
        wins_a=wins, losses_a=losses, ties=ties, n_non_ties=non_ties,
        win_proportion_all=wins / n, loss_proportion_all=losses / n, tie_proportion_all=ties / n,
        probability_a_closer_given_no_tie=probability,
        probability_ci_lower=probability_lower, probability_ci_upper=probability_upper,
        closer_exact_p=sign_pvalue, n_bootstrap=N_BOOTSTRAP, random_seed=RANDOM_SEED,
        bootstrap_test_seed=RANDOM_SEED + 1, bootstrap_batch=BOOTSTRAP_BATCH,
        monte_carlo_p_resolution=1 / (N_BOOTSTRAP + 1), confidence_level=CONFIDENCE_LEVEL,
        intervals_multiplicity_adjusted=False, alpha=ALPHA,
    ))

# %%
psad_pairwise_statistics = pd.DataFrame(psad_statistics_rows)
if not psad_pairwise_statistics.empty:
    primary_p = psad_pairwise_statistics[['rmse_bootstrap_p', 'closer_exact_p']].to_numpy()
    sensitivity_p = psad_pairwise_statistics[['rmse_t_sensitivity_p', 'closer_exact_p']].to_numpy()
    primary_adjusted = multipletests(primary_p.ravel(), method='holm')[1].reshape(-1, 2)
    sensitivity_adjusted = multipletests(
        np.nan_to_num(sensitivity_p, nan=1.0).ravel(), method='holm',
    )[1].reshape(-1, 2)
    psad_pairwise_statistics['holm_family_size'] = primary_p.size
    psad_pairwise_statistics['rmse_bootstrap_p_holm'] = primary_adjusted[:, 0]
    psad_pairwise_statistics['closer_exact_p_holm'] = primary_adjusted[:, 1]
    psad_pairwise_statistics['rmse_t_sensitivity_p_holm'] = np.where(
        np.isfinite(sensitivity_p[:, 0]), sensitivity_adjusted[:, 0], np.nan,
    )
    psad_pairwise_statistics['closer_exact_p_holm_sensitivity'] = sensitivity_adjusted[:, 1]
    # RMSPE is its own family across method pairs, separate from RMSE/closer.
    psad_pairwise_statistics['rmspe_holm_family_size'] = len(psad_pairwise_statistics)
    psad_pairwise_statistics['rmspe_bootstrap_p_holm'] = multipletests(
        psad_pairwise_statistics['rmspe_bootstrap_p'], method='holm',
    )[1]
    rmspe_t_p = psad_pairwise_statistics['rmspe_t_sensitivity_p'].to_numpy()
    rmspe_t_adjusted = multipletests(np.nan_to_num(rmspe_t_p, nan=1.0), method='holm')[1]
    psad_pairwise_statistics['rmspe_t_sensitivity_p_holm'] = np.where(
        np.isfinite(rmspe_t_p), rmspe_t_adjusted, np.nan,
    )
    for endpoint, p_column, effect_column in [
        ('rmspe', 'rmspe_bootstrap_p_holm', 'rmspe_improvement_pp'),
        ('rmspe_t_sensitivity', 'rmspe_t_sensitivity_p_holm', 'rmspe_improvement_pp'),
        ('rmse', 'rmse_bootstrap_p_holm', 'rmse_improvement_psad'),
        ('closer', 'closer_exact_p_holm', 'probability_a_closer_given_no_tie'),
        ('rmse_t_sensitivity', 'rmse_t_sensitivity_p_holm', 'rmse_improvement_psad'),
    ]:
        p = psad_pairwise_statistics[p_column]
        effect = psad_pairwise_statistics[effect_column] - (0.5 if endpoint == 'closer' else 0)
        psad_pairwise_statistics[f'{endpoint}_conclusion'] = np.select(
            [p.isna(), (p <= ALPHA) & effect.gt(0), (p <= ALPHA) & effect.lt(0)],
            ['undefined', 'A better', 'B better'], default='no significant difference',
        )
    psad_pairwise_statistics.loc[
        psad_pairwise_statistics['n_non_ties'].eq(0), 'closer_conclusion'
    ] = 'uninformative_all_ties'
    print('RMSE improvement: positive favors A. Intervals are pointwise and unadjusted.')
    with pd.option_context('display.max_columns', None):
        display(psad_pairwise_statistics[[
            'method_a', 'method_b', 'n', 'rmse_improvement_psad',
            'rmse_improvement_ci_lower_psad', 'rmse_improvement_ci_upper_psad',
            'rmse_bootstrap_p_holm', 'rmse_t_sensitivity_p_holm', 'rmse_conclusion',
        ]])
        print('RMSPE improvement in percentage points: positive favors A. Separate Holm family.')
        display(psad_pairwise_statistics[[
            'method_a', 'method_b', 'n', 'rmspe_a_percent', 'rmspe_b_percent',
            'rmspe_improvement_pp', 'rmspe_improvement_ci_lower_pp', 'rmspe_improvement_ci_upper_pp',
            'rmspe_bootstrap_p_holm', 'rmspe_t_sensitivity_p_holm', 'rmspe_conclusion',
        ]])
        print('Probability A is closer, conditional on no tie. Exact intervals are unadjusted.')
        display(psad_pairwise_statistics[[
            'method_a', 'method_b', 'wins_a', 'losses_a', 'ties',
            'probability_a_closer_given_no_tie', 'probability_ci_lower', 'probability_ci_upper',
            'closer_exact_p_holm', 'closer_conclusion',
        ]])
else:
    # Write an explicit empty result so a previous multi-method run is not stale.
    psad_pairwise_statistics = pd.DataFrame(columns=['method_a', 'method_b', 'n'])
    print('One comparator selected: no pairwise statistical tests.')
psad_pairwise_statistics['density_unit'] = PSAD_UNIT
psad_pairwise_statistics.to_csv(OUTPUT_DIR / 'psa_density_pairwise_statistics.csv', index=False)

# %% [markdown]
# ## PI-RADS 3: biopsy decisions against CSCANCER outcomes
#
# Select normalized reportmaxpirads_ref = 3, without fallback to the entire cohort.
# CSCANCER Y is positive, N is negative; unknown outcomes are excluded only here.
# No ISUP grades are inferred. Recommend biopsy at unrounded PSAD >= 0.15 ng/mL²,
# including equality. Evaluate the automated reference and all comparators on the same patients.
# This retrospective threshold simulation does not describe observed management
# or establish routine-care performance or prevalence. CSCANCER is the recorded
# source outcome, not an independently verified grade.

# %%
BIOPSY_PSAD_THRESHOLD = 0.15
biopsy_methods = ['gt', *SELECTED_COMPARATORS]
biopsy_names = {'gt': REFERENCE_NAME, **{key: COMPARATORS[key][0] for key in SELECTED_COMPARATORS}}
biopsy_cohort = 'reported_PI-RADS_3'
biopsy_audit = psad_audit.copy()
biopsy_selected = biopsy_audit['pirads'].eq(3)
biopsy_audit['biopsy_pirads_status'] = np.select(
    [biopsy_audit['pirads'].isna(), biopsy_selected],
    ['missing_or_invalid_pirads', 'pirads_3'], default='outside_pirads_3',
)
biopsy_outcome_valid = biopsy_audit['cscancer'].isin(['Y', 'N'])
biopsy_audit['biopsy_outcome_status'] = np.where(biopsy_outcome_valid, 'known_cscancer', 'unknown_cscancer')
biopsy_audit['case_cancer_positive'] = biopsy_audit['cscancer'].eq('Y').astype('boolean').where(biopsy_outcome_valid)
biopsy_audit['biopsy_cohort'] = biopsy_cohort
biopsy_audit['biopsy_cutoff_psad'] = BIOPSY_PSAD_THRESHOLD
biopsy_audit['biopsy_analysis_status'] = np.select(
    [~biopsy_selected, ~biopsy_audit['psad_evaluable'], ~biopsy_outcome_valid],
    ['outside_selected_subgroup', 'invalid_psad_inputs', 'unknown_cscancer'], default='included',
)
biopsy_audit.to_csv(OUTPUT_DIR / 'psa_density_biopsy_audit.csv', index=False)
biopsy_cases = biopsy_audit.loc[biopsy_audit['biopsy_analysis_status'].eq('included')].copy()
assert biopsy_cases['amrn'].is_unique, 'Biopsy analysis requires unique patients'
biopsy_truth = biopsy_cases['case_cancer_positive'].astype(bool)
biopsy_n = len(biopsy_cases)
biopsy_positive_n = int(biopsy_truth.sum())
biopsy_negative_n = biopsy_n - biopsy_positive_n
print(f'Biopsy cohort: {biopsy_cohort}; n = {biopsy_n}; '
      f'CSCANCER Y = {biopsy_positive_n}; CSCANCER N = {biopsy_negative_n}')
display(biopsy_audit['biopsy_analysis_status'].value_counts().rename_axis('status').to_frame('n'))
display(biopsy_audit['biopsy_outcome_status'].value_counts().rename_axis('outcome_status').to_frame('n'))

biopsy_summary_rows, biopsy_confusion_rows = [], []
for key in biopsy_methods:
    density = pd.to_numeric(biopsy_cases[f'{key}_psad'], errors='coerce')
    assert (np.isfinite(density) & density.gt(0)).all(), f'Invalid {key} biopsy density'
    recommended = density.ge(BIOPSY_PSAD_THRESHOLD)
    biopsy_cases[f'{key}_biopsy_recommended'] = recommended
    tp = int((biopsy_truth & recommended).sum())
    fp = int((~biopsy_truth & recommended).sum())
    fn = int((biopsy_truth & ~recommended).sum())
    tn = int((~biopsy_truth & ~recommended).sum())
    biopsy_cases[f'{key}_biopsy_outcome'] = np.select(
        [biopsy_truth & recommended, ~biopsy_truth & recommended, biopsy_truth & ~recommended],
        ['TP', 'FP', 'FN'], default='TN',
    )
    biopsy_metadata = dict(
        method=key, name=biopsy_names[key], cohort=biopsy_cohort,
        cutoff_psad=BIOPSY_PSAD_THRESHOLD, density_unit=PSAD_UNIT, n=biopsy_n,
    )
    biopsy_summary_rows.append(dict(
        **biopsy_metadata, n_cancer_positive=biopsy_positive_n, n_cancer_negative=biopsy_negative_n,
        tp=tp, fp=fp, fn=fn, tn=tn,
        biopsies_recommended_n=tp + fp, biopsies_avoided_n=tn + fn,
        cancer_positive_detected_n=tp, cancer_positive_missed_n=fn,
        sensitivity=tp / (tp + fn) if tp + fn else np.nan,
        specificity=tn / (tn + fp) if tn + fp else np.nan,
        ppv=tp / (tp + fp) if tp + fp else np.nan,
        npv=tn / (tn + fn) if tn + fn else np.nan,
        accuracy=(tp + tn) / biopsy_n if biopsy_n else np.nan,
        biopsies_recommended_fraction=(tp + fp) / biopsy_n if biopsy_n else np.nan,
        biopsies_avoided_fraction=(tn + fn) / biopsy_n if biopsy_n else np.nan,
        cancer_positive_missed_fraction=fn / biopsy_positive_n if biopsy_positive_n else np.nan,
    ))
    for truth, decision, outcome, count in [
        (False, False, 'TN', tn), (False, True, 'FP', fp),
        (True, False, 'FN', fn), (True, True, 'TP', tp),
    ]:
        support = biopsy_positive_n if truth else biopsy_negative_n
        biopsy_confusion_rows.append(dict(
            **biopsy_metadata, case_cancer_positive=truth, biopsy_recommended=decision,
            outcome=outcome, count=count, outcome_support=support,
            row_proportion=count / support if support else np.nan,
        ))

biopsy_summary = pd.DataFrame(biopsy_summary_rows)
biopsy_confusion = pd.DataFrame(biopsy_confusion_rows)
biopsy_cases.to_csv(OUTPUT_DIR / 'psa_density_biopsy_per_case.csv', index=False)
biopsy_summary.to_csv(OUTPUT_DIR / 'psa_density_biopsy_summary.csv', index=False)
biopsy_confusion.to_csv(OUTPUT_DIR / 'psa_density_biopsy_confusion.csv', index=False)
with pd.option_context('display.max_columns', None):
    display(biopsy_summary)
if not biopsy_n:
    print('No eligible biopsy-analysis cases; exported empty per-case data and zero-count summaries.')

# %% [markdown]
# ### Biopsy decisions: confusion matrices
#
# Rows are case-level cancer outcomes; columns are simulated biopsy decisions.
# Each cell shows its count and percentage within the outcome row. Sensitivity
# and missed-cancer fraction use CSCANCER-positive cases as denominator; specificity uses
# CSCANCER-negative cases. PPV/NPV use recommended/avoided biopsies, respectively. Biopsy
# fractions and accuracy use all included cases. Undefined ratios are NaN.
# Even an empty subgroup refreshes the figures, with zero counts and no percentages.

# %%
fig, axes = plt.subplots(
    1, len(biopsy_methods), figsize=(3.3 * len(biopsy_methods), 4),
    squeeze=False, constrained_layout=True,
)
for ax, key in zip(axes.ravel(), biopsy_methods):
    method_confusion = biopsy_confusion.loc[biopsy_confusion['method'].eq(key)]
    counts = method_confusion['count'].to_numpy().reshape(2, 2)
    proportions = method_confusion['row_proportion'].to_numpy().reshape(2, 2)
    heatmap = ax.imshow(np.ma.masked_invalid(proportions), vmin=0, vmax=1, cmap='Blues')
    for i, j in np.ndindex(counts.shape):
        percent = f'{proportions[i, j]:.1%}' if np.isfinite(proportions[i, j]) else '—'
        ax.text(j, i, f'{counts[i, j]}\n{percent}', ha='center', va='center',
                color='white' if proportions[i, j] > .5 else 'black')
    ax.set_xticks([0, 1], labels=['Avoid biopsy', 'Recommend biopsy'], rotation=25, ha='right', fontsize=8)
    ax.set_yticks([0, 1], labels=['CSCANCER N', 'CSCANCER Y'], fontsize=8)
    ax.set_xlabel('Biopsy decision', fontsize=9)
    ax.set_ylabel('Case outcome', fontsize=9)
    ax.set_title(f'{biopsy_names[key]}\nn = {biopsy_n}', fontsize=10)
fig.suptitle(f'{biopsy_cohort.replace("_", " ")} | PSAD ≥ {BIOPSY_PSAD_THRESHOLD:g} {PSAD_UNIT}', fontsize=11)
fig.colorbar(heatmap, ax=list(axes.ravel()), label='Proportion within outcome row', shrink=.8)
fig.savefig(OUTPUT_DIR / 'psa_density_biopsy_confusion.png', dpi=300)
fig.savefig(OUTPUT_DIR / 'psa_density_biopsy_confusion.pdf')
plt.show()

# %% [markdown]
# ## Clinical report volume quality check
#
# Compare the reported clinical volume with π/6 × d1 × d2 × d3, using parsed
# report dimensions in cm and volumes in mL. This is a consistency screen, not
# a comparison against the nnUNet702 reference. The assumed range is inclusive:
# **lower = 0.5 × (d1−0.1) × (d2−0.1) × (d3−0.1)** and
# **upper = π/6 × (d1+0.1) × (d2+0.1) × (d3+0.1)**.
# This allows dimension uncertainty of ±0.1 cm (±1 mm) and coefficient choice
# between 0.5 and π/6; it is broader than nearest-0.1-cm rounding alone.
# Compare the reported volume directly, without additional volume rounding.
# Among reports above the upper bound, classify **Missing coefficient** when
# the volume matches d1 × d2 × d3 (coefficient 1), allowing ±0.1 cm on
# each dimension: **(d1−0.1) × (d2−0.1) × (d3−0.1)** through
# **(d1+0.1) × (d2+0.1) × (d3+0.1)** mL, inclusive.
# Flags warrant review and do not establish reporting errors or change eligibility.
# Require every selected case to have positive finite uncensored dimensions
# and valid calculated volumes; stop on invalid inputs rather than omit cases.
# Counts and percentages use the entire selected cohort as their denominator.
#
# The Bland–Altman plot uses pairwise mean volume on x and calculated minus
# reported volume on y. Bias and descriptive 95% limits of agreement (mean
# difference ±1.96 sample SD) include all selected cases, including flags.
# These statistics appear in the table; the plot shows only the zero-difference line.
# These normal-theory limits are not the assumed consistency bounds and can be
# sensitive to extreme values. Both panels share x and y axes: display 10–300 mL
# on logarithmic x-axes and −40 to +40 mL on y. In the first panel, count points
# outside either limit once in the annotation. Display boundaries are inclusive.
# Clipped points remain in the statistics and audit; the outside-view percentage
# uses all selected cases.
# The second panel shows reported volume minus the nnUNet702 reference against
# the automated reference volume, retaining the same consistency groups. Positive
# differences indicate overestimation. Its bias and descriptive 95% limits of
# agreement include all selected cases; the reference-volume axis is logarithmic.
# Identifiers remain in the local audit only. After the plots, display and export
# a table of Below range, Above range, and Missing coefficient cases in cohort
# order. Dimensions are parsed cm; recalculated, reported, and nnUNet702 automated-reference
# volumes are mL. Each corresponding PSAD uses the same consolidated
# PSA divided by volume (ng/mL²). Retain missing PI-RADS and unknown CSCANCER.
# Display dimensions/volumes to two decimals and PSAD to three decimals; keep
# full precision in the dataframe and CSV. The table contains no identifiers.

# %%
report_volume_quality = cases[[
    'ID', 'amrn', 'aano', 'vol_report_ref', *DIMENSION_FIELDS,
    *[f'{field}_numeric' for field in DIMENSION_FIELDS],
    *[f'{field}_censored' for field in DIMENSION_FIELDS],
]].copy()
report_volume_quality['reported_volume_ml'] = cases.clinical_volume_ml
quality_dimensions = report_volume_quality[[f'{field}_numeric' for field in DIMENSION_FIELDS]]
quality_censored = report_volume_quality[[f'{field}_censored' for field in DIMENSION_FIELDS]].any(axis=1)
with np.errstate(over='ignore', invalid='ignore'):
    report_volume_quality['ellipsoid_volume_ml'] = np.pi / 6 * quality_dimensions.prod(axis=1, skipna=False)
    report_volume_quality['lower_bound_ml'] = 0.5 * (quality_dimensions - 0.1).prod(axis=1, skipna=False)
    report_volume_quality['upper_bound_ml'] = np.pi / 6 * (quality_dimensions + 0.1).prod(axis=1, skipna=False)
    report_volume_quality['missing_coefficient_volume_ml'] = quality_dimensions.prod(axis=1, skipna=False)
    report_volume_quality['missing_coefficient_lower_bound_ml'] = (quality_dimensions - 0.1).prod(axis=1, skipna=False)
    report_volume_quality['missing_coefficient_upper_bound_ml'] = (quality_dimensions + 0.1).prod(axis=1, skipna=False)
quality_calculated_fields = [
    'ellipsoid_volume_ml', 'lower_bound_ml', 'upper_bound_ml',
    'missing_coefficient_volume_ml', 'missing_coefficient_lower_bound_ml',
    'missing_coefficient_upper_bound_ml',
]
assert not quality_dimensions.empty and (
    np.isfinite(quality_dimensions).all(axis=1) & quality_dimensions.gt(0).all(axis=1)
    & ~quality_censored
    & np.isfinite(report_volume_quality[quality_calculated_fields]).all(axis=1)
    & report_volume_quality.ellipsoid_volume_ml.gt(0)
).all(), (
    'Every selected case must be assessable: require positive finite uncensored dimensions, '
    'positive finite ellipsoid volumes, and finite bounds; inspect clinical_fields_audit.csv'
)
# Treat numerically equivalent boundary calculations as equal (floating-point
# tolerance only, not an additional allowance for reported-volume rounding).
quality_below_range = (
    report_volume_quality.reported_volume_ml.lt(report_volume_quality.lower_bound_ml)
    & ~np.isclose(report_volume_quality.reported_volume_ml, report_volume_quality.lower_bound_ml,
                  rtol=1e-12, atol=1e-12)
)
quality_above_range = (
    report_volume_quality.reported_volume_ml.gt(report_volume_quality.upper_bound_ml)
    & ~np.isclose(report_volume_quality.reported_volume_ml, report_volume_quality.upper_bound_ml,
                  rtol=1e-12, atol=1e-12)
)
quality_missing_coefficient = quality_above_range & (
    report_volume_quality.reported_volume_ml.ge(report_volume_quality.missing_coefficient_lower_bound_ml)
    | np.isclose(report_volume_quality.reported_volume_ml,
                 report_volume_quality.missing_coefficient_lower_bound_ml, rtol=1e-12, atol=1e-12)
) & (
    report_volume_quality.reported_volume_ml.le(report_volume_quality.missing_coefficient_upper_bound_ml)
    | np.isclose(report_volume_quality.reported_volume_ml,
                 report_volume_quality.missing_coefficient_upper_bound_ml, rtol=1e-12, atol=1e-12)
)
report_volume_quality['classification'] = np.select([
    quality_below_range, quality_missing_coefficient, quality_above_range,
], ['Below range', 'Missing coefficient', 'Above range'], default='Within range')
report_volume_quality['mean_volume_ml'] = (
    report_volume_quality.ellipsoid_volume_ml + report_volume_quality.reported_volume_ml
) / 2
report_volume_quality['difference_ml'] = (
    report_volume_quality.ellipsoid_volume_ml - report_volume_quality.reported_volume_ml
)
report_volume_quality.to_csv(OUTPUT_DIR / 'report_volume_quality_audit.csv', index=False)

report_volume_quality_counts = (
    report_volume_quality.classification.value_counts()
    .reindex(['Within range', 'Below range', 'Above range', 'Missing coefficient'], fill_value=0)
    .rename_axis('classification').rename('n').reset_index()
)
report_volume_quality_counts['percent_of_selected_cohort'] = (
    100 * report_volume_quality_counts.n / len(report_volume_quality)
)
quality_differences = report_volume_quality.difference_ml
quality_bias = quality_differences.mean()
quality_sd = quality_differences.std(ddof=1)
quality_loa_lower, quality_loa_upper = quality_bias - 1.96 * quality_sd, quality_bias + 1.96 * quality_sd
report_volume_quality_statistics = pd.DataFrame([dict(
    n_cases=len(report_volume_quality), bias_ml=quality_bias,
    loa_lower_ml=quality_loa_lower, loa_upper_ml=quality_loa_upper,
)])
display(report_volume_quality_counts, report_volume_quality_statistics)

fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.8), constrained_layout=True, sharex=True, sharey=True)
fig.suptitle(f'Clinical report volume (n = {len(quality_differences)})', fontsize=10)
ax = axes[0]
quality_plot_groups = [
    ('Within range', 'Within expected range', 'tab:blue', 'o'),
    ('Missing coefficient', 'Missing coefficient (d1 × d2 × d3; ±0.1 cm)', 'tab:red', '^'),
    ('Above range', 'Otherwise expected from dimensions < reported volume', 'tab:red', '^'),
    ('Below range', 'Expected from dimensions > reported volume', 'tab:orange', 'v'),
]
for classification, legend_label, color, marker in quality_plot_groups:
    group = report_volume_quality.loc[report_volume_quality.classification.eq(classification)]
    ax.scatter(group.mean_volume_ml, group.difference_ml, edgecolors=color,
               facecolors='none' if classification == 'Missing coefficient' else color, marker=marker,
               s=28, alpha=.75, label=f'{legend_label} (n = {len(group)})')
quality_zero_line = ax.axhline(0, color='0.6', linewidth=.8, label='zero difference')
ax.set_xlabel('Mean of calculated ellipsoid and reported\nvolume (mL; log scale)', fontsize=8)
ax.set_ylabel('Calculated ellipsoid − reported\nvolume (mL)', fontsize=8)
ax.set_title('Consistency with report dimensions\n±0.1 cm; coefficient 0.5–π/6', fontsize=9)
ax.set_xscale('log')
quality_volume_ticks = [10, 30, 100, 300]
ax.set_xticks(quality_volume_ticks, labels=[str(tick) for tick in quality_volume_ticks])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlim(10, 300)
ax.set_ylim(-40, +40)
quality_outside_view = ~(
    report_volume_quality.mean_volume_ml.between(*ax.get_xlim())
    & report_volume_quality.difference_ml.between(*ax.get_ylim())
)
quality_outside_count = int(quality_outside_view.sum())
ax.text(.02, .92,
        f'Outside displayed range:\n{quality_outside_count} / {len(quality_differences)} '
        f'({100 * quality_outside_count / len(quality_differences):.1f}%).',
        transform=ax.transAxes, ha='left', va='top', fontsize=7)
ax.legend(handles=[quality_zero_line], loc='lower right', fontsize=7)

quality_reference_ml = cases.gt_volume_ml
quality_reference_difference_ml = report_volume_quality.reported_volume_ml - quality_reference_ml
quality_reference_bias = quality_reference_difference_ml.mean()
quality_reference_sd = quality_reference_difference_ml.std(ddof=1)
quality_reference_loa_lower = quality_reference_bias - 1.96 * quality_reference_sd
quality_reference_loa_upper = quality_reference_bias + 1.96 * quality_reference_sd
quality_agreement_ax = axes[1]
for classification, legend_label, color, marker in quality_plot_groups:
    group_mask = report_volume_quality.classification.eq(classification)
    quality_agreement_ax.scatter(
        quality_reference_ml.loc[group_mask], quality_reference_difference_ml.loc[group_mask],
        edgecolors=color, facecolors='none' if classification == 'Missing coefficient' else color,
        marker=marker, s=28, alpha=.75, label=f'{legend_label} (n = {int(group_mask.sum())})',
    )
for level, linestyle, label in [
    (quality_reference_bias, '-', f'Mean: {quality_reference_bias:.1f} mL'),
    (quality_reference_loa_lower, '--',
     f'95% LoA: [{quality_reference_loa_lower:.1f}, {quality_reference_loa_upper:.1f}] mL'),
    (quality_reference_loa_upper, '--', '_nolegend_'),
]:
    quality_agreement_ax.axhline(level, color='0.4', linestyle=linestyle, linewidth=.8, label=label)
quality_agreement_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:+g}' if value else '0'))
quality_agreement_ax.set_xlabel('nnUNet702 reference volume\n(mL; log scale)', fontsize=8)
quality_agreement_ax.set_ylabel('Reported volume − automated\nreference (mL)', fontsize=8)
quality_agreement_ax.set_title('Agreement with nnUNet702 reference', fontsize=9)
quality_agreement_ax.legend(handles=quality_agreement_ax.lines[:2], loc='best', fontsize=7)
for panel in axes:
    panel.tick_params(axis='both', labelsize=8)
# Reserve space for one shared group legend below both panels. Keeping long
# group labels out of the axes preserves their width in the 7.5-inch figure.
fig.legend(
    handles=ax.collections, loc='outside lower center', fontsize=7,
    frameon=False, handletextpad=.5, borderaxespad=.5,
)
# Collect outliers from both panels, including the already labeled clipping.
quality_plot_outliers = []
for panel, panel_name, x_values, y_values in [
    (axes[0], 'report dimensions', report_volume_quality.mean_volume_ml,
     report_volume_quality.difference_ml),
    (axes[1], 'nnUNet702 agreement', quality_reference_ml,
     quality_reference_difference_ml),
]:
    x_values = x_values.to_numpy(dtype=float)
    y_values = y_values.to_numpy(dtype=float)
    x_limits = sorted(panel.get_xlim())
    y_limits = sorted(panel.get_ylim())
    visible = np.isfinite(x_values) & np.isfinite(y_values)
    if panel.get_xscale() == 'log':
        visible &= x_values > 0
    if panel.get_yscale() == 'log':
        visible &= y_values > 0
    visible &= (x_values >= x_limits[0]) & (x_values <= x_limits[1])
    visible &= (y_values >= y_limits[0]) & (y_values <= y_limits[1])
    if not visible.all():
        outliers = cases.loc[~visible, ['ID', 'aano']].copy()
        outliers['x'] = x_values[~visible]
        outliers['y'] = y_values[~visible]
        quality_plot_outliers.append((
            f'Clinical report volume / {panel_name}: '
            f'{len(outliers)} / {len(visible)} hidden or invalid points; '
            f'x limits={x_limits}, y limits={y_limits}', outliers,
        ))

fig.savefig(OUTPUT_DIR / 'report_volume_quality.png', dpi=300)
fig.savefig(OUTPUT_DIR / 'report_volume_quality.pdf')
plt.show()
for message, outliers in quality_plot_outliers:
    print(message)
    print(outliers.to_string(index=False))

# %%
quality_flagged = report_volume_quality.classification.isin([
    'Below range', 'Above range', 'Missing coefficient',
])
report_volume_quality_table = pd.DataFrame({
    'dim1': quality_dimensions[DIMENSION_FIELDS[0] + '_numeric'],
    'dim2': quality_dimensions[DIMENSION_FIELDS[1] + '_numeric'],
    'dim3': quality_dimensions[DIMENSION_FIELDS[2] + '_numeric'],
    'recalculated_volume_ml': report_volume_quality.ellipsoid_volume_ml,
    'recalculated_psad': cases.psa_consolidated / report_volume_quality.ellipsoid_volume_ml,
    'report_volume_ml': report_volume_quality.reported_volume_ml,
    'report_psad': cases.psa_consolidated / report_volume_quality.reported_volume_ml,
    'reference_volume_ml': cases.gt_volume_ml,
    'reference_psad': cases.psa_consolidated / cases.gt_volume_ml,
    'PIRADS': cases.pirads,
    'CSCANCER': cases.cscancer,
}).loc[quality_flagged].reset_index(drop=True)
report_volume_quality_table.to_csv(OUTPUT_DIR / 'report_volume_quality_table.csv', index=False)
with pd.option_context(
    'styler.render.max_rows', None, 'styler.render.max_columns', None,
    'styler.render.max_elements', max(1, report_volume_quality_table.size),
):
    display(report_volume_quality_table.style.format({
        'dim1': '{:.2f}', 'dim2': '{:.2f}', 'dim3': '{:.2f}',
        'recalculated_volume_ml': '{:.2f}', 'recalculated_psad': '{:.3f}',
        'report_volume_ml': '{:.2f}', 'report_psad': '{:.3f}',
        'reference_volume_ml': '{:.2f}', 'reference_psad': '{:.3f}',
        'PIRADS': '{:.0f}',
    }, na_rep='—').hide(axis='index'))

# %%
# Summarize the flagged rows using unrounded PSAD: reported → recalculated.
quality_report_stratum = pd.cut(
    report_volume_quality_table.report_psad, PSAD_BINS, labels=PSAD_STRATA, right=False,
)
quality_recalculated_stratum = pd.cut(
    report_volume_quality_table.recalculated_psad, PSAD_BINS, labels=PSAD_STRATA, right=False,
)
quality_category_changed = quality_report_stratum.ne(quality_recalculated_stratum)
quality_category_higher = quality_recalculated_stratum.gt(quality_report_stratum)
quality_category_lower = quality_recalculated_stratum.lt(quality_report_stratum)
quality_pirads3 = report_volume_quality_table.PIRADS.eq(3)
quality_report_above_cutoff = report_volume_quality_table.report_psad.ge(BIOPSY_PSAD_THRESHOLD)
quality_recalculated_above_cutoff = report_volume_quality_table.recalculated_psad.ge(BIOPSY_PSAD_THRESHOLD)
quality_threshold_up = quality_pirads3 & ~quality_report_above_cutoff & quality_recalculated_above_cutoff
quality_threshold_down = quality_pirads3 & quality_report_above_cutoff & ~quality_recalculated_above_cutoff
print('Flagged-case PSAD summary (reported → recalculated; ng/mL²)')
print('Four PSAD categories: ' + ', '.join(PSAD_STRATA))
print(
    f'PSAD category changed: {int(quality_category_changed.sum())} / {len(report_volume_quality_table)} '
    f'(higher: {int(quality_category_higher.sum())}; lower: {int(quality_category_lower.sum())}).'
)
print(
    f'PI-RADS 3 crossing {BIOPSY_PSAD_THRESHOLD:g}: '
    f'{int((quality_threshold_up | quality_threshold_down).sum())} / {int(quality_pirads3.sum())} '
    f'(below → ≥ cutoff: {int(quality_threshold_up.sum())}; '
    f'≥ cutoff → below: {int(quality_threshold_down.sum())}).'
)

# %% [markdown]
# ## Local reproducibility record
#
# Keep the complete record in the ignored output directory: source checksums,
# selected membership fingerprint, saved model metadata, analysis parameters,
# package versions, and this notebook's checksum. No clinical records, identifiers,
# passwords, or generated artifacts should be copied into tracked source files.
# Historical inference settings are documented only to the extent that saved
# metadata provides them. A successful notebook run measures existing masks only.

# %%
notebook_path = REPO_ROOT / 'notebooks/toronto_jdmi_4111_clinical_volume.py'
with notebook_path.open('rb') as stream:
    notebook_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
metadata = dict(
    cohort_sha256=cohort_sha256, n_patients=len(volumes), n_scans=len(volumes),
    reference=REFERENCE_NAME, reference_type='automated_surrogate',
    reference_path_template=str((REFERENCE_ROOT / '{ID}' / REFERENCE_FILENAME).relative_to(REPO_ROOT)),
    reference_foreground_labels=list(REFERENCE_LABELS), allowed_reference_labels=[0, *REFERENCE_LABELS],
    reference_volume_rule='Count union of labels 1,2,3 * abs(det(spatial affine)) / 1000',
    reference_inference_settings='Unknown; existing saved masks only', volume_unit='mL',
    accession_exclusion_rule='Every accession in the explicitly selected exclusion JSON, regardless of reason',
    n_listing_cases=len(listing), n_accession_excluded=int(listing.aano_excluded.sum()),
    n_available_manual_references=int(cases.has_manual_reference.sum()),
    selected_comparators=SELECTED_COMPARATORS, expected_cases=EXPECTED_CASES,
    geometry_tolerance_mm=GEOMETRY_TOLERANCE_MM,
    geometry_warning_threshold_mm=GEOMETRY_WARNING_THRESHOLD_MM,
    unknown_units_rule='Assume mm only on a grid matching a known-mm T2 or selected comparator',
    unit_anchor_tolerance_mm=UNIT_ANCHOR_TOLERANCE_MM,
    unit_assumption_counts=measurement_audit.groupby('method').units_assumed_mm.sum().astype(int).to_dict(),
    n_bootstrap=N_BOOTSTRAP, random_seed=RANDOM_SEED, bootstrap_test_seed=RANDOM_SEED + 1,
    bootstrap_batch=BOOTSTRAP_BATCH, alpha=ALPHA, confidence_level=CONFIDENCE_LEVEL,
    relative_difference_display_limits=RELATIVE_DIFFERENCE_LIMITS,
    packages={name: version(name) for name in (
        'numpy', 'pandas', 'nibabel', 'matplotlib', 'scipy', 'statsmodels', 'tqdm', 'ipython',
        'msoffcrypto-tool', 'openpyxl',
    )},
    python_version=platform.python_version(), notebook_sha256=notebook_sha256,
    input_sha256=input_hashes, model_metadata=model_metadata,
    exclusion_source=str(EXCLUDE_JSON.relative_to(REPO_ROOT)),
    workbook_audit_reuse=workbook_audit_provenance,
    patient_identifier='amrn', examination_identifier='aano',
    workbook_amrn_normalization='Trim whitespace and remove one leading literal AIPR_; preserve leading zeros',
    psad_unit=PSAD_UNIT, n_psad_cases=len(psad_cases),
    psad_stratum_cutoffs=PSAD_BINS[1:-1], psad_strata=PSAD_STRATA,
    psad_threshold_positive_rule='unrounded PSAD >= cutoff',
    biopsy_cohort=biopsy_cohort, biopsy_psad_threshold=BIOPSY_PSAD_THRESHOLD,
    biopsy_outcome='CSCANCER Y positive; N negative; unknown excluded',
    n_biopsy_cases=biopsy_n, n_biopsy_positive=biopsy_positive_n,
    n_biopsy_negative=biopsy_negative_n,
    psad_holm_families='RMSE plus closer; separate RMSPE; separate stratum McNemar; analogous sensitivity families',
    psa_rule='Finite positive psa_report_ref, otherwise finite positive corrected psa',
    psa_correction='Haider documented single-record alternative PSA correction to 28.9 ng/mL',
    psa_source_counts=cases.psa_source.value_counts().to_dict(),
    n_psa_corrections=int(cases.psa_correction_applied.sum()),
    n_censored_psa=int(cases.psa_selected_censored.sum()),
    n_valid_pirads=int(cases.pirads_valid.sum()),
    cancer_outcome_counts=cases.cscancer.value_counts().to_dict(),
)
metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
print(f'Completed Toronto JDMI full-cohort volume, PSAD, and biopsy analysis: '
      f'{len(volumes)} patients, {len(SELECTED_COMPARATORS)} comparators, {biopsy_n} biopsy cases.')
