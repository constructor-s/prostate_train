# %% [markdown]
# # Results for the revised volumetry manuscript
#
# Assemble eight testing cohorts and ask whether the multicenter model is closer to the
# reference when it disagrees with clinical ellipsoid volumetry. All inputs are
# source clinical data, reviewed manifests, and saved masks. No older analysis
# exports, inference, mask modification, or resampling are required.
# Run cells in order. Generated tables and private case audits stay in
# results/paper_new_results/. Manuscript prose is edited in paper_new.md;
# this notebook replaces only the six marked table blocks there.

# %%
from pathlib import Path
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations, product
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

REPO_ROOT = Path(__file__).resolve().parents[1] if '__file__' in globals() else Path.cwd().resolve()
if REPO_ROOT.name == 'notebooks':
    REPO_ROOT = REPO_ROOT.parent
assert (REPO_ROOT / 'pyproject.toml').is_file(), 'Establish REPO_ROOT first'
OUTPUT_DIR = REPO_ROOT / 'results/paper_new_results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPDATE_MANUSCRIPT = globals().get('UPDATE_MANUSCRIPT', True)
# Set tuples such as (10, 300) / (-60, 60) to crop all panels consistently.
# Cropping affects display only; every observation remains in the statistics.
BA_XLIM = globals().get('BA_XLIM', None)
BA_YLIM = globals().get('BA_YLIM', None)
BA_REL_YLIM = globals().get('BA_REL_YLIM', None)
SITES = ['RUMC', 'ZGT', 'PCNN']
# Manual references first, then automated surrogates; counts are derived below.
COHORTS = {
    'prostatex': ('', 'ProstateX RUMC', None, 'Manual whole-gland segmentation'),
    'toronto_manual': ('', 'Toronto JDMI', None, 'Median manual-reader planimetry'),
    'picai_RUMC': ('', 'PI-CAI RUMC', None, 'Bosma22b automated surrogate'),
    'toronto_surrogate': ('', 'Toronto JDMI Additional', None, 'nnUNet702 automated surrogate'),
    'picai_ZGT': ('', 'PI-CAI ZGT', None, 'Bosma22b automated surrogate'),
    'picai_PCNN': ('', 'PI-CAI PCNN (wide field)', None, 'Bosma22b automated surrogate'),
}
METHODS = {'nnunet274': 'Multicenter', 'rumcnotprostatexnoclinical236': 'Single-center', 'clinical': 'Clinical ellipsoid'}
CUTOFFS = np.array([.10, .15, .20])
SOURCE_NOTEBOOK_SHA256 = hashlib.sha256((REPO_ROOT / 'notebooks/paper_new_results.py').read_bytes()).hexdigest()
input_sha256 = {}

# Small I/O helpers keep hashing and Markdown escaping consistent across tables.
def read_input(relative, kind='csv'):
    path = Path(relative)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with path.open('rb') as stream:
        input_sha256[str(path.relative_to(REPO_ROOT))] = hashlib.file_digest(stream, 'sha256').hexdigest()
    if kind == 'json':
        return json.loads(path.read_text())
    return pd.read_csv(path, low_memory=False, float_precision='round_trip')


def markdown(frame):
    def cell(value):
        return str(value).replace('|', r'\|').replace('\n', '<br>')
    return '\n'.join(['| ' + ' | '.join(map(cell, frame.columns)) + ' |',
                      '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
                     + ['| ' + ' | '.join(map(cell, row)) + ' |' for row in frame.itertuples(index=False, name=None)])


def exact_closer_p(wins, losses):
    return binomtest(int(wins), int(wins + losses), .5, alternative='two-sided').pvalue if wins + losses else np.nan


def adjust_six(frame):
    assert len(frame) == 6
    # Undefined tests occupy a family slot conservatively, but remain undefined.
    adjusted = multipletests(frame.p_raw.fillna(1), method='holm')[1]
    frame['p_holm'] = np.where(frame.p_raw.notna(), adjusted, np.nan)


def fmt_p(value):
    return 'Undefined' if pd.isna(value) else ('<0.001' if value < .001 else f'{value:.3f}')

# %% [markdown]
# ## Source cohorts and exclusion audit
#
# These cells read original clinical files and reviewed manifests, never another
# notebook's tables. Toronto accession exclusions apply to every endpoint. A
# positive finite manual volume defines the manual cohort; the remaining eligible
# patients form Additional. Missing PSA or outcomes affect secondary endpoints only.

# %%
DATA_ROOT = REPO_ROOT / 'data/raw'
TORONTO_ROOT = REPO_ROOT / 'data_internal/hlab20260928'
STUDY_ROOT = REPO_ROOT / 'results/20261105_tile0.8_nomirroring'
PICAI_ROOT = DATA_ROOT / 'zenodo/picai/picai_labels'
MODEL_NAMES = {'nnunet274': 'Dataset274_raw_3d_fullres_5fold',
               'rumcnotprostatexnoclinical236': 'Dataset236_raw_3d_fullres_5fold'}
GEOMETRY_TOLERANCE_MM = .5
UNIT_ANCHOR_TOLERANCE_MM = .01
model_metadata = {}
measurement_rows = []
postprocessed_sha256 = {}

listing = read_input(TORONTO_ROOT / 'hlab20260928_t2_image_listing_full_clinical.csv')
assert {'ID', 't2', 'ground_truth_volume_ml', 'vol_report_ref'} <= set(listing)
parts = listing.ID.astype('string').str.extract(r'^AIPR_([^_/\\]+)_(\d{8})_([^_/\\]+)$')
listing[['amrn', 'exam_date', 'aano']] = parts
assert parts.notna().all().all(), 'Invalid Toronto ID'
assert listing.ID.is_unique and listing.amrn.is_unique, 'Repeated Toronto scan/patient'
exclusions = read_input(TORONTO_ROOT / 'exclude.json', 'json')
assert isinstance(exclusions, list) and all(isinstance(e, dict) and isinstance(e.get('aano'), str)
    and e['aano'].strip() for e in exclusions), 'Exclusions require nonempty accession strings'
exclusion_audit = pd.DataFrame(exclusions) if exclusions else pd.DataFrame(columns=['aano'])
exclusion_audit['aano'] = exclusion_audit.aano.str.strip()
exclusion_audit['duplicate_accession'] = exclusion_audit.aano.duplicated(keep=False)
exclusion_audit['matched_source'] = exclusion_audit.aano.isin(listing.aano)
exclusion_audit['exclusion_reason'] = ['; '.join(str(e[k]) for k in ('reason', 'description') if e.get(k))
                                        or 'Listed in exclude.json' for e in exclusions]
exclusion_audit.to_csv(OUTPUT_DIR / 'toronto_exclusion_audit.csv', index=False)
reasons = exclusion_audit.groupby('aano', sort=False).exclusion_reason.agg('; '.join)
listing['aano_excluded'] = listing.aano.isin(reasons.index)
listing['exclusion_reason'] = listing.aano.map(reasons).fillna('')
listing['manual_reference_ml'] = pd.to_numeric(listing.ground_truth_volume_ml, errors='coerce')
listing['has_manual_reference'] = np.isfinite(listing.manual_reference_ml) & listing.manual_reference_ml.gt(0)
listing['included'] = ~listing.aano_excluded
listing['analysis_cohort'] = np.select([listing.aano_excluded, listing.has_manual_reference],
                                     ['excluded', 'toronto_manual'], default='toronto_surrogate')
listing.to_csv(OUTPUT_DIR / 'toronto_cohort_audit.csv', index=False)
flow = pd.DataFrame([
    dict(stage='Source clinical listing', n=len(listing)),
    dict(stage='Accession exclusions', n=int(listing.aano_excluded.sum())),
    dict(stage='Eligible', n=int(listing.included.sum())),
    dict(stage='Manual reference', n=int((listing.included & listing.has_manual_reference).sum())),
    dict(stage='Additional surrogate reference', n=int((listing.included & ~listing.has_manual_reference).sum())),
])
flow.to_csv(OUTPUT_DIR / 'toronto_cohort_flow.csv', index=False)
print(flow.to_string(index=False), flush=True)
print('Unmatched exclusion entries:', int((~exclusion_audit.matched_source).sum()), flush=True)
toronto_cases = listing.loc[listing.included].copy()
toronto_cases['clinical_volume_ml'] = pd.to_numeric(toronto_cases.vol_report_ref, errors='coerce')
toronto_cases['image_path'] = toronto_cases.t2.map(lambda p: str(TORONTO_ROOT / p))
toronto_cases['reference_path'] = toronto_cases.ID.map(
    lambda case_id: str(TORONTO_ROOT / 'PROSATEPSAD' / case_id / 'wp-pz-tz_axt2_nnunet-702.nii.gz'))

marksheet = read_input(PICAI_ROOT / 'clinical_information/marksheet.csv')
assert marksheet[['patient_id', 'study_id']].notna().all().all()
assert not marksheet.duplicated(['patient_id', 'study_id']).any(), 'Duplicate marksheet keys'
rumc_manifest = read_input('configs/rumcnotprostatexnoclinical_folds.json', 'json')
picai_manifest = read_input('configs/picai_test_clinical_folds.json', 'json')
pooled_manifest = read_input('configs/pooled2_folds.json', 'json')
for manifest in [rumc_manifest, picai_manifest, pooled_manifest]:
    for fold, rows in manifest.items():
        ids = [r['id'] for r in rows]
        assert len(ids) == len(set(ids)), ('Duplicate manifest ID', fold)
    excluded = {r['id'] for fold, rows in manifest.items() if fold.startswith('excluded') for r in rows}
    testing = {r['id'] for fold, rows in manifest.items() if fold.startswith('testing') or fold == 'endorectal_coil' for r in rows}
    assert not excluded & testing, 'Testing/exclusion manifest overlap'
assert picai_manifest['training'] == rumc_manifest['training']
index = read_input(DATA_ROOT / 'github/prostatex_masks/index.csv')
assert index.ID.is_unique
mapping = read_input(PICAI_ROOT / 'additional_resources/ProstateX-mapping.json', 'json')
links = pd.DataFrame([(k.split('_')[0], v) for k, v in mapping.items()], columns=['ID', 'picai_key'])
ambiguous = links.ID.duplicated(keep=False)
links[['patient_id', 'study_id']] = links.picai_key.str.split('_', expand=True).astype('int64')
linked = index.merge(links.loc[~ambiguous], on='ID', how='left', validate='one_to_one')
linked = linked.merge(marksheet, on=['patient_id', 'study_id'], how='left', validate='many_to_one', indicator=True)
linked['selected'] = linked.ID.isin([r['id'] for r in rumc_manifest['testing']])
linked.to_csv(OUTPUT_DIR / 'prostatex_linkage_audit.csv', index=False)
prostatex = linked.set_index('ID', drop=False).loc[[r['id'] for r in rumc_manifest['testing']]].copy()
assert prostatex._merge.eq('both').all() and prostatex.patient_id.is_unique
for entry, row in zip(rumc_manifest['testing'], prostatex.itertuples(index=False)):
    assert (DATA_ROOT / entry['label']).resolve() == (DATA_ROOT / 'github/prostatex_masks' / row.label_wg).resolve(), row.ID
    assert (DATA_ROOT / entry['image']).resolve() == (DATA_ROOT / 'github/prostatex_masks' / row.t2).resolve(), row.ID
prostatex['reference_path'] = [str(DATA_ROOT / r['label']) for r in rumc_manifest['testing']]
prostatex['image_path'] = [str(DATA_ROOT / r['image']) for r in rumc_manifest['testing']]
prostatex['clinical_volume_ml'] = pd.to_numeric(prostatex.prostate_volume, errors='coerce')
source_cases = {'prostatex': prostatex.reset_index(drop=True)}
for site in SITES:
    entries = pd.DataFrame(picai_manifest[f'testing_{site}']).rename(columns={'id': 'ID'})
    assert entries.ID.is_unique and len(entries), site
    entries[['patient_id', 'study_id']] = entries.ID.str.extract(r'^picai_(\d+)_(\d+)$').astype('int64')
    cases = entries.merge(marksheet, on=['patient_id', 'study_id'], how='left', validate='one_to_one', indicator=True)
    assert cases._merge.eq('both').all() and cases.center.eq(site).all() and cases.patient_id.is_unique, site
    for manifest in [rumc_manifest, pooled_manifest]:
        training = {int(r['id'].split('_')[1]) for r in manifest['training'] if r['id'].startswith('picai_')}
        assert not set(cases.patient_id) & training, (site, 'Training patient overlap')
    cases['reference_path'] = cases.label.map(lambda p: str(DATA_ROOT / p))
    cases['image_path'] = cases.image.map(lambda p: str(DATA_ROOT / p))
    for row in cases.itertuples(index=False):
        assert Path(row.reference_path).parent == PICAI_ROOT / 'anatomical_delineations/whole_gland/AI/Bosma22b', row.ID
        assert Path(row.reference_path).name == f'{row.patient_id}_{row.study_id}.nii.gz', row.ID
    cases['clinical_volume_ml'] = pd.to_numeric(cases.prostate_volume, errors='coerce')
    cases.to_csv(OUTPUT_DIR / f'picai_{site}_linkage_audit.csv', index=False)
    source_cases[f'picai_{site}'] = cases
source_cases['toronto'] = toronto_cases

source_flow = []
for stage, frame in [
    ('Mapping examinations', links),
    ('Local mask index', linked),
    ('Unambiguous mapping and clinical match', linked.loc[linked._merge.eq('both')]),
    ('Valid clinical volume', linked.loc[linked._merge.eq('both') &
        pd.to_numeric(linked.prostate_volume, errors='coerce').gt(0) &
        np.isfinite(pd.to_numeric(linked.prostate_volume, errors='coerce'))]),
    ('Reviewed testing manifest', prostatex),
]:
    source_flow.append(dict(cohort='prostatex', stage=stage, n_examinations=len(frame),
                            n_patients=frame.patient_id.nunique()))
for dataset, manifest in [('picai', picai_manifest), ('multicenter', pooled_manifest)]:
    for fold, rows in manifest.items():
        source_flow.append(dict(cohort=dataset, stage=fold, n_examinations=len(rows),
            n_patients=len({r['id'].split('_')[1] for r in rows}) if dataset == 'picai' else pd.NA))
pd.DataFrame(source_flow).to_csv(OUTPUT_DIR / 'source_cohort_flow.csv', index=False)

# %% [markdown]
# ## Saved prediction provenance and native-grid measurements
#
# Volumes use foreground voxel count × |det(affine)| / 1,000 mL. DSC uses
# matching grids (maximum corner displacement 0.5 mm), with no resampling.
# Unknown Toronto units require an independently marked mm grid within 0.01 mm.
# Empty predictions are retained. Manual Toronto has scalar references, no DSC.
# Historical Toronto Dataset274 settings are unavailable and explicitly unverified;
# all documented inference runs must match the five-fold study configuration.

# %%
# These helpers centralize spatial validation across every source cohort.
def record_input(path):
    path = Path(path)
    key = str(path.relative_to(REPO_ROOT))
    if key not in input_sha256:
        with path.open('rb') as stream:
            input_sha256[key] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return path


def validate_model(root, method, historical=False):
    root = Path(root)
    key = str(root.relative_to(REPO_ROOT))
    if key in model_metadata:
        return
    bundle = REPO_ROOT / 'bundles' / MODEL_NAMES[method]
    dataset = read_input(root / 'dataset.json', 'json')
    plans = read_input(root / 'plans.json', 'json')
    bundle_dataset = read_input(bundle / 'models/dataset.json', 'json')
    assert dataset['labels'] == bundle_dataset['labels'] == {'background': 0, 'class1': 1}, root
    assert plans['dataset_name'] == MODEL_NAMES[method].removesuffix('_3d_fullres_5fold'), root
    metadata = dict(dataset=dataset, plans=plans, inference_verified=False)
    config_path = root / 'inference_config.json'
    if config_path.is_file():
        config = read_input(config_path, 'json')
        args = config['arguments']
        assert config['folds'] == [0, 1, 2, 3, 4], root
        assert args['tile_step_size'] == .8 and args['use_mirroring'] is False, root
        assert args['use_gaussian'] is True and args['checkpoint_name'] == 'checkpoint_final.pt', root
        assert Path(config['bundle_root']).resolve() == bundle.resolve(), root
        metadata.update(inference_verified=True, inference_config=config)
    else:
        assert historical, f'Missing inference configuration: {root}'
        metadata['inference_limitation'] = 'Historical Toronto Dataset274 settings unavailable; ensemble/settings unverified'
    raw_args = root / 'predict_from_raw_data_args.json'
    if raw_args.is_file():
        metadata['raw_prediction_arguments'] = read_input(raw_args, 'json')
    post_path = root / 'manual_postprocessing.json'
    if not post_path.is_file():
        post_path = root.parent / 'manual_postprocessing.json'
    if post_path.is_file():
        post = read_input(post_path, 'json')
        assert post['status'] == 'complete' and post['processed_files'] == post['expected_files'], post_path
        assert post['kwargs']['labels_or_regions'] == [1], post_path
        audit_path = record_input(post_path.parent / post['audit'])
        assert input_sha256[str(audit_path.relative_to(REPO_ROOT))] == post['audit_sha256'], post_path
        post_audit = pd.read_csv(audit_path)
        assert post_audit.file.is_unique and len(post_audit) == post['processed_files'], post_path
        for row in post_audit.itertuples(index=False):
            postprocessed_sha256[str((post_path.parent / row.file).relative_to(REPO_ROOT))] = row.after_sha256
        metadata['manual_postprocessing'] = post
    if method == 'rumcnotprostatexnoclinical236':
        assert 'manual_postprocessing' in metadata, f'Missing Dataset236 postprocessing record: {root}'
    model_metadata[key] = metadata


def grid_displacement(first, second):
    assert first.shape == second.shape, 'Grid shapes differ'
    corners = np.array(list(product(*[(0, n-1) for n in first.shape])))
    return float(np.linalg.norm(nib.affines.apply_affine(first.affine, corners)
                               - nib.affines.apply_affine(second.affine, corners), axis=1).max())


def measure_source(case, group, prediction_roots, scalar_reference=False):
    audit_rows = []
    paths = {'image': Path(case.image_path), **{m: r / f'{case.ID}.nii.gz' for m, r in prediction_roots.items()}}
    if not scalar_reference:
        paths['gt'] = Path(case.reference_path)
    images, masks, result = {}, {}, {'ID': case.ID}
    for method, path in paths.items():
        record_input(path)
        relative = str(path.relative_to(REPO_ROOT))
        if relative in postprocessed_sha256:
            assert input_sha256[relative] == postprocessed_sha256[relative], (case.ID, method, 'Postprocessed mask hash mismatch')
        img = nib.load(path)
        assert len(img.shape) == 3 and np.isfinite(img.affine).all(), (case.ID, method, 'Invalid 3D affine')
        determinant = abs(float(np.linalg.det(img.affine[:3, :3])))
        assert np.isfinite(determinant) and determinant > 0, (case.ID, method, 'Invalid voxel volume')
        images[method] = img
    for method, img in images.items():
        assert img.shape == images['image'].shape, (case.ID, method, 'Image/mask grid shape mismatch')
        displacement = grid_displacement(img, images['image'])
        assert displacement <= GEOMETRY_TOLERANCE_MM, (case.ID, method, 'Image/mask displacement', displacement)
        units = img.header.get_xyzt_units()[0]
        anchor_path, anchor_displacement = '', 0.
        if units == 'unknown' and group == 'toronto':
            anchors = [(m, other) for m, other in images.items() if other.header.get_xyzt_units()[0] == 'mm'
                       and grid_displacement(img, other) <= UNIT_ANCHOR_TOLERANCE_MM]
            assert anchors, (case.ID, method, 'Unknown units lack a matching known-mm anchor')
            anchor_method, anchor = anchors[0]
            anchor_path = str(paths[anchor_method].relative_to(REPO_ROOT))
            anchor_displacement = grid_displacement(img, anchor)
        else:
            assert units == 'mm', (case.ID, method, 'Spatial units must be mm')
        if method == 'image':
            continue
        foreground = [1, 2, 3] if method == 'gt' and group == 'toronto' else [1]
        data = np.asanyarray(img.dataobj)
        assert np.isin(data, [0, *foreground]).all(), (case.ID, method, 'Unexpected labels')
        mask = np.isin(data, foreground)
        n = int(mask.sum())
        if method == 'gt':
            assert n > 0, (case.ID, 'Empty reference mask')
        volume = n * abs(float(np.linalg.det(img.affine[:3, :3]))) / 1000
        masks[method] = mask
        result[f'{method}_volume_ml'] = volume
        result[f'{method}_prediction_path'] = str(paths[method].relative_to(REPO_ROOT))
        result[f'{method}_unit_anchor_path'] = anchor_path
        result[f'{method}_unit_anchor_max_displacement_mm'] = anchor_displacement
        audit_rows.append(dict(cohort=group, ID=case.ID, method=method, path=str(paths[method].relative_to(REPO_ROOT)),
            foreground_labels=str(foreground), voxel_count=n, volume_ml=volume, spatial_units=units,
            unit_anchor_path=anchor_path, unit_anchor_max_displacement_mm=anchor_displacement,
            image_max_corner_displacement_mm=displacement))
    if scalar_reference:
        result['gt_volume_ml'] = float(case.manual_reference_ml)
        assert np.isfinite(result['gt_volume_ml']) and result['gt_volume_ml'] > 0, (case.ID, 'Invalid scalar reference')
    for method in prediction_roots:
        if scalar_reference:
            result[f'{method}_dsc'] = np.nan
        else:
            displacement = grid_displacement(images[method], images['gt'])
            assert displacement <= GEOMETRY_TOLERANCE_MM, (case.ID, method, 'Reference geometry mismatch', displacement)
            result[f'{method}_dsc'] = 2 * int((masks[method] & masks['gt']).sum()) / (int(masks[method].sum()) + int(masks['gt'].sum()))
    return result, audit_rows


measured = {}
for group, cases in source_cases.items():
    assert np.isfinite(cases.clinical_volume_ml).all() and cases.clinical_volume_ml.gt(0).all(), (group, 'Invalid clinical volume')
    roots = {method: (TORONTO_ROOT / f'seg_out_{name}' if group == 'toronto' else
             STUDY_ROOT / name / ('prostatex_original_train_mapped' if group == 'prostatex' else 'picai_test_clinical'))
             for method, name in MODEL_NAMES.items()}
    for method, root in roots.items():
        validate_model(root, method, historical=(group == 'toronto' and method == 'nnunet274'))
    records = []
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
        results = executor.map(
            lambda case: measure_source(case, group, roots, scalar_reference=(group == 'toronto' and case.has_manual_reference)),
            cases.itertuples(index=False))
        for i, (result, audit_rows) in enumerate(results, 1):
            records.append(result)
            measurement_rows.extend(audit_rows)
            if i % 100 == 0 or i == len(cases):
                print(f'Measured {group}: {i}/{len(cases)}', flush=True)
    measured[group] = cases.merge(pd.DataFrame(records), on='ID', validate='one_to_one')

# Multicenter testing and ERC cohorts have masks and no clinical comparator.
multicenter_metrics = []
for split, folder, label in [('testing', 'pooled2_test', 'Multicenter Data (external coil)'),
                             ('endorectal_coil', 'pooled2_endorectal_coil', 'Multicenter Data (ERC)')]:
    entries = pd.DataFrame(pooled_manifest[split]).rename(columns={'id': 'ID'})
    assert entries.ID.is_unique and len(entries), split
    entries['image_path'] = entries.image.map(lambda p: str(DATA_ROOT / p))
    entries['reference_path'] = entries.label.map(lambda p: str(DATA_ROOT / p))
    roots = {m: STUDY_ROOT / name / folder for m, name in MODEL_NAMES.items()}
    for method, root in roots.items():
        validate_model(root, method)
    for case in entries.itertuples(index=False):
        result, audit_rows = measure_source(case, split, roots)
        measurement_rows.extend(audit_rows)
        for method in roots:
            multicenter_metrics.append(dict(cohort=label, ID=case.ID, method=method,
                gt_volume_ml=result['gt_volume_ml'], predicted_volume_ml=result[f'{method}_volume_ml'], dsc=result[f'{method}_dsc']))
multicenter_metrics = pd.DataFrame(multicenter_metrics)
multicenter_metrics.to_csv(OUTPUT_DIR / 'multicenter_metrics_per_case.csv', index=False)
pd.DataFrame(measurement_rows).to_csv(OUTPUT_DIR / 'mask_measurement_audit.csv', index=False)
for group, cases in measured.items():
    cases.to_csv(OUTPUT_DIR / f'{group}_measured_volumes.csv', index=False)

# %% [markdown]
# ## Toronto clinical workbook linkage
#
# The original workbook is read directly. If encrypted, enter its password at
# the terminal/notebook prompt. The password and decrypted bytes are never saved.
# Reader-zero handling, decimal commas, censored PSA, and the documented PSA
# correction follow the source definitions; no precomputed clinical audit is used.

# %%
from getpass import getpass
from io import BytesIO
import msoffcrypto

workbook_path = record_input(TORONTO_ROOT / 'PROSATELLMVOL20260703-12-WorkingBillShi.xlsx')
with workbook_path.open('rb') as stream:
    office = msoffcrypto.OfficeFile(stream)
    try:
        if office.is_encrypted():
            password = getpass('Toronto Excel password: ')
            try:
                office.load_key(password=password, verify_password=True)
            finally:
                del password
            with BytesIO() as decrypted:
                office.decrypt(decrypted)
                decrypted.seek(0)
                workbook = pd.read_excel(decrypted, engine='openpyxl', dtype=object, keep_default_na=False)
        else:
            workbook = pd.read_excel(workbook_path, engine='openpyxl', dtype=object, keep_default_na=False)
    finally:
        del office
READER_FIELDS = ['p_axt2_ch_vol', 'p_axt2_ah_vol', 'p_axt2_dd_vol', 'p_axt2_ao_vol',
                 'p_axt2_chepg_vol', 'p_axt2_epgch_vol', 'p_axt2_epg_vol']
fields = ['aano', 'amrn', 'psa_report_ref', 'psa', 'reportmaxpirads_ref', 'CSCANCER', *READER_FIELDS]
assert workbook.columns.is_unique and set(fields) <= set(workbook), 'Workbook lacks required clinical fields'
clinical = workbook[fields].copy()
clinical['source_excel_row'] = np.arange(2, len(clinical)+2)
clinical['aano'] = clinical.aano.astype('string').str.strip()
clinical['workbook_amrn'] = clinical.pop('amrn').astype('string').str.strip().str.removeprefix('AIPR_')
assert clinical.aano.notna().all() and clinical.aano.ne('').all() and clinical.aano.is_unique, 'Invalid workbook accession keys'
clinical = measured['toronto'].merge(clinical, on='aano', how='left', validate='one_to_one', indicator='workbook_match')
assert clinical.workbook_match.eq('both').all() and clinical.amrn.eq(clinical.workbook_amrn).all(), 'Workbook linkage mismatch'
for field in ['psa_report_ref', 'psa', 'reportmaxpirads_ref', *READER_FIELDS]:
    raw = clinical[field].astype('string')
    clinical[f'{field}_censored'] = raw.str.contains('>', regex=False, na=False)
    clinical[f'{field}_numeric'] = pd.to_numeric(raw.str.replace(',', '.', regex=False).str.replace('>', '', regex=False), errors='coerce').astype(float)
clinical['psa_correction_applied'] = clinical.aano.eq('333abfe652a8889')
alternative = clinical.psa_numeric.mask(clinical.psa_correction_applied, 28.9)
reported = clinical.psa_report_ref_numeric
valid_report = np.isfinite(reported) & reported.gt(0)
valid_alternative = np.isfinite(alternative) & alternative.gt(0)
clinical['psa_ng_ml'] = reported.where(valid_report, alternative.where(valid_alternative))
clinical['psa_source'] = np.select([valid_report, valid_alternative], ['psa_report_ref', 'psa'], default='unavailable')
clinical['psa_selected_censored'] = np.where(valid_report, clinical.psa_report_ref_censored,
    clinical.psa_censored & ~clinical.psa_correction_applied & valid_alternative)
pirads = clinical.reportmaxpirads_ref_numeric
clinical['pirads_3'] = (np.isfinite(pirads) & pirads.eq(3))
status = clinical.CSCANCER.astype('string').str.strip().str.upper()
missing = status.isin(['', 'NA', 'N/A', '#N/A', '#NA', 'NAN', 'NONE', 'NULL', '<NA>', 'UNKNOWN',
                     'NOT AVAILABLE', 'NOT KNOWN', 'NOT BIOPSIED', 'NO BIOPSY'])
initial = status.mask(missing).str[0]
clinical['outcome'] = initial.where(initial.isin(['Y', 'N']))
clinical['outcome_known'] = clinical.outcome.notna()
clinical['outcome_definition'] = 'CSCANCER (Y/N)'
readers = clinical[[f'{f}_numeric' for f in READER_FIELDS]].copy()
readers.loc[readers.p_axt2_ch_vol_numeric.eq(0), 'p_axt2_ch_vol_numeric'] = np.nan
clinical['invalid_reader_volume'] = (readers.notna() & (~np.isfinite(readers) | readers.le(0))).any(axis=1)
clinical['workbook_reference_volume_ml'] = readers.median(axis=1)
assert not clinical.invalid_reader_volume.any(), 'Invalid reader volume'
assert np.isclose(clinical.workbook_reference_volume_ml, clinical.manual_reference_ml, equal_nan=True, rtol=1e-9, atol=1e-9).all(), 'Workbook/listing manual reference mismatch'
clinical.to_csv(OUTPUT_DIR / 'toronto_clinical_linkage_audit.csv', index=False)

# %% [markdown]
# ## Independent analysis cohorts and secondary endpoints
#
# Remove ProstateX patients from PI-CAI RUMC. Toronto manual and Additional are
# already disjoint by source reference availability. No endpoint-specific missing
# data removes patients from volume analysis. Namespace unrelated patient IDs.

# %%
cohorts = {key: measured[key].copy() for key in COHORTS if key in measured}
cohorts['toronto_manual'] = clinical.loc[clinical.has_manual_reference].copy()
cohorts['toronto_surrogate'] = clinical.loc[~clinical.has_manual_reference].copy()
shared_patients = {site: len(set(cohorts[f'picai_{site}'].patient_id) & set(cohorts['prostatex'].patient_id)) for site in SITES}
membership_tables, provenance = [], []
for key in COHORTS:
    v = cohorts[key]
    patient_column = 'amrn' if key.startswith('toronto') else 'patient_id'
    removed = v.patient_id.isin(cohorts['prostatex'].patient_id) if key == 'picai_RUMC' else pd.Series(False, index=v.index)
    membership_tables.append(pd.DataFrame({'cohort': key, 'ID': v.ID, 'patient_identifier': v[patient_column],
        'selected': ~removed, 'selection_rule': 'Remove ProstateX patients' if key == 'picai_RUMC' else 'Source selection'}))
    v = v.loc[~removed].set_index('ID', drop=False).copy()
    assert len(v) and v.index.is_unique and v[patient_column].notna().all() and v[patient_column].is_unique, key
    for method in ['gt', *METHODS]:
        values = v[f'{method}_volume_ml']
        assert np.isfinite(values).all() and values.ge(0).all(), (key, method)
    assert v.gt_volume_ml.gt(0).all() and v.clinical_volume_ml.gt(0).all(), key
    if not key.startswith('toronto'):
        v['psa_ng_ml'] = pd.to_numeric(v.psa, errors='coerce')
        maximum_pirads = []
        for case in v.itertuples(index=False):
            scores = []
            for token in ([] if pd.isna(case.lesion_PIRADS) else str(case.lesion_PIRADS).split(',')):
                if token.strip().upper() in ('', 'N/A'):
                    continue
                score = float(token)
                assert score in [1, 2, 3, 4, 5], (case.ID, 'Invalid PI-RADS token', token)
                scores.append(score)
            maximum_pirads.append(max(scores) if scores else np.nan)
        v['pirads_3'] = np.array(maximum_pirads) == 3
        outcome = pd.to_numeric(v.case_ISUP, errors='coerce')
        v['outcome_known'] = np.isfinite(outcome) & outcome.isin([0, 1, 2, 3, 4, 5])
        v['outcome'] = outcome.where(v.outcome_known)
        v['outcome_definition'] = 'ISUP grade group (csPCa ≥2)'
    for method in ['gt', 'clinical', 'nnunet274']:
        v[f'{method}_psad'] = v.psa_ng_ml.where(np.isfinite(v.psa_ng_ml) & v.psa_ng_ml.gt(0)) / v[f'{method}_volume_ml'].where(v[f'{method}_volume_ml'].gt(0))
    v['psad_evaluable'] = v.psa_ng_ml.gt(0) & np.isfinite(v[['gt_psad', 'clinical_psad', 'nnunet274_psad']]).all(axis=1)
    v.loc[~v.outcome_known, 'outcome'] = pd.NA
    cohorts[key] = v
    _, name, _, reference = COHORTS[key]
    COHORTS[key] = ('', name, len(v), reference)
    provenance.append(dict(cohort=key, name=name, n=len(v), reference=reference,
        cohort_sha256=hashlib.sha256('\n'.join(sorted(v.index)).encode()).hexdigest(), removed_n=int(removed.sum())))
cohorts = {key: cohorts[key] for key in COHORTS}
membership = pd.concat(membership_tables, ignore_index=True)
membership.to_csv(OUTPUT_DIR / 'analysis_membership.csv', index=False)
membership.loc[membership.selected].to_csv(OUTPUT_DIR / 'selected_membership.csv', index=False)
membership.loc[~membership.selected].to_csv(OUTPUT_DIR / 'removed_membership.csv', index=False)
patient_sets = {key: {(('Toronto' if key.startswith('toronto') else 'Dutch'), str(int(value)))
                     for value in v['amrn' if key.startswith('toronto') else 'patient_id']}
                for key, v in cohorts.items()}
for first, second in combinations(patient_sets, 2):
    assert patient_sets[first].isdisjoint(patient_sets[second]), (first, second, 'Shared analysis patients')
assert set(cohorts['toronto_manual'].index) | set(cohorts['toronto_surrogate'].index) == set(toronto_cases.ID)
assert not set(clinical.aano) & set(reasons.index)
print('Disjoint analysis cohorts:', {k: len(v) for k, v in cohorts.items()}, flush=True)
for key, v in cohorts.items():
    v.to_csv(OUTPUT_DIR / f'{key}_comparator_volumes.csv', index=False)
toronto_seg = pd.DataFrame([dict(ID=case.ID, method=method, dsc=getattr(case, f'{method}_dsc'))
    for case in cohorts['toronto_surrogate'].itertuples(index=False) for method in MODEL_NAMES])
toronto_seg.to_csv(OUTPUT_DIR / 'toronto_additional_segmentation_metrics_per_case.csv', index=False)
toronto_seg.groupby('method').agg(n=('ID', 'size'), mean_dsc=('dsc', 'mean'),
    zero_overlap_cases=('dsc', lambda x: int(x.eq(0).sum()))).to_csv(OUTPUT_DIR / 'toronto_additional_segmentation_summary.csv')

# %% [markdown]
# ## Tables 1–3: testing performance
#
# Volumes are in mL; percentage errors divide by reference volume. Toronto
# manual DSC is unavailable. Toronto Additional and PI-CAI use surrogate masks.

# %%
performance_rows = []
for (label, method), selected in multicenter_metrics.groupby(['cohort', 'method'], sort=False):
    error = selected.predicted_volume_ml - selected.gt_volume_ml
    percent = 100 * error / selected.gt_volume_ml
    performance_rows.append(dict(cohort=label, n=len(selected), method=method, dsc=selected.dsc.mean(),
        mae=error.abs().mean(), rmse=np.sqrt(np.mean(error**2)), mape=percent.abs().mean(), rmspe=np.sqrt(np.mean(percent**2))))
for key, v in cohorts.items():
    _, name, n, reference = COHORTS[key]
    for method in METHODS:
        error = v[f'{method}_volume_ml'] - v.gt_volume_ml
        percent = 100 * error / v.gt_volume_ml
        performance_rows.append(dict(cohort=name, n=n, method=method,
            dsc=v[f'{method}_dsc'].mean() if method != 'clinical' else np.nan,
            mae=error.abs().mean(), rmse=np.sqrt(np.mean(error**2)), mape=percent.abs().mean(), rmspe=np.sqrt(np.mean(percent**2))))
performance = pd.DataFrame(performance_rows)
performance.to_csv(OUTPUT_DIR / 'performance.csv', index=False)
tables = {}
for number, fields in [(1, ['dsc']), (2, ['mae', 'rmse']), (3, ['mape', 'rmspe'])]:
    rows = []
    for (name, n), group in performance.groupby(['cohort', 'n'], sort=False):
        row = {'Dataset': f'{name} (n = {n:,})'}
        for method, title in METHODS.items():
            part = group.loc[group.method.eq(method)]
            values = part.iloc[0][fields] if len(part) else pd.Series([np.nan])
            row[title] = '; '.join(f'{x:.3f}' if number == 1 else f'{x:.2f}' for x in values) if values.notna().all() else '—'
        rows.append(row)
    tables[number] = pd.DataFrame(rows)
    if number in (2, 3):
        tables[number] = tables[number][['Dataset', 'Clinical ellipsoid', 'Multicenter', 'Single-center']]

def outside_both_common(difference, relative, limits):
    absolute_outside = (np.asarray(difference) < limits['ml'][0]) | (np.asarray(difference) > limits['ml'][1])
    relative_outside = (np.asarray(relative) < limits['percent'][0]) | (np.asarray(relative) > limits['percent'][1])
    return absolute_outside & relative_outside


# Exact rounded ProstateX RUMC limits define both analysis and highlighting.
COMMON_LIMITS = {'ml': (-14.28, 25.74), 'percent': (-22.70, 39.60)}

# %% [markdown]
# ## Tables 4–6: disagreement and reference accuracy
#
# D = manual − automated; Q = 100D / pair mean. Use the exact common
# ProstateX limits for all cohorts; retain local mean ±1.96 SD for description.
# The primary subset exceeds BOTH intervals, irrespective of tail. Compare
# absolute reference error there, then compare ordinal PSAD category distance.
# Positive wins mean automated is closer; ties are exact equality. Two-sided
# exact binomial tests exclude ties and test 50% wins. Holm corrections span
# six cohorts separately for volume and PSAD; no error-metric tests are added.

# %%
threshold_rows, volume_rows, psad_rows, case_tables, biopsy_tables = [], [], [], [], []
for key, v in cohorts.items():
    _, name, n, reference = COHORTS[key]
    v['disagreement_ml'] = v.clinical_volume_ml - v.nnunet274_volume_ml
    v['disagreement_percent'] = 100 * v.disagreement_ml / ((v.clinical_volume_ml + v.nnunet274_volume_ml) / 2)
    threshold = dict(cohort=key, name=name, n=n, reference=reference)
    for scale, flag in [('ml', 'absolute'), ('percent', 'relative')]:
        x = v[f'disagreement_{scale}']
        lower, upper = x.mean() - 1.96*x.std(ddof=1), x.mean() + 1.96*x.std(ddof=1)
        common_lower, common_upper = COMMON_LIMITS[scale]
        v[f'flag_{flag}'] = (x < common_lower) | (x > common_upper)
        threshold.update({f'mean_{scale}': x.mean(), f'lower_{scale}': lower, f'upper_{scale}': upper, f'common_lower_{scale}': common_lower, f'common_upper_{scale}': common_upper, f'n_{flag}': int(v[f'flag_{flag}'].sum())})
    v['flag_both'] = outside_both_common(v.disagreement_ml, v.disagreement_percent, COMMON_LIMITS)
    assert np.array_equal(v.flag_both, v.flag_absolute & v.flag_relative)
    threshold['n_both'] = int(v.flag_both.sum())
    threshold_rows.append(threshold)
    flagged = v.loc[v.flag_both].copy()
    manual_error = np.abs(flagged.clinical_volume_ml - flagged.gt_volume_ml)
    auto_error = np.abs(flagged.nnunet274_volume_ml - flagged.gt_volume_ml)
    wins, losses, ties = int((auto_error < manual_error).sum()), int((auto_error > manual_error).sum()), int((auto_error == manual_error).sum())
    row = dict(cohort=key, name=name, cohort_n=n, n=len(flagged), wins=wins, losses=losses, ties=ties, p_raw=exact_closer_p(wins, losses))
    for method, error in [('manual', manual_error), ('automated', auto_error)]:
        pct = 100 * error / flagged.gt_volume_ml
        row.update({f'{method}_mae': error.mean(), f'{method}_rmse': np.sqrt(np.mean(error**2)), f'{method}_mape': pct.mean(), f'{method}_rmspe': np.sqrt(np.mean(pct**2))})
    for metric in ['mae', 'rmse', 'mape', 'rmspe']:
        row[f'change_{metric}'] = row[f'automated_{metric}'] - row[f'manual_{metric}']
    volume_rows.append(row)
    eligible = flagged.loc[flagged.psad_evaluable].copy()
    for method in ['gt', 'clinical', 'nnunet274']:
        eligible[f'{method}_category'] = np.searchsorted(CUTOFFS, eligible[f'{method}_psad'], side='right')
    manual_distance = (eligible.clinical_category - eligible.gt_category).abs()
    auto_distance = (eligible.nnunet274_category - eligible.gt_category).abs()
    wins, losses, ties = int((auto_distance < manual_distance).sum()), int((auto_distance > manual_distance).sum()), int((auto_distance == manual_distance).sum())
    psad_rows.append(dict(cohort=key, name=name, n_flagged=len(flagged), n=len(eligible), excluded_psad=len(flagged)-len(eligible), wins=wins, losses=losses, ties=ties, p_raw=exact_closer_p(wins, losses)))
    # Include every PI-RADS 3 intersection case even if PSA/outcome is missing.
    biopsy = flagged.loc[flagged.pirads_3].copy()
    for method in ['gt', 'clinical', 'nnunet274']:
        value = biopsy[f'{method}_psad']
        biopsy[f'{method}_recommend'] = value.ge(.15).astype('boolean').where(np.isfinite(value), pd.NA)
    biopsy_tables.append(biopsy.assign(cohort=key, name=name, reference=reference))
    case_tables.append(v.assign(cohort=key, reference=reference))
thresholds = pd.DataFrame(threshold_rows)
volume_summary, psad_summary = pd.DataFrame(volume_rows), pd.DataFrame(psad_rows)
adjust_six(volume_summary)
adjust_six(psad_summary)
all_cases = pd.concat(case_tables, ignore_index=True)
biopsy_cases = pd.concat(biopsy_tables, ignore_index=True)
# Table 6 sorts globally by outcome, then reference/manual/automated decisions.
# ISUP grades within each group share a rank. Cohort and ID break ties.
toronto = biopsy_cases.cohort.str.startswith('toronto')
isup = pd.to_numeric(biopsy_cases.outcome, errors='coerce')
outcome_order = np.select([
    biopsy_cases.outcome_known & ~toronto & isup.ge(2),
    biopsy_cases.outcome_known & toronto & biopsy_cases.outcome.eq('Y'),
    biopsy_cases.outcome_known & ~toronto & isup.isin([0, 1]),
    biopsy_cases.outcome_known & toronto & biopsy_cases.outcome.eq('N'),
], [0, 1, 2, 3], default=4)
sort_columns = ['outcome_order', 'gt_order', 'clinical_order', 'nnunet274_order', 'cohort_order', 'ID']
biopsy_cases = biopsy_cases.assign(
    outcome_order=outcome_order,
    cohort_order=biopsy_cases.cohort.map({key: i for i, key in enumerate(COHORTS)}),
    **{f'{method}_order': biopsy_cases[f'{method}_recommend'].map({True: 0, False: 1}).fillna(2)
       for method in ['gt', 'clinical', 'nnunet274']},
).sort_values(sort_columns).drop(columns=sort_columns[:-1]).reset_index(drop=True)

# Save compact case audits without copying private source paths/workbook columns.
case_columns = ['cohort', 'ID', 'reference', 'gt_volume_ml', 'clinical_volume_ml', 'nnunet274_volume_ml', 'disagreement_ml', 'disagreement_percent', 'flag_absolute', 'flag_relative', 'flag_both', 'psa_ng_ml', 'gt_psad', 'clinical_psad', 'nnunet274_psad', 'psad_evaluable', 'pirads_3', 'outcome_known', 'outcome', 'outcome_definition']
all_cases['improvement_ml'] = ((all_cases.clinical_volume_ml - all_cases.gt_volume_ml).abs()
                               - (all_cases.nnunet274_volume_ml - all_cases.gt_volume_ml).abs())
for key in COHORTS:
    all_cases.loc[all_cases.cohort.eq(key) & all_cases.flag_both, case_columns + ['improvement_ml']].sort_values(
        ['improvement_ml', 'ID']).to_csv(OUTPUT_DIR / f'{key}_disagreement_cases.csv', index=False)
all_cases[case_columns].to_csv(OUTPUT_DIR / 'case_audit.csv', index=False)
biopsy_cases[case_columns + ['gt_recommend', 'clinical_recommend', 'nnunet274_recommend']].to_csv(OUTPUT_DIR / 'pirads3_outliers.csv', index=False)
for stem, frame in [('disagreement_thresholds', thresholds), ('volume_closeness', volume_summary), ('psad_closeness', psad_summary)]:
    frame.to_csv(OUTPUT_DIR / f'{stem}.csv', index=False)

rows = []
for r in volume_summary.to_dict('records'):
    row = {'Dataset': r['name'], 'Flagged n/N (%)': f"{r['n']}/{r['cohort_n']:,} ({100*r['n']/r['cohort_n']:.1f}%)"}
    for metric in ['mae', 'rmse', 'mape', 'rmspe']:
        row[metric.upper() + (' (mL)' if metric in ['mae', 'rmse'] else ' (%)')] = f"{r[f'manual_{metric}']:.2f} → {r[f'automated_{metric}']:.2f}"
    row['Improved / all'] = f"{r['wins']}/{r['n']}"
    if volume_summary.ties.gt(0).any():
        row['Ties'] = r['ties']
    row['Adjusted p'] = fmt_p(r['p_holm'])
    rows.append(row)
tables[4] = pd.DataFrame(rows)
tables[5] = pd.DataFrame([{'Dataset': f'{r.name} (n = {r.n}' + (', another missing PSA value' if r.excluded_psad == 1 else f', {r.excluded_psad} missing PSA values' if r.excluded_psad else '') + ')', 'Automated closer': r.wins, 'Equal distance': r.ties, 'Manual closer': r.losses, 'Adjusted p': fmt_p(r.p_holm)} for r in psad_summary.itertuples()])
rows = []
for r in biopsy_cases.to_dict('records'):
    row = {'Dataset': r['name'], 'Case': r['ID']}
    for method, title in [('gt', 'Reference'), ('clinical', 'Manual'), ('nnunet274', 'Automated')]:
        value, recommend = r[f'{method}_psad'], r[f'{method}_recommend']
        row[f'{title}: mL; PSAD; biopsy'] = f"{r[f'{method}_volume_ml']:.2f}; " + (f"{value:.3f}; {'Recommend' if recommend else 'Avoid'}" if pd.notna(value) and pd.notna(recommend) else 'Missing; Unknown')
    row['Recorded outcome'] = ('Unknown' if not r['outcome_known'] else (f"CSCANCER {r['outcome']}" if r['cohort'].startswith('toronto') else f"ISUP {int(r['outcome'])}"))
    rows.append(row)
tables[6] = pd.DataFrame(rows, columns=['Dataset', 'Case', 'Reference: mL; PSAD; biopsy', 'Manual: mL; PSAD; biopsy', 'Automated: mL; PSAD; biopsy', 'Recorded outcome'])
for number, frame in tables.items():
    frame.to_csv(OUTPUT_DIR / f'table_{number}.csv', index=False)
    (OUTPUT_DIR / f'table_{number}.md').write_text(markdown(frame) + '\n')
    print(f'\nTable {number}\n{markdown(frame)}')

# %% [markdown]
# ## Manual-reference volume agreement
#
# Preserve the original 9.5 × 7.5 inch layout. Screening uses the common
# pair-mean criteria, whereas plotted errors use the manual reference.
# Fixed percentage ranges affect display only; audit final limits explicitly.

# %%
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter, FuncFormatter

agreement_rows, agreement_panels, agreement_hidden, agreement_members = [], [], [], []
agreement_settings = {}
method_order = ['clinical', 'nnunet274', 'rumcnotprostatexnoclinical236']
markers = ['o', 's', '^']
line_styles = ['-', '--', ':']
method_colors = ['#4C6A85', '#B88254', '#91A78C']
for key, percentage_limit in [('prostatex', 90), ('toronto_manual', 100)]:
    v = cohorts[key]
    reference = v.gt_volume_ml.to_numpy()
    selected = v.flag_both.to_numpy()
    assert np.array_equal(selected, outside_both_common(v.disagreement_ml, v.disagreement_percent, COMMON_LIMITS))
    agreement_members.append(v[['ID', 'flag_both', 'disagreement_ml', 'disagreement_percent']].assign(cohort=key))
    names = ['Manual ellipsoid', 'Multicenter', 'Single-center']
    reference_name = 'Manual planimetry' if key == 'prostatex' else 'Median manual-reader planimetry'
    fig, axes = plt.subplots(3, 4, figsize=(9.5, 7.5), sharey='row', squeeze=False,
                             constrained_layout=True, gridspec_kw={'width_ratios': [1, 1, 1, .85]})
    fig.get_layout_engine().set(w_pad=.015, h_pad=.015, wspace=.02, hspace=.02)
    fig.suptitle(f'{COHORTS[key][1]} (n = {len(v)} patients)', fontsize=10)
    axes[0, -1].set_axis_off()
    all_volumes = v[['gt_volume_ml', *[f'{m}_volume_ml' for m in method_order]]].to_numpy()
    lower = min(all_volumes[all_volumes > 0].min() * .95, 10)
    upper = max(all_volumes.max() * 1.05, 300)
    differences = [[], []]
    panels = []
    for column, (method, name, marker, linestyle, method_color) in enumerate(zip(method_order, names, markers, line_styles, method_colors)):
        measured = v[f'{method}_volume_ml'].to_numpy()
        error = measured - reference
        relative = 100 * error / reference
        correlation = np.corrcoef(reference, measured)[0, 1]
        record = dict(cohort=key, method=method, n=len(v), selected_n=int(selected.sum()), pearson_r=correlation)
        for row, (values, scale) in enumerate([(measured, 'volume'), (error, 'ml'), (relative, 'percent')]):
            ax = axes[row, column]
            if row == 0:
                label = f'{name}\nPearson r = {correlation:.2f}'
                ax.plot([lower, upper], [lower, upper], 'k--', linewidth=.8)
                ax.set_yscale('log', nonpositive='mask')
                ax.set_yticks([10, 30, 100, 300], labels=['10', '30', '100', '300'])
                ax.yaxis.set_minor_formatter(NullFormatter())
                ax.set_ylim(lower, upper)
            else:
                bias, sd = values.mean(), values.std(ddof=1)
                lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
                record.update({f'bias_{scale}': bias, f'lower_{scale}': lo, f'upper_{scale}': hi})
                label = f'{name}\nMean: {bias:.1f} {"mL" if row == 1 else "%"}\n95% LoA:\n[{lo:.1f}, {hi:.1f}]'
                for level, ls in [(bias, '-'), (lo, '--'), (hi, '--')]:
                    ax.axhline(level, color='0.55', linestyle=ls, linewidth=.8)
                differences[row-1].append(values)
            ax.scatter(reference[~selected], values[~selected], s=16, color=method_color, marker=marker,
                       alpha=.6, edgecolors='none', label=label)
            ax.scatter(reference[selected], values[selected], s=28, color='#D55E00', marker=marker,
                       edgecolors='white', linewidths=.6, alpha=1, zorder=5)
            ax.set_xscale('log')
            ax.set_xticks([10, 30, 100, 300], labels=['10', '30', '100', '300'])
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlim(lower, upper)
            ax.set_xlabel(f'{reference_name} volume\n(mL; log scale)', fontsize=8, labelpad=1)
            ax.set_ylabel([f'{name}\nvolume (mL; log scale)', f'{name}\n− reference (mL)', 'Relative difference (%)'][row], fontsize=8, labelpad=1)
            ax.legend(loc='best', fontsize=7, framealpha=.85, edgecolor='0.6', handlelength=1, handletextpad=.4)
            panels.append((ax, method, scale, reference, values))
        agreement_rows.append(record)
    histogram_records = []
    for row, method_differences in enumerate(differences, start=1):
        ax = axes[row, -1]
        fd_edges = np.histogram_bin_edges(np.asarray(method_differences).ravel(), bins='fd')
        count = max(1, int(np.ceil((len(fd_edges)-1)/2)))
        edges = np.linspace(fd_edges[0], fd_edges[-1], count+1)
        current = axes[row, 0].get_ylim()
        ax.set_ylim(min(current[0], edges[0]), max(current[1], edges[-1]))
        for method, name, values, method_color, ls in zip(method_order, names, method_differences, method_colors, line_styles):
            counts, _ = np.histogram(values, bins=edges)
            assert counts.sum() == len(v), 'Histogram must include every patient'
            ax.stairs(counts, edges, orientation='horizontal', color=method_color, linestyle=ls, linewidth=1.2, label=name)
            histogram_records.append((ax, method, 'ml' if row == 1 else 'percent', values, edges, counts))
        ax.set_xlim(left=0)
        ax.locator_params(axis='x', nbins=3, integer=True)
        ax.legend(fontsize=7, loc='best', framealpha=.85)
    axes[2, 0].set_ylim(-percentage_limit, percentage_limit)
    axes[1, 0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:+g}' if value else '0'))
    axes[2, 0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:+.0f}%' if value else '0%'))
    for ax in axes.flat:
        if ax.axison:
            ax.tick_params(labelsize=8, pad=1.5, labelbottom=True, labelleft=True)
    fig.canvas.draw()
    for ax, method, scale, x, y in panels:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        bad_x = ~np.isfinite(x) | (x <= 0)
        bad_y = ~np.isfinite(y) | ((y <= 0) if ax.get_yscale() == 'log' else False)
        outside_x = (x < xlim[0]) | (x > xlim[1])
        outside_y = (y < ylim[0]) | (y > ylim[1])
        hidden = bad_x | bad_y | outside_x | outside_y
        if hidden.any():
            ax.text(.02, .03, f'Outside displayed limits: {hidden.sum()}/{len(v)}', transform=ax.transAxes,
                    fontsize=6, bbox=dict(facecolor='white', edgecolor='none', alpha=.9))
        agreement_panels.append(dict(cohort=key, method=method, scale=scale, n=len(v), hidden_n=int(hidden.sum()),
                                     x_min=xlim[0], x_max=xlim[1], y_min=ylim[0], y_max=ylim[1]))
        for i in np.flatnonzero(hidden):
            reasons = [name for name, mask in [('invalid_log_x', bad_x), ('invalid_y_or_log_y', bad_y),
                        ('outside_x', outside_x), ('outside_y', outside_y)] if mask[i]]
            agreement_hidden.append(dict(cohort=key, method=method, scale=scale, ID=v.ID.iloc[i],
                                         x=x[i], y=y[i], selected=bool(selected[i]), reason=';'.join(reasons)))
    histogram_settings = []
    for ax, method, scale, values, edges, counts in histogram_records:
        ylim = ax.get_ylim()
        hidden_n = int(((values < ylim[0]) | (values > ylim[1])).sum())
        histogram_settings.append(dict(method=method, scale=scale, edges=edges.tolist(), counts=counts.tolist(), hidden_n=hidden_n))
    for row, scale in [(1, 'ml'), (2, 'percent')]:
        clipped = [f'{names[method_order.index(r["method"])]}: {r["hidden_n"]}' for r in histogram_settings if r['scale'] == scale and r['hidden_n']]
        if clipped:
            axes[row, -1].text(.02, .03, 'Outside displayed limits:\n' + '\n'.join(clipped),
                              transform=axes[row, -1].transAxes, fontsize=6,
                              bbox=dict(facecolor='white', edgecolor='none', alpha=.9))
    stem = f'{key}_comparator_agreement'
    fig.savefig(OUTPUT_DIR / f'{stem}.png', dpi=300, bbox_inches=None)
    fig.savefig(OUTPUT_DIR / f'{stem}.pdf', bbox_inches=None)
    agreement_settings[key] = dict(page_inches=[9.5, 7.5], dpi=300, relative_ylim=[-percentage_limit, percentage_limit],
                                  histogram=histogram_settings, method_colors=dict(zip(method_order, method_colors)), ordinary_points_grayscale=False, highlight_color='#D55E00', selected_n=int(selected.sum()))
    plt.show()
    plt.close(fig)
pd.DataFrame(agreement_rows).to_csv(OUTPUT_DIR / 'manual_reference_agreement_summary.csv', index=False)
pd.DataFrame(agreement_panels).to_csv(OUTPUT_DIR / 'manual_reference_panel_limits.csv', index=False)
pd.DataFrame(agreement_hidden, columns=['cohort', 'method', 'scale', 'ID', 'x', 'y', 'selected', 'reason']).to_csv(OUTPUT_DIR / 'manual_reference_outside_limits.csv', index=False)
pd.concat(agreement_members, ignore_index=True).to_csv(OUTPUT_DIR / 'manual_reference_screening_membership.csv', index=False)

# %% [markdown]
# ## Bland–Altman agreement: multicenter model and clinical ellipsoid
#
# All observations contribute to the mean and sample-SD limits. Shared axis
# settings are display parameters only. Each panel counts observations outside
# either displayed axis once; points on the boundaries remain visible. Rerun
# this cell alone after changing BA_XLIM / BA_YLIM / BA_REL_YLIM.
# Compact portrait output uses constrained layout without a reserved caption area.

# %%
def outside_display(x, y, xlim, ylim):
    return (np.asarray(x) < xlim[0]) | (np.asarray(x) > xlim[1]) | (np.asarray(y) < ylim[0]) | (np.asarray(y) > ylim[1])

fig, axes = plt.subplots(6, 2, figsize=(8.5, 9.75), sharex=True, sharey='col', constrained_layout=True)
fig.get_layout_engine().set(w_pad=.04, h_pad=.04, wspace=.04, hspace=.04)
# Layout-aware headings stay centered over their respective columns.
for ax, heading in zip(axes[0], ['Absolute difference', 'Relative difference']):
    ax.annotate(heading, xy=(.5, 1), xycoords='axes fraction', xytext=(0, 24), textcoords='offset points',
                ha='center', va='bottom', fontsize=10, weight='bold')
axes[2, 0].set_ylabel('Clinical ellipsoid − Multicenter (mL)', fontsize=8)
axes[2, 1].set_ylabel('100 × difference / pair mean (%)', fontsize=8)
plot_rows, panel_data = [], []
for row_index, (key, v) in enumerate(cohorts.items()):
    x = (v.clinical_volume_ml + v.nnunet274_volume_ml) / 2
    difference = v.clinical_volume_ml - v.nnunet274_volume_ml
    common_both = outside_both_common(difference, 100*difference/x, COMMON_LIMITS)
    if 'flag_both' in v:
        assert np.array_equal(common_both, v.flag_both), 'Figure and analysis memberships differ'
    expected = thresholds.set_index('cohort').loc[key]
    for column, (scale, unit, y) in enumerate([('ml', 'mL', difference), ('percent', '%', 100*difference/x)]):
        ax = axes[row_index, column]
        mean, sd = y.mean(), y.std(ddof=1)
        lower, upper = mean-1.96*sd, mean+1.96*sd
        assert np.allclose([mean, lower, upper], expected[[f'mean_{scale}', f'lower_{scale}', f'upper_{scale}']].to_numpy(dtype=float))
        ax.scatter(x[~common_both], y[~common_both], s=4, alpha=.15, color='black', edgecolors='none')
        ax.axhline(mean, color='0.55', linewidth=.8, label=f'Dataset mean: {mean:.2f} {unit}')
        ax.axhline(lower, color='0.55', linestyle=':', linewidth=.7, label=f'Dataset 95% LoA: {lower:.2f} to {upper:.2f} {unit}')
        ax.axhline(upper, color='0.55', linestyle=':', linewidth=.7)
        common_lower, common_upper = COMMON_LIMITS[scale]
        ax.axhline(common_lower, color='black', linestyle='--', linewidth=.8, zorder=4,
                   label=f'Common criteria: {common_lower:.2f} to {common_upper:.2f} {unit}')
        ax.axhline(common_upper, color='black', linestyle='--', linewidth=.8, zorder=4)
        # Reuse the same membership in both panels; draw orange highlighted points last.
        ax.scatter(x[common_both], y[common_both], s=4, alpha=1., color='#D55E00', edgecolors='white', linewidths=.35, zorder=5,
                   label=f'Outside both common criteria (n = {int(common_both.sum())})')
        title = COHORTS[key][1].replace(' excluding ', '\nexcluding ')
        ax.set_title(f'{title} (n = {len(v):,})', fontsize=7, pad=3)
        assert np.isfinite(x).all() and x.gt(0).all(), 'Log x-axis requires positive pair means'
        ax.set_xscale('log')
        # Explicit strings avoid scientific notation; minor ticks stay unlabeled.
        from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
        ax.xaxis.set_major_locator(FixedLocator([10, 30, 100, 300]))
        ax.xaxis.set_major_formatter(FixedFormatter(['10', '30', '100', '300']))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(labelsize=6, length=2, pad=2, labelbottom=True)
        if row_index == len(cohorts)-1:
            ax.set_xlabel('Pair mean volume (mL; log scale)', fontsize=7)
        ax.legend(loc='best', fontsize=5.5, framealpha=.9, borderpad=.3, labelspacing=.2, handlelength=1.6)
        plot_rows.append(dict(cohort=key, name=COHORTS[key][1], scale=scale, unit=unit, n=len(v), mean=mean, lower=lower, upper=upper, common_lower=common_lower, common_upper=common_upper, common_both_n=int(common_both.sum())))
        panel_data.append((ax, x, y))
# Shared x-axis and column-specific y-axes include all data by default.
fig.canvas.draw()
for value, label in [(BA_XLIM, 'BA_XLIM'), (BA_YLIM, 'BA_YLIM'), (BA_REL_YLIM, 'BA_REL_YLIM')]:
    if value is not None:
        assert len(value) == 2 and np.isfinite(value).all() and value[0] < value[1], label
if BA_XLIM is not None:
    assert BA_XLIM[0] > 0, 'BA_XLIM bounds must be positive for log scale'
    axes[0, 0].set_xlim(*BA_XLIM)
if BA_YLIM is not None:
    axes[0, 0].set_ylim(*BA_YLIM)
if BA_REL_YLIM is not None:
    axes[0, 1].set_ylim(*BA_REL_YLIM)
for (ax, x, y), row in zip(panel_data, plot_rows):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    hidden = int(outside_display(x, y, xlim, ylim).sum())
    if hidden > 0:
        ax.text(.02, .03, f'Outside displayed limits: {hidden}/{row["n"]}', transform=ax.transAxes,
                ha='left', va='bottom', fontsize=5.5, bbox=dict(facecolor='white', edgecolor='none', alpha=.8))
    row.update(hidden_n=hidden, x_min=xlim[0], x_max=xlim[1], y_min=ylim[0], y_max=ylim[1])
plot_summary = pd.DataFrame(plot_rows)
plot_summary.to_csv(OUTPUT_DIR / 'bland_altman_summary.csv', index=False)
# Keep the exact compact page dimensions after constrained layout.
fig.savefig(OUTPUT_DIR / 'bland_altman_clinical_nnunet274.png', dpi=300, bbox_inches=None)
fig.savefig(OUTPUT_DIR / 'bland_altman_clinical_nnunet274.pdf', bbox_inches=None)
plt.show()
plt.close(fig)

# %% [markdown]
# ## Arithmetic, boundary cases, and provenance
#
# Ties contribute to displayed denominators but not binomial tests. Empty
# non-tie sets yield undefined p-values. Unknown outcomes stay unknown and
# are retained in the PI-RADS table. No resampling or random seed is needed.

# %%
assert np.array_equal(np.searchsorted(CUTOFFS, [.099, .10, .15, .20], side='right'), [0, 1, 2, 3])
assert not ((np.array([-1., 1.]) < -1) | (np.array([-1., 1.]) > 1)).any()
# x-only, y-only, both axes, and exact-boundary observations.
assert np.array_equal(outside_display([-1, .5, 2, 0, 1], [.5, 2, 2, 0, 1], (0, 1), (0, 1)), [True, True, True, False, False])
assert np.isnan(exact_closer_p(0, 0)) and exact_closer_p(1, 1) == 1
assert exact_closer_p(2, 0) == .5
# Known six-test Holm example, including one undefined/empty non-tie group.
synthetic_family = pd.DataFrame({'p_raw': [.001, .01, .04, .2, .5, np.nan]})
adjust_six(synthetic_family)
assert np.allclose(synthetic_family.p_holm, [.006, .05, .16, .6, 1., np.nan], equal_nan=True)
empty_family = pd.DataFrame({'p_raw': [np.nan] * 6})
adjust_six(empty_family)
assert empty_family.p_holm.isna().all()
for frame in [volume_summary, psad_summary]:
    assert (frame.wins + frame.losses + frame.ties).eq(frame.n).all()
    expected = multipletests(frame.p_raw.fillna(1), method='holm')[1]
    assert np.allclose(frame.p_holm, np.where(frame.p_raw.notna(), expected, np.nan), equal_nan=True)
assert len(biopsy_cases) == int((all_cases.flag_both & all_cases.pirads_3).sum())
assert biopsy_cases.loc[~biopsy_cases.outcome_known, 'outcome'].isna().all()
assert not biopsy_cases.duplicated(['cohort', 'ID']).any()
biopsy_counts = []
for key in COHORTS:
    selected = biopsy_cases.loc[biopsy_cases.cohort.eq(key)]
    changed = selected.clinical_recommend.ne(selected.nnunet274_recommend)
    biopsy_counts.append(dict(cohort=key, n=len(selected), known_outcomes=int(selected.outcome_known.sum()),
                              unknown_outcomes=int((~selected.outcome_known).sum()),
                              recommendations_changed=int(changed.sum())))
pd.DataFrame(biopsy_counts).to_csv(OUTPUT_DIR / 'pirads3_counts.csv', index=False)
assert thresholds.set_index('cohort').n_both.to_dict() == all_cases.groupby('cohort').flag_both.sum().to_dict()
assert plot_summary.groupby('cohort', sort=False).common_both_n.first().tolist() == thresholds.n_both.tolist()
metadata = dict(manual_reference_agreement=agreement_settings, screening=dict(calibration_cohort='ProstateX RUMC', common_limits=COMMON_LIMITS,
                               rule='strictly outside both absolute and relative intervals; equality retained',
                               relative_denominator='pair mean', threshold_uncertainty_in_tests=False), created_utc=datetime.now(timezone.utc).isoformat(), cohorts=provenance,
                input_sha256=input_sha256, model_metadata=model_metadata,
                geometry_tolerance_mm=GEOMETRY_TOLERANCE_MM, unknown_unit_anchor_tolerance_mm=UNIT_ANCHOR_TOLERANCE_MM,
                toronto_flow=flow.to_dict('records'),
                unmatched_toronto_exclusions=exclusion_audit.loc[~exclusion_audit.matched_source, 'aano'].tolist(), source_notebook_sha256=SOURCE_NOTEBOOK_SHA256,
                parent_shared_picai_prostatex_patients=shared_patients, analysis_cohorts_patient_disjoint=True, toronto_selection='Current exclude.json applied to source CSV; eligible positive manual references form Manual; remaining eligible patients form Additional',
                bland_altman=dict(xscale='log', xticks=[10, 30, 100, 300], xlim=BA_XLIM, ylim=BA_YLIM, relative_ylim=BA_REL_YLIM, layout='6x2', page_inches=[8.5, 9.75], common_plot_limits=COMMON_LIMITS, common_criteria_applied_to_analysis=True, highlight=dict(color='#D55E00', edgecolor='white', alpha=1., size=4, criterion='strictly outside both common intervals'), sign='clinical ellipsoid minus multicenter model', statistics_use_all_cases=True),
                tests='two-sided exact binomial p=0.5, non-ties only',
                correction='Holm across six cohorts separately for volume and ordinal PSAD closeness; undefined slots use p=1 then remain undefined',
                error_change='automated minus manual', disagreement_case_improvement='absolute clinical error minus absolute multicenter error (mL); positive improves',
                validation='All assertions passed')
(OUTPUT_DIR / 'run_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
if UPDATE_MANUSCRIPT:
    manuscript_path = REPO_ROOT / 'paper_new.md'
    manuscript = manuscript_path.read_text()
    for number, frame in tables.items():
        start, end = f'<!-- TABLE_{number}_START -->', f'<!-- TABLE_{number}_END -->'
        assert manuscript.count(start) == manuscript.count(end) == 1, f'Missing/duplicate manuscript table markers: {number}'
        before, rest = manuscript.split(start)
        _, after = rest.split(end)
        manuscript = before + start + '\n\n' + markdown(frame) + '\n\n' + end + after
    manuscript_path.write_text(manuscript)
print('Validation passed; outputs:', OUTPUT_DIR)
