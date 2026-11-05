# %% [markdown]
# # Clinically incomplete RUMC training and manual PROSTATEx testing
#
# Restrict the marksheet to RUMC, then require volume, PSA, recorded PSAD,
# PI-RADS, and biopsy grade on the same row for a complete clinical exam.
# Exclude patients with any complete RUMC exam and every PROSTATEx-mapped patient.
# Select the earliest RUMC exam per remaining patient for training.
#
# Keep the 236 training exams from 236 patients and add locally indexed, mapped
# PROSTATEx cases to testing, pairing original T2 images with manual whole-gland
# masks. Training labels remain Bosma22b AI surrogate annotations. Exclude every
# PROSTATEx-mapped patient from training, including ambiguous links and repeats.
# Exclude unmapped PROSTATEx identities from testing and record them in the audit.
#
# Run cells in order, or execute this file with the project Python environment.
# Write the manifest under configs/ and selection audits under
# results/picai_rumc_incomplete_not_prostatex/. All selected training and testing
# paths are checked. No downloads, image processing, preprocessing, training, or
# inference are performed. Selection is deterministic; no random seed is needed.

# %% [markdown]
# ## Load inputs and restrict to RUMC

# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

# Script execution resolves from this file; interactive cells support root or notebooks/.
REPO_ROOT = Path(__file__).resolve().parents[1] if '__file__' in globals() else Path.cwd().resolve()
if REPO_ROOT.name == 'notebooks':
    REPO_ROOT = REPO_ROOT.parent
DATA_ROOT = (REPO_ROOT / 'data/raw').resolve()
PICAI_ROOT = DATA_ROOT / 'zenodo/picai'
LABEL_ROOT = PICAI_ROOT / 'picai_labels'
MARKSHEET_PATH = LABEL_ROOT / 'clinical_information/marksheet.csv'
MAPPING_PATH = LABEL_ROOT / 'additional_resources/ProstateX-mapping.json'
IMAGE_INDEX_PATH = PICAI_ROOT / 'record-6624726/images_nii.csv'
BOSMA_ROOT = LABEL_ROOT / 'anatomical_delineations/whole_gland/AI/Bosma22b'
PROSTATEX_ROOT = DATA_ROOT / 'github/prostatex_masks'
PROSTATEX_INDEX_PATH = PROSTATEX_ROOT / 'index.csv'
MANUAL_ROOT = PROSTATEX_ROOT / 'Files/prostate'
MANIFEST_PATH = REPO_ROOT / 'configs/picai_rumc_incomplete_not_prostatex_folds.json'
OUTPUT_DIR = REPO_ROOT / 'results/picai_rumc_incomplete_not_prostatex'

for path in (MARKSHEET_PATH, MAPPING_PATH, IMAGE_INDEX_PATH, PROSTATEX_INDEX_PATH):
    if not path.is_file():
        raise FileNotFoundError(f'Required local input: {path}')
marksheet = pd.read_csv(MARKSHEET_PATH)
rumc = marksheet.loc[marksheet['center'].eq('RUMC')].copy()
images = pd.read_csv(IMAGE_INDEX_PATH)
mapping = json.loads(MAPPING_PATH.read_text())
prostatex = pd.read_csv(PROSTATEX_INDEX_PATH)
assert prostatex['ID'].notna().all() and prostatex['ID'].is_unique, 'Missing or duplicate PROSTATEx IDs'
assert not prostatex.empty, 'No PROSTATEx testing cases'
assert not rumc[['patient_id', 'study_id']].isna().any().any(), 'Missing RUMC marksheet keys'
assert not rumc.duplicated(['patient_id', 'study_id']).any(), 'Duplicate RUMC marksheet keys'
assert images['ID'].notna().all() and images['ID'].is_unique, 'Missing or duplicate image-index IDs'

# %% [markdown]
# ## Assess clinical completeness for each RUMC exam
#
# Require finite positive volume (mL) and PSA (ng/mL), finite nonnegative recorded
# PSAD (ng/mL²), PI-RADS in 1–5, and biopsy grade. Calculated PSA / volume does not
# replace missing PSAD. Biopsy criteria match picai_clinical_volume.py: `Bx`
# histopathology, case ISUP in 0–5, and usable lesion ISUP or Gleason results.
# Negative biopsies qualify; prostatectomy-only histology does not.
# Ignore empty and N/A lesion tokens; all remaining tokens must be valid.

# %%
audit = rumc.copy()
audit['marksheet_row'] = audit.index.to_numpy()  # Zero-based row in the original marksheet.
audit['ID'] = audit['patient_id'].astype(str) + '_' + audit['study_id'].astype(str)
audit['visit_date'] = pd.to_datetime(audit['mri_date'], errors='raise')
assert audit['visit_date'].notna().all(), 'Missing RUMC MRI date prevents earliest-exam selection'

for column, flag, allowed in [
    ('lesion_PIRADS', 'has_numeric_pirads', range(1, 6)),
    ('lesion_ISUP', 'has_lesion_isup', range(6)),
]:
    flags = []
    for value in audit[column]:
        tokens = [] if pd.isna(value) else [
            token.strip() for token in str(value).split(',')
            if token.strip().upper() not in {'', 'N/A', 'NA', 'NAN'}
        ]
        scores = pd.to_numeric(pd.Series(tokens, dtype='string'), errors='coerce')
        flags.append(bool(tokens) and bool(scores.notna().all() and scores.isin(allowed).all()))
    audit[flag] = flags

gleason_flags = []
for value in audit['lesion_GS']:
    tokens = [] if pd.isna(value) else [
        token.strip() for token in str(value).split(',')
        if token.strip().upper() not in {'', 'N/A', 'NA', 'NAN'}
    ]
    valid = bool(tokens)
    for token in tokens:
        parts = token.split('+')
        try:
            valid = valid and len(parts) == 2 and all(int(part) in range(6) for part in parts)
        except ValueError:
            valid = False
    gleason_flags.append(valid)
audit['has_lesion_gleason'] = gleason_flags
audit['has_biopsy_type'] = audit['histopath_type'].fillna('').str.contains('Bx', regex=False)
audit['has_case_isup'] = pd.to_numeric(audit['case_ISUP'], errors='coerce').isin(range(6))
audit['has_biopsy_results'] = (
    audit['has_biopsy_type'] & audit['has_case_isup']
    & (audit['has_lesion_isup'] | audit['has_lesion_gleason'])
)
volume = pd.to_numeric(audit['prostate_volume'], errors='coerce')
psa = pd.to_numeric(audit['psa'], errors='coerce')
psad = pd.to_numeric(audit['psad'], errors='coerce')
audit['has_valid_clinical_volume'] = np.isfinite(volume) & volume.gt(0)
audit['has_valid_psa'] = np.isfinite(psa) & psa.gt(0)
audit['has_valid_psad'] = np.isfinite(psad) & psad.ge(0)
required_clinical_flags = [
    'has_valid_clinical_volume',
    'has_valid_psa',
    'has_valid_psad',
    'has_numeric_pirads',
    'has_biopsy_results',
]
audit['complete_study_information'] = audit[required_clinical_flags].all(axis=1)
display(audit[required_clinical_flags].sum().rename_axis('required_field').to_frame('available_exams'))

# %% [markdown]
# ## Exclude complete and PROSTATEx-mapped patients
#
# Any complete RUMC exam makes the entire patient ineligible. PROSTATEx exclusion
# uses all mapped PI-CAI patient IDs, without requiring an unambiguous identity,
# local manual mask, or clinical volume. Completeness is assessed before selecting
# an exam, so fields from different visits cannot make a complete row.

# %%
audit['patient_has_complete_exam'] = audit.groupby('patient_id')['complete_study_information'].transform('any')
mapped_patients = {int(value.split('_')[0]) for value in mapping.values()}
audit['prostatex_mapped_patient'] = audit['patient_id'].isin(mapped_patients)
audit['training_eligible_patient'] = ~audit['patient_has_complete_exam'] & ~audit['prostatex_mapped_patient']

# %% [markdown]
# ## Select the earliest exam per eligible patient
#
# Sort by MRI date, then original marksheet row order to resolve date ties.
# Retain every RUMC exam and its selection reason in the audit. All unselected
# RUMC exams are excluded from the manifest; testing uses PROSTATEx only.

# %%
first_exams = audit.sort_values(['patient_id', 'visit_date', 'marksheet_row']).drop_duplicates('patient_id')
audit['is_first_exam'] = audit['marksheet_row'].isin(first_exams['marksheet_row'])
audit['selected_training_exam'] = audit['training_eligible_patient'] & audit['is_first_exam']
audit['assignment'] = np.where(audit['selected_training_exam'], 'training', 'excluded')
audit['selection_reason'] = np.select(
    [audit['selected_training_exam'], audit['patient_has_complete_exam'], audit['prostatex_mapped_patient']],
    ['earliest_exam_of_eligible_patient', 'patient_has_complete_exam', 'prostatex_mapped_patient'],
    default='repeat_exam_of_training_patient',
)
training = audit.loc[audit['selected_training_exam']].copy()

assert audit['center'].eq('RUMC').all()
assert training['patient_id'].is_unique, 'Training must contain one exam per patient'
assert not training['patient_has_complete_exam'].any()
assert not training['prostatex_mapped_patient'].any()
assert training['is_first_exam'].all()
assert set(training['patient_id']) == set(audit.loc[audit['training_eligible_patient'], 'patient_id'])

# %% [markdown]
# ## Add manual PROSTATEx testing cases and check mapped identities
#
# Require a PI-CAI mapping for testing, without clinical completeness requirements.
# Retain all links for ambiguous IDs when checking overlap, since
# choosing one link could miss a shared training patient. Original PROSTATEx T2
# images are paired with mask_prostate annotations, never PI-CAI AI labels.

# %%
testing = prostatex.sort_values('ID').copy()
links = pd.DataFrame(
    [(key.split('_')[0], value, int(value.split('_')[0])) for key, value in mapping.items()],
    columns=['ID', 'picai_key', 'patient_id'],
)
testing_links = links.loc[links['ID'].isin(testing['ID'])].copy()
assert not set(training['patient_id']) & set(testing_links['patient_id']), 'Mapped patient overlap'
assert not set(training['ID']) & set(testing_links['picai_key']), 'Mapped examination overlap'
linked_keys = testing_links.groupby('ID')['picai_key'].agg(lambda values: ','.join(sorted(set(values))))
testing['picai_keys'] = testing['ID'].map(linked_keys).fillna('')
testing['mapping_count'] = testing['ID'].map(testing_links.groupby('ID').size()).fillna(0).astype(int)
testing['linkage_status'] = np.select(
    [testing['mapping_count'].eq(0), testing['mapping_count'].gt(1)],
    ['excluded_no_mapping', 'multiple_links_all_excluded_from_training'],
    default='mapped_patient_excluded_from_training',
)
testing['assignment'] = np.where(testing['mapping_count'].gt(0), 'testing', 'excluded')
testing_audit = testing.copy()
testing = testing.loc[testing['assignment'].eq('testing')].copy()
assert not testing.empty, 'No mapped PROSTATEx testing cases'
assert testing['mapping_count'].gt(0).all(), 'Unmapped testing identity'
display(testing_audit['linkage_status'].value_counts().rename_axis('status').to_frame('prostatex_cases'))

# %% [markdown]
# ## Review cohort flow
#
# Of 792 RUMC patients, 340 have no complete exam. Excluding the 104 mapped to
# PROSTATEx leaves 236 patients. Their 239 eligible exams supply one training
# exam per patient; the other three exams remain excluded.

# %%
no_complete = audit.loc[~audit['patient_has_complete_exam']]
mapped_incomplete = no_complete.loc[no_complete['prostatex_mapped_patient']]
eligible = audit.loc[audit['training_eligible_patient']]
flow = pd.DataFrame([
    {'stage': name, 'examinations': len(frame), 'patients': frame['patient_id'].nunique()}
    for name, frame in [
        ('RUMC examinations only', audit),
        ('RUMC patients with no complete exam', no_complete),
        ('PROSTATEx-mapped patients with no complete exam (excluded)', mapped_incomplete),
        ('Eligible RUMC patients before first-exam selection', eligible),
        ('Training: earliest RUMC exam per eligible patient', training),
        ('Testing: original PROSTATEx images and manual whole-gland masks',
         testing.rename(columns={'ID': 'patient_id'})),
    ]
])
assert (len(audit), audit['patient_id'].nunique()) == (800, 792), 'RUMC cohort changed'
assert (len(no_complete), no_complete['patient_id'].nunique()) == (345, 340)
assert (len(mapped_incomplete), mapped_incomplete['patient_id'].nunique()) == (106, 104)
assert (len(eligible), len(training)) == (239, 236), 'RUMC training counts changed'
display(flow)
display(audit.groupby('selection_reason').agg(examinations=('ID', 'size'), patients=('patient_id', 'nunique')))

# %% [markdown]
# ## Validate selected paths and write outputs
#
# Image-index paths are relative to the index directory, including `../`.
# Resolve paths before making them relative to data/raw for the manifest.
# Check training images/Bosma22b labels and testing images/manual masks.
# All selection and path checks must pass before writing any output.

# %%
training = training.merge(images[['ID', 't2']], on='ID', how='left', validate='one_to_one', indicator='image_index_match')
assert training['image_index_match'].eq('both').all(), 'Selected exam absent from image index'
assert training['t2'].notna().all() and training['t2'].str.strip().ne('').all(), 'Missing selected T2 path'
entries = []
for row in training.sort_values(['patient_id', 'visit_date', 'marksheet_row']).itertuples(index=False):
    image_path = (IMAGE_INDEX_PATH.parent / row.t2).resolve()
    label_path = (BOSMA_ROOT / f'{row.ID}.nii.gz').resolve()
    for path in (image_path, label_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    entries.append({
        'image': image_path.relative_to(DATA_ROOT).as_posix(),
        'label': label_path.relative_to(DATA_ROOT).as_posix(),
        'id': f'picai_{row.ID}',
    })
testing_entries = []
for row in testing.itertuples(index=False):
    assert isinstance(row.t2, str) and row.t2.strip(), 'Missing PROSTATEx T2 path'
    assert isinstance(row.label_wg, str) and row.label_wg.strip(), 'Missing manual whole-gland mask'
    image_path = (PROSTATEX_ROOT / row.t2).resolve()
    label_path = (PROSTATEX_ROOT / row.label_wg).resolve()
    assert image_path.is_relative_to((MANUAL_ROOT / 'Images').resolve()), 'Unexpected PROSTATEx image source'
    assert label_path.is_relative_to((MANUAL_ROOT / 'mask_prostate').resolve()), 'Testing requires manual whole-gland masks'
    for path in (image_path, label_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    testing_entries.append({
        'image': image_path.relative_to(DATA_ROOT).as_posix(),
        'label': label_path.relative_to(DATA_ROOT).as_posix(),
        'id': row.ID,
    })
folds = {'training': entries, 'testing': testing_entries}
for field in ('id', 'image', 'label'):
    train_values = {entry[field] for entry in entries}
    test_values = {entry[field] for entry in testing_entries}
    assert len(train_values) == len(entries), f'Duplicate training {field}'
    assert len(test_values) == len(testing_entries), f'Duplicate testing {field}'
    assert train_values.isdisjoint(test_values), f'Training/testing {field} overlap'
testing['image'] = [entry['image'] for entry in testing_entries]
testing['label'] = [entry['label'] for entry in testing_entries]
manifest_text = json.dumps(folds, indent=4, allow_nan=False) + '\n'

MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH.write_text(manifest_text)
audit.sort_values('marksheet_row').to_csv(OUTPUT_DIR / 'selection_audit.csv', index=False)
flow.to_csv(OUTPUT_DIR / 'cohort_flow.csv', index=False)
testing_audit = testing_audit.merge(testing[['ID', 'image', 'label']], on='ID', how='left', validate='one_to_one')
testing_audit.to_csv(OUTPUT_DIR / 'testing_audit.csv', index=False)
print(f'Wrote {MANIFEST_PATH}')
print(f'Wrote selection_audit.csv, testing_audit.csv, and cohort_flow.csv to {OUTPUT_DIR}')
print(f'Training: {len(entries)} RUMC exams / {training.patient_id.nunique()} patients; testing: {len(testing_entries)} manual PROSTATEx cases')
print(f'PROSTATEx cases excluded without a PI-CAI mapping: {testing_audit.mapping_count.eq(0).sum()}')
