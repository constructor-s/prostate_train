# %% [markdown]
# # Generate the clinically incomplete PI-CAI split
#
# Use all PI-CAI centers. A patient is clinically complete if **any** examination
# has usable volume, PSA, PI-RADS, and biopsy results. Reported PSAD is optional:
# PSA / volume can supply it. Among patients with no complete examination, remove
# everyone appearing anywhere in the PROSTATEx mapping, including ambiguous links.
# Select each remaining patient's earliest MRI examination for training.
#
# Testing contains **every other examination**, including later examinations of
# training patients. This is an examination-disjoint split; four patients occur
# in both splits in the current data. Expected counts are 426 training exams and
# 1,074 testing exams. Both splits use Bosma22b AI whole-gland labels, which are
# automated surrogate annotations rather than independent manual ground truth.
#
# Run cells in order, or run this file with the project Python environment.
# Inputs are local metadata and existing image/label files. The notebook writes
# a training/testing manifest to configs/ and audits to
# results/picai_incomplete_not_prostatex/. It performs no downloads, image
# processing, preprocessing, training, or inference. Selection is deterministic;
# no random seed is needed.

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
MANIFEST_PATH = REPO_ROOT / 'configs/picai_incomplete_not_prostatex_folds.json'
OUTPUT_DIR = REPO_ROOT / 'results/picai_incomplete_not_prostatex'

for path in (MARKSHEET_PATH, MAPPING_PATH, IMAGE_INDEX_PATH):
    if not path.is_file():
        raise FileNotFoundError(f'Required local input: {path}')
marksheet = pd.read_csv(MARKSHEET_PATH)
images = pd.read_csv(IMAGE_INDEX_PATH)
mapping = json.loads(MAPPING_PATH.read_text())
assert not marksheet[['patient_id', 'study_id']].isna().any().any(), 'Missing marksheet keys'
assert not marksheet.duplicated(['patient_id', 'study_id']).any(), 'Duplicate marksheet keys'
assert images['ID'].notna().all() and images['ID'].is_unique, 'Missing or duplicate image-index IDs'

# %% [markdown]
# ## Examination-level clinical completeness
#
# Match picai_clinical_volume.py: finite positive prostate volume (mL) and PSA
# (ng/mL), PI-RADS scores in 1–5, and biopsy type containing `Bx` with case ISUP
# in 0–5 plus usable lesion Gleason or ISUP results. Negative biopsies qualify.
# Ignore empty and N/A lesion tokens; all remaining tokens must be valid.
# Prostatectomy-only histology does not count as a biopsy. Missing values are
# not imputed, and reported `psad` never affects completeness.

# %%
audit = marksheet.copy()
audit['marksheet_row'] = np.arange(len(audit))  # Zero-based data-row position.
audit['ID'] = audit['patient_id'].astype(str) + '_' + audit['study_id'].astype(str)
audit['visit_date'] = pd.to_datetime(audit['mri_date'], errors='raise')
assert audit['visit_date'].notna().all(), 'Missing MRI date prevents earliest-exam selection'

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
audit['has_valid_clinical_volume'] = np.isfinite(volume) & volume.gt(0)
audit['has_valid_psa'] = np.isfinite(psa) & psa.gt(0)
audit['complete_study_information'] = (
    audit['has_valid_clinical_volume'] & audit['has_valid_psa']
    & audit['has_numeric_pirads'] & audit['has_biopsy_results']
)

# %% [markdown]
# ## Classify patients before selecting an examination
#
# Any complete visit makes the entire patient ineligible for training, even if
# their other visits are incomplete. PROSTATEx exclusion uses **all** mapped
# PI-CAI patient IDs; it does not depend on a local manual mask, clinical volume,
# or an unambiguous PROSTATEx identity. Among eligible patients, sort by MRI date
# and original CSV row order. Assign every unselected examination to testing.

# %%
audit['patient_has_complete_exam'] = audit.groupby('patient_id')['complete_study_information'].transform('any')
mapped_patients = {int(value.split('_')[0]) for value in mapping.values()}
audit['prostatex_mapped_patient'] = audit['patient_id'].isin(mapped_patients)
audit['training_eligible_patient'] = ~audit['patient_has_complete_exam'] & ~audit['prostatex_mapped_patient']
first_exams = audit.sort_values(['patient_id', 'visit_date', 'marksheet_row']).drop_duplicates('patient_id')
audit['is_first_exam'] = audit['marksheet_row'].isin(first_exams['marksheet_row'])
audit['selected_training_exam'] = audit['training_eligible_patient'] & audit['is_first_exam']
audit['assignment'] = np.where(audit['selected_training_exam'], 'training', 'testing')
audit['selection_reason'] = np.select(
    [audit['selected_training_exam'], audit['patient_has_complete_exam'], audit['prostatex_mapped_patient']],
    ['earliest_exam_of_eligible_patient', 'patient_has_complete_exam', 'prostatex_mapped_patient'],
    default='repeat_exam_of_training_patient',
)
training = audit.loc[audit['assignment'].eq('training')].copy()
testing = audit.loc[audit['assignment'].eq('testing')].copy()
shared_patient_ids = sorted(set(training['patient_id']) & set(testing['patient_id']))
audit['patient_in_both_splits'] = audit['patient_id'].isin(shared_patient_ids)

assert training['patient_id'].is_unique, 'Training must contain one exam per patient'
assert not training['patient_has_complete_exam'].any()
assert not training['prostatex_mapped_patient'].any()
assert training['is_first_exam'].all()
assert set(training['patient_id']) == set(audit.loc[audit['training_eligible_patient'], 'patient_id'])
assert not set(training['ID']) & set(testing['ID']), 'Examination overlap'
assert set(training['ID']) | set(testing['ID']) == set(audit['ID']), 'Unassigned examination'
assert len(training) + len(testing) == len(marksheet)

no_complete = audit.loc[~audit['patient_has_complete_exam']]
mapped_incomplete = no_complete.loc[no_complete['prostatex_mapped_patient']]
eligible = audit.loc[audit['training_eligible_patient']]
flow = pd.DataFrame([
    {'stage': name, 'examinations': len(frame), 'patients': frame['patient_id'].nunique()}
    for name, frame in [
        ('Entire PI-CAI marksheet', audit),
        ('Patients with no complete exam', no_complete),
        ('PROSTATEx-mapped patients with no complete exam (excluded from training)', mapped_incomplete),
        ('Eligible patients before first-exam selection', eligible),
        ('Training: earliest exam per eligible patient', training),
        ('Testing: every remaining exam', testing),
        ('Patients shared between training and testing: all their exams',
         audit.loc[audit['patient_in_both_splits']]),
    ]
])
assert (len(audit), audit['patient_id'].nunique()) == (1500, 1476), 'PI-CAI cohort changed'
assert (no_complete['patient_id'].nunique(), mapped_incomplete['patient_id'].nunique()) == (516, 90)
assert (len(training), len(testing), testing['patient_id'].nunique(), len(shared_patient_ids)) == (426, 1074, 1054, 4), 'Split counts changed'
display(flow)
print(f'Patients with exams in both splits: {shared_patient_ids}')
display(audit.loc[audit['patient_in_both_splits'],
                  ['patient_id', 'study_id', 'mri_date', 'assignment', 'selection_reason']])

# %% [markdown]
# ## Validate paths and write the manifest and audits
#
# The image index stores paths relative to its own directory, including `../`.
# Resolve those paths before making them relative to data/raw. Labels always
# come from whole_gland/AI/Bosma22b. Check every image and label before writing
# outputs; unavailable inputs stop generation rather than silently removing exams.
# All original marksheet columns and row order remain in the selection audit.

# %%
audit = audit.merge(images[['ID', 't2']], on='ID', how='left', validate='one_to_one', indicator='image_index_match')
assert audit['image_index_match'].eq('both').all(), 'Examination absent from image index'
assert audit['t2'].notna().all() and audit['t2'].str.strip().ne('').all(), 'Missing T2 path'
image_paths = []
label_paths = []
for row in audit.itertuples(index=False):
    image_path = (IMAGE_INDEX_PATH.parent / row.t2).resolve()
    label_path = (BOSMA_ROOT / f'{row.ID}.nii.gz').resolve()
    for path in (image_path, label_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    image_paths.append(image_path.relative_to(DATA_ROOT).as_posix())
    label_paths.append(label_path.relative_to(DATA_ROOT).as_posix())
audit['image'] = image_paths
audit['label'] = label_paths
audit['id'] = 'picai_' + audit['ID']

folds = {}
for split in ('training', 'testing'):
    frame = audit.loc[audit['assignment'].eq(split)].sort_values(['patient_id', 'visit_date', 'marksheet_row'])
    folds[split] = frame[['image', 'label', 'id']].to_dict(orient='records')
manifest_text = json.dumps(folds, indent=4, allow_nan=False) + '\n'

# All metadata, membership, and path checks above must pass before writing.
MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH.write_text(manifest_text)
audit.sort_values('marksheet_row').to_csv(OUTPUT_DIR / 'selection_audit.csv', index=False)
flow.to_csv(OUTPUT_DIR / 'cohort_flow.csv', index=False)
audit.loc[audit['patient_in_both_splits']].sort_values(
    ['patient_id', 'visit_date', 'marksheet_row'],
).to_csv(OUTPUT_DIR / 'shared_patients.csv', index=False)
print(f'Wrote {MANIFEST_PATH}')
print(f'Wrote selection_audit.csv, cohort_flow.csv, and shared_patients.csv to {OUTPUT_DIR}')
print(f'Training: {len(folds["training"])} exams / {training.patient_id.nunique()} patients')
print(f'Testing: {len(folds["testing"])} exams / {testing.patient_id.nunique()} patients')

# %% [markdown]
# Current inputs yield 426 training exams (232 RUMC, 174 ZGT, 20 PCNN), one per
# eligible patient. All other 1,074 exams from 1,054 patients are testing cases.
# The four patients with exams in both splits are 10193, 10580, 10629, and 10634.
# This overlap is intentional: the manifest assigns all 1,500 exams exactly once.
