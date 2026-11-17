# %% [markdown]
# # PI-CAI clinical cohort with Bosma22b surrogate references
#
# Include all centres in the local marksheet. Bosma22b label 1 is an automated
# whole-gland surrogate reference, not independent manual ground truth.
# Clinical completeness requires finite positive volume (mL), PSA (ng/mL),
# PI-RADS 1–5 and biopsy results; recorded PSAD is optional. Biopsy requires
# Bx histopathology, case ISUP 0–5 and valid lesion ISUP or Gleason tokens.
# Negative biopsies qualify; empty/N/A tokens are ignored.
#
# Preserve Dataset236 training exactly and reserve all its patients first.
# For other patients, exclude the six manually identified poor Bosma22b segmentations
# in a separate fold, then any mask contacting a superior/inferior outer face (>0 voxels).
# This operational rule suggests possible incomplete coverage, not proven truncation.
# Among remaining visits prefer complete clinicals, earliest MRI date, then CSV
# row order. Other visits enter the duplicate fold. Patients with only contacting
# visits have no selected exam. Existing models are not retrained.
#
# Execute cells in order using the project environment. Inputs remain local;
# no downloads, preprocessing, training or inference occur. No randomness is used.

# %%
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from IPython.display import display

REPO_ROOT = Path(__file__).resolve().parents[1] if '__file__' in globals() else Path.cwd().resolve()
if REPO_ROOT.name == 'notebooks':
    REPO_ROOT = REPO_ROOT.parent
DATA_ROOT = (REPO_ROOT / 'data/raw').resolve()
PICAI_ROOT = DATA_ROOT / 'zenodo/picai'
MARKSHEET_PATH = PICAI_ROOT / 'picai_labels/clinical_information/marksheet.csv'
IMAGE_INDEX_PATH = PICAI_ROOT / 'record-6624726/images_nii.csv'
BOSMA_ROOT = PICAI_ROOT / 'picai_labels/anatomical_delineations/whole_gland/AI/Bosma22b'
MANIFEST_PATH = REPO_ROOT / 'configs/picai_test_clinical_folds.json'
OUTPUT_DIR = REPO_ROOT / 'results/picai_test_clinical'
TRAINING_MANIFEST_PATH = REPO_ROOT / 'configs/rumcnotprostatexnoclinical_folds.json'
POOR_BOSMA22B_IDS = {
    '11050_1001070', '10096_1000096', '10211_1000215',
    '10355_1000361', '10473_1000481', '10730_1000746',
}
training_entries = json.loads(TRAINING_MANIFEST_PATH.read_text())['training']
training_ids = {entry['id'].removeprefix('picai_') for entry in training_entries}
training_patients = {int(case.split('_')[0]) for case in training_ids}
assert len(training_ids) == len(training_entries) == len(training_patients), 'Invalid training cohort'

marksheet = pd.read_csv(MARKSHEET_PATH)
images = pd.read_csv(IMAGE_INDEX_PATH)
assert not marksheet.empty, 'Empty marksheet'
assert not marksheet[['patient_id', 'study_id', 'center']].isna().any().any()
assert not marksheet.duplicated(['patient_id', 'study_id']).any(), 'Duplicate marksheet keys'
assert images['ID'].notna().all() and images['ID'].is_unique, 'Invalid image-index IDs'
audit = marksheet.copy()
audit['marksheet_row'] = np.arange(len(audit))
audit['ID'] = audit['patient_id'].astype(str) + '_' + audit['study_id'].astype(str)
audit['visit_date'] = pd.to_datetime(audit['mri_date'], errors='raise')
assert audit['visit_date'].notna().all(), 'Missing MRI date'

# %% [markdown]
# ## Assess clinical completeness independently on each visit

# %%
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
audit['clinical_volume_ml'] = pd.to_numeric(audit['prostate_volume'], errors='coerce')
audit_psa = pd.to_numeric(audit['psa'], errors='coerce')
audit['has_valid_clinical_volume'] = (
    np.isfinite(audit['clinical_volume_ml']) & audit['clinical_volume_ml'].gt(0)
)
audit['has_valid_psa'] = np.isfinite(audit_psa) & audit_psa.gt(0)
audit['complete_study_information'] = (
    audit['has_valid_clinical_volume'] & audit['has_valid_psa']
    & audit['has_numeric_pirads'] & audit['has_biopsy_results']
)

# %% [markdown]
# ## Validate every image/reference pair and assess S/I contact
#
# Require nonempty binary masks and matching 3D grids in millimetres. Maximum
# image/mask corner displacement must be at most 0.5 mm. Resolve image-index
# paths relative to its directory. Contact uses the mask affine's S/I axis.

# %%
audit = audit.merge(images[['ID', 't2']], on='ID', how='left', validate='one_to_one', indicator=True)
assert audit['_merge'].eq('both').all(), 'Marksheet exam missing from image index'
audit = audit.drop(columns='_merge')
path_rows = []
for row in audit.itertuples(index=False):
    assert isinstance(row.t2, str) and row.t2.strip(), (row.ID, 'Missing T2 path')
    image_path = (IMAGE_INDEX_PATH.parent / row.t2).resolve()
    label_path = (BOSMA_ROOT / f'{row.ID}.nii.gz').resolve()
    for path in (image_path, label_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    image = nib.load(image_path)
    mask_img = nib.load(label_path)
    assert len(mask_img.shape) == 3 and mask_img.shape == image.shape, (row.ID, 'Grid shape mismatch')
    assert mask_img.header.get_xyzt_units()[0] == image.header.get_xyzt_units()[0] == 'mm', row.ID
    corners = np.array(np.meshgrid(*[(0, n - 1) for n in mask_img.shape], indexing='ij')).reshape(3, -1).T
    displacement = np.linalg.norm(nib.affines.apply_affine(mask_img.affine, corners)
                                  - nib.affines.apply_affine(image.affine, corners), axis=1).max()
    assert np.isfinite(displacement) and displacement <= .5, (row.ID, 'Geometry mismatch')
    data = np.asanyarray(mask_img.dataobj)
    assert np.isin(data, [0, 1]).all() and (data == 1).any(), (row.ID, 'Expected nonempty binary mask')
    axis_codes = nib.aff2axcodes(mask_img.affine)
    si_axis = next(axis for axis, code in enumerate(axis_codes) if code in ('S', 'I'))
    directions = ('I', 'S') if axis_codes[si_axis] == 'S' else ('S', 'I')
    faces = [direction for direction, index in zip(directions, [0, -1])
             if np.count_nonzero(np.take(data, index, axis=si_axis)) > 0]
    path_rows.append(dict(ID=row.ID, image=image_path.relative_to(DATA_ROOT).as_posix(),
                          label=label_path.relative_to(DATA_ROOT).as_posix(),
                          boundary_contact_faces=';'.join(faces), max_corner_displacement_mm=displacement))
audit = audit.merge(pd.DataFrame(path_rows), on='ID', validate='one_to_one')
audit['has_boundary_contact'] = audit['boundary_contact_faces'].ne('')

# %% [markdown]
# ## Select one noncontact visit per patient and partition all exams

# %%
assert training_ids <= set(audit['ID']), 'Training exam absent from marksheet'
audit['in_training'] = audit['ID'].isin(training_ids)
audit['training_patient'] = audit['patient_id'].isin(training_patients)
assert POOR_BOSMA22B_IDS <= set(audit['ID']), 'Poor-segmentation exam absent from marksheet'
assert POOR_BOSMA22B_IDS.isdisjoint(training_ids), 'Cannot exclude fixed training exams'
audit['poor_bosma22b_segmentation'] = audit['ID'].isin(POOR_BOSMA22B_IDS)
eligible = ~audit['training_patient'] & ~audit['poor_bosma22b_segmentation'] & ~audit['has_boundary_contact']
ranked = audit.loc[eligible].sort_values(
    ['patient_id', 'complete_study_information', 'visit_date', 'marksheet_row'],
    ascending=[True, False, True, True],
)
selected = ranked.drop_duplicates('patient_id')
audit['selected'] = audit['ID'].isin(selected['ID'])
audit['selected_study_id'] = audit['patient_id'].map(selected.set_index('patient_id')['study_id']).astype('Int64')
audit['assignment'] = np.select(
    [audit['in_training'], audit['training_patient'], audit['poor_bosma22b_segmentation'], audit['has_boundary_contact'],
     ~audit['selected'], audit['complete_study_information']],
    ['training', 'excluded_due_to_training_patient', 'excluded_due_to_poor_bosma22b_segmentation', 'excluded_due_to_boundary_contact',
     'excluded_due_to_duplicate_patient', 'testing'],
    default='incomplete_clinicals',
)
audit['selection_reason'] = np.select(
    [audit['in_training'], audit['training_patient'], audit['poor_bosma22b_segmentation'], audit['has_boundary_contact'],
     ~audit['selected'], audit['complete_study_information']],
    ['fixed_dataset236_training_exam', 'other_exam_of_dataset236_training_patient',
     'manual_review_poor_bosma22b_segmentation',
     'bosma22b_superior_or_inferior_boundary_contact', 'lower_ranked_noncontact_visit',
     'selected_earliest_complete_noncontact_visit'],
    default='selected_earliest_incomplete_noncontact_visit',
)
testing_centres = ('PCNN', 'RUMC', 'ZGT')
testing = audit['assignment'].eq('testing')
assert audit.loc[testing, 'center'].isin(testing_centres).all(), 'Unknown testing institution'
audit.loc[testing, 'assignment'] = 'testing_' + audit.loc[testing, 'center']
fold_names = ['training', 'excluded_due_to_training_patient',
              *[f'testing_{centre}' for centre in testing_centres], 'incomplete_clinicals', 'excluded_due_to_boundary_contact',
              'excluded_due_to_duplicate_patient', 'excluded_due_to_poor_bosma22b_segmentation']
folds = {}
for name in fold_names:
    frame = audit.loc[audit['assignment'].eq(name)].sort_values(['patient_id', 'visit_date', 'marksheet_row'])
    folds[name] = [dict(image=row.image, label=row.label, id=f'picai_{row.ID}')
                   for row in frame.itertuples(index=False)]
assert {entry['id']: entry for entry in folds['training']} == {entry['id']: entry for entry in training_entries}
folds['training'] = training_entries  # Preserve original entry order as well as values.
assert folds['training'] == training_entries
assert not set(audit.loc[audit['selected'], 'patient_id']) & training_patients
assert audit.loc[audit['selected'], 'patient_id'].is_unique
assert audit['selected'].sum() == audit.loc[eligible, 'patient_id'].nunique()
assert {entry['id'].removeprefix('picai_') for entry in folds['excluded_due_to_poor_bosma22b_segmentation']} == POOR_BOSMA22B_IDS
assert set(audit.loc[audit.assignment.isin([f'testing_{centre}' for centre in testing_centres]), 'patient_id']).isdisjoint(
    audit.loc[audit.assignment.eq('incomplete_clinicals'), 'patient_id'])
assert audit.loc[audit.assignment.isin([f'testing_{centre}' for centre in testing_centres]), 'complete_study_information'].all()
assert not audit.loc[audit.assignment.eq('incomplete_clinicals'), 'complete_study_information'].any()
for field in ('id', 'image', 'label'):
    values = [entry[field] for entries in folds.values() for entry in entries]
    assert len(values) == len(audit) == len(set(values)), f'Partition or duplicate {field}'

# %% [markdown]
# ## Review computed cohort counts and export
#
# Cohort counts are recomputed below from validated inputs, overall and by centre,
# including testing_PCNN, testing_RUMC, testing_ZGT and poor-segmentation exclusions.
# Exclusion-fold patient counts can overlap selected-fold patient counts.
# All validation completes before creating outputs.

# %%
flow_rows = []
for centre, source in [('ALL', audit), *list(audit.groupby('center', sort=True))]:
    for stage, frame in [('input', source), *[(name, source.loc[source.assignment.eq(name)]) for name in fold_names]]:
        flow_rows.append(dict(center=centre, stage=stage, examinations=len(frame), patients=frame.patient_id.nunique()))
flow = pd.DataFrame(flow_rows)
display(flow)
display(audit.groupby(['assignment', 'selection_reason']).size().to_frame('examinations'))
manifest_text = json.dumps(folds, indent=4, allow_nan=False) + '\n'
MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH.write_text(manifest_text)
audit.sort_values('marksheet_row').to_csv(OUTPUT_DIR / 'selection_audit.csv', index=False)
flow.to_csv(OUTPUT_DIR / 'cohort_flow.csv', index=False)
print(f'Wrote {MANIFEST_PATH}')
print(f'Wrote selection_audit.csv and cohort_flow.csv to {OUTPUT_DIR}')
