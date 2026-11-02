# %% [markdown]
# # PI-CAI RUMC independent of the linked PROSTATEx cohort
#
# Identify RUMC studies absent from the linked clinical-volume cohort, then
# exclude every study belonging to those linked patients. Independence is based
# on PI-CAI patient IDs, not an independent identity verification. Run in cell
# order. This exploration writes no outputs and performs no image processing.

# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

REPO_ROOT = Path.cwd().resolve()
if REPO_ROOT.name == 'notebooks':
    REPO_ROOT = REPO_ROOT.parent
assert (REPO_ROOT / 'src/generate_pooled_folds.py').is_file(), 'Run from repository root or notebooks/'
PICAI_ROOT = REPO_ROOT / 'data/raw/zenodo/picai'
LABEL_ROOT = PICAI_ROOT / 'picai_labels'
IMAGE_INDEX_PATH = PICAI_ROOT / 'record-6624726/images_nii.csv'
# All paths in the image index are relative to this directory (including ../).
IMAGE_PATH_BASE = IMAGE_INDEX_PATH.parent
marksheet = pd.read_csv(LABEL_ROOT / 'clinical_information/marksheet.csv')
cases = pd.read_csv(REPO_ROOT / 'data/raw/github/prostatex_masks/index.csv')
mapping = json.loads((LABEL_ROOT / 'additional_resources/ProstateX-mapping.json').read_text())
images = pd.read_csv(IMAGE_INDEX_PATH)
assert cases['ID'].is_unique
assert not marksheet[['patient_id', 'study_id']].isna().any().any()
assert not marksheet.duplicated(['patient_id', 'study_id']).any()
assert images['ID'].notna().all() and images['ID'].is_unique

# %% [markdown]
# ## Reconstruct the comparison cohort
#
# Apply the same rules as prostatex_linked_clinical_volume.py: discard ambiguous
# mappings, missing links/marksheet matches, and nonpositive or nonfinite clinical
# volumes. PSA availability is not required. This volume criterion defines the
# comparison cohort only; it does not filter candidate PI-CAI studies.

# %%
links = pd.DataFrame(
    [(key, key.split('_')[0], value) for key, value in mapping.items()],
    columns=['mapping_key', 'ID', 'picai_key'],
)
ambiguous_ids = links.loc[links['ID'].duplicated(keep=False), 'ID'].unique()
unique_links = links.loc[~links['ID'].isin(ambiguous_ids)].copy()
unique_links[['patient_id', 'study_id']] = unique_links['picai_key'].str.split('_', expand=True).astype('int64')
cohort = cases.merge(unique_links, on='ID', how='left', validate='one_to_one')
cohort = cohort.merge(marksheet, on=['patient_id', 'study_id'], how='left',
                      validate='many_to_one', indicator='marksheet_match')
clinical = pd.to_numeric(cohort['prostate_volume'], errors='coerce')
cohort['selection_status'] = np.select(
    [cohort['ID'].isin(ambiguous_ids), cohort['picai_key'].isna(),
     cohort['marksheet_match'].ne('both'), ~np.isfinite(clinical) | clinical.le(0)],
    ['ambiguous_mapping', 'no_mapping', 'no_marksheet', 'invalid_clinical_volume'],
    default='included',
)
linked = cohort.loc[cohort['selection_status'].eq('included')].copy()
assert len(linked) == 190, 'Linked cohort changed; inspect selection before continuing'
assert not linked.duplicated(['patient_id', 'study_id']).any()
display(cohort['selection_status'].value_counts().rename_axis('status').to_frame('studies'))

# %% [markdown]
# ## Sequential study and patient exclusions
#
# A case is the pair (patient_id, study_id). Patient exclusion additionally removes
# other examinations from linked patients. Keep all marksheet fields and explicit
# flags so the selection can later feed a CSV-generating function.

# %%
audit = marksheet.copy()
audit['picai_key'] = audit['patient_id'].astype(str) + '_' + audit['study_id'].astype(str)
audit['is_rumc'] = audit['center'].eq('RUMC')
audit['linked_study_overlap'] = audit['picai_key'].isin(linked['picai_key'])
audit['linked_patient_overlap'] = audit['patient_id'].isin(linked['patient_id'])
assert (~audit['linked_study_overlap'] | audit['linked_patient_overlap']).all()
audit['exclusion_reason'] = np.select(
    [~audit['is_rumc'], audit['linked_study_overlap'], audit['linked_patient_overlap']],
    ['non_rumc', 'linked_study', 'other_study_of_linked_patient'], default='eligible',
)
rumc = audit.loc[audit['is_rumc']]
exact_overlap = rumc.loc[rumc['linked_study_overlap']]
case_disjoint = rumc.loc[~rumc['linked_study_overlap']]
additional_exclusion = case_disjoint.loc[case_disjoint['linked_patient_overlap']]
eligible = audit.loc[audit['exclusion_reason'].eq('eligible')].copy()
summary = pd.DataFrame([
    {'stage': name, 'studies': len(frame), 'patients': frame['patient_id'].nunique()}
    for name, frame in [
        ('Total PI-CAI', audit), ('RUMC', rumc),
        ('RUMC included in linked PROSTATEx (removed)', exact_overlap),
        ('RUMC after exact-study exclusion', case_disjoint),
        ('Additional shared-patient studies (removed)', additional_exclusion),
        ('Final patient-disjoint RUMC', eligible),
    ]
])
display(summary)
display(audit.groupby('exclusion_reason').agg(studies=('picai_key', 'size'), patients=('patient_id', 'nunique')))
assert len(rumc) == len(exact_overlap) + len(case_disjoint)
assert len(case_disjoint) == len(additional_exclusion) + len(eligible)
assert not set(eligible['picai_key']) & set(linked['picai_key'])
assert not set(eligible['patient_id']) & set(linked['patient_id'])
# Show both the additional excluded examination and its linked counterpart.
display(audit.loc[audit['patient_id'].isin(additional_exclusion['patient_id']),
                  ['patient_id', 'study_id', 'center', 'linked_study_overlap', 'exclusion_reason']])

# %% [markdown]
# ## Retain paths for future manifest generation
#
# Join the eligible metadata to the image index without choosing a label source.
# These segmentation columns are AI-generated anatomical delineations, not manual
# ground truth; zonal sources would require an explicit whole-gland conversion.
# Availability is reported separately and does not silently shrink the cohort.
# No train/test split or CSV is generated here. A future split must group the
# retained repeated examinations by patient.

# %%
eligible = eligible.merge(images.rename(columns={'ID': 'image_index_id'}),
                          left_on='picai_key', right_on='image_index_id',
                          how='left', validate='one_to_one')
assert eligible['image_index_id'].notna().all(), 'Eligible study absent from image index'
path_columns = [column for column in images.columns if column != 'ID']
availability = []
for column in path_columns:
    present = eligible[column].notna() & eligible[column].ne('')
    exists = eligible[column].map(
        lambda value: isinstance(value, str) and bool(value) and (IMAGE_PATH_BASE / value).is_file()
    )
    eligible[f'{column}_exists'] = exists
    availability.append({'source': column, 'eligible_studies': len(eligible),
                         'paths_present': int(present.sum()), 'files_existing': int(exists.sum())})
display(pd.DataFrame(availability))
display(eligible)
assert eligible[[f'{column}_exists' for column in path_columns]].all().all(), 'Inspect unavailable paths'
print(f'Eligible: {len(eligible)} studies from {eligible.patient_id.nunique()} patients')
print(f'Image-index path base: {IMAGE_PATH_BASE}')

# %% [markdown]
# The current local data yield 609 eligible studies from 602 patients. Excluding
# exact studies alone leaves 610 studies: patient 11383 also has study 1001407,
# while study 1001406 belongs to the linked cohort. Clinical-volume availability
# has not been imposed on the candidates. The in-memory `audit`, `summary`, and
# `eligible` tables preserve selection provenance for subsequent manifest work.
