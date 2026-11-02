# %% [markdown]
# # Clinical and ground-truth volumes in the linked PROSTATEx cohort
#
# Reconstruct the 190-case cohort from the local PROSTATEx mask index and PI-CAI
# linkage, retain **all marksheet columns**, and compare clinical prostate volume
# with manual whole-gland segmentation volume. Clinical `prostate_volume` is in
# mL (equivalently cc); it is a separate clinical measurement, not a segmentation.
# Inputs are existing local downloads; this notebook performs no downloads or
# inference. Run cells in order. Outputs go to `results/prostatex_linked_volume/`.

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from IPython.display import display

# Select one or more methods; order controls display order.
SELECTED_COMPARATORS = ['clinical', 'nnunet274', 'bosma22b'] #, 'guerbet23', 'heviai23', 'yuan23']
COMPARATORS = {
    'clinical': ('Reported clinical volume', None, [1]),
    'nnunet274': ('Dataset274 nnUNet', None, [1]),
    'bosma22b': ('Bosma22b', 'whole_gland/AI/Bosma22b', [1]),
    'guerbet23': ('Guerbet23', 'whole_gland/AI/Guerbet23', [1]),
    'heviai23': ('HeviAI23', 'zonal_pz_tz/AI/HeviAI23', [1, 2]),
    'yuan23': ('Yuan23', 'zonal_pz_tz/AI/Yuan23', [1, 2]),
}
assert SELECTED_COMPARATORS, 'Select at least one comparator'
assert len(set(SELECTED_COMPARATORS)) == len(SELECTED_COMPARATORS), 'Repeated comparator'
assert set(SELECTED_COMPARATORS) <= COMPARATORS.keys(), 'Unknown comparator'
COLORS = {key: f'C{i}' for i, key in enumerate(COMPARATORS)}

# Run from notebooks/; the repository root is its parent.
REPO_ROOT = Path.cwd().resolve()
if REPO_ROOT.name == 'notebooks':
    REPO_ROOT = REPO_ROOT.parent
MASK_ROOT = REPO_ROOT / 'data/raw/github/prostatex_masks'
PICAI_ROOT = REPO_ROOT / 'data/raw/zenodo/picai/picai_labels'
INDEX_PATH = MASK_ROOT / 'index.csv'
MAPPING_PATH = PICAI_ROOT / 'additional_resources/ProstateX-mapping.json'
MARKSHEET_PATH = PICAI_ROOT / 'clinical_information/marksheet.csv'
OUTPUT_DIR = REPO_ROOT / 'results/prostatex_linked_volume'
EXPECTED_CASES = 190

for path in (INDEX_PATH, MAPPING_PATH, MARKSHEET_PATH):
    if not path.is_file():
        raise FileNotFoundError(f'Required local input: {path}')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Link the mask index to the complete clinical marksheet
#
# Mapping keys include acquisition dates, whereas index IDs do not. Exclude IDs
# with multiple mapping entries rather than choosing a study arbitrarily. Join on
# both PI-CAI patient and study IDs; require unique marksheet keys to avoid row
# multiplication. PSA availability is not a selection criterion for this analysis.

# %%
cases = pd.read_csv(INDEX_PATH)
marksheet = pd.read_csv(MARKSHEET_PATH)
assert cases['ID'].is_unique, 'Duplicate PROSTATEx IDs in mask index'
assert not marksheet[['patient_id', 'study_id']].isna().any().any()
assert not marksheet.duplicated(['patient_id', 'study_id']).any(), 'Duplicate marksheet keys'

mapping = json.loads(MAPPING_PATH.read_text())
links = pd.DataFrame(
    [(key, key.split('_')[0], value) for key, value in mapping.items()],
    columns=['mapping_key', 'ID', 'picai_key'],
)
ambiguous_ids = links.loc[links['ID'].duplicated(keep=False), 'ID'].unique()
unique_links = links.loc[~links['ID'].isin(ambiguous_ids)].copy()
unique_links[['patient_id', 'study_id']] = (
    unique_links['picai_key'].str.split('_', expand=True).astype('int64')
)
cohort = cases.merge(unique_links, on='ID', how='left', validate='one_to_one')
cohort = cohort.merge(
    marksheet, on=['patient_id', 'study_id'], how='left',
    validate='many_to_one', indicator='marksheet_match',
)
clinical = pd.to_numeric(cohort['prostate_volume'], errors='coerce')
cohort['clinical_volume_ml'] = clinical
cohort['selection_status'] = np.select(
    [cohort['ID'].isin(ambiguous_ids), cohort['picai_key'].isna(),
     cohort['marksheet_match'].ne('both'), ~np.isfinite(clinical) | clinical.le(0)],
    ['ambiguous_mapping', 'no_mapping', 'no_marksheet', 'invalid_clinical_volume'],
    default='included',
)
cohort.to_csv(OUTPUT_DIR / 'cohort_linkage_audit.csv', index=False)
linked = cohort.loc[cohort['selection_status'].eq('included')].copy().reset_index(drop=True)
display(cohort['selection_status'].value_counts().rename_axis('status').to_frame('n'))
print(f'Included: {len(linked)}; expected: {EXPECTED_CASES}')
assert len(linked) == EXPECTED_CASES, 'Cohort changed; inspect linkage audit before analysis'
assert not linked.duplicated(['patient_id', 'study_id']).any(), 'Repeated linked study'
display(linked[['ID', *marksheet.columns]])
display(linked[marksheet.columns].isna().sum().to_frame('missing_values'))

# %% [markdown]
# ## Calculate reference whole-gland volume
#
# `label_wg` points to the manual binary whole-gland mask (background 0, gland 1),
# as documented in the downloaded mask repository README. Do not substitute a TZ
# or PZ mask. Volume = gland voxel count × voxel volume in mm³ / 1,000, yielding
# mL. The affine determinant accounts for the voxel grid; NIfTI spatial units must
# be millimetres. No image resampling or spatial overlap comparison is performed.
# Missing or invalid masks stop execution rather than silently shrinking the cohort.

# %%
reference_rows = []
for row in linked.itertuples(index=False):
    mask_path = MASK_ROOT / row.label_wg
    image = nib.load(mask_path)
    mask = np.asanyarray(image.dataobj)
    assert mask.ndim == 3, f'{row.ID}: expected a 3D mask'
    assert image.header.get_xyzt_units()[0] == 'mm', f'{row.ID}: spatial units are not mm'
    assert np.isin(mask, [0, 1]).all(), f'{row.ID}: unexpected whole-gland labels'
    voxel_volume_mm3 = float(abs(np.linalg.det(image.affine[:3, :3])))
    assert np.isfinite(voxel_volume_mm3) and voxel_volume_mm3 > 0
    voxel_count = int(np.count_nonzero(mask == 1))
    assert voxel_count > 0, f'{row.ID}: empty reference mask'
    reference_rows.append({
        'ID': row.ID,
        'gt_voxel_count': voxel_count,
        'gt_voxel_volume_mm3': voxel_volume_mm3,
        'gt_volume_ml': voxel_count * voxel_volume_mm3 / 1000,
    })

volumes = linked.merge(pd.DataFrame(reference_rows), on='ID', validate='one_to_one')

# %% [markdown]
# ## Selected measurements on native grids
#
# Clinical volume is reported in the marksheet; its calculation method is not
# established here. PI-CAI masks are matched by patient/study ID. Bosma22b and
# Guerbet23 use whole-gland label 1; HeviAI23 and Yuan23 use PZ=1 and TZ=2.
# Their union is measured in memory. No resampling or spatial evaluation is
# performed. Empty predictions remain zero-volume failures; invalid inputs stop.

# %%
from itertools import combinations

MODEL_ROOT = REPO_ROOT / 'results/prostatex_pred_pooled2_3d_fullres'
if 'nnunet274' in SELECTED_COMPARATORS:
    assert json.loads((MODEL_ROOT / 'plans.json').read_text())['dataset_name'] == 'Dataset274_raw'
    assert json.loads((MODEL_ROOT / 'dataset.json').read_text())['labels'] == {'background': 0, 'class1': 1}
for key in SELECTED_COMPARATORS:
    name, source, foreground = COMPARATORS[key]
    if key == 'clinical':
        continue
    rows = []
    for case in linked.itertuples(index=False):
        path = (MODEL_ROOT / f'{case.ID}.nii.gz' if key == 'nnunet274' else
                PICAI_ROOT / 'anatomical_delineations' / source / f'{int(case.patient_id)}_{int(case.study_id)}.nii.gz')
        image = nib.load(path)
        mask = np.asanyarray(image.dataobj)
        assert mask.ndim == 3 and np.isfinite(mask).all(), path
        assert image.header.get_xyzt_units()[0] == 'mm', path
        assert np.isin(mask, [0, *foreground]).all(), path
        count = int(np.count_nonzero(np.isin(mask, foreground)))
        assert count == sum(int(np.count_nonzero(mask == label)) for label in foreground)
        voxel_volume = float(abs(np.linalg.det(image.affine[:3, :3])))
        assert np.isfinite(voxel_volume) and voxel_volume > 0, path
        reference = nib.load(MASK_ROOT / case.label_wg)
        rows.append({'ID': case.ID, f'{key}_volume_ml': count * voxel_volume / 1000,
                     f'{key}_path': str(path.relative_to(REPO_ROOT)),
                     f'{key}_foreground_labels': str(foreground), f'{key}_voxel_count': count,
                     f'{key}_voxel_volume_mm3': voxel_volume,
                     f'{key}_gt_shape_match': image.shape == reference.shape,
                     f'{key}_gt_affine_max_abs_difference': float(np.abs(image.affine-reference.affine).max())})
    volumes = volumes.merge(pd.DataFrame(rows), on='ID', validate='one_to_one')

# %% [markdown]
# ## Volume accuracy against manual whole-gland ground truth
#
# Positive signed errors indicate overestimation. Limits of agreement are bias
# ± 1.96 sample SD, not confidence intervals. Rank by MAE on this linked cohort;
# volume accuracy alone does not establish spatial segmentation quality.

# %%
summary_rows = []
fig, axes = plt.subplots(1, 3, figsize=(21, 6), constrained_layout=True)
gt = volumes['gt_volume_ml']
for key in SELECTED_COMPARATORS:
    name = COMPARATORS[key][0]
    value = volumes[f'{key}_volume_ml']
    error = value - gt
    relative = 100 * error / gt
    volumes[f'{key}_minus_gt_ml'] = error
    volumes[f'{key}_absolute_error_ml'] = error.abs()
    volumes[f'{key}_absolute_percentage_error'] = relative.abs()
    summary_rows.append(dict(method=key, name=name, n=len(value), bias_ml=error.mean(),
        mae_ml=error.abs().mean(), rmse_ml=np.sqrt((error**2).mean()),
        mean_absolute_percentage_error=relative.abs().mean(), pearson_r=value.corr(gt),
        loa_lower_ml=error.mean()-1.96*error.std(), loa_upper_ml=error.mean()+1.96*error.std(),
        relative_bias_percent=relative.mean(), relative_loa_lower_percent=relative.mean()-1.96*relative.std(),
        relative_loa_upper_percent=relative.mean()+1.96*relative.std()))
    axes[0].scatter(gt, value, s=12, alpha=.5, color=COLORS[key], label=name)
    for ax, differences in zip(axes[1:], [error, relative]):
        ax.scatter(gt, differences, s=12, alpha=.5, color=COLORS[key], label=name)
        for level, style in [(differences.mean(), '-'), (differences.mean()-1.96*differences.std(), '--'),
                             (differences.mean()+1.96*differences.std(), '--')]:
            ax.axhline(level, color=COLORS[key], linestyle=style, linewidth=.7)
upper = max(gt.max(), *(volumes[f'{k}_volume_ml'].max() for k in SELECTED_COMPARATORS)) * 1.05
axes[0].plot([0, upper], [0, upper], 'k--')
for ax, ylabel in zip(axes, ['Measured volume (mL)', 'Measurement − ground truth (mL)', 'Relative difference (%)']):
    ax.set(xlabel='Manual whole-gland volume (mL)', ylabel=ylabel)
    ax.legend(fontsize=8)
summary = pd.DataFrame(summary_rows).sort_values('mae_ml')
display(summary)
volumes.to_csv(OUTPUT_DIR / 'comparator_volumes.csv', index=False)
summary.to_csv(OUTPUT_DIR / 'comparator_summary.csv', index=False)
fig.savefig(OUTPUT_DIR / 'comparator_agreement.png', dpi=180)
plt.show()

# %% [markdown]
# ## Paired absolute-error comparisons
#
# Positive improvement means method A is closer than method B. Differences within
# 1e-6 mL are numerical ties. All methods use the same 190 cases.

# %%
TIE_TOLERANCE_ML = 1e-6
pair_tables, count_rows = [], []
for a, b in combinations(SELECTED_COMPARATORS, 2):
    pair = volumes[['ID', 'patient_id', 'study_id']].copy()
    pair['method_a'], pair['method_b'] = a, b
    pair['absolute_error_improvement_ml'] = volumes[f'{b}_absolute_error_ml'] - volumes[f'{a}_absolute_error_ml']
    delta = pair['absolute_error_improvement_ml']
    pair['a_vs_b'] = np.select([delta > TIE_TOLERANCE_ML, delta < -TIE_TOLERANCE_ML], ['Closer', 'Farther'], default='Tie')
    pair_tables.append(pair)
    for outcome in ['Closer', 'Farther', 'Tie']:
        n = int(pair['a_vs_b'].eq(outcome).sum())
        count_rows.append(dict(method_a=a, method_b=b, outcome=outcome, n=n, percent=100*n/len(pair)))
if pair_tables:
    pairwise = pd.concat(pair_tables, ignore_index=True)
    pairwise_counts = pd.DataFrame(count_rows)
    display(pairwise_counts)
    pairwise.to_csv(OUTPUT_DIR / 'comparator_pairwise_per_case.csv', index=False)
    pairwise_counts.to_csv(OUTPUT_DIR / 'comparator_pairwise_counts.csv', index=False)
