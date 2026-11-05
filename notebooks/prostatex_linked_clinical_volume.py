# %% [markdown]
# # Prostate volume and PSA density in the linked PROSTATEx cohort
#
# Reconstruct the linked cohort, retain **all marksheet columns**, and compare clinical prostate volume
# with manual whole-gland segmentation volume. Clinical `prostate_volume` is in
# mL (equivalently cc); it is a separate clinical measurement, not a segmentation.
# Then compare PSA densities computed from each volume and investigate the
# marksheet's reported PSA density against recalculated and manual-mask densities.
# Inputs are existing local downloads; this notebook performs no downloads or
# inference. Run cells in order. Outputs go to `results/prostatex_linked_volume/`.
#
# ## Study population and cohort flow
#
# Start from the 328 examinations in `ProstateX-mapping.json`, representing 327
# unique subjects. The local mask index and PI-CAI public clinical marksheet
# supply the manual whole-gland masks and clinical measurements for this analysis.
# Expected counts are recomputed from local metadata immediately below:
#
# | Stage | Examinations | Explanation |
# |---|---:|---|
# | PROSTATEx mapping JSON | 328 | 327 subjects; one has two mapped examinations |
# | Local manual whole-gland mask available | 194 | 134 examinations lack a local mask |
# | Unambiguous linkage | 192 | Remove both examinations of ProstateX-0025 |
# | Matched clinical marksheet | 192 | All remaining examinations match |
# | Valid clinical volume: volume analysis | **190** | Two lack usable clinical volume |
# | PSA available: subsequent density analysis | **189** | ProstateX-0202 lacks PSA |
#
# Keep all 190 volume cases, including suspicious records, for volume analysis.
# Only at the density step remove missing/nonfinite PSA. Eight cases with PSA
# but missing reported PSAD remain in recalculated density comparisons. The
# separate reported-PSAD diagnostic uses its 181 complete valid clinical triples.
# Flag |reported PSAD − PSA / clinical volume| > **0.01 ng/mL²** for review only
# (23 current cases). This diagnostic threshold is not documented source precision
# and a flag does not identify which clinical field, if any, is wrong.

# %%
import json
from itertools import combinations
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.ticker import FuncFormatter, NullFormatter

# Select one or more methods; order controls display order.
SELECTED_COMPARATORS = ['clinical', 'nnunet274', 'rumcnotprostatex609']
COMPARATORS = {
    'clinical': ('Manual ellipsoid', None, [1]),
    'nnunet274': ('External (multi-site)', None, [1]),
    'rumcnotprostatex609': ('Internal (RUMC)', None, [1]),
    'bosma22b': ('Bosma22b', 'whole_gland/AI/Bosma22b', [1]),
    'guerbet23': ('Guerbet23', 'whole_gland/AI/Guerbet23', [1]),
    'heviai23': ('HeviAI23', 'zonal_pz_tz/AI/HeviAI23', [1, 2]),
    'yuan23': ('Yuan23', 'zonal_pz_tz/AI/Yuan23', [1, 2]),
}
assert SELECTED_COMPARATORS, 'Select at least one comparator'
assert len(set(SELECTED_COMPARATORS)) == len(SELECTED_COMPARATORS), 'Repeated comparator'
assert set(SELECTED_COMPARATORS) <= COMPARATORS.keys(), 'Unknown comparator'

# Script execution resolves from this file; interactive cells support the root or notebooks/.
REPO_ROOT = Path(__file__).resolve().parents[1] if '__file__' in globals() else Path.cwd().resolve()
if REPO_ROOT.name == 'notebooks':
    REPO_ROOT = REPO_ROOT.parent
MASK_ROOT = REPO_ROOT / 'data/raw/github/prostatex_masks'
PICAI_ROOT = REPO_ROOT / 'data/raw/zenodo/picai/picai_labels'
INDEX_PATH = MASK_ROOT / 'index.csv'
MAPPING_PATH = PICAI_ROOT / 'additional_resources/ProstateX-mapping.json'
MARKSHEET_PATH = PICAI_ROOT / 'clinical_information/marksheet.csv'
OUTPUT_DIR = REPO_ROOT / 'results/prostatex_linked_volume'
EXPECTED_CASES = 190
PSAD_REPORTING_TOLERANCE = 0.01  # ng/mL²; diagnostic review threshold only
QC_NUMERICAL_TOLERANCE = 1e-12

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
# multiplication. PSA and reported PSAD do not affect volume-analysis eligibility.

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
links[['patient_id', 'study_id']] = links['picai_key'].str.split('_', expand=True).astype('int64')
unique_links = links.loc[~links['ID'].isin(ambiguous_ids)].copy()
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
assert len(linked) == EXPECTED_CASES, 'Cohort changed; inspect linkage audit before analysis'
assert not linked.duplicated(['patient_id', 'study_id']).any(), 'Repeated linked study'

# %% [markdown]
# ### Cohort flow from the PROSTATEx mapping
#
# Examination and unique-patient counts are distinct. Each exclusion is relative
# to the preceding stage. Anticipate PSA availability for the later density step
# without filtering the volume cohort here. Invalid PSA or density denominators
# will stop that step explicitly rather than silently remove additional cases.

# %%
mapped_masks = links.loc[links['ID'].isin(cases['ID'])]
unambiguous_masks = mapped_masks.loc[~mapped_masks['ID'].isin(ambiguous_ids)]
matched_marksheet = cohort.loc[cohort['picai_key'].notna() & cohort['marksheet_match'].eq('both')]
psa_available = np.isfinite(pd.to_numeric(linked['psa'], errors='coerce'))
flow_rows = []
previous_examinations = previous_patients = None
for stage, frame, reason in [
    ('PROSTATEx mapping JSON', links, ''),
    ('Local manual whole-gland mask available', mapped_masks, 'No local manual whole-gland mask'),
    ('Unambiguous linkage', unambiguous_masks, 'Ambiguous PROSTATEx mapping'),
    ('Matched clinical marksheet', matched_marksheet, 'No matching clinical marksheet'),
    ('Valid clinical volume: volume analysis', linked, 'Missing/nonfinite/nonpositive clinical volume'),
    ('PSA available: subsequent density analysis', linked.loc[psa_available], 'Missing/nonfinite PSA'),
]:
    n_examinations = len(frame)
    n_patients = frame['patient_id'].nunique()
    flow_rows.append(dict(
        stage=stage, n_examinations=n_examinations, n_patients=n_patients,
        excluded_examinations=previous_examinations - n_examinations if previous_examinations is not None else pd.NA,
        excluded_patients=previous_patients - n_patients if previous_patients is not None else pd.NA,
        exclusion_reason=reason, source='Local metadata',
    ))
    previous_examinations, previous_patients = n_examinations, n_patients
cohort_flow = pd.DataFrame(flow_rows)
cohort_flow.to_csv(OUTPUT_DIR / 'cohort_flow.csv', index=False)
display(cohort_flow)
print(f'Volume-analysis cohort: {len(linked)}; expected: {EXPECTED_CASES}')
display(cohort['selection_status'].value_counts().rename_axis('status').to_frame('n'))
display(linked[['ID', *marksheet.columns]])
display(linked[marksheet.columns].isna().sum().to_frame('missing_values'))

# %% [markdown]
# ## Clinical consistency flags for review only
#
# Recalculate PSA / clinical volume whenever PSA and volume are finite positive,
# even when reported PSAD is missing. Assess consistency only when reported PSAD
# is also finite nonnegative. Flag absolute residuals above the configured
# 0.01 ng/mL² threshold; equality passes, ignoring numerical roundoff. Missing or
# invalid values have separate assessment statuses. No assessment removes a case
# from volume or recalculated density analysis. Display suspicious cases after
# measuring manual-mask and selected comparator volumes.

# %%
clinical_qc = linked.copy()
qc_psa = pd.to_numeric(clinical_qc['psa'], errors='coerce')
qc_volume = pd.to_numeric(clinical_qc['clinical_volume_ml'], errors='coerce')
qc_psad = pd.to_numeric(clinical_qc['psad'], errors='coerce')
qc_missing = ~np.isfinite(qc_psa) | ~np.isfinite(qc_volume) | ~np.isfinite(qc_psad)
qc_invalid = qc_psa.le(0) | qc_volume.le(0) | qc_psad.lt(0)
qc_valid = ~qc_missing & ~qc_invalid
clinical_qc['clinical_qc_status'] = np.select(
    [qc_missing, qc_invalid], ['missing_or_nonfinite_clinical_values', 'invalid_clinical_values'],
    default='within_reporting_threshold',
)
valid_ratio_inputs = np.isfinite(qc_psa) & qc_psa.gt(0) & np.isfinite(qc_volume) & qc_volume.gt(0)
clinical_qc['recalculated_clinical_psad'] = (
    qc_psa / qc_volume.where(qc_volume.gt(0))
).where(valid_ratio_inputs)
clinical_qc['reported_minus_recalculated_psad'] = (
    qc_psad - clinical_qc['recalculated_clinical_psad']
).where(qc_valid)
clinical_qc['absolute_reporting_residual_psad'] = clinical_qc['reported_minus_recalculated_psad'].abs()
clinical_qc['reporting_residual_percent_of_recalculated'] = (
    100 * clinical_qc['reported_minus_recalculated_psad'] / clinical_qc['recalculated_clinical_psad']
)
clinical_qc['volume_implied_by_reported_psad_ml'] = (
    qc_psa / qc_psad.where(qc_psad.gt(0))
).where(qc_valid)
clinical_qc['implied_minus_clinical_volume_ml'] = (
    clinical_qc['volume_implied_by_reported_psad_ml'] - qc_volume
)
clinical_qc['reporting_tolerance_psad'] = PSAD_REPORTING_TOLERANCE
qc_residual = clinical_qc['absolute_reporting_residual_psad']
clinical_qc['beyond_reporting_tolerance'] = (
    qc_residual.gt(PSAD_REPORTING_TOLERANCE)
    & ~np.isclose(qc_residual, PSAD_REPORTING_TOLERANCE, rtol=0, atol=QC_NUMERICAL_TOLERANCE)
).astype('boolean').where(qc_valid)
clinical_qc.loc[
    clinical_qc['beyond_reporting_tolerance'].fillna(False), 'clinical_qc_status'
] = 'suspicious_reporting_residual'
clinical_qc.to_csv(OUTPUT_DIR / 'clinical_consistency_audit.csv', index=False)
display(clinical_qc['clinical_qc_status'].value_counts().rename_axis('assessment').to_frame('n'))

# %%
cohort = cohort.merge(
    clinical_qc[['ID', 'clinical_qc_status', 'beyond_reporting_tolerance']],
    on='ID', how='left', validate='one_to_one',
)
cohort.to_csv(OUTPUT_DIR / 'cohort_linkage_audit.csv', index=False)
linked = clinical_qc.copy()  # Measure every eligible volume case, irrespective of review flags.

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
# Clinical volume is reported in the marksheet and labeled "Manual ellipsoid"
# in this analysis. Dataset274 and Dataset609 predictions are matched by PROSTATEx
# ID and use whole-gland label 1. PI-CAI masks are matched by patient/study ID.
# Bosma22b and Guerbet23 use whole-gland label 1; HeviAI23 and Yuan23 use PZ=1 and TZ=2.
# Their union is measured in memory. No resampling or spatial evaluation is
# performed. Empty predictions remain zero-volume failures; invalid inputs stop.

# %%
MODEL_ROOTS = {
    'nnunet274': (REPO_ROOT / 'results/prostatex_pred_pooled2_3d_fullres', 'Dataset274_raw'),
    'rumcnotprostatex609': (
        REPO_ROOT / 'results/prostatex_pred_rumcnotprostatex609_3d_fullres', 'Dataset609_raw',
    ),
}
for key in SELECTED_COMPARATORS:
    if key == 'clinical':
        continue
    _, source, foreground = COMPARATORS[key]
    if key in MODEL_ROOTS:
        model_root, dataset_name = MODEL_ROOTS[key]
        assert json.loads((model_root / 'plans.json').read_text())['dataset_name'] == dataset_name, key
        assert json.loads((model_root / 'dataset.json').read_text())['labels'] == {'background': 0, 'class1': 1}, key
    prediction_rows = []
    for case in linked.itertuples(index=False):
        if key in MODEL_ROOTS:
            path = model_root / f'{case.ID}.nii.gz'
        else:
            filename = f'{int(case.patient_id)}_{int(case.study_id)}.nii.gz'
            path = PICAI_ROOT / 'anatomical_delineations' / source / filename
        image = nib.load(path)
        mask = np.asanyarray(image.dataobj)
        assert mask.ndim == 3 and np.isfinite(mask).all(), path
        assert image.header.get_xyzt_units()[0] == 'mm', path
        assert np.isin(mask, [0, *foreground]).all(), path
        voxel_count = int(np.count_nonzero(np.isin(mask, foreground)))
        assert voxel_count == sum(int(np.count_nonzero(mask == label)) for label in foreground)
        voxel_volume_mm3 = float(abs(np.linalg.det(image.affine[:3, :3])))
        assert np.isfinite(voxel_volume_mm3) and voxel_volume_mm3 > 0, path
        reference = nib.load(MASK_ROOT / case.label_wg)
        prediction_rows.append({
            'ID': case.ID,
            f'{key}_volume_ml': voxel_count * voxel_volume_mm3 / 1000,
            f'{key}_path': str(path.relative_to(REPO_ROOT)),
            f'{key}_foreground_labels': str(foreground),
            f'{key}_voxel_count': voxel_count,
            f'{key}_voxel_volume_mm3': voxel_volume_mm3,
            f'{key}_gt_shape_match': image.shape == reference.shape,
            f'{key}_gt_affine_max_abs_difference': float(np.abs(image.affine - reference.affine).max()),
        })
    volumes = volumes.merge(pd.DataFrame(prediction_rows), on='ID', validate='one_to_one')

# %% [markdown]
# ## Suspicious clinical cases retained in analysis
#
# Display every suspicious record in descending absolute reporting residual,
# with clinical and manual-mask volumes in mL and densities in ng/mL². Export
# all marksheet fields, paths, measurements, and comparator errors for review.
# All cases remain in volume analysis, including those missing PSA or reported
# PSAD. Ground-truth volume is shown for context and does not determine flags.

# %%
volumes_all = volumes.copy()
volumes_all['manual_mask_path'] = volumes_all['label_wg'].map(lambda path: str(MASK_ROOT / path))
volumes_all['mask_index_path'] = str(INDEX_PATH)
volumes_all['marksheet_path'] = str(MARKSHEET_PATH)
volumes_all['mapping_path'] = str(MAPPING_PATH)
for key in SELECTED_COMPARATORS:
    error = volumes_all[f'{key}_volume_ml'] - volumes_all['gt_volume_ml']
    volumes_all[f'{key}_minus_gt_ml'] = error
    volumes_all[f'{key}_absolute_error_ml'] = error.abs()
    volumes_all[f'{key}_absolute_percentage_error'] = 100 * error.abs() / volumes_all['gt_volume_ml']
volumes_all['manual_mask_psad'] = (
    pd.to_numeric(volumes_all['psa'], errors='coerce').where(
        np.isfinite(pd.to_numeric(volumes_all['psa'], errors='coerce'))
        & pd.to_numeric(volumes_all['psa'], errors='coerce').gt(0)
    )
    / volumes_all['gt_volume_ml']
)
volumes_all.to_csv(OUTPUT_DIR / 'comparator_volumes_all_cases.csv', index=False)
clinical_case_review = volumes_all.loc[
    volumes_all['beyond_reporting_tolerance'].fillna(False)
].sort_values('absolute_reporting_residual_psad', ascending=False).copy()
clinical_case_review['review_reason'] = 'Absolute reported-minus-recalculated PSAD exceeds review threshold'
clinical_case_review.to_csv(OUTPUT_DIR / 'clinical_case_review.csv', index=False)
suspicious_columns = [
    'ID', 'patient_id', 'study_id', 'psa', 'psad', 'clinical_volume_ml', 'gt_volume_ml',
    'recalculated_clinical_psad', 'reported_minus_recalculated_psad',
    'absolute_reporting_residual_psad', 'reporting_residual_percent_of_recalculated',
    'volume_implied_by_reported_psad_ml', 'implied_minus_clinical_volume_ml',
    'reporting_tolerance_psad',
]
clinical_suspicious_cases = clinical_case_review[suspicious_columns].copy()
clinical_suspicious_cases.to_csv(OUTPUT_DIR / 'clinical_suspicious_cases.csv', index=False)
print(f'Suspicious clinical cases: {len(clinical_case_review)}; all retained in volume analysis.')
print(clinical_suspicious_cases.to_string(index=False))
print('Detailed clinical case review, including masks and comparator errors (all rows and columns):')
print(clinical_case_review.to_string(index=False))
volumes = volumes_all.copy().reset_index(drop=True)
print(f'Volume-analysis cohort: {len(volumes)}; no clinical review flags excluded')
assert len(volumes) >= 2, 'Fewer than two eligible volume cases'

# %% [markdown]
# ## Volume accuracy against manual whole-gland ground truth
#
# **Manual ellipsoid** is the reported clinical volume. **External (multi-site)**
# identifies Dataset274 nnUNet. **Internal (RUMC)** identifies Dataset609 nnUNet
# predictions from the RUMC training cohort excluding PROSTATEx. Bosma22b remains
# available as an optional comparator under its own name.
#
# Positive signed errors indicate overestimation. Limits of agreement are bias
# ± 1.96 sample SD, not confidence intervals. Rank by MAE on this linked cohort;
# volume accuracy alone does not establish spatial segmentation quality.
# Pearson's r uses the original volumes, before the logarithmic axis display.

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
        pearson_r=value.corr(gt),
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
FIGURE_SIZE = (9.5, 8.5)
VOLUME_TICKS = [10, 30, 100, 300]
RELATIVE_DIFFERENCE_LIMITS = (-90, 90)

# Keep each method's color, marker, and histogram line style together.
PLOT_STYLES = {
    'clinical': dict(color='#4C6A85', marker='o', linestyle='-'),
    'nnunet274': dict(color='#B88254', marker='s', linestyle='--'),
    'rumcnotprostatex609': dict(color='#91A78C', marker='^', linestyle=':'),
    'bosma22b': dict(color='#7D8995', marker='X', linestyle=(0, (1, 1))),
    'guerbet23': dict(color='#806A88', marker='D', linestyle='-.'),
    'heviai23': dict(color='#AD7773', marker='v', linestyle=(0, (5, 1, 1, 1))),
    'yuan23': dict(color='#C2B18A', marker='P', linestyle=(0, (3, 1, 1, 1, 1, 1))),
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
axes[0, -1].set_axis_off()

volume_values = volumes[['gt_volume_ml', *volume_columns]].to_numpy()
lower = min(volume_values[volume_values > 0].min() * 0.95, 10)
upper = max(volume_values.max() * 1.05, 300)
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
        f'{name}\n− manual ground truth (mL)',
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
            'Manual ground-truth volume\n(mL; log scale)',
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

fig.savefig(OUTPUT_DIR / 'comparator_agreement.png', dpi=300)
fig.savefig(OUTPUT_DIR / 'comparator_agreement.pdf')
plt.show()

# %% [markdown]
# ## Paired absolute-error comparisons
#
# Positive improvement means method A is closer than method B. Exactly equal
# absolute errors are ties. All methods use the complete volume-analysis cohort.

# %%
pair_tables = []
count_rows = []
for a, b in combinations(SELECTED_COMPARATORS, 2):
    pair = volumes[['ID', 'patient_id', 'study_id']].copy()
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
# ## Paired post hoc inference: RMSE and probability of being closer
#
# These are two different questions: a method can have lower population RMSE
# without being closer on most patients. Positive RMSE improvement = RMSE(B) −
# RMSE(A), in mL, favors A. Equality of population RMSE is equivalent to zero mean
# paired squared-error difference, d = error(B)² − error(A)² (mL²).
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
# representative of the population of interest. This selected linked cohort is
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
assert volumes['patient_id'].is_unique, 'Patient-level resampling requires unique patients'


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
        n=n, rmse_a_ml=float(np.sqrt(np.mean(error_a**2))),
        rmse_b_ml=float(np.sqrt(np.mean(error_b**2))), rmse_improvement_ml=improvement,
        rmse_improvement_ci_lower_ml=float(ci_lower), rmse_improvement_ci_upper_ml=float(ci_upper),
        rmse_ci_method=ci_method, mean_squared_error_improvement_ml2=d_mean,
        rmse_test_method=test_method, rmse_bootstrap_p=bootstrap_pvalue,
        rmse_t_sensitivity_p=t_pvalue, rmse_t_sensitivity_status=t_status,
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
    for endpoint, p_column, effect_column in [
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
# ## PSA density: calculations and continuous summaries
#
# PSA density = PSA (ng/mL) / whole-gland volume (mL), in **ng/mL²**.
# Use the same PSA for every volume measurement. The reference is PSA divided by
# manual-mask volume; it is a derived reference, not an independent PSA assay.
# Recalculate densities rather than substituting the marksheet's reported `psad`.
# A separate analysis at the end examines that reported value.
#
# Retain all volume-analysis cases in this audit, then remove missing/nonfinite
# PSA only (currently ProstateX-0202), giving 189 density cases. Reported PSAD and
# review flags do not affect eligibility. Finite PSA must be positive and reference
# and selected-method volumes must be finite positive; unexpected invalid values
# stop execution rather than silently shrink the cohort. These requirements allow
# relative errors and log axes. Nothing is imputed or removed from volume analysis.

# %%
PSAD_UNIT = 'ng/mL²'
PSAD_BINS = [0, 0.10, 0.15, 0.20, np.inf]
PSAD_STRATA = ['0–<0.10', '0.10–<0.15', '0.15–<0.20', '≥0.20']
psad_audit = volumes.copy()
psad_psa = pd.to_numeric(psad_audit['psa'], errors='coerce')
psad_audit['psa_numeric_ng_ml'] = psad_psa
psad_audit['psa_density_status'] = np.select(
    [~np.isfinite(psad_psa), psad_psa.lt(0), psad_psa.eq(0)],
    ['missing_or_nonfinite_psa', 'negative_psa', 'zero_psa'], default='included',
)
for key in ['gt', *SELECTED_COMPARATORS]:
    value = pd.to_numeric(psad_audit[f'{key}_volume_ml'], errors='coerce')
    valid_volume = np.isfinite(value) & value.gt(0)
    psad_audit[f'{key}_psad'] = (
        psad_psa.where(np.isfinite(psad_psa) & psad_psa.ge(0)) / value.where(valid_volume)
    )
    psad_audit[f'{key}_valid_psad_volume'] = valid_volume
    invalid = ~valid_volume | ~np.isfinite(psad_audit[f'{key}_psad'])
    psad_audit.loc[
        invalid & psad_audit['psa_density_status'].eq('included'), 'psa_density_status'
    ] = f'invalid_{key}_density_or_volume'

psad_audit.to_csv(OUTPUT_DIR / 'psa_density_cohort_audit.csv', index=False)
display(psad_audit['psa_density_status'].value_counts().rename_axis('status').to_frame('n'))
psa_present = np.isfinite(psad_psa)
assert psad_psa.loc[psa_present].gt(0).all(), 'Nonpositive PSA; inspect the density audit'
for key in ['gt', *SELECTED_COMPARATORS]:
    assert psad_audit.loc[psa_present, f'{key}_valid_psad_volume'].all(), (
        f'Invalid {key} volume for PSA-density analysis; inspect the density audit'
    )
    assert np.isfinite(psad_audit.loc[psa_present, f'{key}_psad']).all(), (
        f'Nonfinite {key} PSA density; inspect the density audit'
    )
psad_cases = psad_audit.loc[psa_present].copy()
print(f'PSA-density comparisons: {len(psad_cases)} / {len(psad_audit)} cases')
assert len(psad_cases) >= 2, 'Fewer than two eligible PSA-density cases; inspect the audit'
assert psad_cases['patient_id'].is_unique, 'Patient-level resampling requires unique patients'
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
        ax.set_xlabel(f'Manual-mask PSA density\n({PSAD_UNIT}; log scale)', fontsize=8)
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
fig.savefig(OUTPUT_DIR / 'psa_density_agreement.png', dpi=300)
fig.savefig(OUTPUT_DIR / 'psa_density_agreement.pdf')
plt.show()

# %% [markdown]
# ## PSA density: strata and threshold statistics
#
# Use unrounded densities and left-closed strata: **[0, 0.10), [0.10, 0.15),
# [0.15, 0.20), [0.20, ∞)**. Confusion-matrix rows are manual-mask reference
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
    ax.set_ylabel('Manual-mask reference stratum')
    ax.set_title(f'{COMPARATORS[key][0]}\nn = {len(psad_cases)}', fontsize=10)
fig.colorbar(heatmap, ax=list(axes.flat), label='Proportion within reference stratum', shrink=.8)
fig.savefig(OUTPUT_DIR / 'psa_density_confusion.png', dpi=300)
fig.savefig(OUTPUT_DIR / 'psa_density_confusion.pdf')
plt.show()

# %% [markdown]
# ## PSA density: paired stratum-accuracy comparisons
#
# Which method assigns more patients to the same stratum as the manual-mask
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
    pair = psad_cases[['ID', 'patient_id', 'study_id']].copy()
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
    pd.DataFrame(columns=['ID', 'patient_id', 'study_id', 'method_a', 'method_b',
                          'absolute_error_improvement_psad', 'a_vs_b']).to_csv(
        OUTPUT_DIR / 'psa_density_pairwise_per_case.csv', index=False,
    )
    pd.DataFrame(columns=['method_a', 'method_b', 'outcome', 'n', 'percent']).to_csv(
        OUTPUT_DIR / 'psa_density_pairwise_counts.csv', index=False,
    )

# %% [markdown]
# ## PSA density: paired post hoc inference: RMSE and probability of being closer
#
# These are two different questions: a method can have lower population RMSE
# without being closer on most patients. Positive RMSE improvement = RMSE(B) −
# RMSE(A), in ng/mL², favors A. Equality of population RMSE is equivalent to zero mean
# paired squared-error difference, d = error(B)² − error(A)² ((ng/mL²)²).
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
assert psad_cases['patient_id'].is_unique, 'Patient-level resampling requires unique patients'


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
        n=n, rmse_a_psad=float(np.sqrt(np.mean(error_a**2))),
        rmse_b_psad=float(np.sqrt(np.mean(error_b**2))), rmse_improvement_psad=improvement,
        rmse_improvement_ci_lower_psad=float(ci_lower), rmse_improvement_ci_upper_psad=float(ci_upper),
        rmse_ci_method=ci_method, mean_squared_error_improvement_psad2=d_mean,
        rmse_test_method=test_method, rmse_bootstrap_p=bootstrap_pvalue,
        rmse_t_sensitivity_p=t_pvalue, rmse_t_sensitivity_status=t_status,
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
    for endpoint, p_column, effect_column in [
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
# ## Reported marksheet PSA density: reporting and volume components
#
# Examine the marksheet's **reported `psad`** separately, using its own complete
# subset of the complete volume-analysis cohort, including suspicious cases,
# independent of selected prediction methods. This is a reporting diagnostic,
# requiring reported PSAD only for these comparisons. Compare it with both
# **PSA / clinical volume** (arithmetic consistency) and **PSA / manual-mask volume**
# (total disagreement). Also compare recalculated clinical density with manual-mask
# density on these same cases so missing reported values cannot confound that comparison.
#
# For each case the signed differences satisfy the exact arithmetic identity:
#
# reported − manual-mask = (reported − recalculated clinical)
#                         + (recalculated clinical − manual-mask).
#
# The first component is a **reporting residual**, not proof of a transcription
# error: the source PSA, volume, and density may have different rounding or may
# have been recorded from different measurements. The second is the density
# difference attributable to using clinical rather than manual-mask volume for
# the same PSA. The decomposition is arithmetic, not causal attribution.
#
# Flag absolute reporting residuals **> 0.01 ng/mL²** as a diagnostic, using
# the same configured review threshold as the early consistency assessment,
# not a known precision shared by all reported values. Rounded PSA
# and volume can also produce residuals beyond this band. Inspect signed/relative
# residuals, implied volumes, and the largest discrepancies rather than labeling
# every flag an error. Reported zero density remains valid; implied volume is
# undefined for it, and it cannot appear on a logarithmic density axis.

# %%
reported_psad_audit = volumes_all.copy()
reported_psa = pd.to_numeric(reported_psad_audit['psa'], errors='coerce')
reported_value = pd.to_numeric(reported_psad_audit['psad'], errors='coerce')
reported_psad_audit['reported_psad'] = reported_value
reported_psad_audit['reported_psad_status'] = np.select(
    [~np.isfinite(reported_psa), reported_psa.le(0),
     ~np.isfinite(reported_value), reported_value.lt(0)],
    ['missing_or_nonfinite_psa', 'nonpositive_psa',
     'missing_or_nonfinite_reported_psad', 'negative_reported_psad'], default='included',
)
for source, density_column in [
    ('clinical', 'recalculated_clinical_psad'), ('gt', 'manual_mask_psad'),
]:
    value = pd.to_numeric(reported_psad_audit[f'{source}_volume_ml'], errors='coerce')
    valid_volume = np.isfinite(value) & value.gt(0)
    density = reported_psa.where(np.isfinite(reported_psa) & reported_psa.ge(0)) / value.where(valid_volume)
    reported_psad_audit[density_column] = density
    reported_psad_audit.loc[
        (~valid_volume | ~np.isfinite(density)) & reported_psad_audit['reported_psad_status'].eq('included'),
        'reported_psad_status',
    ] = f'invalid_{source}_density_or_volume'
reporting_included = reported_psad_audit['reported_psad_status'].eq('included')
reported_psad_audit['reported_minus_recalculated_psad'] = (
    reported_value - reported_psad_audit['recalculated_clinical_psad']
).where(reporting_included)
reported_psad_audit['recalculated_minus_manual_mask_psad'] = (
    reported_psad_audit['recalculated_clinical_psad'] - reported_psad_audit['manual_mask_psad']
).where(reporting_included)
reported_psad_audit['reported_minus_manual_mask_psad'] = (
    reported_value - reported_psad_audit['manual_mask_psad']
).where(reporting_included)
reporting_residual = reported_psad_audit['reported_minus_recalculated_psad']
reported_psad_audit['absolute_reporting_residual_psad'] = reporting_residual.abs()
reported_psad_audit['reporting_change_in_absolute_manual_mask_error_psad'] = (
    reported_psad_audit['reported_minus_manual_mask_psad'].abs()
    - reported_psad_audit['recalculated_minus_manual_mask_psad'].abs()
)
reported_psad_audit['reporting_residual_percent_of_recalculated'] = (
    100 * reporting_residual / reported_psad_audit['recalculated_clinical_psad'].where(
        reported_psad_audit['recalculated_clinical_psad'].gt(0),
    )
)
# Ignore only numerical roundoff at the diagnostic tolerance itself.
reported_psad_audit['beyond_reporting_tolerance'] = (
    reporting_residual.abs().gt(PSAD_REPORTING_TOLERANCE)
    & ~np.isclose(reporting_residual.abs(), PSAD_REPORTING_TOLERANCE, rtol=0, atol=QC_NUMERICAL_TOLERANCE)
).astype('boolean').where(reporting_included)
reported_psad_audit['volume_implied_by_reported_psad_ml'] = (
    reported_psa / reported_value.where(reported_value.gt(0))
).where(reporting_included)
reported_psad_audit['implied_minus_clinical_volume_ml'] = (
    reported_psad_audit['volume_implied_by_reported_psad_ml'] - reported_psad_audit['clinical_volume_ml']
)
reported_psad_audit.to_csv(OUTPUT_DIR / 'psa_density_reported_audit.csv', index=False)
reported_psad_cases = reported_psad_audit.loc[reporting_included].copy()
display(reported_psad_audit['reported_psad_status'].value_counts().rename_axis('status').to_frame('n'))
print(f'Reported-density comparisons: {len(reported_psad_cases)} / {len(reported_psad_audit)} cases')
np.testing.assert_allclose(
    reported_psad_cases['reported_minus_manual_mask_psad'],
    reported_psad_cases['reported_minus_recalculated_psad']
    + reported_psad_cases['recalculated_minus_manual_mask_psad'], rtol=1e-12, atol=1e-12,
)
for column in ['reported_psad', 'recalculated_clinical_psad', 'manual_mask_psad']:
    reported_psad_cases[f'{column}_stratum'] = pd.cut(
        reported_psad_cases[column], PSAD_BINS, labels=PSAD_STRATA, right=False,
    )
reported_psad_cases['reporting_changes_stratum'] = (
    reported_psad_cases['reported_psad_stratum'] != reported_psad_cases['recalculated_clinical_psad_stratum']
)
reported_psad_cases['clinical_volume_changes_stratum'] = (
    reported_psad_cases['recalculated_clinical_psad_stratum'] != reported_psad_cases['manual_mask_psad_stratum']
)
reported_psad_cases['reported_differs_from_manual_mask_stratum'] = (
    reported_psad_cases['reported_psad_stratum'] != reported_psad_cases['manual_mask_psad_stratum']
)
reported_psad_cases.to_csv(OUTPUT_DIR / 'psa_density_reported_per_case.csv', index=False)

# %%
REPORTING_COMPARISONS = [
    ('reported_vs_recalculated', 'reported_psad', 'recalculated_clinical_psad',
     'Reported', 'Recalculated clinical'),
    ('reported_vs_manual_mask', 'reported_psad', 'manual_mask_psad',
     'Reported', 'Manual-mask reference'),
    ('recalculated_vs_manual_mask', 'recalculated_clinical_psad', 'manual_mask_psad',
     'Recalculated clinical', 'Manual-mask reference'),
]
reported_summary_rows, reported_confusion_rows = [], []
for comparison, estimate_column, reference_column, estimate_name, reference_name in REPORTING_COMPARISONS:
    estimate, reference = reported_psad_cases[estimate_column], reported_psad_cases[reference_column]
    error = estimate - reference
    relative = 100 * error / reference
    bias, sd = error.mean(), error.std(ddof=1)
    reported_summary_rows.append(dict(
        comparison=comparison, estimate=estimate_name, reference=reference_name,
        n=len(error), density_unit=PSAD_UNIT, bias_psad=bias, mae_psad=error.abs().mean(),
        rmse_psad=np.sqrt((error**2).mean()),
        pearson_r=estimate.corr(reference) if estimate.nunique() > 1 and reference.nunique() > 1 else np.nan,
        mean_absolute_percentage_error=relative.abs().mean(),
        loa_lower_psad=bias - 1.96 * sd, loa_upper_psad=bias + 1.96 * sd,
        stratum_accuracy=reported_psad_cases[f'{estimate_column}_stratum'].eq(
            reported_psad_cases[f'{reference_column}_stratum'],
        ).mean(),
    ))
    if reported_psad_cases.empty:
        matrix = np.zeros((4, 4), dtype=int)
    else:
        matrix = pd.crosstab(
            reported_psad_cases[f'{reference_column}_stratum'],
            reported_psad_cases[f'{estimate_column}_stratum'], dropna=False,
        ).reindex(index=PSAD_STRATA, columns=PSAD_STRATA, fill_value=0).to_numpy(dtype=int)
    assert matrix.sum() == len(reported_psad_cases)
    for i, reference_stratum in enumerate(PSAD_STRATA):
        support = int(matrix[i].sum())
        for j, estimated_stratum in enumerate(PSAD_STRATA):
            reported_confusion_rows.append(dict(
                comparison=comparison, reference_stratum=reference_stratum,
                estimated_stratum=estimated_stratum, n=int(matrix[i, j]), reference_support=support,
                row_percent=100 * matrix[i, j] / support if support else np.nan,
            ))
reported_psad_summary = pd.DataFrame(reported_summary_rows)
reported_psad_confusion = pd.DataFrame(reported_confusion_rows)
reported_psad_summary.to_csv(OUTPUT_DIR / 'psa_density_reported_summary.csv', index=False)
reported_psad_confusion.to_csv(OUTPUT_DIR / 'psa_density_reported_confusion.csv', index=False)
reported_n = len(reported_psad_cases)
reported_diagnostic_rows = []
for column in ['beyond_reporting_tolerance', 'reporting_changes_stratum',
               'clinical_volume_changes_stratum', 'reported_differs_from_manual_mask_stratum']:
    count = int(reported_psad_cases[column].sum())
    reported_diagnostic_rows.append(dict(
        diagnostic=column, count=count, n=reported_n,
        proportion=count / reported_n if reported_n else np.nan,
        reporting_tolerance_psad=PSAD_REPORTING_TOLERANCE, density_unit=PSAD_UNIT,
    ))
reported_psad_diagnostics = pd.DataFrame(reported_diagnostic_rows)
reported_psad_diagnostics.to_csv(OUTPUT_DIR / 'psa_density_reported_diagnostics.csv', index=False)
with pd.option_context('display.max_columns', None):
    display(reported_psad_summary)
    display(reported_psad_diagnostics)
    display(reported_psad_cases.sort_values('absolute_reporting_residual_psad', ascending=False)[[
        'ID', 'psa', 'psad', 'clinical_volume_ml', 'gt_volume_ml',
        'recalculated_clinical_psad', 'manual_mask_psad', 'reported_minus_recalculated_psad',
        'recalculated_minus_manual_mask_psad', 'reported_minus_manual_mask_psad',
        'reporting_change_in_absolute_manual_mask_error_psad',
        'beyond_reporting_tolerance', 'volume_implied_by_reported_psad_ml',
    ]])

# %% [markdown]
# ### Reported-density agreement and propagation diagnostic
#
# All three agreement columns use the same complete cases. In the final scatter,
# x is the clinical-volume component and y is total reported-minus-manual-mask
# disagreement. Vertical displacement from the identity line equals the reporting
# residual. Color shows that residual with a symmetric scale. The ±0.01 band in
# the reporting-residual plot is a diagnostic review threshold only.
# The exported change in absolute manual-mask error is positive when reporting
# increases that error and negative when the reporting residual offsets some of
# the clinical-volume disagreement. Neither sign establishes the reporting cause.

# %%
if reported_n:
    report_values = reported_psad_cases[['reported_psad', 'recalculated_clinical_psad', 'manual_mask_psad']].to_numpy()
    report_lower, report_upper = report_values[report_values > 0].min() * .85, report_values.max() * 1.15
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.7), constrained_layout=True)
    for column, (comparison, estimate_column, reference_column, estimate_name, reference_name) in enumerate(REPORTING_COMPARISONS):
        estimate = reported_psad_cases[estimate_column]
        reference = reported_psad_cases[reference_column]
        metrics = reported_psad_summary.set_index('comparison').loc[comparison]
        ax = axes[0, column]
        ax.scatter(reference, estimate, s=16, alpha=.6, color='#4C6A85', edgecolors='none', linewidths=0)
        ax.plot([report_lower, report_upper], [report_lower, report_upper], 'k--', linewidth=.8)
        ax.set(xscale='log', yscale='log', xlim=(report_lower, report_upper), ylim=(report_lower, report_upper),
               ylabel=f'{estimate_name}\ndensity ({PSAD_UNIT})')
        ax.set_title(f'{estimate_name}\nvs {reference_name}\nn = {reported_n}; r = {metrics.pearson_r:.3f}', fontsize=10)
        ax = axes[1, column]
        ax.scatter(reference, estimate - reference, s=16, alpha=.6, color='#4C6A85', edgecolors='none', linewidths=0,
                   label=f'MAE = {metrics.mae_psad:.3g}\nRMSE = {metrics.rmse_psad:.3g}')
        for level, linestyle in [(metrics.bias_psad, '-'), (metrics.loa_lower_psad, '--'), (metrics.loa_upper_psad, '--')]:
            ax.axhline(level, color='#4C6A85', linestyle=linestyle, linewidth=.8)
        if column == 0:
            ax.axhspan(-PSAD_REPORTING_TOLERANCE, PSAD_REPORTING_TOLERANCE,
                       color='#91A78C', alpha=.2, label=f'±{PSAD_REPORTING_TOLERANCE:g} diagnostic band')
        ax.set(xscale='log', xlim=(report_lower, report_upper), ylabel=f'Estimate − reference\n({PSAD_UNIT})')
        ax.legend(fontsize=8)
        for row in range(2):
            axes[row, column].set_xlabel(f'{reference_name} density\n({PSAD_UNIT}; log scale)', fontsize=9)
            axes[row, column].yaxis.label.set_fontsize(9)
            axes[row, column].tick_params(labelsize=8)
    fig.savefig(OUTPUT_DIR / 'psa_density_reported_agreement.png', dpi=300)
    fig.savefig(OUTPUT_DIR / 'psa_density_reported_agreement.pdf')
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    residual = reported_psad_cases['reported_minus_recalculated_psad']
    axes[0].hist(residual, bins='fd', color='#4C6A85', edgecolor='white')
    for level in [-PSAD_REPORTING_TOLERANCE, PSAD_REPORTING_TOLERANCE]:
        axes[0].axvline(level, color='black', linestyle='--', linewidth=.8)
    axes[0].set(xlabel=f'Reported − recalculated clinical ({PSAD_UNIT})', ylabel='Cases',
                title='Reporting-residual distribution')
    x = reported_psad_cases['recalculated_minus_manual_mask_psad']
    y = reported_psad_cases['reported_minus_manual_mask_psad']
    color_limit = max(residual.abs().max(), PSAD_REPORTING_TOLERANCE)
    scatter = axes[1].scatter(x, y, c=residual, cmap='coolwarm', vmin=-color_limit, vmax=color_limit,
                              s=20, edgecolors='none', linewidths=0, alpha=.6)
    lower, upper = min(x.min(), y.min()), max(x.max(), y.max())
    if lower == upper:
        lower, upper = lower - PSAD_REPORTING_TOLERANCE, upper + PSAD_REPORTING_TOLERANCE
    axes[1].plot([lower, upper], [lower, upper], 'k--', linewidth=.8)
    axes[1].set(xlabel=f'Recalculated clinical − manual-mask ({PSAD_UNIT})',
                ylabel=f'Reported − manual-mask ({PSAD_UNIT})', title='Total vs clinical-volume component')
    fig.colorbar(scatter, ax=axes[1], label=f'Reporting residual ({PSAD_UNIT})')
    fig.savefig(OUTPUT_DIR / 'psa_density_reported_components.png', dpi=300)
    fig.savefig(OUTPUT_DIR / 'psa_density_reported_components.pdf')
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 4.3), constrained_layout=True)
    for ax, (comparison, _, _, estimate_name, reference_name) in zip(axes, REPORTING_COMPARISONS):
        matrix = reported_psad_confusion.loc[
            reported_psad_confusion['comparison'].eq(comparison), 'n'
        ].to_numpy().reshape(4, 4)
        support = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, support, out=np.full(matrix.shape, np.nan), where=support > 0)
        heatmap = ax.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
        for i, j in np.ndindex(matrix.shape):
            percent = f'{normalized[i, j]:.1%}' if support[i, 0] else '—'
            ax.text(j, i, f'{matrix[i, j]}\n{percent}', ha='center', va='center', fontsize=8,
                    color='white' if normalized[i, j] > .5 else 'black')
        ax.set_xticks(range(4), labels=PSAD_STRATA, rotation=35, ha='right', fontsize=8)
        ax.set_yticks(range(4), labels=PSAD_STRATA, fontsize=8)
        ax.set_xlabel(f'{estimate_name} stratum', fontsize=9)
        ax.set_ylabel(f'{reference_name} stratum', fontsize=9)
        ax.set_title(f'{estimate_name}\nvs {reference_name}\nn = {reported_n}', fontsize=10)
    fig.colorbar(heatmap, ax=list(axes), label='Proportion within reference stratum', shrink=.8)
    fig.savefig(OUTPUT_DIR / 'psa_density_reported_confusion.png', dpi=300)
    fig.savefig(OUTPUT_DIR / 'psa_density_reported_confusion.pdf')
    plt.show()
else:
    print('No complete reported-density cases: reporting figures were not generated.')

# %% [markdown]
# ## PI-RADS 3: biopsy decisions against clinically significant cancer
#
# The July 2026 labels release adds comma-separated `lesion_PIRADS` scores.
# Define a case's PI-RADS as its maximum numeric lesion score and select maximum
# **3**, excluding cases with a coexisting PI-RADS 4/5 lesion. Ignore missing and
# `N/A` entries; do not assign a numeric score when none is available. Scores mix
# PI-RADS v1/2/2.1 according to the guidelines used at examination. Fall back to
# the entire eligible linked cohort only when the column is absent, never when
# the selected subgroup is empty. Current local data yield 26 eligible cases:
# 5 with `case_ISUP >= 2` (csPCa) and 21 with `case_ISUP <= 1`.
#
# Simulate recommending biopsy at **unrounded PSA density >= 0.15 ng/mL²**;
# equality recommends biopsy. Evaluate manual-mask density and every selected
# comparator using the same PSA and patients. The outcome reference is case-level
# ISUP, including grade 0; manual-mask density is another evaluated measurement.
# Reported marksheet density is not used. Retain clinical review flags. Audit all
# volume cases, including missing PSA, unavailable scores, and invalid outcomes.
#
# This is a retrospective threshold simulation, not observed biopsy management.
# The source dataset uses convenience sampling, mixes PI-RADS versions, and can
# include MRI-confirmed negatives without tissue sampling or follow-up. Results
# describe this selected cohort, not routine-care performance or prevalence.

# %%
BIOPSY_PSAD_THRESHOLD = 0.15  # ng/mL²; compare unrounded calculated values
biopsy_methods = ['gt', *SELECTED_COMPARATORS]
biopsy_names = {'gt': 'Manual mask', **{key: COMPARATORS[key][0] for key in SELECTED_COMPARATORS}}
biopsy_audit = psad_audit.copy()
biopsy_audit['case_max_pirads'] = np.nan
if 'lesion_PIRADS' in biopsy_audit:
    biopsy_cohort = 'maximum_PI-RADS_3'
    for index, row in biopsy_audit.iterrows():
        raw_scores = row['lesion_PIRADS']
        scores = []
        for token in ([] if pd.isna(raw_scores) else str(raw_scores).split(',')):
            token = token.strip()
            if not token or token.upper() == 'N/A':
                continue
            try:
                score = float(token)
            except ValueError as error:
                raise ValueError(f'{row.ID}: invalid lesion_PIRADS token {token!r}') from error
            if not np.isfinite(score) or score not in [1, 2, 3, 4, 5]:
                raise ValueError(f'{row.ID}: lesion_PIRADS score outside integers 1–5: {token!r}')
            scores.append(int(score))
        if scores:
            biopsy_audit.loc[index, 'case_max_pirads'] = max(scores)
    biopsy_selected = biopsy_audit['case_max_pirads'].eq(3)
    biopsy_audit['biopsy_pirads_status'] = np.select(
        [biopsy_audit['case_max_pirads'].isna(), biopsy_selected],
        ['no_numeric_pirads', 'maximum_pirads_3'], default='outside_pirads_3',
    )
else:
    biopsy_cohort = 'entire_linked_cohort_PI-RADS_unavailable'
    biopsy_selected = pd.Series(True, index=biopsy_audit.index)
    biopsy_audit['biopsy_pirads_status'] = 'pirads_column_absent'

biopsy_isup_raw = biopsy_audit['case_ISUP']
biopsy_isup = pd.to_numeric(biopsy_isup_raw, errors='coerce')
biopsy_isup_missing = (
    biopsy_isup_raw.isna() | biopsy_isup_raw.astype(str).str.strip().str.upper().isin(['', 'N/A'])
)
biopsy_isup_valid = np.isfinite(biopsy_isup) & biopsy_isup.isin([0, 1, 2, 3, 4, 5])
biopsy_audit['case_isup_numeric'] = biopsy_isup
biopsy_audit['biopsy_outcome_status'] = np.select(
    [biopsy_isup_missing, ~biopsy_isup_valid],
    ['missing_case_isup', 'invalid_case_isup'], default='valid_case_isup',
)
biopsy_audit['case_cspca'] = biopsy_isup.ge(2).astype('boolean').where(biopsy_isup_valid)
biopsy_audit['biopsy_cohort'] = biopsy_cohort
biopsy_audit['biopsy_cutoff_psad'] = BIOPSY_PSAD_THRESHOLD
biopsy_audit['biopsy_analysis_status'] = np.select(
    [~biopsy_selected, biopsy_audit['psa_density_status'].ne('included'), ~biopsy_isup_valid],
    ['outside_selected_subgroup', biopsy_audit['psa_density_status'], biopsy_audit['biopsy_outcome_status']],
    default='included',
)
biopsy_audit.to_csv(OUTPUT_DIR / 'psa_density_biopsy_audit.csv', index=False)
biopsy_cases = biopsy_audit.loc[biopsy_audit['biopsy_analysis_status'].eq('included')].copy()
assert biopsy_cases['patient_id'].is_unique, 'Biopsy analysis requires unique patients'
biopsy_truth = biopsy_cases['case_cspca'].astype(bool)
biopsy_n = len(biopsy_cases)
biopsy_positive_n = int(biopsy_truth.sum())
biopsy_negative_n = biopsy_n - biopsy_positive_n
print(f'Biopsy cohort: {biopsy_cohort}; n = {biopsy_n}; '
      f'csPCa = {biopsy_positive_n}; no csPCa = {biopsy_negative_n}')
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
        **biopsy_metadata, n_cspca=biopsy_positive_n, n_non_cspca=biopsy_negative_n,
        tp=tp, fp=fp, fn=fn, tn=tn,
        biopsies_recommended_n=tp + fp, biopsies_avoided_n=tn + fn,
        cspca_detected_n=tp, cspca_missed_n=fn,
        sensitivity=tp / (tp + fn) if tp + fn else np.nan,
        specificity=tn / (tn + fp) if tn + fp else np.nan,
        ppv=tp / (tp + fp) if tp + fp else np.nan,
        npv=tn / (tn + fn) if tn + fn else np.nan,
        accuracy=(tp + tn) / biopsy_n if biopsy_n else np.nan,
        biopsies_recommended_fraction=(tp + fp) / biopsy_n if biopsy_n else np.nan,
        biopsies_avoided_fraction=(tn + fn) / biopsy_n if biopsy_n else np.nan,
        cspca_missed_fraction=fn / biopsy_positive_n if biopsy_positive_n else np.nan,
    ))
    for truth, decision, outcome, count in [
        (False, False, 'TN', tn), (False, True, 'FP', fp),
        (True, False, 'FN', fn), (True, True, 'TP', tp),
    ]:
        support = biopsy_positive_n if truth else biopsy_negative_n
        biopsy_confusion_rows.append(dict(
            **biopsy_metadata, case_cspca=truth, biopsy_recommended=decision,
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
# and missed-cancer fraction use csPCa cases as denominator; specificity uses
# non-csPCa cases. PPV/NPV use recommended/avoided biopsies, respectively. Biopsy
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
    ax.set_yticks([0, 1], labels=['No csPCa\n(ISUP 0–1)', 'csPCa\n(ISUP 2–5)'], fontsize=8)
    ax.set_xlabel('Biopsy decision', fontsize=9)
    ax.set_ylabel('Case outcome', fontsize=9)
    ax.set_title(f'{biopsy_names[key]}\nn = {biopsy_n}', fontsize=10)
fig.suptitle(f'{biopsy_cohort.replace("_", " ")} | PSAD ≥ {BIOPSY_PSAD_THRESHOLD:g} {PSAD_UNIT}', fontsize=11)
fig.colorbar(heatmap, ax=list(axes.ravel()), label='Proportion within outcome row', shrink=.8)
fig.savefig(OUTPUT_DIR / 'psa_density_biopsy_confusion.png', dpi=300)
fig.savefig(OUTPUT_DIR / 'psa_density_biopsy_confusion.pdf')
plt.show()
