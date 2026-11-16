# %% [markdown]
# # Toronto JDMI clinical ellipsoid volume tolerance
#
# Assess whether reported clinical prostate volumes are consistent with the
# recorded dimensions, allowing ±0.1 cm per dimension and a volume coefficient
# between 0.5 and 5π/24. The cohort starts from the raw full clinical listing and
# excludes every accession in the shared `data_internal/hlab20260928/exclude.json`.
# This is a consistency screen; flagged rows need review and are not proof of an
# error. Outputs, including a proposed exclusion list, are written locally under
# `results/toronto_jdmi_ellipsoid_tolerance/`. The shared exclusion file is input
# only and is never modified.

# %%
from getpass import getpass
from io import BytesIO
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
import msoffcrypto
import numpy as np
import pandas as pd
from IPython.display import display

REPO_ROOT = Path(__file__).resolve().parents[1] if '__file__' in globals() else Path.cwd().resolve()
if REPO_ROOT.name == 'notebooks':
    REPO_ROOT = REPO_ROOT.parent
assert (REPO_ROOT / 'pyproject.toml').is_file(), 'Set REPO_ROOT to the repository root'
DATA_ROOT = REPO_ROOT / 'data_internal/hlab20260928'
CLINICAL_CSV = DATA_ROOT / 'hlab20260928_t2_image_listing_full_clinical.csv'
WORKBOOK = DATA_ROOT / 'PROSATELLMVOL20260703-12-WorkingBillShi.xlsx'
EXCLUDE_JSON = DATA_ROOT / 'exclude.json'
OUTPUT_DIR = REPO_ROOT / 'results/toronto_jdmi_ellipsoid_tolerance'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DIMENSION_FIELDS = ['dim1_report_ref', 'dim2_report_ref', 'dim3_report_ref']
DIMENSION_TOLERANCE_CM = 0.1

# %% [markdown]
# ## Cohort and accession exclusions
#
# Preserve raw listing order. The global exclusion JSON is keyed by AANO and
# can include either `reason` or `description`. Do not require manual planimetry
# or select a different visit. Save the full audit before stopping on malformed
# or repeated identifiers.

# %%
listing = pd.read_csv(CLINICAL_CSV, dtype={'ID': 'string'})
required_listing = {'ID', 'vol_report_ref'}
assert required_listing <= set(listing.columns), 'Clinical listing lacks ID or vol_report_ref'
parts = listing.ID.str.extract(r'^AIPR_([^_]+)_([^_]+)_([^_]+)$')
listing['amrn'] = parts[0]
listing['exam_date'] = parts[1]
listing['aano'] = parts[2]
listing['invalid_id'] = parts.isna().any(axis=1) | listing.ID.str.contains(r'[/\\]', na=True)
listing['duplicate_scan_id'] = listing.ID.duplicated(keep=False)
listing['duplicate_amrn'] = listing.amrn.duplicated(keep=False)

exclusions = json.loads(EXCLUDE_JSON.read_text())
assert isinstance(exclusions, list) and all(
    isinstance(entry, dict) and isinstance(entry.get('aano'), str) and entry['aano'].strip()
    for entry in exclusions
), 'Exclusion JSON requires a list of entries with nonempty aano strings'
exclusion_reasons = {}
for entry in exclusions:
    detail = '; '.join(str(entry[key]) for key in ('reason', 'description') if entry.get(key))
    exclusion_reasons.setdefault(entry['aano'].strip(), []).append(detail or 'Listed in exclude.json')
listing['aano_excluded'] = listing.aano.isin(exclusion_reasons)
listing['accession_exclusion_reason'] = listing.aano.map(
    {key: '; '.join(value) for key, value in exclusion_reasons.items()}
).fillna('')
listing.to_csv(OUTPUT_DIR / 'cohort_audit.csv', index=False)
assert not listing[['invalid_id', 'duplicate_scan_id', 'duplicate_amrn']].any().any(), (
    'Invalid or repeated scan/patient IDs; inspect cohort_audit.csv'
)
cases = listing.loc[~listing.aano_excluded].copy().reset_index(drop=True)
cases['reported_volume_ml'] = pd.to_numeric(cases.vol_report_ref, errors='coerce')
assert (np.isfinite(cases.reported_volume_ml) & cases.reported_volume_ml.gt(0)).all(), (
    'Eligible cases require positive finite reported volumes; inspect cohort_audit.csv'
)

cohort_flow = pd.DataFrame([
    {'stage': 'Raw full clinical listing', 'n': len(listing)},
    {'stage': 'After global accession exclusions', 'n': len(cases)},
])
cohort_flow['excluded_since_previous'] = cohort_flow.n.shift() - cohort_flow.n
cohort_flow.to_csv(OUTPUT_DIR / 'cohort_flow.csv', index=False)
display(cohort_flow)

# %% [markdown]
# ## Workbook linkage and dimension audit
#
# Read only the AANO, AMRN, and report dimensions needed for this analysis.
# Decrypt an encrypted workbook in memory. Keep the listing order, verify the
# patient key, and stop with a local audit if any eligible row cannot be assessed.
# Dimensions are in cm; reported and derived volumes are in mL.

# %%
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
                workbook = pd.read_excel(
                    decrypted, engine='openpyxl', dtype=object, keep_default_na=False,
                )
        else:
            workbook = pd.read_excel(
                WORKBOOK, engine='openpyxl', dtype=object, keep_default_na=False,
            )
    finally:
        del office

required_workbook = {'aano', 'amrn', *DIMENSION_FIELDS}
assert workbook.columns.is_unique and required_workbook <= set(workbook.columns), (
    'Workbook requires unique columns, aano, amrn, and all three report dimensions'
)
clinical_fields = workbook[['aano', 'amrn', *DIMENSION_FIELDS]].copy()
clinical_fields['source_excel_row'] = np.arange(2, len(clinical_fields) + 2)
clinical_fields['aano'] = clinical_fields.aano.astype('string').str.strip()
clinical_fields['amrn'] = clinical_fields.amrn.astype('string').str.strip().str.removeprefix('AIPR_')
clinical_fields['invalid_aano'] = clinical_fields.aano.isna() | clinical_fields.aano.eq('')
clinical_fields['duplicate_aano'] = clinical_fields.aano.duplicated(keep=False)
clinical_fields['amrn'] = clinical_fields.amrn.mask(clinical_fields.amrn.eq(''))
clinical_fields.to_csv(OUTPUT_DIR / 'workbook_key_audit.csv', index=False)
assert not clinical_fields[['invalid_aano', 'duplicate_aano']].any().any(), (
    'Invalid or duplicate workbook accession keys; inspect workbook_key_audit.csv'
)

clinical_fields = clinical_fields.rename(columns={'amrn': 'workbook_amrn'})
quality = cases.merge(
    clinical_fields, on='aano', how='left', sort=False, validate='one_to_one',
    indicator='workbook_match',
)
assert quality.ID.tolist() == cases.ID.tolist(), 'Workbook linkage changed listing order'
quality['amrn_matches'] = quality.amrn.eq(quality.workbook_amrn).fillna(False)
for field in DIMENSION_FIELDS:
    raw = quality[field].astype('string')
    quality[f'{field}_censored'] = raw.str.contains('>', regex=False, na=False)
    quality[f'{field}_numeric'] = pd.to_numeric(
        raw.str.replace(',', '.', regex=False).str.replace('>', '', regex=False), errors='coerce',
    ).astype(float)

dimensions = quality[[f'{field}_numeric' for field in DIMENSION_FIELDS]]
quality['dimensions_valid'] = np.isfinite(dimensions).all(axis=1) & dimensions.gt(0).all(axis=1)
quality['dimensions_censored'] = quality[[f'{field}_censored' for field in DIMENSION_FIELDS]].any(axis=1)
quality['ellipsoid_volume_ml'] = np.nan
quality['lower_bound_ml'] = np.nan
quality['upper_bound_ml'] = np.nan
quality['missing_coefficient_lower_bound_ml'] = np.nan
quality['missing_coefficient_upper_bound_ml'] = np.nan
valid = quality.dimensions_valid & ~quality.dimensions_censored
with np.errstate(over='ignore', invalid='ignore'):
    d = dimensions.loc[valid]
    quality.loc[valid, 'ellipsoid_volume_ml'] = np.pi / 6 * d.prod(axis=1, skipna=False)
    quality.loc[valid, 'lower_bound_ml'] = 0.5 * (
        d - DIMENSION_TOLERANCE_CM
    ).prod(axis=1, skipna=False)
    quality.loc[valid, 'upper_bound_ml'] = np.pi * 5 / 24 * ( # np.pi / 6 * (
        d + DIMENSION_TOLERANCE_CM
    ).prod(axis=1, skipna=False)
    quality.loc[valid, 'missing_coefficient_lower_bound_ml'] = (
        d - DIMENSION_TOLERANCE_CM
    ).prod(axis=1, skipna=False)
    quality.loc[valid, 'missing_coefficient_upper_bound_ml'] = (
        d + DIMENSION_TOLERANCE_CM
    ).prod(axis=1, skipna=False)

quality.to_csv(OUTPUT_DIR / 'report_volume_quality_audit.csv', index=False)
assert quality.workbook_match.eq('both').all() and quality.amrn_matches.all(), (
    'Workbook linkage or AMRN mismatch; inspect report_volume_quality_audit.csv'
)
bound_fields = [
    'ellipsoid_volume_ml', 'lower_bound_ml', 'upper_bound_ml',
    'missing_coefficient_lower_bound_ml', 'missing_coefficient_upper_bound_ml',
]
assert (quality.dimensions_valid & ~quality.dimensions_censored).all() and (
    np.isfinite(quality[bound_fields]).all(axis=1)
).all(), 'Invalid dimensions or bounds; inspect report_volume_quality_audit.csv'

# %% [markdown]
# ## Inclusive tolerance bounds and proposed exclusions
#
# Boundary comparisons use a tiny floating-point equivalence tolerance only;
# no extra volume rounding is applied. A report above the upper bound is
# classified as `Missing coefficient` when it lies within the coefficient-1
# bounds. The generated exclusion list contains the three review classifications.

# %%
below = quality.reported_volume_ml.lt(quality.lower_bound_ml) & ~np.isclose(
    quality.reported_volume_ml, quality.lower_bound_ml, rtol=1e-12, atol=1e-12,
)
above = quality.reported_volume_ml.gt(quality.upper_bound_ml) & ~np.isclose(
    quality.reported_volume_ml, quality.upper_bound_ml, rtol=1e-12, atol=1e-12,
)
missing_coefficient = (
    above
    & (
        quality.reported_volume_ml.ge(quality.missing_coefficient_lower_bound_ml)
        | np.isclose(quality.reported_volume_ml, quality.missing_coefficient_lower_bound_ml,
                     rtol=1e-12, atol=1e-12)
    )
    & (
        quality.reported_volume_ml.le(quality.missing_coefficient_upper_bound_ml)
        | np.isclose(quality.reported_volume_ml, quality.missing_coefficient_upper_bound_ml,
                     rtol=1e-12, atol=1e-12)
    )
)
quality['classification'] = np.select(
    [below, missing_coefficient, above],
    ['Below range', 'Missing coefficient', 'Above range'],
    default='Within range',
)
quality['mean_volume_ml'] = (quality.ellipsoid_volume_ml + quality.reported_volume_ml) / 2
quality['difference_ml'] = quality.ellipsoid_volume_ml - quality.reported_volume_ml
quality.to_csv(OUTPUT_DIR / 'report_volume_quality_audit.csv', index=False)

counts = (
    quality.groupby('classification', sort=False).size()
    .rename('n').reset_index()
)
counts['percent_of_eligible_cohort'] = 100 * counts.n / len(quality)
display(counts)

# Print descriptive volume statistics within each classification. The 99% bounds
# are empirical central quantiles (0.5th and 99.5th percentiles), not confidence
# intervals. Keep zero-count groups in the output with missing statistics.
DESCRIPTIVE_METRICS = [
    'reported_volume_ml', 'ellipsoid_volume_ml', 'difference_ml',
    'lower_bound_ml', 'upper_bound_ml',
]
descriptive_rows = []
for classification, subgroup in quality.groupby('classification', sort=False):
    for metric in DESCRIPTIVE_METRICS:
        values = subgroup[metric].dropna()
        descriptive_rows.append({
            'classification': classification,
            'metric': metric,
            'n': len(values),
            'mean': values.mean(),
            'sd': values.std(ddof=1),
            'median': values.median(),
            'lower_99_bound_p0_5': values.quantile(0.005),
            'upper_99_bound_p99_5': values.quantile(0.995),
            'min': values.min(),
            'max': values.max(),
        })
descriptive_statistics = pd.DataFrame(descriptive_rows)
print('Within-classification descriptive statistics; 99% bounds are the p0.5–p99.5 interval:')
print(descriptive_statistics.to_string(index=False, float_format=lambda value: f'{value:.3f}'))

flagged_mask = quality.classification.isin(['Below range', 'Above range', 'Missing coefficient'])
exclude_dataframe = quality.loc[flagged_mask, [
    'aano', 'ID', 'classification', 'reported_volume_ml', 'ellipsoid_volume_ml',
    'lower_bound_ml', 'upper_bound_ml', 'missing_coefficient_lower_bound_ml',
    'missing_coefficient_upper_bound_ml', *DIMENSION_FIELDS,
]].copy()
exclude_dataframe['reason'] = exclude_dataframe.apply(
    lambda row: (
        f"{row['classification']} under ±{DIMENSION_TOLERANCE_CM:g} cm dimension tolerance "
        'and coefficient 0.5–5π/24 bounds'
        + ('; report is within coefficient-1 bounds' if row['classification'] == 'Missing coefficient' else '')
    ),
    axis=1,
)
exclude_dataframe.to_csv(OUTPUT_DIR / 'exclude_consider.csv', index=False)
exclude_json_object = exclude_dataframe[['aano', 'reason']].to_dict(orient='records')
assert all(
    isinstance(entry['aano'], str) and entry['aano'].strip()
    and isinstance(entry['reason'], str) and entry['reason'].strip()
    for entry in exclude_json_object
), 'Generated exclusion entries require nonempty aano and reason strings'
(OUTPUT_DIR / 'exclude_consider.json').write_text(
    json.dumps(exclude_json_object, indent=2, ensure_ascii=False) + '\n'
)
# %% [markdown]
# ## Full-range plot with a focused inset
#
# Show every eligible case across the full observed data range, then add a zoom
# of the 10–300 mL and −40 to +40 mL region. Marker shapes, colors, and group
# labels match the tolerance classification used in the summary above.

# %%
quality_plot_groups = [
    ('Within range', 'Within expected range', 'tab:blue', 'o'),
    ('Missing coefficient', 'Missing coefficient (reported ≈ d1 × d2 × d3)', 'tab:red', '^'),
    ('Above range', 'Expected from dimensions < reported volume, other than missing coefficient', 'tab:red', '^'),
    ('Below range', 'Expected from dimensions > reported volume', 'tab:orange', 'v'),
]
fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
fig.get_layout_engine().set(h_pad=0.02)
for classification, legend_label, color, marker in quality_plot_groups:
    group = quality.loc[quality.classification.eq(classification)]
    ax.scatter(
        group.mean_volume_ml, group.difference_ml, edgecolors=color if classification != 'Within range' else 'none',
        facecolors='none' if classification == 'Missing coefficient' else color,
        marker=marker, s=28, alpha=.1 if classification == 'Within range' else 1,
        label=f'{legend_label} (n = {len(group)})',
    )
ax.axhline(0, color='0.6', linewidth=.8)
ax.set_xscale('log')
ax.set_xlabel(r'Mean of $\frac{\pi}{6} d_1 d_2 d_3$ and reported volume (mL; log scale)', fontsize=8)
ax.set_ylabel(r'$\frac{\pi}{6} d_1 d_2 d_3 \minus$ reported volume (mL)', fontsize=8)
# ax.set_title('Report-volume tolerance: full data range with focused inset', fontsize=9)
ax.tick_params(axis='both', labelsize=8)

inset_ax = ax.inset_axes([.08, .35, .58, .59])
for classification, _, color, marker in quality_plot_groups:
    group = quality.loc[quality.classification.eq(classification)]
    inset_ax.scatter(
        group.mean_volume_ml, group.difference_ml, edgecolors=color if classification != 'Within range' else 'none',
        facecolors='none' if classification == 'Missing coefficient' else color,
        marker=marker, s=18, alpha=.1 if classification == 'Within range' else 1,
    )
inset_ax.axhline(0, color='0.6', linewidth=.7)
inset_ax.set_xscale('log')
inset_ax.set_xlim(10, 300)
inset_ax.set_ylim(-60, 60)
inset_ax.set_xticks([10, 30, 100, 300], labels=['10', '30', '100', '300'])
inset_ax.tick_params(axis='both', labelsize=6)
# inset_ax.set_title('Focused view', fontsize=7)
focus_box = Rectangle(
    (10, -60), 300 - 10, 60 - (-60), fill=False,
    edgecolor='0.4', linewidth=.8, zorder=2,
)
ax.add_patch(focus_box)
for focus_corner, inset_corner in [((10, 60), (0, 0)), ((300, 60), (1, 0))]:
    fig.add_artist(ConnectionPatch(
        xyA=focus_corner, coordsA='data', axesA=ax,
        xyB=inset_corner, coordsB='axes fraction', axesB=inset_ax,
        color='0.55', linewidth=.8, clip_on=False, zorder=.5,
    ))
fig.legend(
    handles=ax.collections, loc='outside lower center', fontsize=7,
    frameon=True
)
fig.savefig(OUTPUT_DIR / 'report_volume_quality_full_range_inset.png', dpi=300)
fig.savefig(OUTPUT_DIR / 'report_volume_quality_full_range_inset.pdf')
plt.show()



# %% [markdown]
# ## Histogram of out-of-bound reported-to-dimension ratios
#
# Include Below range, Above range, and Missing coefficient cases. The ratio is
# reported volume divided by d1 × d2 × d3, with dimension product in cm³ (mL).
# Histogram bins are 0.1 wide and aligned to multiples of 0.1.

# %%
out_of_bound_mask = quality.classification.isin([
    'Below range', 'Above range', 'Missing coefficient',
])
dimension_product_ml = dimensions.prod(axis=1, skipna=False)
out_of_bound_ratio = (
    quality.loc[out_of_bound_mask, 'reported_volume_ml']
    / dimension_product_ml.loc[out_of_bound_mask]
)
assert len(out_of_bound_ratio) > 0 and np.isfinite(out_of_bound_ratio).all(), (
    'Expected finite reported-volume ratios for out-of-bound cases'
)
bin_width = 0.1
bin_lower = np.floor(out_of_bound_ratio.min() / bin_width) * bin_width
bin_upper = np.ceil(out_of_bound_ratio.max() / bin_width) * bin_width
if bin_upper <= bin_lower:
    bin_upper = bin_lower + bin_width
bin_count = int(round((bin_upper - bin_lower) / bin_width))
histogram_bins = np.linspace(bin_lower, bin_upper, bin_count + 1)

fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
ax.hist(out_of_bound_ratio, bins=histogram_bins, color='tab:red', edgecolor='black')
ax.set_xlabel('Reported volume / (d1 × d2 × d3)')
ax.set_ylabel('Number of cases')
ax.set_title(f'Out-of-bound reported-volume ratios (n = {len(out_of_bound_ratio)})')
ax.set_xticks(histogram_bins)
ax.tick_params(axis='x', labelrotation=45)
ax.grid(axis='y', alpha=.25)
fig.savefig(OUTPUT_DIR / 'out_of_bound_reported_to_dimension_product_histogram.png', dpi=300)
fig.savefig(OUTPUT_DIR / 'out_of_bound_reported_to_dimension_product_histogram.pdf')
plt.show()

# %%
print(f'Proposed exclusion entries: {len(exclude_json_object)}')
print(f'CSV: {OUTPUT_DIR / "exclude_consider.csv"}')
print(f'JSON: {OUTPUT_DIR / "exclude_consider.json"}')

# %% [markdown]
# ## Dataset274 accuracy in outliers with manual ground truth
#
# Pool Below range, Above range, and Missing coefficient cases from the existing
# screen, retaining only positive finite manual-reader planimetry references.
# Compare Dataset274 whole-gland segmentation, π/6 × d1 × d2 × d3, and reported
# volume on exactly the same patients. Dimensions are cm; volumes are mL.
# Missing, invalid, and empty predictions stop the analysis after saving an audit.
# This measures volume accuracy in a selected outlier cohort, not spatial accuracy
# or performance in an unselected population. No inference is run.

# %%
import itertools
import warnings

import nibabel as nib
from scipy import stats
from statsmodels.stats.multitest import multipletests

reference_ml = pd.to_numeric(quality.ground_truth_volume_ml, errors='coerce')
outlier_mask = quality.classification.isin(['Below range', 'Above range', 'Missing coefficient'])
reference_valid = np.isfinite(reference_ml) & reference_ml.gt(0)
outlier_volumes = quality.loc[outlier_mask & reference_valid].copy()
outlier_volumes['gt_volume_ml'] = reference_ml.loc[outlier_volumes.index]
assert outlier_volumes.ID.is_unique and outlier_volumes.amrn.is_unique
outlier_flow = pd.DataFrame([
    dict(stage='Flagged outliers after shared exclusions', n=int(outlier_mask.sum())),
    dict(stage='Outliers with positive finite ground truth', n=len(outlier_volumes)),
])
display(outlier_flow)
outlier_flow.to_csv(OUTPUT_DIR / 'outlier_comparison_cohort_flow.csv', index=False)

MODEL_ROOT = DATA_ROOT / 'seg_out_Dataset274_raw_3d_fullres_5fold'
GEOMETRY_TOLERANCE_MM = 0.5
plans = json.loads((MODEL_ROOT / 'plans.json').read_text())
dataset = json.loads((MODEL_ROOT / 'dataset.json').read_text())
assert plans.get('dataset_name') == 'Dataset274_raw', 'Unexpected Dataset274 identity'
assert dataset.get('labels') == {'background': 0, 'class1': 1}, 'Expected binary whole-gland labels'

measurement_rows = []
for case in outlier_volumes.itertuples(index=False):
    prediction_path = MODEL_ROOT / f'{case.ID}.nii.gz'
    image_path = (DATA_ROOT / case.t2).resolve()
    record = dict(ID=case.ID, prediction_path=str(prediction_path), image_path=str(image_path),
                  status='invalid', error='', volume_ml=np.nan)
    try:
        pred_img, image = nib.load(prediction_path), nib.load(image_path)
        if len(pred_img.shape) != 3 or pred_img.shape != image.shape:
            raise ValueError('Prediction and T2 shapes differ or are not 3D')
        for img in (pred_img, image):
            if img.header.get_xyzt_units()[0] != 'mm':
                raise ValueError('Spatial units must be mm')
            determinant = abs(np.linalg.det(img.affine[:3, :3]))
            if not np.isfinite(img.affine).all() or not np.isfinite(determinant) or determinant <= 0:
                raise ValueError('Invalid spatial affine')
        corners = np.array(list(itertools.product(*[(0, n - 1) for n in image.shape])))
        displacement = float(np.linalg.norm(
            nib.affines.apply_affine(pred_img.affine, corners)
            - nib.affines.apply_affine(image.affine, corners), axis=1,
        ).max())
        record['max_corner_displacement_mm'] = displacement
        if not np.isfinite(displacement) or displacement > GEOMETRY_TOLERANCE_MM:
            raise ValueError('Geometry displacement exceeds tolerance')
        mask = np.asanyarray(pred_img.dataobj)
        if not np.isin(mask, [0, 1]).all():
            raise ValueError('Expected finite binary labels 0 and 1')
        voxel_count = int(np.count_nonzero(mask == 1))
        voxel_volume_mm3 = float(abs(np.linalg.det(pred_img.affine[:3, :3])))
        volume_ml = voxel_count * voxel_volume_mm3 / 1000
        record.update(voxel_count=voxel_count, voxel_volume_mm3=voxel_volume_mm3, volume_ml=volume_ml)
        if not np.isfinite(volume_ml) or volume_ml <= 0:
            raise ValueError('Prediction volume must be positive and finite')
        record.update(status='measured')
    except (OSError, ValueError, nib.filebasedimages.ImageFileError) as error:
        record['error'] = str(error)
    measurement_rows.append(record)
outlier_measurement_audit = pd.DataFrame(
    measurement_rows, columns=['ID', 'prediction_path', 'image_path', 'status', 'error',
                              'volume_ml', 'voxel_count', 'voxel_volume_mm3',
                              'max_corner_displacement_mm'],
)
outlier_measurement_audit['volume_ml'] = pd.to_numeric(outlier_measurement_audit.volume_ml, errors='coerce')
outlier_measurement_audit.to_csv(OUTPUT_DIR / 'outlier_prediction_audit.csv', index=False)
assert len(outlier_measurement_audit) == len(outlier_volumes) and (
    outlier_measurement_audit.status.eq('measured')
    & np.isfinite(outlier_measurement_audit.volume_ml)
    & outlier_measurement_audit.volume_ml.gt(0)
).all(), 'Missing, invalid, or zero-volume prediction; inspect outlier_prediction_audit.csv'
outlier_volumes['nnunet274_volume_ml'] = outlier_volumes.ID.map(
    outlier_measurement_audit.set_index('ID').volume_ml,
)

# %% [markdown]
# ### Descriptive errors and paired comparisons
#
# Errors are estimate minus reference; MAPE and RMSPE are percentages. Positive
# RMSE improvement (mL) or RMSPE improvement (percentage points) favors Dataset274.
# Paired BCa 95% intervals are pointwise and unadjusted. Null-centered bootstrap-t
# tests compare squared errors; paired t-tests are sensitivity analyses. Degenerate
# BCa intervals fall back to paired percentile intervals and are marked accordingly.
# Exact two-sided sign tests assess P(Dataset274 closer | no tie) = 0.5; ties use
# exact equality of absolute errors. All-tie comparisons are uninformative.
# Holm correction covers RMSE and closer jointly across the two comparisons;
# RMSPE has a separate family, with analogous sensitivity families. Undefined
# sensitivity p-values occupy a family slot as 1 but remain NaN in outputs.
# Resampling assumes independent patients and fixed predictions. Interpret inference
# only within this selected cohort. Fewer than two patients precludes inference.

# %%
OUTLIER_METHODS = {
    'nnunet274': ('Dataset274', 'nnunet274_volume_ml'),
    'ellipsoid': ('Recalculated ellipsoid', 'ellipsoid_volume_ml'),
    'reported': ('Reported volume', 'reported_volume_ml'),
}
outlier_summary_rows = []
for key, (name, column) in OUTLIER_METHODS.items():
    error = outlier_volumes[column] - outlier_volumes.gt_volume_ml
    relative = 100 * error / outlier_volumes.gt_volume_ml
    outlier_volumes[f'{key}_minus_gt_ml'] = error
    outlier_volumes[f'{key}_absolute_error_ml'] = error.abs()
    outlier_volumes[f'{key}_absolute_percentage_error'] = relative.abs()
    outlier_summary_rows.append(dict(
        method=key, name=name, n=len(error), bias_ml=error.mean(),
        mae_ml=error.abs().mean(), rmse_ml=np.sqrt((error**2).mean()),
        mape_percent=relative.abs().mean(), rmspe_percent=np.sqrt((relative**2).mean()),
    ))
outlier_summary = pd.DataFrame(outlier_summary_rows)
outlier_closer_rows = []
for comparator in ['ellipsoid', 'reported']:
    improvement = (outlier_volumes[f'{comparator}_absolute_error_ml']
                   - outlier_volumes.nnunet274_absolute_error_ml)
    outlier_volumes[f'nnunet274_vs_{comparator}_absolute_improvement_ml'] = improvement
    outlier_closer_rows.append(dict(
        comparator=comparator, n=len(improvement),
        wins=int(improvement.gt(0).sum()), losses=int(improvement.lt(0).sum()),
        ties=int(improvement.eq(0).sum()),
    ))
outlier_closer = pd.DataFrame(outlier_closer_rows)
display(outlier_summary, outlier_closer)
outlier_volumes.to_csv(OUTPUT_DIR / 'outlier_comparator_volumes.csv', index=False)
outlier_summary.to_csv(OUTPUT_DIR / 'outlier_comparator_summary.csv', index=False)
outlier_closer.to_csv(OUTPUT_DIR / 'outlier_closer_counts.csv', index=False)

# %%
N_BOOTSTRAP = 100_000
RANDOM_SEED = 20261102
BOOTSTRAP_BATCH = 1_000
ALPHA = 0.05
CONFIDENCE_LEVEL = 0.95

# A vectorized statistic is needed by SciPy for paired BCa resampling/jackknifing.
def rmse_improvement(error_a, error_b, axis=-1):
    return np.sqrt(np.mean(error_b**2, axis=axis)) - np.sqrt(np.mean(error_a**2, axis=axis))


# %%
statistics_rows = []
for a, b in ([('nnunet274', 'ellipsoid'), ('nnunet274', 'reported')] if len(outlier_volumes) >= 2 else []):
    error_a = (outlier_volumes[OUTLIER_METHODS[a][1]] - outlier_volumes['gt_volume_ml']).to_numpy(dtype=float)
    error_b = (outlier_volumes[OUTLIER_METHODS[b][1]] - outlier_volumes['gt_volume_ml']).to_numpy(dtype=float)
    assert np.isfinite(error_a).all() and np.isfinite(error_b).all(), (a, b)
    n = len(error_a)
    reference = outlier_volumes['gt_volume_ml'].to_numpy(dtype=float)
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
        method_a=a, name_a=OUTLIER_METHODS[a][0], method_b=b, name_b=OUTLIER_METHODS[b][0],
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
outlier_pairwise_statistics = pd.DataFrame(statistics_rows)
if not outlier_pairwise_statistics.empty:
    primary_p = outlier_pairwise_statistics[['rmse_bootstrap_p', 'closer_exact_p']].to_numpy()
    sensitivity_p = outlier_pairwise_statistics[['rmse_t_sensitivity_p', 'closer_exact_p']].to_numpy()
    primary_adjusted = multipletests(primary_p.ravel(), method='holm')[1].reshape(-1, 2)
    sensitivity_adjusted = multipletests(
        np.nan_to_num(sensitivity_p, nan=1.0).ravel(), method='holm',
    )[1].reshape(-1, 2)
    outlier_pairwise_statistics['holm_family_size'] = primary_p.size
    outlier_pairwise_statistics['rmse_bootstrap_p_holm'] = primary_adjusted[:, 0]
    outlier_pairwise_statistics['closer_exact_p_holm'] = primary_adjusted[:, 1]
    outlier_pairwise_statistics['rmse_t_sensitivity_p_holm'] = np.where(
        np.isfinite(sensitivity_p[:, 0]), sensitivity_adjusted[:, 0], np.nan,
    )
    outlier_pairwise_statistics['closer_exact_p_holm_sensitivity'] = sensitivity_adjusted[:, 1]
    # RMSPE is its own family across method pairs, separate from RMSE/closer.
    outlier_pairwise_statistics['rmspe_holm_family_size'] = len(outlier_pairwise_statistics)
    outlier_pairwise_statistics['rmspe_bootstrap_p_holm'] = multipletests(
        outlier_pairwise_statistics['rmspe_bootstrap_p'], method='holm',
    )[1]
    rmspe_t_p = outlier_pairwise_statistics['rmspe_t_sensitivity_p'].to_numpy()
    rmspe_t_adjusted = multipletests(np.nan_to_num(rmspe_t_p, nan=1.0), method='holm')[1]
    outlier_pairwise_statistics['rmspe_t_sensitivity_p_holm'] = np.where(
        np.isfinite(rmspe_t_p), rmspe_t_adjusted, np.nan,
    )
    for endpoint, p_column, effect_column in [
        ('rmspe', 'rmspe_bootstrap_p_holm', 'rmspe_improvement_pp'),
        ('rmspe_t_sensitivity', 'rmspe_t_sensitivity_p_holm', 'rmspe_improvement_pp'),
        ('rmse', 'rmse_bootstrap_p_holm', 'rmse_improvement_ml'),
        ('closer', 'closer_exact_p_holm', 'probability_a_closer_given_no_tie'),
        ('rmse_t_sensitivity', 'rmse_t_sensitivity_p_holm', 'rmse_improvement_ml'),
    ]:
        p = outlier_pairwise_statistics[p_column]
        effect = outlier_pairwise_statistics[effect_column] - (0.5 if endpoint == 'closer' else 0)
        outlier_pairwise_statistics[f'{endpoint}_conclusion'] = np.select(
            [p.isna(), (p <= ALPHA) & effect.gt(0), (p <= ALPHA) & effect.lt(0)],
            ['undefined', 'A better', 'B better'], default='no significant difference',
        )
    outlier_pairwise_statistics.loc[
        outlier_pairwise_statistics['n_non_ties'].eq(0), 'closer_conclusion'
    ] = 'uninformative_all_ties'
    print('RMSE improvement: positive favors A. Intervals are pointwise and unadjusted.')
    with pd.option_context('display.max_columns', None):
        display(outlier_pairwise_statistics[[
            'method_a', 'method_b', 'n', 'rmse_improvement_ml',
            'rmse_improvement_ci_lower_ml', 'rmse_improvement_ci_upper_ml',
            'rmse_bootstrap_p_holm', 'rmse_t_sensitivity_p_holm', 'rmse_conclusion',
        ]])
        print('RMSPE improvement in percentage points: positive favors A. Separate Holm family.')
        display(outlier_pairwise_statistics[[
            'method_a', 'method_b', 'n', 'rmspe_a_percent', 'rmspe_b_percent',
            'rmspe_improvement_pp', 'rmspe_improvement_ci_lower_pp', 'rmspe_improvement_ci_upper_pp',
            'rmspe_bootstrap_p_holm', 'rmspe_t_sensitivity_p_holm', 'rmspe_conclusion',
        ]])
        print('Probability A is closer, conditional on no tie. Exact intervals are unadjusted.')
        display(outlier_pairwise_statistics[[
            'method_a', 'method_b', 'wins_a', 'losses_a', 'ties',
            'probability_a_closer_given_no_tie', 'probability_ci_lower', 'probability_ci_upper',
            'closer_exact_p_holm', 'closer_conclusion',
        ]])
else:
    # Write an explicit empty result so a previous eligible-cohort run is not stale.
    outlier_pairwise_statistics = pd.DataFrame(columns=['method_a', 'method_b', 'n'])
    print('Fewer than two outliers with ground truth: paired inference skipped.')
outlier_pairwise_statistics.to_csv(OUTPUT_DIR / 'outlier_comparator_pairwise_statistics.csv', index=False)

# %% [markdown]
# ## Dataset274 agreement with Dataset702 in all flagged outliers
#
# Use every Below range, Above range, and Missing coefficient case after shared
# accession exclusions, regardless of manual ground-truth availability. Preserve
# the existing classifications and order. Dataset702 is an automated surrogate
# reference, not manual ground truth; agreement does not establish manual accuracy.
# Historical Dataset702 training and inference settings are unknown.
#
# Read PROSATEPSAD/{ID}/wp-pz-tz_axt2_nnunet-702.nii.gz. Whole gland is the union
# of labels 1, 2, and 3, not their numeric sum. Dataset274 uses foreground label 1.
# Volumes are foreground count × |det(spatial affine)| / 1,000, in mL.
# Masks and T2 must share a 3D shape and agree within 0.5 mm at voxel corners.
# Unknown spatial units require a matching known-mm T2 or Dataset274 grid within
# 0.01 mm, following toronto_jdmi_4111_clinical_volume.py. Audit every assumption.
# Missing, invalid, nonfinite, or empty masks stop after saving the measurement
# audit; no patients are silently removed and no model inference is performed.

# %%
import itertools
import warnings

import nibabel as nib
from scipy import stats
from statsmodels.stats.multitest import multipletests

outlier_702_cases = quality.loc[quality.classification.isin([
    'Below range', 'Above range', 'Missing coefficient',
])].copy()
assert outlier_702_cases.ID.is_unique and outlier_702_cases.amrn.is_unique
outlier_702_cases['image_path'] = outlier_702_cases.t2.map(
    lambda path: str((DATA_ROOT / path).resolve()),
)
outlier_702_flow = pd.DataFrame([
    dict(stage='Eligible cases after shared exclusions', n=len(quality)),
    dict(stage='All flagged outliers; manual reference not required', n=len(outlier_702_cases)),
])
display(outlier_702_flow)
outlier_702_flow.to_csv(OUTPUT_DIR / 'outlier_702_cohort_flow.csv', index=False)
OUTLIER_702_REFERENCE_ROOT = DATA_ROOT / 'PROSATEPSAD'
OUTLIER_702_REFERENCE_FILENAME = 'wp-pz-tz_axt2_nnunet-702.nii.gz'
OUTLIER_702_REFERENCE_LABELS = (1, 2, 3)
OUTLIER_702_MODEL_ROOT = DATA_ROOT / 'seg_out_Dataset274_raw_3d_fullres_5fold'
OUTLIER_702_GEOMETRY_TOLERANCE_MM = 0.5
OUTLIER_702_UNIT_ANCHOR_TOLERANCE_MM = 0.01
outlier_702_plans = json.loads((OUTLIER_702_MODEL_ROOT / 'plans.json').read_text())
outlier_702_dataset = json.loads((OUTLIER_702_MODEL_ROOT / 'dataset.json').read_text())
assert outlier_702_plans.get('dataset_name') == 'Dataset274_raw', 'Unexpected Dataset274 identity'
assert outlier_702_dataset.get('labels') == {'background': 0, 'class1': 1}, (
    'Expected Dataset274 binary whole-gland labels'
)
outlier_702_measurement_rows = []
for case in outlier_702_cases.itertuples(index=False):
    mask_paths = {
        'reference_702': OUTLIER_702_REFERENCE_ROOT / case.ID / OUTLIER_702_REFERENCE_FILENAME,
        'nnunet274': OUTLIER_702_MODEL_ROOT / f'{case.ID}.nii.gz',
    }
    for key, path in mask_paths.items():
        foreground_labels = OUTLIER_702_REFERENCE_LABELS if key == 'reference_702' else (1,)
        record = dict(ID=case.ID, method=key, prediction_path=str(path.relative_to(REPO_ROOT)),
                      volume_ml=np.nan)
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
                    for candidate_path in (mask_paths['nnunet274'],):
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
                        if (np.isfinite(candidate_displacement) and candidate_displacement <= OUTLIER_702_UNIT_ANCHOR_TOLERANCE_MM
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
                        or max(anchor_displacements) > OUTLIER_702_UNIT_ANCHOR_TOLERANCE_MM):
                    raise ValueError('Unknown-unit grid differs from known-mm anchor')
            displacement = float(np.linalg.norm(
                nib.affines.apply_affine(pred_img.affine, corners)
                - nib.affines.apply_affine(image.affine, corners), axis=1,
            ).max())
            record['max_corner_displacement_mm'] = displacement
            if not np.isfinite(displacement) or displacement > OUTLIER_702_GEOMETRY_TOLERANCE_MM:
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
            if not np.isfinite(volume_ml) or volume_ml <= 0:
                raise ValueError('Reference and Dataset274 volumes must be positive and finite')
            record.update(status='measured', error='')
        except (OSError, ValueError, nib.filebasedimages.ImageFileError) as error:
            record.update(status='invalid', error=str(error))
        outlier_702_measurement_rows.append(record)

outlier_702_measurement_audit = pd.DataFrame(outlier_702_measurement_rows)
# Explicit dtypes also support an empty flagged cohort.
if outlier_702_measurement_audit.empty:
    outlier_702_measurement_audit = pd.DataFrame(columns=[
        'ID', 'method', 'prediction_path', 'status', 'error', 'volume_ml',
    ])
outlier_702_measurement_audit['volume_ml'] = pd.to_numeric(
    outlier_702_measurement_audit.volume_ml, errors='coerce',
)
outlier_702_measurement_audit.to_csv(OUTPUT_DIR / 'outlier_702_measurement_audit.csv', index=False)
assert len(outlier_702_measurement_audit) == 2 * len(outlier_702_cases) and (
    outlier_702_measurement_audit.status.eq('measured')
    & np.isfinite(outlier_702_measurement_audit.volume_ml)
    & outlier_702_measurement_audit.volume_ml.gt(0)
).all(), 'Missing, invalid, or empty mask; inspect outlier_702_measurement_audit.csv'
assert not outlier_702_measurement_audit.duplicated(['ID', 'method']).any()
outlier_702_volumes = outlier_702_cases.copy()
for key in ['reference_702', 'nnunet274']:
    measured = outlier_702_measurement_audit.loc[outlier_702_measurement_audit.method.eq(key)]
    assert measured.ID.tolist() == outlier_702_cases.ID.tolist(), 'Incomplete measurement cohort'
    outlier_702_volumes[f'{key}_volume_ml'] = pd.to_numeric(
        outlier_702_volumes.ID.map(measured.set_index('ID').volume_ml), errors='coerce',
    )
assert outlier_702_volumes.ID.tolist() == outlier_702_cases.ID.tolist(), 'Measurement changed cohort order'

# %% [markdown]
# ### Errors and paired inference against the Dataset702 surrogate
#
# Use the same metrics and paired tests as the manual-reference comparison above,
# replacing the reference with Dataset702 and including all flagged outliers.
# Bias is estimate minus Dataset702 (mL); MAPE and RMSPE are percentages. Positive
# RMSE/RMSPE improvement favors Dataset274; RMSPE improvement is percentage points.
# Resample paired patients 100,000 times with seed 20261102. BCa 95% intervals are
# pointwise and unadjusted; degenerate cases use the marked fallback. Bootstrap-t
# tests are null-centered, with paired t-test sensitivity analyses. Exact sign
# tests and probability intervals exclude exact ties; win/loss/tie counts include
# everyone. All ties are uninformative; fewer than two patients skips inference.
# Holm families are separate from the manual-reference analysis: RMSE and closer
# jointly across the two comparisons, RMSPE separately, and analogous sensitivity
# families. Undefined sensitivity p-values occupy a family slot as 1 but stay NaN.
# Interpretation assumes independent patients and fixed predictions and is limited
# to agreement with this automated surrogate within the selected outlier cohort.

# %%
OUTLIER_702_METHODS = {
    'nnunet274': ('Dataset274', 'nnunet274_volume_ml'),
    'ellipsoid': ('Recalculated ellipsoid', 'ellipsoid_volume_ml'),
    'reported': ('Reported volume', 'reported_volume_ml'),
}
outlier_702_summary_rows = []
for key, (name, column) in OUTLIER_702_METHODS.items():
    error = outlier_702_volumes[column] - outlier_702_volumes.reference_702_volume_ml
    relative = 100 * error / outlier_702_volumes.reference_702_volume_ml
    outlier_702_volumes[f'{key}_minus_reference_702_ml'] = error
    outlier_702_volumes[f'{key}_absolute_error_ml'] = error.abs()
    outlier_702_volumes[f'{key}_absolute_percentage_error'] = relative.abs()
    outlier_702_summary_rows.append(dict(
        method=key, name=name, n=len(error), bias_ml=error.mean(),
        mae_ml=error.abs().mean(), rmse_ml=np.sqrt((error**2).mean()),
        mape_percent=relative.abs().mean(), rmspe_percent=np.sqrt((relative**2).mean()),
    ))
outlier_702_summary = pd.DataFrame(outlier_702_summary_rows)
outlier_702_closer_rows = []
for comparator in ['ellipsoid', 'reported']:
    improvement = (outlier_702_volumes[f'{comparator}_absolute_error_ml']
                   - outlier_702_volumes.nnunet274_absolute_error_ml)
    outlier_702_volumes[f'nnunet274_vs_{comparator}_absolute_improvement_ml'] = improvement
    outlier_702_closer_rows.append(dict(
        comparator=comparator, n=len(improvement),
        wins=int(improvement.gt(0).sum()), losses=int(improvement.lt(0).sum()),
        ties=int(improvement.eq(0).sum()),
    ))
outlier_702_closer = pd.DataFrame(outlier_702_closer_rows)
display(outlier_702_summary, outlier_702_closer)
outlier_702_volumes.to_csv(OUTPUT_DIR / 'outlier_702_comparator_volumes.csv', index=False)
outlier_702_summary.to_csv(OUTPUT_DIR / 'outlier_702_comparator_summary.csv', index=False)
outlier_702_closer.to_csv(OUTPUT_DIR / 'outlier_702_closer_counts.csv', index=False)

# %%
OUTLIER_702_N_BOOTSTRAP = 100_000
OUTLIER_702_RANDOM_SEED = 20261102
OUTLIER_702_BOOTSTRAP_BATCH = 1_000
OUTLIER_702_ALPHA = 0.05
OUTLIER_702_CONFIDENCE_LEVEL = 0.95

# A vectorized statistic is needed by SciPy for paired BCa resampling/jackknifing.
def outlier_702_rmse_improvement(error_a, error_b, axis=-1):
    return np.sqrt(np.mean(error_b**2, axis=axis)) - np.sqrt(np.mean(error_a**2, axis=axis))


# %%
outlier_702_statistics_rows = []
for a, b in ([('nnunet274', 'ellipsoid'), ('nnunet274', 'reported')] if len(outlier_702_volumes) >= 2 else []):
    error_a = (outlier_702_volumes[OUTLIER_702_METHODS[a][1]] - outlier_702_volumes['reference_702_volume_ml']).to_numpy(dtype=float)
    error_b = (outlier_702_volumes[OUTLIER_702_METHODS[b][1]] - outlier_702_volumes['reference_702_volume_ml']).to_numpy(dtype=float)
    assert np.isfinite(error_a).all() and np.isfinite(error_b).all(), (a, b)
    n = len(error_a)
    reference = outlier_702_volumes['reference_702_volume_ml'].to_numpy(dtype=float)
    assert np.isfinite(reference).all() and (reference > 0).all(), 'RMSPE requires positive finite references'
    raw_error_a, raw_error_b = error_a, error_b
    endpoint_statistics = {}
    for endpoint in ['rmse', 'rmspe']:
        error_a = raw_error_a if endpoint == 'rmse' else 100 * raw_error_a / reference
        error_b = raw_error_b if endpoint == 'rmse' else 100 * raw_error_b / reference
        assert np.isfinite(error_a).all() and np.isfinite(error_b).all(), (a, b, endpoint)
        # Reset for each pair so reordering methods preserves the sampled patients.
        ci_rng = np.random.default_rng(OUTLIER_702_RANDOM_SEED)
        test_rng = np.random.default_rng(OUTLIER_702_RANDOM_SEED + 1)
        improvement = float(outlier_702_rmse_improvement(error_a, error_b))
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
                    (error_a, error_b), outlier_702_rmse_improvement, paired=True, vectorized=True,
                    n_resamples=OUTLIER_702_N_BOOTSTRAP, batch=OUTLIER_702_BOOTSTRAP_BATCH, rng=ci_rng,
                    confidence_level=OUTLIER_702_CONFIDENCE_LEVEL, method='BCa',
                )
            ci_lower, ci_upper = bootstrap_ci.confidence_interval
            ci_method = 'paired_BCa'
            if not np.isfinite([ci_lower, ci_upper]).all():
                # BCa acceleration can be undefined for a degenerate jackknife.
                ci_lower, ci_upper = np.quantile(
                    bootstrap_ci.bootstrap_distribution,
                    [(1 - OUTLIER_702_CONFIDENCE_LEVEL) / 2, (1 + OUTLIER_702_CONFIDENCE_LEVEL) / 2],
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
        for start in range(0, OUTLIER_702_N_BOOTSTRAP, OUTLIER_702_BOOTSTRAP_BATCH):
            batch_size = min(OUTLIER_702_BOOTSTRAP_BATCH, OUTLIER_702_N_BOOTSTRAP - start)
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
        bootstrap_pvalue = (extreme + 1) / (OUTLIER_702_N_BOOTSTRAP + 1)

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
            confidence_level=OUTLIER_702_CONFIDENCE_LEVEL, method='exact',
        )
        sign_pvalue = float(sign_test.pvalue)
    else:
        probability = probability_lower = probability_upper = np.nan
        sign_pvalue = 1.0

    outlier_702_statistics_rows.append(dict(
        method_a=a, name_a=OUTLIER_702_METHODS[a][0], method_b=b, name_b=OUTLIER_702_METHODS[b][0],
        n=n, **endpoint_statistics,
        wins_a=wins, losses_a=losses, ties=ties, n_non_ties=non_ties,
        win_proportion_all=wins / n, loss_proportion_all=losses / n, tie_proportion_all=ties / n,
        probability_a_closer_given_no_tie=probability,
        probability_ci_lower=probability_lower, probability_ci_upper=probability_upper,
        closer_exact_p=sign_pvalue, n_bootstrap=OUTLIER_702_N_BOOTSTRAP, random_seed=OUTLIER_702_RANDOM_SEED,
        bootstrap_test_seed=OUTLIER_702_RANDOM_SEED + 1, bootstrap_batch=OUTLIER_702_BOOTSTRAP_BATCH,
        monte_carlo_p_resolution=1 / (OUTLIER_702_N_BOOTSTRAP + 1), confidence_level=OUTLIER_702_CONFIDENCE_LEVEL,
        intervals_multiplicity_adjusted=False, alpha=OUTLIER_702_ALPHA,
    ))

# %%
outlier_702_pairwise_statistics = pd.DataFrame(outlier_702_statistics_rows)
if not outlier_702_pairwise_statistics.empty:
    primary_p = outlier_702_pairwise_statistics[['rmse_bootstrap_p', 'closer_exact_p']].to_numpy()
    sensitivity_p = outlier_702_pairwise_statistics[['rmse_t_sensitivity_p', 'closer_exact_p']].to_numpy()
    primary_adjusted = multipletests(primary_p.ravel(), method='holm')[1].reshape(-1, 2)
    sensitivity_adjusted = multipletests(
        np.nan_to_num(sensitivity_p, nan=1.0).ravel(), method='holm',
    )[1].reshape(-1, 2)
    outlier_702_pairwise_statistics['holm_family_size'] = primary_p.size
    outlier_702_pairwise_statistics['rmse_bootstrap_p_holm'] = primary_adjusted[:, 0]
    outlier_702_pairwise_statistics['closer_exact_p_holm'] = primary_adjusted[:, 1]
    outlier_702_pairwise_statistics['rmse_t_sensitivity_p_holm'] = np.where(
        np.isfinite(sensitivity_p[:, 0]), sensitivity_adjusted[:, 0], np.nan,
    )
    outlier_702_pairwise_statistics['closer_exact_p_holm_sensitivity'] = sensitivity_adjusted[:, 1]
    # RMSPE is its own family across method pairs, separate from RMSE/closer.
    outlier_702_pairwise_statistics['rmspe_holm_family_size'] = len(outlier_702_pairwise_statistics)
    outlier_702_pairwise_statistics['rmspe_bootstrap_p_holm'] = multipletests(
        outlier_702_pairwise_statistics['rmspe_bootstrap_p'], method='holm',
    )[1]
    rmspe_t_p = outlier_702_pairwise_statistics['rmspe_t_sensitivity_p'].to_numpy()
    rmspe_t_adjusted = multipletests(np.nan_to_num(rmspe_t_p, nan=1.0), method='holm')[1]
    outlier_702_pairwise_statistics['rmspe_t_sensitivity_p_holm'] = np.where(
        np.isfinite(rmspe_t_p), rmspe_t_adjusted, np.nan,
    )
    for endpoint, p_column, effect_column in [
        ('rmspe', 'rmspe_bootstrap_p_holm', 'rmspe_improvement_pp'),
        ('rmspe_t_sensitivity', 'rmspe_t_sensitivity_p_holm', 'rmspe_improvement_pp'),
        ('rmse', 'rmse_bootstrap_p_holm', 'rmse_improvement_ml'),
        ('closer', 'closer_exact_p_holm', 'probability_a_closer_given_no_tie'),
        ('rmse_t_sensitivity', 'rmse_t_sensitivity_p_holm', 'rmse_improvement_ml'),
    ]:
        p = outlier_702_pairwise_statistics[p_column]
        effect = outlier_702_pairwise_statistics[effect_column] - (0.5 if endpoint == 'closer' else 0)
        outlier_702_pairwise_statistics[f'{endpoint}_conclusion'] = np.select(
            [p.isna(), (p <= OUTLIER_702_ALPHA) & effect.gt(0), (p <= OUTLIER_702_ALPHA) & effect.lt(0)],
            ['undefined', 'A better', 'B better'], default='no significant difference',
        )
    outlier_702_pairwise_statistics.loc[
        outlier_702_pairwise_statistics['n_non_ties'].eq(0), 'closer_conclusion'
    ] = 'uninformative_all_ties'
    print('RMSE improvement: positive favors A. Intervals are pointwise and unadjusted.')
    with pd.option_context('display.max_columns', None):
        display(outlier_702_pairwise_statistics[[
            'method_a', 'method_b', 'n', 'rmse_improvement_ml',
            'rmse_improvement_ci_lower_ml', 'rmse_improvement_ci_upper_ml',
            'rmse_bootstrap_p_holm', 'rmse_t_sensitivity_p_holm', 'rmse_conclusion',
        ]])
        print('RMSPE improvement in percentage points: positive favors A. Separate Holm family.')
        display(outlier_702_pairwise_statistics[[
            'method_a', 'method_b', 'n', 'rmspe_a_percent', 'rmspe_b_percent',
            'rmspe_improvement_pp', 'rmspe_improvement_ci_lower_pp', 'rmspe_improvement_ci_upper_pp',
            'rmspe_bootstrap_p_holm', 'rmspe_t_sensitivity_p_holm', 'rmspe_conclusion',
        ]])
        print('Probability A is closer, conditional on no tie. Exact intervals are unadjusted.')
        display(outlier_702_pairwise_statistics[[
            'method_a', 'method_b', 'wins_a', 'losses_a', 'ties',
            'probability_a_closer_given_no_tie', 'probability_ci_lower', 'probability_ci_upper',
            'closer_exact_p_holm', 'closer_conclusion',
        ]])
else:
    # Write an explicit empty result so a previous eligible-cohort run is not stale.
    outlier_702_pairwise_statistics = pd.DataFrame(columns=['method_a', 'method_b', 'n'])
    print('Fewer than two flagged outliers: paired inference skipped.')
outlier_702_pairwise_statistics.to_csv(OUTPUT_DIR / 'outlier_702_comparator_pairwise_statistics.csv', index=False)

# %% [markdown]
# ## Paired signed errors in inconsistent cases
#
# Each line connects the calculated ellipsoid or reported volume error to the nnUNet274 error for the same patient and reference volume.
# Signed error is estimate minus reference (mL); distance from zero is absolute
# error. Solid green lines indicate reduced absolute error, dashed orange
# lines increased error, and gray
# an exact tie. Lines are clipped at the displayed axis limits; statistics
# still include every case. All inconsistent cases (Below range, Above range, Missing
# coefficient) in each existing reference cohort are retained. Manual-reference
# plots use available manual measurements; nnUNet702 is a surrogate reference
# and its plots use all flagged cases. Cohorts can therefore differ.

# %%
from matplotlib.lines import Line2D

paired_error_specs = [
    (outlier_volumes, 'gt_volume_ml', 'Manual segmentation', 'manual'),
    (outlier_702_volumes, 'reference_702_volume_ml', 'Reference automated segmentation', 'nnunet702'),
]
paired_error_panels = []
for cohort, reference_column, reference_name, reference_key in paired_error_specs:
    assert cohort.classification.isin(['Below range', 'Above range', 'Missing coefficient']).all()
    assert cohort.ID.is_unique
    for comparator_column, comparator_name, comparator_key in [
        ('ellipsoid_volume_ml', 'Calculated ellipsoid', 'calculated'),
        ('reported_volume_ml', 'Reported', 'reported'),
    ]:
        reference = cohort[reference_column].to_numpy(dtype=float)
        start_error = cohort[comparator_column].to_numpy(dtype=float) - reference
        end_error = cohort.nnunet274_volume_ml.to_numpy(dtype=float) - reference
        assert np.isfinite(np.column_stack([reference, start_error, end_error])).all()
        assert (reference > 0).all()
        paired_error_panels.append(dict(
            reference=reference, start_error=start_error, end_error=end_error,
            reference_name=reference_name, reference_key=reference_key,
            comparator_name=comparator_name, comparator_key=comparator_key,
        ))

# Shared limits allow direct comparison without clipping any selected cases.
paired_all_references = np.concatenate([panel['reference'] for panel in paired_error_panels])
paired_all_errors = np.concatenate([
    panel[key] for panel in paired_error_panels for key in ['start_error', 'end_error']
])
paired_x_max = max(float(paired_all_references.max()), 1.0) if paired_all_references.size else 1.0
paired_y_max = max(float(np.abs(paired_all_errors).max()), 1.0) if paired_all_errors.size else 1.0
# Edit these shared limits to adjust all four figures. Keep x limits positive.
paired_x_limits = (min(20.0, float(paired_all_references.min()) / 1.08) if paired_all_references.size else 10.0,
                   max(200.0, 1.08 * paired_x_max))
paired_y_limits = (-100, +100)
paired_error_plot_summary = []

with plt.rc_context({'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6}):
    for panel in paired_error_panels:
        reference = panel['reference']
        start_error, end_error = panel['start_error'], panel['end_error']
        improvement = np.abs(start_error) - np.abs(end_error)
        improved, worsened, tied = improvement > 0, improvement < 0, improvement == 0
        colors = np.where(improved, '#238b45', np.where(worsened, '#d97706', '#777777'))
        n = len(reference)
        start_mae = float(np.abs(start_error).mean()) if n else np.nan
        end_mae = float(np.abs(end_error).mean()) if n else np.nan
        paired_error_plot_summary.append(dict(
            reference=panel['reference_key'], comparator=panel['comparator_key'], n=n,
            improved=int(improved.sum()), worsened=int(worsened.sum()), tied=int(tied.sum()),
            comparator_mae_ml=start_mae, nnunet274_mae_ml=end_mae,
        ))
        assert int(improved.sum() + worsened.sum() + tied.sum()) == n
        fig, ax = plt.subplots(figsize=(4, 2.5), constrained_layout=True)
        ax.axhline(0, color='black', linewidth=.8, zorder=1)
        line_styles = np.where(worsened, '--', '-')
        for x, start, end, color, line_style in zip(reference, start_error, end_error, colors, line_styles):
            ax.plot([x, x], [start, end], color=color, linestyle=line_style,
                    linewidth=.65, alpha=.65, clip_on=True, zorder=2)
        ax.scatter(reference, start_error, c=colors, marker='o', s=12,
                   alpha=.8, linewidths=0, zorder=3)
        ax.scatter(reference, end_error, c=colors, marker='^', s=16,
                   alpha=.9, linewidths=0, zorder=4)
        ax.set_xscale('log')
        ax.set(xlim=paired_x_limits,
               ylim=paired_y_limits,
               xlabel=f"{panel['reference_name']} volume (mL)",
               ylabel='Estimate − reference (mL)')
        ax.set_xticks([20, 30, 100, 200], labels=['20', '30', '100', '200'])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.grid(alpha=.15, linewidth=.5)
        handles = [
            Line2D([], [], color='black', marker='o', linestyle='none', markersize=3,
                   label=panel['comparator_name']),
            Line2D([], [], color='black', marker='^', linestyle='none', markersize=3,
                   label='nnUNet274'),
            Line2D([], [], color='#238b45', lw=1, linestyle='-', label='Reduced absolute error'),
            Line2D([], [], color='#d97706', lw=1, linestyle='--', label='Increased absolute error'),
        ]
        if tied.any():
            handles.append(Line2D([], [], color='#777777', lw=1, label='Unchanged absolute error'))
        ax.legend(handles=handles, loc='lower left',
                  ncol=1, frameon=True, framealpha=.85, edgecolor='none', handlelength=1.8)
        output_stem = f"paired_signed_error_{panel['comparator_key']}_vs_{panel['reference_key']}"
        fig.savefig(OUTPUT_DIR / f'{output_stem}.png', dpi=300)
        fig.savefig(OUTPUT_DIR / f'{output_stem}.pdf')
        plt.show()
        plt.close(fig)

paired_error_plot_summary = pd.DataFrame(paired_error_plot_summary)
display(paired_error_plot_summary)
paired_error_plot_summary.to_csv(OUTPUT_DIR / 'paired_signed_error_plot_summary.csv', index=False)

# %% [markdown]
# ## Whole-cohort descriptive agreement for Appendix C
#
# Use exactly the 4,109 examinations in the saved consistency-screen audit. Reuse
# existing nnUNet702 surrogate and nnUNet274 whole-gland volumes where available;
# measure missing cases from existing masks with the same geometry, label, and
# unit checks as the flagged-case analysis. No inference or hypothesis tests are
# run. Reported and recalculated volumes come from the current screen audit.
# Errors are estimate minus nnUNet702 reference (mL); MAPE and RMSPE are percentages.
# All positive finite reference volumes are retained, including those below 1 mL;
# percentage errors can be sensitive to small denominators. These comparisons
# measure agreement with an automated surrogate, not manual-reference accuracy.

# %%
from pathlib import Path
import itertools
import json

import nibabel as nib
import numpy as np
import pandas as pd

# The consistency-screen audit, rather than the expanded-analysis cohort, defines inclusion.
whole_cohort = pd.read_csv(OUTPUT_DIR / 'report_volume_quality_audit.csv', dtype={'ID': 'string'})
assert len(whole_cohort) == 4109 and whole_cohort.ID.is_unique
saved_volume_path = REPO_ROOT / 'results/toronto_jdmi_4111_clinical_volume/comparator_volumes.csv'
whole_cohort_saved = pd.read_csv(saved_volume_path, dtype={'ID': 'string'}, low_memory=False)
assert whole_cohort_saved.ID.is_unique
whole_cohort_saved = whole_cohort_saved[['ID', 'gt_volume_ml', 'nnunet274_volume_ml']].rename(
    columns={'gt_volume_ml': 'reference_702_volume_ml'},
)
whole_cohort = whole_cohort.merge(whole_cohort_saved, on='ID', how='left', validate='one_to_one', sort=False)
whole_cohort['image_path'] = whole_cohort.t2.map(lambda path: str((DATA_ROOT / path).resolve()))
whole_cohort_missing = whole_cohort.loc[
    whole_cohort[['reference_702_volume_ml', 'nnunet274_volume_ml']].isna().any(axis=1)
].copy()
print(f'Reused complete measurements: {len(whole_cohort) - len(whole_cohort_missing)}; '
      f'cases requiring mask measurement: {len(whole_cohort_missing)}')
OUTLIER_702_REFERENCE_ROOT = DATA_ROOT / 'PROSATEPSAD'
OUTLIER_702_REFERENCE_FILENAME = 'wp-pz-tz_axt2_nnunet-702.nii.gz'
OUTLIER_702_REFERENCE_LABELS = (1, 2, 3)
OUTLIER_702_MODEL_ROOT = DATA_ROOT / 'seg_out_Dataset274_raw_3d_fullres_5fold'
OUTLIER_702_GEOMETRY_TOLERANCE_MM = 0.5
OUTLIER_702_UNIT_ANCHOR_TOLERANCE_MM = 0.01
assert json.loads((OUTLIER_702_MODEL_ROOT / 'plans.json').read_text())['dataset_name'] == 'Dataset274_raw'
assert json.loads((OUTLIER_702_MODEL_ROOT / 'dataset.json').read_text())['labels'] == {'background': 0, 'class1': 1}
whole_cohort_measurement_rows = []
for case in whole_cohort_missing.itertuples(index=False):
    mask_paths = {
        'reference_702': OUTLIER_702_REFERENCE_ROOT / case.ID / OUTLIER_702_REFERENCE_FILENAME,
        'nnunet274': OUTLIER_702_MODEL_ROOT / f'{case.ID}.nii.gz',
    }
    for key, path in mask_paths.items():
        foreground_labels = OUTLIER_702_REFERENCE_LABELS if key == 'reference_702' else (1,)
        record = dict(ID=case.ID, method=key, prediction_path=str(path.relative_to(REPO_ROOT)),
                      volume_ml=np.nan)
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
                    for candidate_path in (mask_paths['nnunet274'],):
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
                        if (np.isfinite(candidate_displacement) and candidate_displacement <= OUTLIER_702_UNIT_ANCHOR_TOLERANCE_MM
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
                        or max(anchor_displacements) > OUTLIER_702_UNIT_ANCHOR_TOLERANCE_MM):
                    raise ValueError('Unknown-unit grid differs from known-mm anchor')
            displacement = float(np.linalg.norm(
                nib.affines.apply_affine(pred_img.affine, corners)
                - nib.affines.apply_affine(image.affine, corners), axis=1,
            ).max())
            record['max_corner_displacement_mm'] = displacement
            if not np.isfinite(displacement) or displacement > OUTLIER_702_GEOMETRY_TOLERANCE_MM:
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
            if not np.isfinite(volume_ml) or volume_ml <= 0:
                raise ValueError('Reference and Dataset274 volumes must be positive and finite')
            record.update(status='measured', error='')
        except (OSError, ValueError, nib.filebasedimages.ImageFileError) as error:
            record.update(status='invalid', error=str(error))
        whole_cohort_measurement_rows.append(record)

whole_cohort_measurement_audit = pd.DataFrame(whole_cohort_measurement_rows,
    columns=['ID', 'method', 'volume_ml', 'status', 'error', 'prediction_path',
             'mask_spatial_units', 'image_spatial_units', 'units_assumed_mm',
             'unit_anchor_path', 'unit_anchor_max_displacement_mm', 'max_corner_displacement_mm',
             'observed_labels', 'label_0_voxel_count', 'label_1_voxel_count',
             'label_2_voxel_count', 'label_3_voxel_count', 'voxel_count', 'voxel_volume_mm3'])
whole_cohort_measurement_audit.to_csv(OUTPUT_DIR / 'whole_cohort_4109_new_measurement_audit.csv', index=False)
assert len(whole_cohort_measurement_audit) == 2 * len(whole_cohort_missing)
assert whole_cohort_measurement_audit.status.eq('measured').all(), 'Inspect new measurement audit'
for method in ['reference_702', 'nnunet274']:
    measured = whole_cohort_measurement_audit.loc[whole_cohort_measurement_audit.method.eq(method)]
    assert not measured.ID.duplicated().any()
    column = f'{method}_volume_ml'
    whole_cohort.loc[whole_cohort.ID.isin(measured.ID), column] = (
        whole_cohort.loc[whole_cohort.ID.isin(measured.ID), 'ID'].map(measured.set_index('ID').volume_ml)
    )
whole_cohort['measurement_source'] = np.where(
    whole_cohort.ID.isin(whole_cohort_missing.ID), 'existing masks measured for this analysis',
    'saved expanded-cohort comparator_volumes.csv',
)
whole_cohort_columns = ['reference_702_volume_ml', 'nnunet274_volume_ml',
                        'reported_volume_ml', 'ellipsoid_volume_ml']
assert np.isfinite(whole_cohort[whole_cohort_columns]).all().all()
assert whole_cohort[whole_cohort_columns].gt(0).all().all()
assert whole_cohort.ID.tolist() == pd.read_csv(
    OUTPUT_DIR / 'report_volume_quality_audit.csv', dtype={'ID': 'string'}).ID.tolist()
# Independently saved flagged-case measurements must agree with the reused volumes.
flagged_saved = pd.read_csv(OUTPUT_DIR / 'outlier_702_comparator_volumes.csv', dtype={'ID': 'string'})
flagged_overlap = whole_cohort.merge(flagged_saved[['ID', 'reference_702_volume_ml', 'nnunet274_volume_ml']],
    on='ID', suffixes=('', '_flagged'), validate='one_to_one')
assert len(flagged_overlap) == 65
for column in ['reference_702_volume_ml', 'nnunet274_volume_ml']:
    assert np.allclose(flagged_overlap[column], flagged_overlap[f'{column}_flagged'], rtol=1e-10, atol=1e-10)
whole_cohort_summary_rows = []
for method, column in [('nnunet274', 'nnunet274_volume_ml'),
                       ('reported', 'reported_volume_ml'), ('ellipsoid', 'ellipsoid_volume_ml')]:
    error = whole_cohort[column] - whole_cohort.reference_702_volume_ml
    relative = 100 * error / whole_cohort.reference_702_volume_ml
    whole_cohort[f'{method}_minus_reference_702_ml'] = error
    whole_cohort_summary_rows.append(dict(method=method, n=len(error), bias_ml=error.mean(),
        mae_ml=error.abs().mean(), rmse_ml=np.sqrt((error**2).mean()),
        mape_percent=relative.abs().mean(), rmspe_percent=np.sqrt((relative**2).mean())))
whole_cohort_summary = pd.DataFrame(whole_cohort_summary_rows)
whole_cohort.to_csv(OUTPUT_DIR / 'whole_cohort_4109_comparator_volumes.csv', index=False)
whole_cohort_summary.to_csv(OUTPUT_DIR / 'whole_cohort_4109_comparator_summary.csv', index=False)
display(whole_cohort_summary)

# %% [markdown]
# ## Error magnitudes by paired-change subgroup
#
# Among the 65 flagged examinations with nnUNet702 surrogate references, compare
# reported-volume and nnUNet274 absolute errors. Groups are defined by the exact
# sign of model minus reported absolute error. Paired changes are model minus
# reported absolute error (mL), so negative changes indicate improvement.
# These descriptive, outcome-defined groups do not identify causes of disagreement
# or measure spatial segmentation accuracy. No inference or bootstrap analysis is run.

# %%
# Read saved per-case volumes so this calculation does not require mask remeasurement.
subgroup_volumes = pd.read_csv(OUTPUT_DIR / 'outlier_702_comparator_volumes.csv')
assert len(subgroup_volumes) == 65 and subgroup_volumes.ID.is_unique
subgroup_reference = subgroup_volumes.reference_702_volume_ml
assert np.isfinite(subgroup_reference).all() and subgroup_reference.gt(0).all()
subgroup_reported_error = (subgroup_volumes.reported_volume_ml - subgroup_reference).abs()
subgroup_model_error = (subgroup_volumes.nnunet274_volume_ml - subgroup_reference).abs()
subgroup_change = subgroup_model_error - subgroup_reported_error
assert np.isfinite(np.column_stack([subgroup_reported_error, subgroup_model_error])).all()
subgroup_summary_rows = []
for group, selected in [('reduced', subgroup_change.lt(0)),
                         ('increased', subgroup_change.gt(0)),
                         ('unchanged', subgroup_change.eq(0))]:
    row = dict(group=group, n=int(selected.sum()))
    for metric, values in [('reported_absolute_error_ml', subgroup_reported_error),
                            ('nnunet274_absolute_error_ml', subgroup_model_error),
                            ('paired_change_ml', subgroup_change)]:
        row[f'{metric}_median'] = values.loc[selected].median()
        row[f'{metric}_min'] = values.loc[selected].min()
        row[f'{metric}_max'] = values.loc[selected].max()
    subgroup_summary_rows.append(row)
subgroup_error_summary = pd.DataFrame(subgroup_summary_rows)
assert subgroup_error_summary.n.sum() == len(subgroup_volumes)
subgroup_error_summary.to_csv(OUTPUT_DIR / 'outlier_702_error_subgroup_summary.csv', index=False)
display(subgroup_error_summary)
