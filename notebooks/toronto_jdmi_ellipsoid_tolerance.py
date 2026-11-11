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
fig, ax = plt.subplots(figsize=(7.5, 4), constrained_layout=True)
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
    frameon=False, handletextpad=.5, borderaxespad=.5,
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
