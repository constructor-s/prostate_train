# %% [markdown]
# # Whole-gland volume agreement in Toronto JDMI
#
# Compare clinical manual ellipsoid estimates, Multicenter Dataset274, and RUMC
# Dataset236 against median manual-reader planimetry, in mL. The cleaned listing
# supplies one examination per patient; after accession exclusions and requiring
# positive finite reference volumes, the expected cohort is **383 patients**.
# Patient selection is performed upstream in `_hlab20260928_clinical_image_listing.py`.
# Do not select another visit here, including when a reference is unavailable.
#
# Run cells in order. This notebook reads existing internal masks directly and
# performs no downloads or inference. No manual reference masks are available
# here, so these are volume comparisons; DSC is not calculated. PSA density and
# biopsy analyses are outside this version. All outputs remain local under the
# Git-ignored `results/toronto_jdmi_clinical_volume/` directory. Display only
# aggregates; patient identifiers and image paths belong in the local audits.

# %%
import hashlib
from importlib.metadata import version
import itertools
from itertools import combinations
import json
from pathlib import Path
import platform
from textwrap import fill

import matplotlib.pyplot as plt
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
EXCLUDE_JSON = DATA_ROOT / 'exclude.json'
OUTPUT_DIR = REPO_ROOT / 'results/toronto_jdmi_clinical_volume'
EXPECTED_CASES = 383
GEOMETRY_TOLERANCE_MM = 0.5
GEOMETRY_WARNING_THRESHOLD_MM = 1e-3
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
# `exclude.json`, regardless of its reason, and require positive finite manual
# reference volumes. Ellipsoid validity is a measurement check, not an additional
# selection criterion. IDs have the structure `AIPR_{patient_id}_{date}_{scan_id}`.
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
if not required <= set(listing.columns):
    raise ValueError('Clinical listing lacks required columns')
parts = listing.ID.str.extract(r'^AIPR_([^_]+)_([^_]+)_([^_]+)$')
listing['patient_id'] = parts[0]
listing['exam_date'] = parts[1]
listing['accession'] = parts[2]
listing['invalid_id'] = parts.isna().any(axis=1) | listing.ID.str.contains(r'[/\\]', na=True)
listing['duplicate_scan_id'] = listing.ID.duplicated(keep=False)
listing['duplicate_patient_id'] = listing.patient_id.duplicated(keep=False)
exclusions = json.loads(EXCLUDE_JSON.read_text())
if not isinstance(exclusions, list) or any(
    not isinstance(entry, dict) or not isinstance(entry.get('aano'), str)
    or not entry['aano'].strip() for entry in exclusions
):
    raise ValueError('Exclusion JSON requires a list of entries with nonempty aano strings')
exclusion_reasons = {}
for entry in exclusions:
    detail = '; '.join(str(entry[k]) for k in ('reason', 'description') if entry.get(k))
    exclusion_reasons.setdefault(entry['aano'], []).append(detail or 'Listed in exclude.json')
reference_ml = pd.to_numeric(listing.ground_truth_volume_ml, errors='coerce')
listing['invalid_reference_volume'] = ~np.isfinite(reference_ml) | reference_ml.le(0)
listing['accession_excluded'] = listing.accession.isin(exclusion_reasons)
listing['exclusion_reason'] = listing.accession.map(
    {key: '; '.join(value) for key, value in exclusion_reasons.items()}
).fillna('')
listing['included'] = ~listing.invalid_reference_volume & ~listing.accession_excluded
listing.to_csv(OUTPUT_DIR / 'cohort_audit.csv', index=False)
if listing[['invalid_id', 'duplicate_scan_id', 'duplicate_patient_id']].any().any():
    raise ValueError('Invalid or repeated scan/patient IDs; inspect local cohort_audit.csv')
cases = listing.loc[listing.included].copy().reset_index(drop=True)
cases['gt_volume_ml'] = reference_ml.loc[listing.included].to_numpy()
flow = pd.DataFrame([
    dict(stage='Cleaned clinical listing', n=len(listing)),
    dict(stage='After accession exclusions', n=int((~listing.accession_excluded).sum())),
    dict(stage='Positive finite manual reference', n=len(cases)),
])
flow['excluded_since_previous'] = flow.n.shift() - flow.n
flow.to_csv(OUTPUT_DIR / 'cohort_flow.csv', index=False)
display(flow)
if len(cases) != EXPECTED_CASES:
    raise ValueError('Cohort count changed; inspect local cohort_audit.csv')
cohort_sha256 = hashlib.sha256('\n'.join(sorted(cases.ID)).encode()).hexdigest()
metadata_path = OUTPUT_DIR / 'run_metadata.json'
if metadata_path.is_file():
    previous = json.loads(metadata_path.read_text())
    if previous['cohort_sha256'] != cohort_sha256:
        raise ValueError('Cohort membership changed; inspect local cohort_audit.csv')
cases['clinical_volume_ml'] = pd.to_numeric(cases.vol_report_ref, errors='coerce')
if (~np.isfinite(cases.clinical_volume_ml) | cases.clinical_volume_ml.le(0)).any():
    raise ValueError('Invalid ellipsoid measurements; inspect local cohort_audit.csv')
if cases.t2.isna().any() or cases.t2.str.strip().eq('').any():
    raise ValueError('Missing T2 paths; inspect local cohort_audit.csv')
cases['image_path'] = cases.t2.map(lambda p: str((DATA_ROOT / p).resolve()))
if cases.image_path.duplicated().any():
    raise ValueError('Repeated eligible T2 paths; inspect local cohort_audit.csv')

# %% [markdown]
# ## Direct measurements on native prediction grids
#
# Model metadata must identify the expected dataset and whole-gland labels 0/1.
# Both masks and T2 images must be 3D, with spatial units mm and finite,
# invertible affines. Equal shapes and maximum corresponding voxel-corner
# displacement <= 0.5 mm are accepted, matching `model_study_performance.py`.
# Displacements > 0.001 mm are counted below; every displacement is audited.
# No resampling occurs. Whole-gland mL = label-1 voxels × |det(affine)| / 1,000.
# Empty predictions remain zero-volume failures in every statistical calculation.
#
# Hash each input, including compressed masks and T2 images, for local provenance.
# Saved inference metadata is recorded when available, without assuming that
# historical settings were identical across models or independently certified.

# %%
input_hashes = {}
for path in (CLINICAL_CSV, EXCLUDE_JSON):
    with path.open('rb') as stream:
        input_hashes[str(path.relative_to(REPO_ROOT))] = hashlib.file_digest(stream, 'sha256').hexdigest()
model_metadata = {}
model_roots = {}
for key in SELECTED_COMPARATORS:
    model = COMPARATORS[key][1]
    if model is None:
        continue
    model_root = DATA_ROOT / f'seg_out_{model}'
    model_roots[key] = model_root
    saved_metadata = {}
    for filename in ('plans.json', 'dataset.json', 'inference_config.json', 'predict_from_raw_data_args.json'):
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

measurement_rows = []
for case in tqdm(cases.itertuples(index=False), total=len(cases), desc='Measuring Toronto volumes'):
    for key in model_roots:
        path = model_roots[key] / f'{case.ID}.nii.gz'
        record = dict(ID=case.ID, method=key, prediction_path=str(path.relative_to(REPO_ROOT)))
        try:
            pred_img, image = nib.load(path), nib.load(case.image_path)
            if len(pred_img.shape) != 3 or pred_img.shape != image.shape:
                raise ValueError('Prediction and T2 shapes differ or are not 3D')
            for img in (pred_img, image):
                if img.header.get_xyzt_units()[0] != 'mm':
                    raise ValueError('Spatial units must be mm')
                if not np.isfinite(img.affine).all() or abs(np.linalg.det(img.affine[:3, :3])) <= 0:
                    raise ValueError('Invalid spatial affine')
            corners = np.array(list(itertools.product(*[(0, n - 1) for n in image.shape])))
            displacement = float(np.linalg.norm(
                nib.affines.apply_affine(pred_img.affine, corners)
                - nib.affines.apply_affine(image.affine, corners), axis=1,
            ).max())
            record['max_corner_displacement_mm'] = displacement
            if displacement > GEOMETRY_TOLERANCE_MM:
                raise ValueError('Geometry displacement exceeds tolerance')
            mask = np.asanyarray(pred_img.dataobj)
            if not np.isin(mask, [0, 1]).all():
                raise ValueError('Expected binary whole-gland labels 0 and 1')
            voxel_count = int(np.count_nonzero(mask == 1))
            voxel_volume_mm3 = float(abs(np.linalg.det(pred_img.affine[:3, :3])))
            record.update(status='measured', error='', voxel_count=voxel_count,
                          voxel_volume_mm3=voxel_volume_mm3,
                          volume_ml=voxel_count * voxel_volume_mm3 / 1000)
            for input_path in (path, Path(case.image_path)):
                relative_path = str(input_path.relative_to(REPO_ROOT))
                if relative_path not in input_hashes:
                    with input_path.open('rb') as stream:
                        input_hashes[relative_path] = hashlib.file_digest(stream, 'sha256').hexdigest()
        except (OSError, ValueError, nib.filebasedimages.ImageFileError) as error:
            # Details may contain identifiers; keep them in the local audit only.
            record.update(status='invalid', error=str(error))
        measurement_rows.append(record)
measurement_audit = pd.DataFrame(measurement_rows)
measurement_audit.to_csv(OUTPUT_DIR / 'measurement_audit.csv', index=False)
if measurement_audit.status.ne('measured').any():
    raise ValueError('Incomplete or invalid measurements; inspect local measurement_audit.csv')
volumes = cases.copy()
for key in model_roots:
    measured = measurement_audit.loc[measurement_audit.method.eq(key)]
    measured = measured.drop(columns=['method', 'status', 'error']).rename(
        columns={column: f'{key}_{column}' for column in measured.columns if column != 'ID'}
    )
    volumes = volumes.merge(measured, on='ID', validate='one_to_one', how='left')
display(measurement_audit.groupby('method').agg(
    measured=('status', 'size'), empty_predictions=('voxel_count', lambda s: int(s.eq(0).sum())),
    geometry_warnings=('max_corner_displacement_mm', lambda s: int(s.gt(GEOMETRY_WARNING_THRESHOLD_MM).sum())),
))

# %% [markdown]
# ## Volume accuracy against median manual-reader planimetry
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
RELATIVE_DIFFERENCE_LIMITS = (-100, 100)

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
fig.suptitle(f'Toronto JDMI (n = {len(volumes)} patients)', fontsize=10)
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
        f'{name}\n− manual planimetry (mL)',
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
            'Manual planimetry volume\n(mL; log scale)',
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
    pair = volumes[['ID', 'patient_id', 'exam_date']].copy()
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
# ## Local reproducibility record
#
# Keep the complete record in the ignored output directory: source checksums,
# selected membership fingerprint, saved model metadata, analysis parameters,
# package versions, and this notebook's checksum. No clinical records, identifiers,
# passwords, or generated artifacts should be copied into tracked source files.
# Historical inference settings are documented only to the extent that saved
# metadata provides them. A successful notebook run measures existing masks only.

# %%
notebook_path = REPO_ROOT / 'notebooks/toronto_jdmi_clinical_volume.py'
with notebook_path.open('rb') as stream:
    notebook_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
metadata = dict(
    cohort_sha256=cohort_sha256, n_patients=len(volumes), n_scans=len(volumes),
    reference='Median manual-reader planimetry', volume_unit='mL',
    selected_comparators=SELECTED_COMPARATORS, expected_cases=EXPECTED_CASES,
    geometry_tolerance_mm=GEOMETRY_TOLERANCE_MM,
    geometry_warning_threshold_mm=GEOMETRY_WARNING_THRESHOLD_MM,
    n_bootstrap=N_BOOTSTRAP, random_seed=RANDOM_SEED, bootstrap_test_seed=RANDOM_SEED + 1,
    bootstrap_batch=BOOTSTRAP_BATCH, alpha=ALPHA, confidence_level=CONFIDENCE_LEVEL,
    relative_difference_display_limits=RELATIVE_DIFFERENCE_LIMITS,
    packages={name: version(name) for name in (
        'numpy', 'pandas', 'nibabel', 'matplotlib', 'scipy', 'statsmodels', 'tqdm', 'ipython',
    )},
    python_version=platform.python_version(), notebook_sha256=notebook_sha256,
    input_sha256=input_hashes, model_metadata=model_metadata,
)
metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
print(f'Completed Toronto JDMI volume analysis: {len(volumes)} patients, three methods.')
