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
SELECTED_COMPARATORS = ['clinical', 'nnunet274', 'bosma22b']
COMPARATORS = {
    'clinical': ('Manual ellipsoid', None, [1]),
    'nnunet274': ('External (public)', None, [1]),
    'bosma22b': ('Internal (institutional)', 'whole_gland/AI/Bosma22b', [1]),
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
# Clinical volume is reported in the marksheet and labeled "Manual ellipsoid"
# in this analysis. PI-CAI masks are matched by patient/study ID. Bosma22b and
# Guerbet23 use whole-gland label 1; HeviAI23 and Yuan23 use PZ=1 and TZ=2.
# Their union is measured in memory. No resampling or spatial evaluation is
# performed. Empty predictions remain zero-volume failures; invalid inputs stop.

# %%
MODEL_ROOT = REPO_ROOT / 'results/prostatex_pred_pooled2_3d_fullres'
if 'nnunet274' in SELECTED_COMPARATORS:
    assert json.loads((MODEL_ROOT / 'plans.json').read_text())['dataset_name'] == 'Dataset274_raw'
    assert json.loads((MODEL_ROOT / 'dataset.json').read_text())['labels'] == {'background': 0, 'class1': 1}
for key in SELECTED_COMPARATORS:
    if key == 'clinical':
        continue
    _, source, foreground = COMPARATORS[key]
    prediction_rows = []
    for case in linked.itertuples(index=False):
        if key == 'nnunet274':
            path = MODEL_ROOT / f'{case.ID}.nii.gz'
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
# ## Volume accuracy against manual whole-gland ground truth
#
# **Manual ellipsoid** is the reported clinical volume. **External (public)**
# identifies Dataset274 nnUNet, trained on public data external to this evaluation
# dataset. **Internal (institutional)** identifies Bosma22b, from the same
# institution as the evaluation dataset.
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
    'bosma22b': dict(color='#91A78C', marker='^', linestyle=':'),
    'guerbet23': dict(color='#806A88', marker='D', linestyle='-.'),
    'heviai23': dict(color='#AD7773', marker='v', linestyle=(0, (5, 1, 1, 1))),
    'yuan23': dict(color='#C2B18A', marker='P', linestyle=(0, (3, 1, 1, 1, 1, 1))),
}
scatter_options = dict(s=16, edgecolors='0.1', linewidths=.4)
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
    panel_name = fill(name, width=16)
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
            linestyle=style['linestyle'], linewidth=1.2, label=fill(COMPARATORS[key][0], width=16),
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
# absolute errors are ties. All methods use the same 190 cases.

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
