# %% [markdown]
# # Quality improvement through manual–automated volume disagreement
#
# Can unusually discordant manual ellipsoid and nnUNet274 volumes identify
# patients who may benefit from measurement review? Screen without reference
# volumes, then evaluate paired accuracy against manual references (ProstateX
# and Toronto) or automated surrogates (expanded Toronto and PI-CAI). A flag does not
# establish which measurement is wrong or identify an error's cause.
#
# Read existing local exports; run picai_test_clinical_volume.py and
# prostatex_linked_clinical_volume.py first
# after changing the reviewed testing manifest, then run these cells in order.
# The manifest separates superior/inferior mask-boundary exclusions from
# mapping/clinical ineligibility; those cases remain in upstream audits but
# do not contribute to thresholds, outlier tables, PSAD, or biopsy analyses.
# No downloads, inference,
# workbook decryption, or patient reselection are performed. The expanded
# directory retains its historical `4111` name; current membership comes from
# the source notebook and saved provenance. Toronto cohorts overlap and must
# not be pooled as independent observations. Local review tables print outlier
# case identifiers, available image paths, and method volumes for inspection.

# %% [markdown]
# Select one institution; all estimates and outputs below use this site only.
# Run independently for PCNN, RUMC, and ZGT. Existing pooled exports are historical.

# %%
SITE = globals().get('SITE', 'PCNN')
assert SITE in ('PCNN', 'RUMC', 'ZGT'), SITE
TESTING_FOLD = f'testing_{SITE}'
BUILD_ALL_SITE_FIGURE = globals().get('BUILD_ALL_SITE_FIGURE', False)

# %%
import ast
import hashlib
import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

REPO_ROOT = Path(__file__).resolve().parents[1] if '__file__' in globals() else Path.cwd().resolve()
if REPO_ROOT.name == 'notebooks':
    REPO_ROOT = REPO_ROOT.parent
assert (REPO_ROOT / 'pyproject.toml').is_file(), 'Establish the repository root'
OUTPUT_DIR = REPO_ROOT / 'results/volume_disagreement_quality_improvement' / SITE
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
COHORTS = {
    'prostatex': ('prostatex_linked_volume', 'ProstateX', 'Manual whole-gland segmentation'),
    'toronto_manual': ('toronto_jdmi_clinical_volume', 'Toronto manual', 'Median manual-reader planimetry'),
    'toronto_surrogate': ('toronto_jdmi_4111_clinical_volume', 'Toronto surrogate', 'nnUNet702 automated surrogate'),
    'picai': ('picai_test_clinical_volume', 'PI-CAI', 'Bosma22b automated surrogate'),
}
METHODS = ['clinical', 'nnunet274']
SCREENS = ['both', 'absolute', 'relative', 'either']
PSAD_CUTOFFS = np.array([.10, .15, .20])
PSAD_STRATA = ['0–<0.10', '0.10–<0.15', '0.15–<0.20', '≥0.20']
N_BOOTSTRAP = 100_000
RANDOM_SEED = 20261102
BOOTSTRAP_BATCH = 1_000
ALPHA = .05
CONFIDENCE_LEVEL = .95

# %% [markdown]
# ## Verify inputs and cohort membership
#
# One row per patient/examination; all volume and biopsy audits must describe
# exactly the same selected cohort. Compare saved membership hashes and source
# notebook counts. Reference volumes are in mL; they must be finite positive.
# A zero automated volume remains a volume failure but cannot yield valid PSAD.
# Hash every input for reproducibility. Reject stale PSAD/biopsy exports rather
# than silently mixing runs. Source notebooks remain responsible for exclusion
# selection, mask labels, spatial units, and native-grid volume measurements.

# %%
cohorts = {}
provenance = []
for cohort_key, (directory, label, reference_name) in COHORTS.items():
    root = REPO_ROOT / 'results' / directory
    if cohort_key == 'picai':
        root = root / SITE
        label = f'PI-CAI {SITE}'
    tables = {}
    input_hashes = {}
    for table_name in ['comparator_volumes', 'psa_density_per_case',
                       'psa_density_biopsy_audit', 'psa_density_biopsy_per_case']:
        path = root / f'{table_name}.csv'
        input_hashes[str(path.relative_to(REPO_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        tables[table_name] = pd.read_csv(path, low_memory=False, float_precision='round_trip')
        assert tables[table_name].ID.is_unique, (cohort_key, table_name, 'duplicate ID')
    volume = tables['comparator_volumes'].set_index('ID', drop=False)
    audit = tables['psa_density_biopsy_audit'].set_index('ID', drop=False)
    assert set(volume.index) == set(audit.index), (cohort_key, 'volume/biopsy membership')
    audit = audit.loc[volume.index]
    for column in ['gt_volume_ml', 'clinical_volume_ml', 'nnunet274_volume_ml']:
        assert np.isfinite(volume[column]).all(), (cohort_key, column)
        assert np.allclose(volume[column], audit[column], rtol=0, atol=1e-10), (cohort_key, column, 'stale audit')
    assert volume.gt_volume_ml.gt(0).all() and volume.clinical_volume_ml.gt(0).all()
    assert volume.nnunet274_volume_ml.ge(0).all()
    excluded_ids = []
    excluded_folds = {}
    membership_hash = hashlib.sha256('\n'.join(sorted(volume.index)).encode()).hexdigest()
    if cohort_key in ('prostatex', 'picai'):
        manifest_path = REPO_ROOT / ('configs/picai_test_clinical_folds.json' if cohort_key == 'picai'
                                     else 'configs/rumcnotprostatexnoclinical_folds.json')
        manifest = json.loads(manifest_path.read_text())
        testing_folds = (TESTING_FOLD,) if cohort_key == 'picai' else ('testing',)
        testing_ids = [row['id'] for fold in testing_folds for row in manifest[fold]]
        excluded_folds = {name: [row['id'] for row in rows] for name, rows in manifest.items()
                          if name == 'excluded' or name.startswith('excluded_')}
        excluded_ids = [case_id for ids in excluded_folds.values() for case_id in ids]
        assert len(excluded_ids) == len(set(excluded_ids)), 'Duplicate IDs across exclusion folds'
        assert testing_ids and len(testing_ids) == len(set(testing_ids)), 'Empty or duplicate testing IDs'
        assert not set(testing_ids) & set(excluded_ids), 'Testing/excluded manifest overlap'
        assert set(volume.index) == set(testing_ids), (
            f'{cohort_key} exports differ from the reviewed testing fold; rerun its source notebook')
        assert volume.patient_id.is_unique
        source_metadata_path = root / 'run_metadata.json'
        source_metadata = json.loads(source_metadata_path.read_text())
        manual_postprocessing = source_metadata.get('manual_postprocessing')
        if manual_postprocessing is not None:
            manual_record_path = REPO_ROOT / manual_postprocessing['path']
            assert hashlib.sha256(manual_record_path.read_bytes()).hexdigest() == manual_postprocessing['sha256'], 'Stale postprocessing provenance'
        input_hashes[str(source_metadata_path.relative_to(REPO_ROOT))] = hashlib.sha256(source_metadata_path.read_bytes()).hexdigest()
        if cohort_key == 'picai':
            assert volume.center.eq(SITE).all()
            reserved_patients = {int(row['id'].split('_')[1]) for row in manifest['training']}
            assert not set(volume.patient_id) & reserved_patients, 'Training-patient leakage'
            metadata_path = root / 'run_metadata.json'
            metadata = json.loads(metadata_path.read_text())
            assert metadata['cohort_sha256'] == membership_hash and metadata['n_patients'] == len(volume)
            for relative_path, digest in metadata['input_sha256'].items():
                assert hashlib.sha256((REPO_ROOT / relative_path).read_bytes()).hexdigest() == digest, 'Stale PI-CAI inputs'
            input_hashes[str(metadata_path.relative_to(REPO_ROOT))] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
        input_hashes[str(manifest_path.relative_to(REPO_ROOT))] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    else:
        metadata_path = root / 'run_metadata.json'
        metadata = json.loads(metadata_path.read_text())
        assert metadata['cohort_sha256'] == membership_hash
        assert metadata['n_patients'] == len(volume)
        source_path = REPO_ROOT / 'notebooks' / f'{directory}.py'
        source_tree = ast.parse(source_path.read_text())
        source_count = next(ast.literal_eval(node.value) for node in source_tree.body
                            if isinstance(node, ast.Assign)
                            and any(isinstance(target, ast.Name) and target.id == 'EXPECTED_CASES'
                                    for target in node.targets))
        assert source_count == len(volume), 'Source notebook and export counts differ; refresh upstream outputs'
        assert volume.amrn.is_unique
        input_hashes[str(metadata_path.relative_to(REPO_ROOT))] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
        input_hashes[str(source_path.relative_to(REPO_ROOT))] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    # Preserve the source's selected PSA, including Toronto report/alternative
    # consolidation and documented correction; raw `psa` is not equivalent.
    psa_column = 'psa_numeric_ng_ml' if cohort_key in ('prostatex', 'picai') else 'psa_consolidated'
    volume['psa'] = pd.to_numeric(audit[psa_column], errors='coerce')
    for method in ['gt', *METHODS]:
        volume[f'{method}_psad'] = volume.psa / volume[f'{method}_volume_ml'].where(volume[f'{method}_volume_ml'].gt(0))
    volume['psad_evaluable'] = (
        np.isfinite(volume.psa) & volume.psa.gt(0)
        & np.isfinite(volume[['gt_psad', 'clinical_psad', 'nnunet274_psad']]).all(axis=1)
    )
    saved_psad = tables['psa_density_per_case'].set_index('ID')
    assert set(saved_psad.index) == set(volume.index[volume.psad_evaluable])
    for method in ['gt', *METHODS]:
        assert np.allclose(saved_psad[f'{method}_psad'], volume.loc[saved_psad.index, f'{method}_psad'])
        recalculated = np.searchsorted(PSAD_CUTOFFS, saved_psad[f'{method}_psad'], side='right')
        saved_categories = saved_psad[f'{method}_psad_stratum'].map(dict(zip(PSAD_STRATA, range(4))))
        assert np.array_equal(recalculated, saved_categories)
        recomputed_categories = np.searchsorted(PSAD_CUTOFFS, volume.loc[saved_psad.index, f'{method}_psad'], side='right')
        assert np.array_equal(recomputed_categories, saved_categories), 'Recomputed PSAD crosses a saved category boundary'
    volume['pirads_3'] = audit.biopsy_pirads_status.isin(['maximum_pirads_3', 'pirads_3'])
    if cohort_key in ('prostatex', 'picai'):
        volume['outcome_known'] = audit.biopsy_outcome_status.eq('valid_case_isup')
        volume['cancer_positive'] = audit.case_isup_numeric.ge(2)
    else:
        volume['outcome_known'] = audit.biopsy_outcome_status.eq('known_cscancer')
        volume['cancer_positive'] = audit.case_cancer_positive.eq(True)
    known_biopsy = volume.psad_evaluable & volume.pirads_3 & volume.outcome_known
    assert set(volume.index[known_biopsy]) == set(tables['psa_density_biopsy_per_case'].ID)
    saved_biopsy = tables['psa_density_biopsy_per_case'].set_index('ID')
    for method in ['gt', *METHODS]:
        assert np.array_equal(volume.loc[saved_biopsy.index, f'{method}_psad'].ge(.15),
                              saved_biopsy[f'{method}_biopsy_recommended']), 'Saved biopsy decisions differ'
    cohorts[cohort_key] = volume
    provenance.append(dict(cohort=cohort_key, name=label, n=len(volume), reference=reference_name,
                           cohort_sha256=membership_hash, input_sha256=input_hashes,
                           excluded_ids=excluded_ids, excluded_folds=excluded_folds,
                           manual_postprocessing=source_metadata.get('manual_postprocessing')
                           if cohort_key in ('prostatex', 'picai') else None))

assert set(cohorts['toronto_manual'].index) <= set(cohorts['toronto_surrogate'].index)
print('Toronto manual-reference patients are included in expanded Toronto; analyses are not independent.')
shared_picai_prostatex = set(cohorts['picai'].patient_id) & set(cohorts['prostatex'].patient_id)
print(f'PI-CAI and ProstateX share {len(shared_picai_prostatex)} patients; do not pool cohorts.')
display(pd.DataFrame(provenance)[['cohort', 'n', 'reference']])

# %% [markdown]
# ## Define disagreement screens without reference volumes
#
# D = manual − automated (mL); Q = 100 D / pair mean (%).
# Fit mean ± 1.96 sample SD once on each COMPLETE volume cohort. Flag strict
# exceedances of both intervals for the primary analysis, irrespective of tail.
# Individual-scale and union screens are descriptive sensitivities. These are
# empirical disagreement limits, not clinical acceptability thresholds. Extreme
# observations can inflate SD and mask other discrepancies.

# %%
threshold_rows = []
flag_tables = []
for cohort_key, volume in cohorts.items():
    difference = volume.clinical_volume_ml - volume.nnunet274_volume_ml
    pair_mean = (volume.clinical_volume_ml + volume.nnunet274_volume_ml) / 2
    volume['disagreement_ml'] = difference
    volume['disagreement_percent'] = 100 * difference / pair_mean
    for scale, screen in [('ml', 'absolute'), ('percent', 'relative')]:
        values = volume[f'disagreement_{scale}']
        bias, sd = values.mean(), values.std(ddof=1)
        lower, upper = bias - 1.96 * sd, bias + 1.96 * sd
        volume[f'flag_{screen}'] = values.lt(lower) | values.gt(upper)
        threshold_rows.append(dict(cohort=cohort_key, scale=scale, n=len(volume),
                                   bias=bias, sd=sd, loa_lower=lower, loa_upper=upper))
    volume['flag_both'] = volume.flag_absolute & volume.flag_relative
    volume['flag_either'] = volume.flag_absolute | volume.flag_relative
    for method in METHODS:
        volume[f'{method}_error_ml'] = volume[f'{method}_volume_ml'] - volume.gt_volume_ml
        volume[f'{method}_absolute_error_ml'] = volume[f'{method}_error_ml'].abs()
    volume['absolute_error_reduction_ml'] = volume.clinical_absolute_error_ml - volume.nnunet274_absolute_error_ml
    export_columns = ['ID', 'gt_volume_ml', 'clinical_volume_ml', 'nnunet274_volume_ml',
                      'disagreement_ml', 'disagreement_percent', *[f'flag_{s}' for s in SCREENS],
                      'clinical_error_ml', 'nnunet274_error_ml', 'absolute_error_reduction_ml',
                      'psad_evaluable', 'pirads_3', 'outcome_known']
    flag_tables.append(volume[export_columns].assign(cohort=cohort_key))

thresholds = pd.DataFrame(threshold_rows)
flags = pd.concat(flag_tables, ignore_index=True)
flags.to_csv(OUTPUT_DIR / 'case_audit.csv', index=False)
thresholds.to_csv(OUTPUT_DIR / 'disagreement_thresholds.csv', index=False)
for flag_cohort in ['prostatex', 'toronto_manual', 'picai']:
    plot_root = REPO_ROOT / 'results' / COHORTS[flag_cohort][0]
    if flag_cohort == 'picai':
        plot_root = plot_root / SITE
    plot_flag_path = plot_root / 'manual_automated_disagreement_flags.csv'
    if not plot_flag_path.is_file():
        raise FileNotFoundError('Run the corresponding source notebook through its agreement-figure cell first')
    plot_flags = pd.read_csv(plot_flag_path, float_precision='round_trip').set_index('ID')
    px = cohorts[flag_cohort]
    assert set(plot_flags.index) == set(px.index)
    for source_column, analysis_column in [('outside_ml', 'flag_absolute'), ('outside_percent', 'flag_relative'),
                                            ('outside_both', 'flag_both'), ('outside_either', 'flag_either')]:
        assert np.array_equal(plot_flags.loc[px.index, source_column], px[analysis_column]), 'Figure/analysis flags differ'
    for column in ['disagreement_ml', 'disagreement_percent']:
        assert np.allclose(plot_flags.loc[px.index, column], px[column], rtol=0, atol=1e-10)
    flag_provenance = next(row for row in provenance if row['cohort'] == flag_cohort)
    flag_provenance['input_sha256'][str(plot_flag_path.relative_to(REPO_ROOT))] = hashlib.sha256(plot_flag_path.read_bytes()).hexdigest()
display(thresholds)

# %% [markdown]
# ## Volume accuracy and review burden
#
# Evaluate against manual references in ProstateX/Toronto, nnUNet702 in
# expanded Toronto, and Bosma22b in PI-CAI. All percentage errors use the positive reference volume,
# not the pair mean used for screening. Positive paired reduction favors nnUNet274.
# A patient's winner is identical on absolute and percentage error scales.
# Improvement among all flagged cases and closer probability among non-ties
# have different denominators; report both. Worsening summaries use only losses.
#
# First inspect every primary (strict-intersection) outlier. Include all automated
# method volumes available in each source export, even though the accuracy
# comparison below is restricted to clinical ellipsoid and nnUNet274. All volumes
# are in mL; `gt_volume_ml` is the cohort's reference, which is an automated
# surrogate in expanded Toronto and PI-CAI. Image paths are retained as saved upstream.

# %%
outlier_case_tables = []
for cohort_key, volume in cohorts.items():
    _, cohort_name, reference_name = COHORTS[cohort_key]
    review_columns = ['ID']
    image_column = next((column for column in ['image_path', 't2'] if column in volume), None)
    if image_column is not None:
        review_columns.append(image_column)
    review_columns.extend(['gt_volume_ml', 'clinical_volume_ml'])
    # Automated measurements have a matching saved mask/prediction path. This
    # avoids confusing raw/manual-reference aliases or volume differences with
    # additional automated methods.
    for column in volume.columns:
        if column.endswith('_volume_ml') and column not in review_columns:
            method = column.removesuffix('_volume_ml')
            if f'{method}_path' in volume or f'{method}_prediction_path' in volume:
                review_columns.append(column)
    outlier_cases = volume.loc[volume.flag_both, review_columns].copy()
    outlier_cases.insert(0, 'reference', reference_name)
    outlier_cases.insert(0, 'cohort', cohort_key)
    outlier_case_tables.append(outlier_cases)
    print(f'\n{cohort_name}: strict-intersection outliers (n = {len(outlier_cases)}); volumes in mL')
    print(f'Reference: {reference_name}')
    print(outlier_cases.drop(columns=['cohort', 'reference']).to_string(
        index=False, float_format=lambda value: f'{value:.3f}',
    ))
outlier_case_volumes = pd.concat(outlier_case_tables, ignore_index=True)
outlier_case_volumes.to_csv(OUTPUT_DIR / 'outlier_case_volumes.csv', index=False)

# %%
volume_rows = []
paired_rows = []
for cohort_key, volume in cohorts.items():
    for screen in SCREENS:
        selected = volume.loc[volume[f'flag_{screen}']]
        n = len(selected)
        for method in METHODS:
            error = selected[f'{method}_error_ml'].to_numpy()
            percent = 100 * error / selected.gt_volume_ml.to_numpy()
            volume_rows.append(dict(cohort=cohort_key, screen=screen, method=method, n=n,
                                    mae_ml=np.mean(abs(error)) if n else np.nan,
                                    rmse_ml=np.sqrt(np.mean(error**2)) if n else np.nan,
                                    mape_percent=np.mean(abs(percent)) if n else np.nan,
                                    rmspe_percent=np.sqrt(np.mean(percent**2)) if n else np.nan))
        reduction = selected.absolute_error_reduction_ml
        wins, losses, ties = int(reduction.gt(0).sum()), int(reduction.lt(0).sum()), int(reduction.eq(0).sum())
        non_ties = wins + losses
        if non_ties:
            sign = stats.binomtest(wins, non_ties, p=.5)
            ci = sign.proportion_ci(confidence_level=CONFIDENCE_LEVEL, method='exact')
            probability, ci_lower, ci_upper, sign_p = wins / non_ties, ci.low, ci.high, sign.pvalue
        else:
            probability = ci_lower = ci_upper = np.nan
            sign_p = 1.0 if n else np.nan
        worsening = -reduction[reduction.lt(0)]
        manual_metrics, automated_metrics = volume_rows[-2:]
        metric_reductions = {
            output: manual_metrics[metric] - automated_metrics[metric]
            for metric, output in [('mae_ml', 'mae_reduction_ml'), ('rmse_ml', 'rmse_reduction_ml'),
                                   ('mape_percent', 'mape_reduction_pp'), ('rmspe_percent', 'rmspe_reduction_pp')]
        }
        paired_rows.append(dict(cohort=cohort_key, screen=screen, n_cohort=len(volume), n=n,
                                review_fraction=n / len(volume), improved=wins, worsened=losses, tied=ties,
                                improved_fraction_all=wins / n if n else np.nan,
                                n_non_ties=non_ties, closer_probability=probability,
                                closer_ci_lower=ci_lower, closer_ci_upper=ci_upper,
                                closer_exact_p=sign_p if screen == 'both' else np.nan,
                                mean_absolute_error_reduction_ml=reduction.mean(),
                                median_absolute_error_reduction_ml=reduction.median(),
                                worsening_median_ml=worsening.median(), worsening_max_ml=worsening.max(),
                                clinical_mae_in_worsened_ml=selected.loc[reduction.lt(0), 'clinical_absolute_error_ml'].mean(),
                                automated_mae_in_worsened_ml=selected.loc[reduction.lt(0), 'nnunet274_absolute_error_ml'].mean(),
                                **metric_reductions))
volume_summary = pd.DataFrame(volume_rows)
paired_summary = pd.DataFrame(paired_rows)
volume_summary.to_csv(OUTPUT_DIR / 'volume_summary.csv', index=False)
paired_summary.to_csv(OUTPUT_DIR / 'paired_volume_summary.csv', index=False)
display(volume_summary.loc[volume_summary.screen.eq('both')])
display(paired_summary.loc[paired_summary.screen.eq('both')])

# %% [markdown]
# ## Paired volume inference in the primary intersection
#
# Retain the source notebook's null-centered bootstrap-t tests, paired BCa
# intervals (percentile fallback if degenerate), exact sign test, and paired
# squared-error t-test sensitivity. Use 100,000 resamples and seeds
# 20261102/20261103. Thresholds and selected patients are fixed for inference:
# intervals/tests do not capture threshold estimation, model training, or
# post-selection uncertainty and are exploratory, especially at small n.
# Holm jointly covers RMSE and closer tests within each cohort (two tests).
# RMSPE is a separate one-comparison family. Intervals are pointwise, unadjusted.
# Secondary screens receive no significance tests.

# %%
# SciPy requires a vectorized statistic for paired resampling and jackknifing.
def rmse_improvement(error_a, error_b, axis=-1):
    return np.sqrt(np.mean(error_b**2, axis=axis)) - np.sqrt(np.mean(error_a**2, axis=axis))


# %%
statistics_rows = []
for cohort_key, cohort_volume in cohorts.items():
    a, b = 'nnunet274', 'clinical'
    volumes = cohort_volume.loc[cohort_volume.flag_both]
    if len(volumes) < 2:
        paired = paired_summary.loc[paired_summary.cohort.eq(cohort_key) & paired_summary.screen.eq('both')].iloc[0]
        statistics_rows.append(dict(cohort=cohort_key, n=len(volumes),
                                    inference_status='unavailable_fewer_than_two',
                                    closer_exact_p=paired.closer_exact_p))
        continue
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
        cohort=cohort_key, inference_status='defined',
        method_a=a, name_a='nnUNet274', method_b=b, name_b='Manual ellipsoid',
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
volume_statistics = pd.DataFrame(statistics_rows)
for column in ['rmse_bootstrap_p', 'rmse_t_sensitivity_p', 'rmspe_bootstrap_p', 'rmspe_t_sensitivity_p']:
    if column not in volume_statistics:
        volume_statistics[column] = np.nan
for index, row in volume_statistics.iterrows():
    primary = np.array([row.rmse_bootstrap_p, row.closer_exact_p], dtype=float)
    sensitivity = np.array([row.rmse_t_sensitivity_p, row.closer_exact_p], dtype=float)
    adjusted = multipletests(np.nan_to_num(primary, nan=1.), method='holm')[1]
    sensitivity_adjusted = multipletests(np.nan_to_num(sensitivity, nan=1.), method='holm')[1]
    for endpoint_index, endpoint in enumerate(['rmse_bootstrap', 'closer_exact']):
        volume_statistics.loc[index, f'{endpoint}_p_holm'] = adjusted[endpoint_index] if np.isfinite(primary[endpoint_index]) else np.nan
    volume_statistics.loc[index, 'rmse_t_sensitivity_p_holm'] = sensitivity_adjusted[0] if np.isfinite(sensitivity[0]) else np.nan
    volume_statistics.loc[index, 'closer_exact_p_holm_sensitivity'] = sensitivity_adjusted[1] if np.isfinite(sensitivity[1]) else np.nan
# One method pair per cohort: the separate RMSPE family has size one.
volume_statistics['rmspe_bootstrap_p_holm'] = volume_statistics.rmspe_bootstrap_p
volume_statistics['rmspe_t_sensitivity_p_holm'] = volume_statistics.rmspe_t_sensitivity_p
volume_statistics['holm_family_size'] = 2
volume_statistics['rmspe_holm_family_size'] = 1
volume_statistics.to_csv(OUTPUT_DIR / 'primary_volume_statistics.csv', index=False)
display(volume_statistics)


# %% [markdown]
# ## PSAD category changes within the fixed volume flags
#
# Use the same PSA for all volume methods. Cutoff equality enters the higher
# stratum. Exact correctness and ordinal distance answer different questions;
# a move closer can remain incorrect. Record corrected/newly incorrect,
# both-correct/both-incorrect, and lower/same/higher category transitions.
# Exact McNemar tests compare correctness only in the primary intersection,
# separately by cohort (one-comparison Holm family). Unknown cancer outcomes
# do not affect this analysis.

# %%
psad_rows, transition_rows, psad_case_tables = [], [], []
for cohort_key, volume in cohorts.items():
    for screen in SCREENS:
        selected = volume.loc[volume[f'flag_{screen}'] & volume.psad_evaluable].copy()
        n = len(selected)
        for method in ['gt', *METHODS]:
            selected[f'{method}_stratum'] = np.searchsorted(PSAD_CUTOFFS, selected[f'{method}_psad'], side='right')
        manual_correct = selected.clinical_stratum.eq(selected.gt_stratum)
        auto_correct = selected.nnunet274_stratum.eq(selected.gt_stratum)
        corrected = int((~manual_correct & auto_correct).sum())
        newly_incorrect = int((manual_correct & ~auto_correct).sum())
        both_correct = int((manual_correct & auto_correct).sum())
        both_incorrect = int((~manual_correct & ~auto_correct).sum())
        distance_reduction = ((selected.clinical_stratum - selected.gt_stratum).abs()
                              - (selected.nnunet274_stratum - selected.gt_stratum).abs())
        changed = selected.clinical_stratum.ne(selected.nnunet274_stratum)
        table = np.array([[both_correct, newly_incorrect], [corrected, both_incorrect]])
        p = (float(mcnemar(table, exact=True).pvalue) if corrected + newly_incorrect else 1.0) if n else np.nan
        psad_rows.append(dict(cohort=cohort_key, screen=screen, n_flagged=int(volume[f'flag_{screen}'].sum()),
                              n=n, excluded_psad=int((volume[f'flag_{screen}'] & ~volume.psad_evaluable).sum()),
                              category_changed=int(changed.sum()), clinical_correct=int(manual_correct.sum()),
                              automated_correct=int(auto_correct.sum()), corrected=corrected,
                              newly_incorrect=newly_incorrect, both_correct=both_correct, both_incorrect=both_incorrect,
                              ordinal_closer=int(distance_reduction.gt(0).sum()), ordinal_farther=int(distance_reduction.lt(0).sum()),
                              ordinal_equal=int(distance_reduction.eq(0).sum()),
                              mcnemar_exact_p=p if screen == 'both' else np.nan,
                              mcnemar_exact_p_holm=p if screen == 'both' else np.nan,
                              holm_family_size=1 if screen == 'both' else 0))
        for manual_stratum in range(4):
            for auto_stratum in range(4):
                transition_rows.append(dict(cohort=cohort_key, screen=screen, clinical_stratum=manual_stratum,
                                             automated_stratum=auto_stratum,
                                             n=int((selected.clinical_stratum.eq(manual_stratum)
                                                    & selected.nnunet274_stratum.eq(auto_stratum)).sum())))
        psad_case_tables.append(selected[['ID', 'gt_psad', 'clinical_psad', 'nnunet274_psad',
                                           'gt_stratum', 'clinical_stratum', 'nnunet274_stratum']]
                                .assign(cohort=cohort_key, screen=screen, ordinal_distance_reduction=distance_reduction))
psad_summary = pd.DataFrame(psad_rows)
psad_summary.to_csv(OUTPUT_DIR / 'psad_category_summary.csv', index=False)
pd.DataFrame(transition_rows).to_csv(OUTPUT_DIR / 'psad_category_transitions.csv', index=False)
pd.concat(psad_case_tables, ignore_index=True).to_csv(OUTPUT_DIR / 'psad_per_case.csv', index=False)
display(psad_summary.loc[psad_summary.screen.eq('both')])

# %% [markdown]
# ## PI-RADS 3: reference decisions and recorded cancer outcomes
#
# First compare unrounded PSAD ≥ 0.15 decisions with reference-volume decisions
# for ALL evaluable PI-RADS 3 outliers, even with unknown cancer outcomes.
# Then assess known outcomes separately: ProstateX ISUP ≥ 2; Toronto CSCANCER
# Y versus N. Reference decisions are derived from volume, not cancer labels.
# Avoidance includes false negatives. Sensitivity/specificity use positive/
# negative outcome denominators; undefined rates are NaN. No new biopsy tests.

# %%
decision_rows, outcome_rows, eligibility_rows, biopsy_case_tables = [], [], [], []
for cohort_key, volume in cohorts.items():
    for screen in SCREENS:
        flagged = volume.loc[volume[f'flag_{screen}']]
        eligible = flagged.loc[flagged.psad_evaluable & flagged.pirads_3].copy()
        n = len(eligible)
        for method in ['gt', *METHODS]:
            eligible[f'{method}_recommend'] = eligible[f'{method}_psad'].ge(.15)
        manual_correct = eligible.clinical_recommend.eq(eligible.gt_recommend)
        auto_correct = eligible.nnunet274_recommend.eq(eligible.gt_recommend)
        decision_rows.append(dict(cohort=cohort_key, screen=screen, n=n,
                                  decisions_changed=int(eligible.clinical_recommend.ne(eligible.nnunet274_recommend).sum()),
                                  clinical_reference_agreement=int(manual_correct.sum()),
                                  automated_reference_agreement=int(auto_correct.sum()),
                                  corrected=int((~manual_correct & auto_correct).sum()),
                                  newly_incorrect=int((manual_correct & ~auto_correct).sum()),
                                  both_correct=int((manual_correct & auto_correct).sum()),
                                  both_incorrect=int((~manual_correct & ~auto_correct).sum())))
        known = eligible.loc[eligible.outcome_known]
        eligibility_rows.append(dict(cohort=cohort_key, screen=screen, n_flagged=len(flagged),
                                     excluded_psad=int((~flagged.psad_evaluable).sum()),
                                     excluded_not_pirads3=int((flagged.psad_evaluable & ~flagged.pirads_3).sum()),
                                     n_reference_decisions=n, excluded_unknown_outcome=int((~eligible.outcome_known).sum()),
                                     n_known_outcomes=len(known)))
        for method in ['gt', *METHODS]:
            recommend = known[f'{method}_recommend']
            positive = known.cancer_positive
            tp, fn = int((recommend & positive).sum()), int((~recommend & positive).sum())
            fp, tn = int((recommend & ~positive).sum()), int((~recommend & ~positive).sum())
            total, n_positive, n_negative = len(known), tp + fn, fp + tn
            outcome_rows.append(dict(cohort=cohort_key, screen=screen, method=method, n=total,
                                      tp=tp, fn=fn, fp=fp, tn=tn, n_positive=n_positive, n_negative=n_negative,
                                      sensitivity=tp / n_positive if n_positive else np.nan,
                                      specificity=tn / n_negative if n_negative else np.nan,
                                      biopsies_avoided=fn + tn, avoidance_fraction=(fn + tn) / total if total else np.nan,
                                      outcome_accuracy=(tp + tn) / total if total else np.nan))
        biopsy_case_tables.append(eligible[['ID', 'gt_psad', 'clinical_psad', 'nnunet274_psad',
                                             'gt_recommend', 'clinical_recommend', 'nnunet274_recommend',
                                             'outcome_known', 'cancer_positive']].assign(cohort=cohort_key, screen=screen))
decision_summary = pd.DataFrame(decision_rows)
biopsy_summary = pd.DataFrame(outcome_rows)
eligibility_summary = pd.DataFrame(eligibility_rows)
decision_summary.to_csv(OUTPUT_DIR / 'biopsy_reference_decision_summary.csv', index=False)
biopsy_summary.to_csv(OUTPUT_DIR / 'biopsy_outcome_summary.csv', index=False)
eligibility_summary.to_csv(OUTPUT_DIR / 'biopsy_eligibility.csv', index=False)
biopsy_cases = pd.concat(biopsy_case_tables, ignore_index=True)
# Unknown outcomes remain missing in the case export, never implied negative.
biopsy_cases['cancer_positive'] = biopsy_cases.cancer_positive.astype('boolean')
biopsy_cases.loc[~biopsy_cases.outcome_known, 'cancer_positive'] = pd.NA
biopsy_cases.to_csv(OUTPUT_DIR / 'biopsy_per_case.csv', index=False)
display(eligibility_summary.loc[eligibility_summary.screen.eq('both')])
display(decision_summary.loc[decision_summary.screen.eq('both')])
display(biopsy_summary.loc[biopsy_summary.screen.eq('both')])

# %% [markdown]
# ## Paired volume errors and interpretation
#
# Connect signed errors before/after using automated measurement, not observed
# corrections to clinical reports. Green indicates lower absolute reference
# error; orange indicates worsening. Expanded Toronto and PI-CAI measure surrogate
# agreement. Small downstream groups cannot establish improved cancer detection
# or a deployed mitigation workflow. Selecting by disagreement is compatible
# with either method failing; prospective calibration and adjudication remain
# necessary to establish error causes and routine-care performance.

# %%
fig, axes = plt.subplots(1, len(COHORTS), figsize=(4 * len(COHORTS), 4), constrained_layout=True)
for ax, (cohort_key, (_, label, _)) in zip(axes, COHORTS.items()):
    selected = cohorts[cohort_key].loc[cohorts[cohort_key].flag_both]
    for row in selected.itertuples():
        color = '#228833' if row.absolute_error_reduction_ml > 0 else '#EE7733' if row.absolute_error_reduction_ml < 0 else '0.5'
        ax.plot([row.gt_volume_ml, row.gt_volume_ml], [row.clinical_error_ml, row.nnunet274_error_ml],
                color=color, alpha=.65, linewidth=.9)
        ax.scatter(row.gt_volume_ml, row.clinical_error_ml, marker='o', s=20, facecolors='none', edgecolors=color)
        ax.scatter(row.gt_volume_ml, row.nnunet274_error_ml, marker='^', s=22, color=color)
    ax.axhline(0, color='black', linewidth=.7)
    ax.set_xscale('log')
    ax.set_title(f'{label}\nStrict intersection: n = {len(selected)}', fontsize=10)
    ax.set_xlabel('Reference volume (mL; log scale)')
    ax.set_ylabel('Estimated − reference volume (mL)')
from matplotlib.lines import Line2D

fig.legend(handles=[Line2D([], [], marker='o', markerfacecolor='none', color='0.3', linestyle='none', label='Manual ellipsoid'),
                    Line2D([], [], marker='^', color='0.3', linestyle='none', label='nnUNet274'),
                    Line2D([], [], color='#228833', label='Lower absolute error'),
                    Line2D([], [], color='#EE7733', label='Higher absolute error')],
           loc='outside lower center', ncol=4, fontsize=8)
fig.savefig(OUTPUT_DIR / 'paired_volume_errors.png', dpi=300)
fig.savefig(OUTPUT_DIR / 'paired_volume_errors.pdf')
plt.show()

# %% [markdown]
# ## Reconciliation and provenance
#
# These checks verify arithmetic and eligibility, without hardcoding the
# exploratory outlier counts. Run-specific input hashes and inference settings
# permit later reconciliation after upstream cohort updates.

# %%
assert np.array_equal(np.searchsorted(PSAD_CUTOFFS, [0, .10, .15, .20], side='right'), [0, 1, 2, 3])
for cohort_key, volume in cohorts.items():
    assert (volume.flag_both <= volume.flag_either).all()
    for scale in ['ml', 'percent']:
        threshold = thresholds.loc[thresholds.cohort.eq(cohort_key) & thresholds.scale.eq(scale)].iloc[0]
        boundaries = np.array([threshold.loa_lower, threshold.loa_upper])
        assert not ((boundaries < threshold.loa_lower) | (boundaries > threshold.loa_upper)).any()
    for screen in SCREENS:
        paired = paired_summary.loc[paired_summary.cohort.eq(cohort_key) & paired_summary.screen.eq(screen)].iloc[0]
        assert paired.improved + paired.worsened + paired.tied == paired.n
        psad = psad_summary.loc[psad_summary.cohort.eq(cohort_key) & psad_summary.screen.eq(screen)].iloc[0]
        assert psad.corrected + psad.newly_incorrect + psad.both_correct + psad.both_incorrect == psad.n
        eligibility = eligibility_summary.loc[eligibility_summary.cohort.eq(cohort_key) & eligibility_summary.screen.eq(screen)].iloc[0]
        assert eligibility.excluded_psad + eligibility.excluded_not_pirads3 + eligibility.n_reference_decisions == eligibility.n_flagged
        assert eligibility.excluded_unknown_outcome + eligibility.n_known_outcomes == eligibility.n_reference_decisions
        for outcome in biopsy_summary.loc[biopsy_summary.cohort.eq(cohort_key) & biopsy_summary.screen.eq(screen)].itertuples():
            assert outcome.tp + outcome.fn + outcome.fp + outcome.tn == outcome.n
metadata = dict(site=SITE, fold=TESTING_FOLD, cohorts=provenance, primary_screen='both', secondary_screens=SCREENS[1:],
                disagreement_sign='manual minus nnUNet274', relative_denominator='pair mean',
                loa='mean +/- 1.96 sample SD; strict outside; full cohort',
                n_bootstrap=N_BOOTSTRAP, ci_seed=RANDOM_SEED, test_seed=RANDOM_SEED + 1,
                bootstrap_batch=BOOTSTRAP_BATCH, confidence_level=CONFIDENCE_LEVEL,
                inference_conditions='fixed thresholds, fixed selected patients, fixed predictions',
                toronto_manual_subset_of_expanded=True,
                picai_prostatex_shared_patients=len(shared_picai_prostatex),
                source_notebook_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest() if '__file__' in globals() else None)
(OUTPUT_DIR / 'run_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
print('Reconciled figure flags, input cohorts, category cutoffs, paired counts, and biopsy denominators.')


# %% [markdown]
# ## Optional six-dataset manuscript figure
#
# Set BUILD_ALL_SITE_FIGURE=True after running this notebook for PCNN, RUMC,
# and ZGT. This section can also run independently with Path, REPO_ROOT and
# BUILD_ALL_SITE_FIGURE established. It reads saved exports, validates their
# membership and provenance, and preserves results/paper_picai_sites/ paths.
# It does not rerun thresholds, statistics, inference, or patient selection.

# %%
if BUILD_ALL_SITE_FIGURE:
    import hashlib
    import json
    from pathlib import Path

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import pandas as pd
    import numpy as np

    FIGURE_OUTPUT_DIR = REPO_ROOT / 'results/paper_picai_sites'
    SOURCE_ROOT = REPO_ROOT / 'results/volume_disagreement_quality_improvement'
    SITES = ('PCNN', 'RUMC', 'ZGT')
    manifest_path = REPO_ROOT / 'configs/picai_test_clinical_folds.json'
    manifest = json.loads(manifest_path.read_text())
    input_hashes = {str(manifest_path.relative_to(REPO_ROOT)): hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    required = [SOURCE_ROOT / site / filename for site in SITES
                for filename in ('case_audit.csv', 'run_metadata.json')]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError('Run quality improvement for all three sites before building the figure: '
                                + ', '.join(missing))
    tables = {}
    for site in SITES:
        path = SOURCE_ROOT / site / 'case_audit.csv'
        metadata_path = SOURCE_ROOT / site / 'run_metadata.json'
        metadata = json.loads(metadata_path.read_text())
        assert metadata['site'] == site and metadata['fold'] == f'testing_{site}'
        for source in metadata['cohorts']:
            for relative_path, digest in source['input_sha256'].items():
                assert hashlib.sha256((REPO_ROOT / relative_path).read_bytes()).hexdigest() == digest, relative_path
        frame = pd.read_csv(path)
        expected_ids = {entry['id'] for entry in manifest[f'testing_{site}']}
        site_frame = frame.loc[frame.cohort.eq('picai')]
        assert site_frame.ID.is_unique and set(site_frame.ID) == expected_ids
        for source in (path, metadata_path):
            input_hashes[str(source.relative_to(REPO_ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
        tables[site] = frame

    # Require identical non-PI-CAI rows before using one copy.
    panels = []
    for key, label in [('prostatex', 'ProstateX'), ('toronto_manual', 'Toronto manual'),
                       ('toronto_surrogate', 'Expanded Toronto')]:
        baseline = tables['PCNN'].loc[tables['PCNN'].cohort.eq(key)].sort_values('ID').reset_index(drop=True)
        for site in SITES[1:]:
            comparison = tables[site].loc[tables[site].cohort.eq(key)].sort_values('ID').reset_index(drop=True)
            pd.testing.assert_frame_equal(baseline, comparison)
        panels.append((label, baseline.loc[baseline.flag_both]))
    for site in SITES:
        frame = tables[site]
        panels.append((f'PI-CAI {site}', frame.loc[frame.cohort.eq('picai') & frame.flag_both]))

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for letter, ax, (label, selected) in zip('abcdef', axes.flat, panels):
        for row in selected.itertuples():
            color = '#228833' if row.absolute_error_reduction_ml > 0 else '#EE7733' if row.absolute_error_reduction_ml < 0 else '0.5'
            ax.plot([row.gt_volume_ml, row.gt_volume_ml], [row.clinical_error_ml, row.nnunet274_error_ml],
                    color=color, alpha=.65, linewidth=.9)
            ax.scatter(row.gt_volume_ml, row.clinical_error_ml, marker='o', s=20, facecolors='none', edgecolors=color)
            ax.scatter(row.gt_volume_ml, row.nnunet274_error_ml, marker='^', s=22, color=color)
        ax.axhline(0, color='black', linewidth=.7)
        ax.set_xscale('log')
        if not selected.empty:
            ticks = np.geomspace(selected.gt_volume_ml.min(), selected.gt_volume_ml.max(), 3)
            ax.set_xticks(ticks, labels=[f'{value:.0f}' for value in ticks])
            ax.set_xticks([], minor=True)
        ax.set_title(f'({letter}) {label}\nStrict intersection: n = {len(selected)}', fontsize=11)
        ax.set_xlabel('Reference volume (mL; log scale)')
        ax.set_ylabel('Estimated − reference volume (mL)')
    fig.legend(handles=[
        Line2D([], [], marker='o', markerfacecolor='none', color='0.3', linestyle='none', label='Clinical ellipsoid'),
        Line2D([], [], marker='^', color='0.3', linestyle='none', label='Multicenter'),
        Line2D([], [], color='#228833', label='Lower absolute error'),
        Line2D([], [], color='#EE7733', label='Higher absolute error')],
        loc='outside lower center', ncol=4, fontsize=9)
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_OUTPUT_DIR / 'paired_volume_errors.png', dpi=300)
    fig.savefig(FIGURE_OUTPUT_DIR / 'paired_volume_errors.pdf')
    plt.show()

    figure_metadata = dict(input_sha256=input_hashes, panels=[dict(dataset=label, n=len(frame), ids=frame.ID.tolist())
        for label, frame in panels], volume_units='mL', error_sign='estimated minus reference',
        screen='strict intersection; independently estimated within each PI-CAI institution',
        source_notebook_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest() if '__file__' in globals() else None)
    (FIGURE_OUTPUT_DIR / 'run_metadata.json').write_text(json.dumps(figure_metadata, indent=2, allow_nan=False) + '\n')
    print(pd.DataFrame([dict(dataset=label, flagged=len(frame)) for label, frame in panels]))
