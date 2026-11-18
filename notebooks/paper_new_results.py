# %% [markdown]
# # Results for the revised volumetry manuscript
#
# Assemble eight testing cohorts and ask whether the multicenter model is closer to the
# reference when it disagrees with clinical ellipsoid volumetry. All inputs are
# saved analysis exports: no inference, cohort reselection, or mask processing.
# Run cells in order. Generated tables and private case audits stay in
# results/paper_new_results/. Manuscript prose is edited in paper_new.md;
# this notebook replaces only the six marked table blocks there.

# %%
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

REPO_ROOT = Path(__file__).resolve().parents[1] if '__file__' in globals() else Path.cwd().resolve()
if REPO_ROOT.name == 'notebooks':
    REPO_ROOT = REPO_ROOT.parent
assert (REPO_ROOT / 'pyproject.toml').is_file(), 'Establish REPO_ROOT first'
OUTPUT_DIR = REPO_ROOT / 'results/paper_new_results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPDATE_MANUSCRIPT = globals().get('UPDATE_MANUSCRIPT', True)
# Set tuples such as (10, 300) / (-60, 60) to crop all panels consistently.
# Cropping affects display only; every observation remains in the statistics.
BA_XLIM = globals().get('BA_XLIM', None)
BA_YLIM = globals().get('BA_YLIM', None)
BA_REL_YLIM = globals().get('BA_REL_YLIM', None)
SITES = ['RUMC', 'ZGT', 'PCNN']
# Manual references first, then automated surrogates; PCNN is the failure analysis. Expanded Toronto's directory name is historical.
COHORTS = {
    'prostatex': ('prostatex_linked_volume', 'ProstateX RUMC', 157, 'Manual whole-gland segmentation'),
    'toronto_manual': ('toronto_jdmi_clinical_volume', 'Toronto JDMI', 383, 'Median manual-reader planimetry'),
    'picai_RUMC': ('picai_test_clinical_volume/RUMC', 'PI-CAI RUMC', 391, 'Bosma22b automated surrogate'),
    'toronto_surrogate': ('toronto_jdmi_4111_clinical_volume', 'Toronto JDMI Expanded', 4105, 'nnUNet702 automated surrogate'),
    'picai_ZGT': ('picai_test_clinical_volume/ZGT', 'PI-CAI ZGT', 168, 'Bosma22b automated surrogate'),
    'picai_PCNN': ('picai_test_clinical_volume/PCNN', 'PI-CAI PCNN (wide field)', 315, 'Bosma22b automated surrogate'),
}
METHODS = {'nnunet274': 'Multicenter', 'rumcnotprostatexnoclinical236': 'Single-center', 'clinical': 'Clinical ellipsoid'}
CUTOFFS = np.array([.10, .15, .20])
input_sha256 = {}
provenance_reviews = []

# Small I/O helpers keep hashing and Markdown escaping consistent across tables.
def read_input(relative, kind='csv'):
    path = REPO_ROOT / relative
    input_sha256[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if kind == 'json':
        return json.loads(path.read_text())
    return pd.read_csv(path, low_memory=False, float_precision='round_trip')


def check_record(record):
    for relative, digest in record.get('input_sha256', {}).items():
        path = REPO_ROOT / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            # Two reviewed provenance drifts are checked semantically below:
            # archived selection is rederived from the full saved cohort audit;
            # current exclusions are compared explicitly, never substituted.
            reviewed_current = {
                'notebooks/toronto_jdmi_clinical_volume.py': '561b65240686f90ae6474d99024be9e7c72e2e0f12c83675254b9a40d30bdf68',
                'data_internal/hlab20260928/exclude.json': '9880cb4617c175d27fc1f907d5f0f2560c5e1059d927848a4ad1a9c4a115895f',
            }
            assert actual == reviewed_current.get(relative), f'Stale upstream input: {relative}'
            review = dict(path=relative, recorded_sha256=digest, current_sha256=actual,
                          resolution='Revalidated archived selection against full cohort audit; current exclusion differences exported; requested calculations recomputed from unchanged saved measurements')
            if review not in provenance_reviews:
                provenance_reviews.append(review)
        input_sha256[relative] = actual
    postprocessing = record.get('manual_postprocessing')
    if postprocessing:
        path = REPO_ROOT / postprocessing['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == postprocessing['sha256']
        input_sha256[postprocessing['path']] = postprocessing['sha256']


def markdown(frame):
    def cell(value):
        return str(value).replace('|', r'\|').replace('\n', '<br>')
    return '\n'.join(['| ' + ' | '.join(map(cell, frame.columns)) + ' |',
                      '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
                     + ['| ' + ' | '.join(map(cell, row)) + ' |' for row in frame.itertuples(index=False, name=None)])


def exact_closer_p(wins, losses):
    return binomtest(int(wins), int(wins + losses), .5, alternative='two-sided').pvalue if wins + losses else np.nan


def adjust_six(frame):
    assert len(frame) == 6
    # Undefined tests occupy a family slot conservatively, but remain undefined.
    adjusted = multipletests(frame.p_raw.fillna(1), method='holm')[1]
    frame['p_holm'] = np.where(frame.p_raw.notna(), adjusted, np.nan)


def fmt_p(value):
    return 'Undefined' if pd.isna(value) else ('<0.001' if value < .001 else f'{value:.3f}')

# %% [markdown]
# ## Reconcile current exports before combining cohorts
#
# All three site runs repeat ProstateX and Toronto. Require identical case
# audits for those repeated cohorts, then include each only once. Preserve the
# requested archived Toronto membership; audit the current exclusion-file drift
# (4109 current versus 4105 archived) explicitly. Verify saved
# provenance hashes, current manifests, one case per patient, source PSAD and
# biopsy decisions. Fail rather than combine stale or inconsistent exports.

# %%
qi = {}
for site in SITES:
    root = f'results/volume_disagreement_quality_improvement/{site}'
    meta = read_input(f'{root}/run_metadata.json', 'json')
    assert meta['site'] == site
    for record in meta['cohorts']:
        check_record(record)
    qi[site] = read_input(f'{root}/case_audit.csv')
    assert not qi[site].duplicated(['cohort', 'ID']).any()
for key in ['prostatex', 'toronto_manual', 'toronto_surrogate']:
    baseline = qi['RUMC'].loc[qi['RUMC'].cohort.eq(key)].sort_values('ID').reset_index(drop=True)
    for site in ['ZGT', 'PCNN']:
        other = qi[site].loc[qi[site].cohort.eq(key)].sort_values('ID').reset_index(drop=True)
        pd.testing.assert_frame_equal(baseline, other, check_exact=True)

cohorts, audits, provenance = {}, {}, []
for key, (directory, name, expected_n, reference) in COHORTS.items():
    root = f'results/{directory}'
    v = read_input(f'{root}/comparator_volumes.csv').set_index('ID', drop=False)
    audit = read_input(f'{root}/psa_density_biopsy_audit.csv').set_index('ID', drop=False)
    psad = read_input(f'{root}/psa_density_per_case.csv').set_index('ID')
    biopsy = read_input(f'{root}/psa_density_biopsy_per_case.csv').set_index('ID')
    meta = read_input(f'{root}/run_metadata.json', 'json')
    check_record(meta)
    assert v.index.is_unique and audit.index.is_unique and psad.index.is_unique and biopsy.index.is_unique
    assert len(v) == expected_n and set(v.index) == set(audit.index)
    audit = audit.loc[v.index]
    membership_hash = hashlib.sha256('\n'.join(sorted(v.index)).encode()).hexdigest()
    if 'cohort_sha256' in meta:
        assert meta['cohort_sha256'] == membership_hash
    if 'n_patients' in meta:
        assert meta['n_patients'] == expected_n
    if key.startswith('picai') or key == 'prostatex':
        manifest_path = 'configs/picai_test_clinical_folds.json' if key.startswith('picai') else 'configs/rumcnotprostatexnoclinical_folds.json'
        manifest = read_input(manifest_path, 'json')
        fold = f'testing_{key.split("_")[1]}' if key.startswith('picai') else 'testing'
        assert set(v.index) == {row['id'] for row in manifest[fold]}
        assert v.patient_id.is_unique
        if key.startswith('picai'):
            assert v.center.eq(key.split('_')[1]).all()
            training = {int(row['id'].split('_')[1]) for row in manifest['training']}
            assert not set(v.patient_id) & training
    else:
        assert v.amrn.is_unique
        listing = read_input(f'{root}/cohort_audit.csv')
        exclusions = read_input('data_internal/hlab20260928/exclude.json', 'json')
        excluded_accessions = {entry['aano'] for entry in exclusions}
        assert len(excluded_accessions) == 11
        include = ~listing.aano.isin(excluded_accessions)
        if key == 'toronto_manual':
            reference_volume = pd.to_numeric(listing.ground_truth_volume_ml, errors='coerce')
            include &= np.isfinite(reference_volume) & reference_volume.gt(0)
        archived_include = listing.included
        assert archived_include.dtype == bool
        assert set(listing.loc[archived_include, 'ID']) == set(v.index), 'Saved Toronto selection differs from full audit'
        difference = listing.loc[include.ne(archived_include), ['ID', 'aano', 'exclusion_reason']].copy()
        difference['selected_saved'] = archived_include.loc[difference.index]
        difference['selected_current'] = include.loc[difference.index]
        difference.to_csv(OUTPUT_DIR / f'{key}_exclusion_membership_difference.csv', index=False)
        if key == 'toronto_manual':
            assert difference.empty, 'Manual Toronto membership changed'
        else:
            assert len(difference) == 8 and int(include.sum()) == 4109, 'Unreviewed expanded Toronto selection drift'
        listing.loc[listing.aano_excluded, ['ID', 'aano', 'exclusion_reason']].to_csv(OUTPUT_DIR / f'{key}_archived_exclusions.csv', index=False)
    for method in ['gt', *METHODS]:
        col = f'{method}_volume_ml'
        assert np.isfinite(v[col]).all() and v[col].ge(0).all()
        assert np.allclose(v[col], audit[col], rtol=0, atol=1e-10)
    assert v.gt_volume_ml.gt(0).all() and v.clinical_volume_ml.gt(0).all()
    psa_col = 'psa_consolidated' if key.startswith('toronto') else 'psa_numeric_ng_ml'
    v['psa_ng_ml'] = pd.to_numeric(audit[psa_col], errors='coerce')
    for method in ['gt', 'clinical', 'nnunet274']:
        v[f'{method}_psad'] = v.psa_ng_ml / v[f'{method}_volume_ml'].where(v[f'{method}_volume_ml'].gt(0))
    v['psad_evaluable'] = (v.psa_ng_ml.gt(0) & np.isfinite(v[['gt_psad', 'clinical_psad', 'nnunet274_psad']]).all(axis=1))
    assert set(psad.index) == set(v.index[v.psad_evaluable])
    for method in ['gt', 'clinical', 'nnunet274']:
        assert np.allclose(psad[f'{method}_psad'], v.loc[psad.index, f'{method}_psad'])
        strata = psad[f'{method}_psad_stratum'].map({'0–<0.10': 0, '0.10–<0.15': 1, '0.15–<0.20': 2, '≥0.20': 3})
        assert np.array_equal(strata, np.searchsorted(CUTOFFS, v.loc[psad.index, f'{method}_psad'], side='right'))
    v['pirads_3'] = audit.biopsy_pirads_status.isin(['maximum_pirads_3', 'pirads_3'])
    toronto = key.startswith('toronto')
    v['outcome_known'] = audit.biopsy_outcome_status.eq('known_cscancer' if toronto else 'valid_case_isup')
    v['outcome'] = audit.cscancer if toronto else audit.case_isup_numeric
    v['outcome_definition'] = 'CSCANCER (Y/N)' if toronto else 'ISUP grade group (csPCa ≥2)'
    v.loc[~v.outcome_known, 'outcome'] = pd.NA
    assert set(biopsy.index) == set(v.index[v.psad_evaluable & v.pirads_3 & v.outcome_known])
    for method in ['gt', 'clinical', 'nnunet274']:
        assert np.array_equal(v.loc[biopsy.index, f'{method}_psad'].ge(.15), biopsy[f'{method}_biopsy_recommended'])
    cohorts[key], audits[key] = v, audit
    provenance.append(dict(cohort=key, n=len(v), reference=reference, cohort_sha256=membership_hash))
assert set(cohorts['toronto_manual'].index) <= set(cohorts['toronto_surrogate'].index)
shared_patients = {site: len(set(cohorts[f'picai_{site}'].patient_id) & set(cohorts['prostatex'].patient_id)) for site in SITES}
print(pd.DataFrame(provenance).to_string(index=False))
print('PI-CAI patients also in ProstateX:', shared_patients)

# %% [markdown]
# ## Derive disjoint analysis cohorts from validated parent exports
#
# Subtract patients, not examination strings. Keep the parent measurements and
# original configs intact. All following descriptive limits are calculated on
# retained subsets; screening uses the fixed common ProstateX criteria. Namespace Dutch and Toronto patient identifiers because
# their identifier systems are unrelated.

# %%
parents = {key: value.copy() for key, value in cohorts.items()}
parent_definitions = COHORTS.copy()
parent_provenance = provenance
# Validate parent disagreement flags before deriving subsets.
for key, parent in parents.items():
    site = key.split('_')[1] if key.startswith('picai') else 'RUMC'
    saved = qi[site].loc[qi[site].cohort.eq('picai' if key.startswith('picai') else key)].set_index('ID').loc[parent.index]
    parent_flags = []
    for values, column in [(parent.clinical_volume_ml-parent.nnunet274_volume_ml, 'flag_absolute'),
                           (200*(parent.clinical_volume_ml-parent.nnunet274_volume_ml)/(parent.clinical_volume_ml+parent.nnunet274_volume_ml), 'flag_relative')]:
        flag = (values < values.mean()-1.96*values.std(ddof=1)) | (values > values.mean()+1.96*values.std(ddof=1))
        assert np.array_equal(flag, saved[column])
        parent_flags.append(flag)
    assert np.array_equal(parent_flags[0] & parent_flags[1], saved.flag_both)

subtractions = {
    'picai_RUMC': ('prostatex', 'patient_id', 275, 'PI-CAI RUMC excluding ProstateX RUMC'),
    'toronto_surrogate': ('toronto_manual', 'amrn', 3722, 'Toronto JDMI Additional'),
}
membership_tables, provenance = [], []
for key, parent in parents.items():
    patient_column = 'amrn' if key.startswith('toronto') else 'patient_id'
    removed = pd.Series(False, index=parent.index)
    rule = 'Full validated parent cohort'
    if key in subtractions:
        subtract_key, patient_column, expected, name = subtractions[key]
        removed = parent[patient_column].isin(parents[subtract_key][patient_column])
        rule = f'Parent minus {subtract_key}, matched by {patient_column}'
        directory, _, _, reference = COHORTS[key]
        COHORTS[key] = (directory, name, expected, reference)
    retained = parent.loc[~removed].copy()
    assert len(retained) == COHORTS[key][2]
    assert retained[patient_column].notna().all() and retained[patient_column].is_unique
    assert set(retained.index).isdisjoint(parent.index[removed])
    assert set(retained.index) | set(parent.index[removed]) == set(parent.index)
    cohorts[key] = retained
    membership = pd.DataFrame({'cohort': key, 'ID': parent.ID, 'patient_identifier': parent[patient_column],
                               'selected': ~removed, 'selection_rule': rule})
    membership_tables.append(membership)
    provenance.append(dict(cohort=key, name=COHORTS[key][1], n=len(retained), reference=COHORTS[key][3],
                           parent_n=len(parent), parent_cohort_sha256=hashlib.sha256('\n'.join(sorted(parent.index)).encode()).hexdigest(),
                           cohort_sha256=hashlib.sha256('\n'.join(sorted(retained.index)).encode()).hexdigest(),
                           removed_n=int(removed.sum()), selection_rule=rule))
membership = pd.concat(membership_tables, ignore_index=True)
membership.to_csv(OUTPUT_DIR / 'analysis_membership.csv', index=False)
membership.loc[membership.selected].to_csv(OUTPUT_DIR / 'selected_membership.csv', index=False)
membership.loc[~membership.selected].to_csv(OUTPUT_DIR / 'removed_membership.csv', index=False)
patient_sets = {}
for key, v in cohorts.items():
    if key.startswith('toronto'):
        patient_sets[key] = {('Toronto', str(value)) for value in v.amrn}
    else:
        patient_sets[key] = {('Dutch', str(int(value))) for value in v.patient_id}
for first, second in combinations(patient_sets, 2):
    assert patient_sets[first].isdisjoint(patient_sets[second]), (first, second, 'shared analysis patients')
print('Disjoint analysis cohorts:', {key: len(v) for key, v in cohorts.items()})

# %% [markdown]
# ## Tables 1–3: testing performance
#
# Volumes are in mL. Percentage errors divide by the reference volume.
# Toronto manual references are scalar planimetry, so DSC is unavailable.
# Expanded Toronto uses saved scalar volumes here; no DSC is inferred from
# volume agreement. PI-CAI DSC is agreement with an automated surrogate.

# %%
performance_root = 'results/20261105_tile0.8_nomirroring/tables'
check_record(read_input(f'{performance_root}/run_metadata.json', 'json'))
metrics = read_input(f'{performance_root}/per_case_metrics.csv')
performance_rows = []
for split, label, n in [('testing', 'Multicenter Data (external coil)', 22), ('endorectal_coil', 'Multicenter Data (ERC)', 13)]:
    for model, method in [('Dataset274_raw_3d_fullres_5fold', 'nnunet274'), ('Dataset236_raw_3d_fullres_5fold', 'rumcnotprostatexnoclinical236')]:
        selected = metrics.loc[metrics.dataset.eq('pooled2') & metrics.split.eq(split) & metrics.model.eq(model) & metrics.region.eq('whole_gland')]
        assert len(selected) == n and selected.case_id.is_unique and selected.status.eq('evaluated').all()
        error = selected.predicted_volume_ml.to_numpy() - selected.reference_volume_ml.to_numpy()
        percent = 100 * error / selected.reference_volume_ml.to_numpy()
        performance_rows.append(dict(cohort=label, n=n, method=method, dsc=selected.dsc.mean(), mae=np.abs(error).mean(), rmse=np.sqrt(np.mean(error**2)), mape=np.abs(percent).mean(), rmspe=np.sqrt(np.mean(percent**2))))
for key, v in cohorts.items():
    directory, name, n, reference = COHORTS[key]
    dsc = {}
    if key.startswith('picai'):
        root = f'results/{directory}'
        seg_meta = read_input(f'{root}/segmentation_run_metadata.json', 'json')
        check_record(seg_meta)
        assert seg_meta['cohort_sha256'] == hashlib.sha256('\n'.join(sorted(parents[key].index)).encode()).hexdigest()
        seg = read_input(f'{root}/segmentation_metrics_per_case.csv')
        for method in ['nnunet274', 'rumcnotprostatexnoclinical236']:
            part = seg.loc[seg.method.eq(method)]
            assert part.ID.is_unique and set(part.ID) == set(parents[key].index)
            part = part.loc[part.ID.isin(v.index)]
            assert len(part) == n and part.ID.is_unique and set(part.ID) == set(v.index)
            assert part.dsc.between(0, 1).all()
            dsc[method] = part.dsc.mean()
    elif key == 'prostatex':
        for model, method in [('Dataset274_raw_3d_fullres_5fold', 'nnunet274'), ('Dataset236_raw_3d_fullres_5fold', 'rumcnotprostatexnoclinical236')]:
            part = metrics.loc[metrics.dataset.eq('rumcnotprostatexnoclinical') & metrics.split.eq('testing') & metrics.model.eq(model) & metrics.region.eq('whole_gland')].set_index('case_id')
            assert len(part) == n and part.index.is_unique and set(part.index) == set(v.index)
            assert np.allclose(part.predicted_volume_ml, v.loc[part.index, f'{method}_volume_ml'])
            assert np.allclose(part.reference_volume_ml, v.loc[part.index, 'gt_volume_ml'])
            dsc[method] = part.dsc.mean()
    for method in METHODS:
        error = (v[f'{method}_volume_ml'] - v.gt_volume_ml).to_numpy()
        percent = 100 * error / v.gt_volume_ml.to_numpy()
        performance_rows.append(dict(cohort=name, n=n, method=method, dsc=dsc.get(method, np.nan), mae=np.abs(error).mean(), rmse=np.sqrt(np.mean(error**2)), mape=np.abs(percent).mean(), rmspe=np.sqrt(np.mean(percent**2))))
performance = pd.DataFrame(performance_rows)
# Independently reconcile all four volume endpoints to saved source summaries.
for key, (directory, name, n, reference) in COHORTS.items():
    summary = read_input(f'results/{directory}/comparator_summary.csv').set_index('method')
    for method in METHODS:
        # Source summaries describe the full parents, not the derived subsets.
        parent = parents[key]
        error = parent[f'{method}_volume_ml'] - parent.gt_volume_ml
        percent = 100 * error / parent.gt_volume_ml
        row = dict(mae=np.abs(error).mean(), rmse=np.sqrt(np.mean(error**2)),
                   mape=np.abs(percent).mean(), rmspe=np.sqrt(np.mean(percent**2)))
        assert summary.loc[method, 'n'] == len(parent)
        for metric, saved_column in [('mae', 'mae_ml'), ('rmse', 'rmse_ml'), ('mape', 'mean_absolute_percentage_error'), ('rmspe', 'rmspe_percent')]:
            assert np.isclose(row[metric], summary.loc[method, saved_column], rtol=0, atol=1e-9), (key, method, metric)
performance.to_csv(OUTPUT_DIR / 'performance.csv', index=False)
tables = {}
for number, fields in [(1, ['dsc']), (2, ['mae', 'rmse']), (3, ['mape', 'rmspe'])]:
    rows = []
    for (name, n), group in performance.groupby(['cohort', 'n'], sort=False):
        row = {'Dataset': f'{name} (n = {n:,})'}
        for method, title in METHODS.items():
            part = group.loc[group.method.eq(method)]
            values = part.iloc[0][fields] if len(part) else pd.Series([np.nan])
            row[title] = '; '.join(f'{x:.3f}' if number == 1 else f'{x:.2f}' for x in values) if values.notna().all() else '—'
        rows.append(row)
    tables[number] = pd.DataFrame(rows)
    if number in (2, 3):
        tables[number] = tables[number][['Dataset', 'Clinical ellipsoid', 'Multicenter', 'Single-center']]

def outside_both_common(difference, relative, limits):
    absolute_outside = (np.asarray(difference) < limits['ml'][0]) | (np.asarray(difference) > limits['ml'][1])
    relative_outside = (np.asarray(relative) < limits['percent'][0]) | (np.asarray(relative) > limits['percent'][1])
    return absolute_outside & relative_outside


# Exact rounded ProstateX RUMC limits define both analysis and highlighting.
COMMON_LIMITS = {'ml': (-14.28, 25.74), 'percent': (-22.70, 39.60)}

# %% [markdown]
# ## Tables 4–6: disagreement and reference accuracy
#
# D = manual − automated; Q = 100D / pair mean. Use the exact common
# ProstateX limits for all cohorts; retain local mean ±1.96 SD for description.
# The primary subset exceeds BOTH intervals, irrespective of tail. Compare
# absolute reference error there, then compare ordinal PSAD category distance.
# Positive wins mean automated is closer; ties are exact equality. Two-sided
# exact binomial tests exclude ties and test 50% wins. Holm corrections span
# six cohorts separately for volume and PSAD; no error-metric tests are added.

# %%
threshold_rows, volume_rows, psad_rows, case_tables, biopsy_tables = [], [], [], [], []
for key, v in cohorts.items():
    _, name, n, reference = COHORTS[key]
    v['disagreement_ml'] = v.clinical_volume_ml - v.nnunet274_volume_ml
    v['disagreement_percent'] = 100 * v.disagreement_ml / ((v.clinical_volume_ml + v.nnunet274_volume_ml) / 2)
    threshold = dict(cohort=key, name=name, n=n, reference=reference)
    for scale, flag in [('ml', 'absolute'), ('percent', 'relative')]:
        x = v[f'disagreement_{scale}']
        lower, upper = x.mean() - 1.96*x.std(ddof=1), x.mean() + 1.96*x.std(ddof=1)
        common_lower, common_upper = COMMON_LIMITS[scale]
        v[f'flag_{flag}'] = (x < common_lower) | (x > common_upper)
        threshold.update({f'mean_{scale}': x.mean(), f'lower_{scale}': lower, f'upper_{scale}': upper, f'common_lower_{scale}': common_lower, f'common_upper_{scale}': common_upper, f'n_{flag}': int(v[f'flag_{flag}'].sum())})
    v['flag_both'] = outside_both_common(v.disagreement_ml, v.disagreement_percent, COMMON_LIMITS)
    assert np.array_equal(v.flag_both, v.flag_absolute & v.flag_relative)
    threshold['n_both'] = int(v.flag_both.sum())
    threshold_rows.append(threshold)
    site = key.split('_')[1] if key.startswith('picai') else 'RUMC'
    old_key = 'picai' if key.startswith('picai') else key
    saved = qi[site].loc[qi[site].cohort.eq(old_key)].set_index('ID').loc[v.index]
    for column in ['gt_volume_ml', 'clinical_volume_ml', 'nnunet274_volume_ml', 'disagreement_ml', 'disagreement_percent']:
        assert np.allclose(v[column], saved[column], rtol=0, atol=1e-10)
    for column in ['psad_evaluable', 'pirads_3', 'outcome_known']:
        assert np.array_equal(v[column], saved[column]), (key, column)
    flagged = v.loc[v.flag_both].copy()
    manual_error = np.abs(flagged.clinical_volume_ml - flagged.gt_volume_ml)
    auto_error = np.abs(flagged.nnunet274_volume_ml - flagged.gt_volume_ml)
    wins, losses, ties = int((auto_error < manual_error).sum()), int((auto_error > manual_error).sum()), int((auto_error == manual_error).sum())
    row = dict(cohort=key, name=name, cohort_n=n, n=len(flagged), wins=wins, losses=losses, ties=ties, p_raw=exact_closer_p(wins, losses))
    for method, error in [('manual', manual_error), ('automated', auto_error)]:
        pct = 100 * error / flagged.gt_volume_ml
        row.update({f'{method}_mae': error.mean(), f'{method}_rmse': np.sqrt(np.mean(error**2)), f'{method}_mape': pct.mean(), f'{method}_rmspe': np.sqrt(np.mean(pct**2))})
    for metric in ['mae', 'rmse', 'mape', 'rmspe']:
        row[f'change_{metric}'] = row[f'automated_{metric}'] - row[f'manual_{metric}']
    volume_rows.append(row)
    eligible = flagged.loc[flagged.psad_evaluable].copy()
    for method in ['gt', 'clinical', 'nnunet274']:
        eligible[f'{method}_category'] = np.searchsorted(CUTOFFS, eligible[f'{method}_psad'], side='right')
    manual_distance = (eligible.clinical_category - eligible.gt_category).abs()
    auto_distance = (eligible.nnunet274_category - eligible.gt_category).abs()
    wins, losses, ties = int((auto_distance < manual_distance).sum()), int((auto_distance > manual_distance).sum()), int((auto_distance == manual_distance).sum())
    psad_rows.append(dict(cohort=key, name=name, n_flagged=len(flagged), n=len(eligible), excluded_psad=len(flagged)-len(eligible), wins=wins, losses=losses, ties=ties, p_raw=exact_closer_p(wins, losses)))
    # Include every PI-RADS 3 intersection case even if PSA/outcome is missing.
    biopsy = flagged.loc[flagged.pirads_3].copy()
    for method in ['gt', 'clinical', 'nnunet274']:
        value = biopsy[f'{method}_psad']
        biopsy[f'{method}_recommend'] = value.ge(.15).astype('boolean').where(np.isfinite(value), pd.NA)
    biopsy_tables.append(biopsy.assign(cohort=key, name=name, reference=reference))
    case_tables.append(v.assign(cohort=key, reference=reference))
thresholds = pd.DataFrame(threshold_rows)
volume_summary, psad_summary = pd.DataFrame(volume_rows), pd.DataFrame(psad_rows)
adjust_six(volume_summary)
adjust_six(psad_summary)
all_cases = pd.concat(case_tables, ignore_index=True)
biopsy_cases = pd.concat(biopsy_tables, ignore_index=True)
# Table 6 sorts globally by outcome, then reference/manual/automated decisions.
# ISUP grades within each group share a rank. Cohort and ID break ties.
toronto = biopsy_cases.cohort.str.startswith('toronto')
isup = pd.to_numeric(biopsy_cases.outcome, errors='coerce')
outcome_order = np.select([
    biopsy_cases.outcome_known & ~toronto & isup.ge(2),
    biopsy_cases.outcome_known & toronto & biopsy_cases.outcome.eq('Y'),
    biopsy_cases.outcome_known & ~toronto & isup.isin([0, 1]),
    biopsy_cases.outcome_known & toronto & biopsy_cases.outcome.eq('N'),
], [0, 1, 2, 3], default=4)
sort_columns = ['outcome_order', 'gt_order', 'clinical_order', 'nnunet274_order', 'cohort_order', 'ID']
biopsy_cases = biopsy_cases.assign(
    outcome_order=outcome_order,
    cohort_order=biopsy_cases.cohort.map({key: i for i, key in enumerate(COHORTS)}),
    **{f'{method}_order': biopsy_cases[f'{method}_recommend'].map({True: 0, False: 1}).fillna(2)
       for method in ['gt', 'clinical', 'nnunet274']},
).sort_values(sort_columns).drop(columns=sort_columns[:-1]).reset_index(drop=True)

# Save compact case audits without copying private source paths/workbook columns.
case_columns = ['cohort', 'ID', 'reference', 'gt_volume_ml', 'clinical_volume_ml', 'nnunet274_volume_ml', 'disagreement_ml', 'disagreement_percent', 'flag_absolute', 'flag_relative', 'flag_both', 'psa_ng_ml', 'gt_psad', 'clinical_psad', 'nnunet274_psad', 'psad_evaluable', 'pirads_3', 'outcome_known', 'outcome', 'outcome_definition']
all_cases[case_columns].to_csv(OUTPUT_DIR / 'case_audit.csv', index=False)
biopsy_cases[case_columns + ['gt_recommend', 'clinical_recommend', 'nnunet274_recommend']].to_csv(OUTPUT_DIR / 'pirads3_outliers.csv', index=False)
for stem, frame in [('disagreement_thresholds', thresholds), ('volume_closeness', volume_summary), ('psad_closeness', psad_summary)]:
    frame.to_csv(OUTPUT_DIR / f'{stem}.csv', index=False)

rows = []
for r in volume_summary.to_dict('records'):
    row = {'Dataset': r['name'], 'Flagged n/N (%)': f"{r['n']}/{r['cohort_n']:,} ({100*r['n']/r['cohort_n']:.1f}%)"}
    for metric in ['mae', 'rmse', 'mape', 'rmspe']:
        row[metric.upper() + (' (mL)' if metric in ['mae', 'rmse'] else ' (%)')] = f"{r[f'manual_{metric}']:.2f} → {r[f'automated_{metric}']:.2f}"
    row['Improved / all'] = f"{r['wins']}/{r['n']}"
    if volume_summary.ties.gt(0).any():
        row['Ties'] = r['ties']
    row['Adjusted p'] = fmt_p(r['p_holm'])
    rows.append(row)
tables[4] = pd.DataFrame(rows)
tables[5] = pd.DataFrame([{'Dataset': f'{r.name} (n = {r.n}' + (', another missing PSA value' if r.excluded_psad == 1 else f', {r.excluded_psad} missing PSA values' if r.excluded_psad else '') + ')', 'Automated closer': r.wins, 'Equal distance': r.ties, 'Manual closer': r.losses, 'Adjusted p': fmt_p(r.p_holm)} for r in psad_summary.itertuples()])
rows = []
for r in biopsy_cases.to_dict('records'):
    row = {'Dataset': r['name'], 'Case': r['ID']}
    for method, title in [('gt', 'Reference'), ('clinical', 'Manual'), ('nnunet274', 'Automated')]:
        value, recommend = r[f'{method}_psad'], r[f'{method}_recommend']
        row[f'{title}: mL; PSAD; biopsy'] = f"{r[f'{method}_volume_ml']:.2f}; " + (f"{value:.3f}; {'Recommend' if recommend else 'Avoid'}" if pd.notna(value) and pd.notna(recommend) else 'Missing; Unknown')
    row['Recorded outcome'] = ('Unknown' if not r['outcome_known'] else (f"CSCANCER {r['outcome']}" if r['cohort'].startswith('toronto') else f"ISUP {int(r['outcome'])}"))
    rows.append(row)
tables[6] = pd.DataFrame(rows, columns=['Dataset', 'Case', 'Reference: mL; PSAD; biopsy', 'Manual: mL; PSAD; biopsy', 'Automated: mL; PSAD; biopsy', 'Recorded outcome'])
for number, frame in tables.items():
    frame.to_csv(OUTPUT_DIR / f'table_{number}.csv', index=False)
    (OUTPUT_DIR / f'table_{number}.md').write_text(markdown(frame) + '\n')
    print(f'\nTable {number}\n{markdown(frame)}')

# %% [markdown]
# ## Manual-reference volume agreement
#
# Preserve the original 9.5 × 7.5 inch layout. Screening uses the common
# pair-mean criteria, whereas plotted errors use the manual reference.
# Fixed percentage ranges affect display only; audit final limits explicitly.

# %%
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter, FuncFormatter

agreement_rows, agreement_panels, agreement_hidden, agreement_members = [], [], [], []
agreement_settings = {}
method_order = ['clinical', 'nnunet274', 'rumcnotprostatexnoclinical236']
markers = ['o', 's', '^']
line_styles = ['-', '--', ':']
method_colors = ['#4C6A85', '#B88254', '#91A78C']
for key, percentage_limit in [('prostatex', 90), ('toronto_manual', 100)]:
    v = cohorts[key]
    reference = v.gt_volume_ml.to_numpy()
    selected = v.flag_both.to_numpy()
    assert selected.sum() == {'prostatex': 2, 'toronto_manual': 10}[key]
    assert np.array_equal(selected, outside_both_common(v.disagreement_ml, v.disagreement_percent, COMMON_LIMITS))
    agreement_members.append(v[['ID', 'flag_both', 'disagreement_ml', 'disagreement_percent']].assign(cohort=key))
    names = ['Manual ellipsoid', 'Multicenter', 'Single-center']
    reference_name = 'Manual planimetry' if key == 'prostatex' else 'Median manual-reader planimetry'
    fig, axes = plt.subplots(3, 4, figsize=(9.5, 7.5), sharey='row', squeeze=False,
                             constrained_layout=True, gridspec_kw={'width_ratios': [1, 1, 1, .85]})
    fig.get_layout_engine().set(w_pad=.015, h_pad=.015, wspace=.02, hspace=.02)
    fig.suptitle(f'{COHORTS[key][1]} (n = {len(v)} patients)', fontsize=10)
    axes[0, -1].set_axis_off()
    all_volumes = v[['gt_volume_ml', *[f'{m}_volume_ml' for m in method_order]]].to_numpy()
    lower = min(all_volumes[all_volumes > 0].min() * .95, 10)
    upper = max(all_volumes.max() * 1.05, 300)
    differences = [[], []]
    panels = []
    for column, (method, name, marker, linestyle, method_color) in enumerate(zip(method_order, names, markers, line_styles, method_colors)):
        measured = v[f'{method}_volume_ml'].to_numpy()
        error = measured - reference
        relative = 100 * error / reference
        correlation = np.corrcoef(reference, measured)[0, 1]
        record = dict(cohort=key, method=method, n=len(v), selected_n=int(selected.sum()), pearson_r=correlation)
        for row, (values, scale) in enumerate([(measured, 'volume'), (error, 'ml'), (relative, 'percent')]):
            ax = axes[row, column]
            if row == 0:
                label = f'{name}\nPearson r = {correlation:.2f}'
                ax.plot([lower, upper], [lower, upper], 'k--', linewidth=.8)
                ax.set_yscale('log', nonpositive='mask')
                ax.set_yticks([10, 30, 100, 300], labels=['10', '30', '100', '300'])
                ax.yaxis.set_minor_formatter(NullFormatter())
                ax.set_ylim(lower, upper)
            else:
                bias, sd = values.mean(), values.std(ddof=1)
                lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
                record.update({f'bias_{scale}': bias, f'lower_{scale}': lo, f'upper_{scale}': hi})
                label = f'{name}\nMean: {bias:.1f} {"mL" if row == 1 else "%"}\n95% LoA:\n[{lo:.1f}, {hi:.1f}]'
                for level, ls in [(bias, '-'), (lo, '--'), (hi, '--')]:
                    ax.axhline(level, color='0.55', linestyle=ls, linewidth=.8)
                differences[row-1].append(values)
            ax.scatter(reference[~selected], values[~selected], s=16, color=method_color, marker=marker,
                       alpha=.6, edgecolors='none', label=label)
            ax.scatter(reference[selected], values[selected], s=28, color='#D55E00', marker=marker,
                       edgecolors='white', linewidths=.6, alpha=1, zorder=5)
            ax.set_xscale('log')
            ax.set_xticks([10, 30, 100, 300], labels=['10', '30', '100', '300'])
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlim(lower, upper)
            ax.set_xlabel(f'{reference_name} volume\n(mL; log scale)', fontsize=8, labelpad=1)
            ax.set_ylabel([f'{name}\nvolume (mL; log scale)', f'{name}\n− reference (mL)', 'Relative difference (%)'][row], fontsize=8, labelpad=1)
            ax.legend(loc='best', fontsize=7, framealpha=.85, edgecolor='0.6', handlelength=1, handletextpad=.4)
            panels.append((ax, method, scale, reference, values))
        agreement_rows.append(record)
    histogram_records = []
    for row, method_differences in enumerate(differences, start=1):
        ax = axes[row, -1]
        fd_edges = np.histogram_bin_edges(np.asarray(method_differences).ravel(), bins='fd')
        count = max(1, int(np.ceil((len(fd_edges)-1)/2)))
        edges = np.linspace(fd_edges[0], fd_edges[-1], count+1)
        current = axes[row, 0].get_ylim()
        ax.set_ylim(min(current[0], edges[0]), max(current[1], edges[-1]))
        for method, name, values, method_color, ls in zip(method_order, names, method_differences, method_colors, line_styles):
            counts, _ = np.histogram(values, bins=edges)
            assert counts.sum() == len(v), 'Histogram must include every patient'
            ax.stairs(counts, edges, orientation='horizontal', color=method_color, linestyle=ls, linewidth=1.2, label=name)
            histogram_records.append((ax, method, 'ml' if row == 1 else 'percent', values, edges, counts))
        ax.set_xlim(left=0)
        ax.locator_params(axis='x', nbins=3, integer=True)
        ax.legend(fontsize=7, loc='best', framealpha=.85)
    axes[2, 0].set_ylim(-percentage_limit, percentage_limit)
    axes[1, 0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:+g}' if value else '0'))
    axes[2, 0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:+.0f}%' if value else '0%'))
    for ax in axes.flat:
        if ax.axison:
            ax.tick_params(labelsize=8, pad=1.5, labelbottom=True, labelleft=True)
    fig.canvas.draw()
    for ax, method, scale, x, y in panels:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        bad_x = ~np.isfinite(x) | (x <= 0)
        bad_y = ~np.isfinite(y) | ((y <= 0) if ax.get_yscale() == 'log' else False)
        outside_x = (x < xlim[0]) | (x > xlim[1])
        outside_y = (y < ylim[0]) | (y > ylim[1])
        hidden = bad_x | bad_y | outside_x | outside_y
        if hidden.any():
            ax.text(.02, .03, f'Outside displayed limits: {hidden.sum()}/{len(v)}', transform=ax.transAxes,
                    fontsize=6, bbox=dict(facecolor='white', edgecolor='none', alpha=.9))
        agreement_panels.append(dict(cohort=key, method=method, scale=scale, n=len(v), hidden_n=int(hidden.sum()),
                                     x_min=xlim[0], x_max=xlim[1], y_min=ylim[0], y_max=ylim[1]))
        for i in np.flatnonzero(hidden):
            reasons = [name for name, mask in [('invalid_log_x', bad_x), ('invalid_y_or_log_y', bad_y),
                        ('outside_x', outside_x), ('outside_y', outside_y)] if mask[i]]
            agreement_hidden.append(dict(cohort=key, method=method, scale=scale, ID=v.ID.iloc[i],
                                         x=x[i], y=y[i], selected=bool(selected[i]), reason=';'.join(reasons)))
    histogram_settings = []
    for ax, method, scale, values, edges, counts in histogram_records:
        ylim = ax.get_ylim()
        hidden_n = int(((values < ylim[0]) | (values > ylim[1])).sum())
        histogram_settings.append(dict(method=method, scale=scale, edges=edges.tolist(), counts=counts.tolist(), hidden_n=hidden_n))
    for row, scale in [(1, 'ml'), (2, 'percent')]:
        clipped = [f'{names[method_order.index(r["method"])]}: {r["hidden_n"]}' for r in histogram_settings if r['scale'] == scale and r['hidden_n']]
        if clipped:
            axes[row, -1].text(.02, .03, 'Outside displayed limits:\n' + '\n'.join(clipped),
                              transform=axes[row, -1].transAxes, fontsize=6,
                              bbox=dict(facecolor='white', edgecolor='none', alpha=.9))
    stem = f'{key}_comparator_agreement'
    fig.savefig(OUTPUT_DIR / f'{stem}.png', dpi=300, bbox_inches=None)
    fig.savefig(OUTPUT_DIR / f'{stem}.pdf', bbox_inches=None)
    agreement_settings[key] = dict(page_inches=[9.5, 7.5], dpi=300, relative_ylim=[-percentage_limit, percentage_limit],
                                  histogram=histogram_settings, method_colors=dict(zip(method_order, method_colors)), ordinary_points_grayscale=False, highlight_color='#D55E00', selected_n=int(selected.sum()))
    plt.show()
    plt.close(fig)
pd.DataFrame(agreement_rows).to_csv(OUTPUT_DIR / 'manual_reference_agreement_summary.csv', index=False)
pd.DataFrame(agreement_panels).to_csv(OUTPUT_DIR / 'manual_reference_panel_limits.csv', index=False)
pd.DataFrame(agreement_hidden, columns=['cohort', 'method', 'scale', 'ID', 'x', 'y', 'selected', 'reason']).to_csv(OUTPUT_DIR / 'manual_reference_outside_limits.csv', index=False)
pd.concat(agreement_members, ignore_index=True).to_csv(OUTPUT_DIR / 'manual_reference_screening_membership.csv', index=False)

# %% [markdown]
# ## Bland–Altman agreement: multicenter model and clinical ellipsoid
#
# All observations contribute to the mean and sample-SD limits. Shared axis
# settings are display parameters only. Each panel counts observations outside
# either displayed axis once; points on the boundaries remain visible. Rerun
# this cell alone after changing BA_XLIM / BA_YLIM / BA_REL_YLIM.
# Compact portrait output uses constrained layout without a reserved caption area.

# %%
def outside_display(x, y, xlim, ylim):
    return (np.asarray(x) < xlim[0]) | (np.asarray(x) > xlim[1]) | (np.asarray(y) < ylim[0]) | (np.asarray(y) > ylim[1])

fig, axes = plt.subplots(6, 2, figsize=(8.5, 9.75), sharex=True, sharey='col', constrained_layout=True)
fig.get_layout_engine().set(w_pad=.04, h_pad=.04, wspace=.04, hspace=.04)
# Layout-aware headings stay centered over their respective columns.
for ax, heading in zip(axes[0], ['Absolute difference', 'Relative difference']):
    ax.annotate(heading, xy=(.5, 1), xycoords='axes fraction', xytext=(0, 24), textcoords='offset points',
                ha='center', va='bottom', fontsize=10, weight='bold')
axes[2, 0].set_ylabel('Clinical ellipsoid − Multicenter (mL)', fontsize=8)
axes[2, 1].set_ylabel('100 × difference / pair mean (%)', fontsize=8)
plot_rows, panel_data = [], []
for row_index, (key, v) in enumerate(cohorts.items()):
    x = (v.clinical_volume_ml + v.nnunet274_volume_ml) / 2
    difference = v.clinical_volume_ml - v.nnunet274_volume_ml
    common_both = outside_both_common(difference, 100*difference/x, COMMON_LIMITS)
    if 'flag_both' in v:
        assert np.array_equal(common_both, v.flag_both), 'Figure and analysis memberships differ'
    expected = thresholds.set_index('cohort').loc[key]
    for column, (scale, unit, y) in enumerate([('ml', 'mL', difference), ('percent', '%', 100*difference/x)]):
        ax = axes[row_index, column]
        mean, sd = y.mean(), y.std(ddof=1)
        lower, upper = mean-1.96*sd, mean+1.96*sd
        assert np.allclose([mean, lower, upper], expected[[f'mean_{scale}', f'lower_{scale}', f'upper_{scale}']].to_numpy(dtype=float))
        ax.scatter(x[~common_both], y[~common_both], s=4, alpha=.15, color='black', edgecolors='none')
        ax.axhline(mean, color='0.55', linewidth=.8, label=f'Dataset mean: {mean:.2f} {unit}')
        ax.axhline(lower, color='0.55', linestyle=':', linewidth=.7, label=f'Dataset 95% LoA: {lower:.2f} to {upper:.2f} {unit}')
        ax.axhline(upper, color='0.55', linestyle=':', linewidth=.7)
        common_lower, common_upper = COMMON_LIMITS[scale]
        ax.axhline(common_lower, color='black', linestyle='--', linewidth=.8, zorder=4,
                   label=f'Common criteria: {common_lower:.2f} to {common_upper:.2f} {unit}')
        ax.axhline(common_upper, color='black', linestyle='--', linewidth=.8, zorder=4)
        # Reuse the same membership in both panels; draw orange highlighted points last.
        ax.scatter(x[common_both], y[common_both], s=4, alpha=1., color='#D55E00', edgecolors='white', linewidths=.35, zorder=5,
                   label=f'Outside both common criteria (n = {int(common_both.sum())})')
        title = COHORTS[key][1].replace(' excluding ', '\nexcluding ')
        ax.set_title(f'{title} (n = {len(v):,})', fontsize=7, pad=3)
        assert np.isfinite(x).all() and x.gt(0).all(), 'Log x-axis requires positive pair means'
        ax.set_xscale('log')
        # Explicit strings avoid scientific notation; minor ticks stay unlabeled.
        from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
        ax.xaxis.set_major_locator(FixedLocator([10, 30, 100, 300]))
        ax.xaxis.set_major_formatter(FixedFormatter(['10', '30', '100', '300']))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(labelsize=6, length=2, pad=2, labelbottom=True)
        if row_index == len(cohorts)-1:
            ax.set_xlabel('Pair mean volume (mL; log scale)', fontsize=7)
        ax.legend(loc='best', fontsize=5.5, framealpha=.9, borderpad=.3, labelspacing=.2, handlelength=1.6)
        plot_rows.append(dict(cohort=key, name=COHORTS[key][1], scale=scale, unit=unit, n=len(v), mean=mean, lower=lower, upper=upper, common_lower=common_lower, common_upper=common_upper, common_both_n=int(common_both.sum())))
        panel_data.append((ax, x, y))
# Shared x-axis and column-specific y-axes include all data by default.
fig.canvas.draw()
for value, label in [(BA_XLIM, 'BA_XLIM'), (BA_YLIM, 'BA_YLIM'), (BA_REL_YLIM, 'BA_REL_YLIM')]:
    if value is not None:
        assert len(value) == 2 and np.isfinite(value).all() and value[0] < value[1], label
if BA_XLIM is not None:
    assert BA_XLIM[0] > 0, 'BA_XLIM bounds must be positive for log scale'
    axes[0, 0].set_xlim(*BA_XLIM)
if BA_YLIM is not None:
    axes[0, 0].set_ylim(*BA_YLIM)
if BA_REL_YLIM is not None:
    axes[0, 1].set_ylim(*BA_REL_YLIM)
for (ax, x, y), row in zip(panel_data, plot_rows):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    hidden = int(outside_display(x, y, xlim, ylim).sum())
    if hidden > 0:
        ax.text(.02, .03, f'Outside displayed limits: {hidden}/{row["n"]}', transform=ax.transAxes,
                ha='left', va='bottom', fontsize=5.5, bbox=dict(facecolor='white', edgecolor='none', alpha=.8))
    row.update(hidden_n=hidden, x_min=xlim[0], x_max=xlim[1], y_min=ylim[0], y_max=ylim[1])
plot_summary = pd.DataFrame(plot_rows)
plot_summary.to_csv(OUTPUT_DIR / 'bland_altman_summary.csv', index=False)
# Keep the exact compact page dimensions after constrained layout.
fig.savefig(OUTPUT_DIR / 'bland_altman_clinical_nnunet274.png', dpi=300, bbox_inches=None)
fig.savefig(OUTPUT_DIR / 'bland_altman_clinical_nnunet274.pdf', bbox_inches=None)
plt.show()
plt.close(fig)

# %% [markdown]
# ## Arithmetic, boundary cases, and provenance
#
# Ties contribute to displayed denominators but not binomial tests. Empty
# non-tie sets yield undefined p-values. Unknown outcomes stay unknown and
# are retained in the PI-RADS table. No resampling or random seed is needed.

# %%
assert np.array_equal(np.searchsorted(CUTOFFS, [.099, .10, .15, .20], side='right'), [0, 1, 2, 3])
assert not ((np.array([-1., 1.]) < -1) | (np.array([-1., 1.]) > 1)).any()
# x-only, y-only, both axes, and exact-boundary observations.
assert np.array_equal(outside_display([-1, .5, 2, 0, 1], [.5, 2, 2, 0, 1], (0, 1), (0, 1)), [True, True, True, False, False])
assert np.isnan(exact_closer_p(0, 0)) and exact_closer_p(1, 1) == 1
assert exact_closer_p(2, 0) == .5
# Known six-test Holm example, including one undefined/empty non-tie group.
synthetic_family = pd.DataFrame({'p_raw': [.001, .01, .04, .2, .5, np.nan]})
adjust_six(synthetic_family)
assert np.allclose(synthetic_family.p_holm, [.006, .05, .16, .6, 1., np.nan], equal_nan=True)
empty_family = pd.DataFrame({'p_raw': [np.nan] * 6})
adjust_six(empty_family)
assert empty_family.p_holm.isna().all()
for frame in [volume_summary, psad_summary]:
    assert (frame.wins + frame.losses + frame.ties).eq(frame.n).all()
    expected = multipletests(frame.p_raw.fillna(1), method='holm')[1]
    assert np.allclose(frame.p_holm, np.where(frame.p_raw.notna(), expected, np.nan), equal_nan=True)
assert len(biopsy_cases) == int((all_cases.flag_both & all_cases.pirads_3).sum())
assert biopsy_cases.loc[~biopsy_cases.outcome_known, 'outcome'].isna().all()
assert not biopsy_cases.duplicated(['cohort', 'ID']).any()
biopsy_counts = []
for key in COHORTS:
    selected = biopsy_cases.loc[biopsy_cases.cohort.eq(key)]
    changed = selected.clinical_recommend.ne(selected.nnunet274_recommend)
    biopsy_counts.append(dict(cohort=key, n=len(selected), known_outcomes=int(selected.outcome_known.sum()),
                              unknown_outcomes=int((~selected.outcome_known).sum()),
                              recommendations_changed=int(changed.sum())))
pd.DataFrame(biopsy_counts).to_csv(OUTPUT_DIR / 'pirads3_counts.csv', index=False)
assert thresholds.set_index('cohort').n_both.to_dict() == {
    'prostatex': 2, 'toronto_manual': 10, 'picai_RUMC': 11,
    'toronto_surrogate': 143, 'picai_ZGT': 7, 'picai_PCNN': 60,
}
assert plot_summary.groupby('cohort', sort=False).common_both_n.first().tolist() == thresholds.n_both.tolist()
metadata = dict(manual_reference_agreement=agreement_settings, screening=dict(calibration_cohort='ProstateX RUMC', common_limits=COMMON_LIMITS,
                               rule='strictly outside both absolute and relative intervals; equality retained',
                               relative_denominator='pair mean', threshold_uncertainty_in_tests=False), created_utc=datetime.now(timezone.utc).isoformat(), cohorts=provenance, parent_cohorts=parent_provenance,
                input_sha256=input_sha256, provenance_reviews=provenance_reviews, source_notebook_sha256=hashlib.sha256((REPO_ROOT / 'notebooks/paper_new_results.py').read_bytes()).hexdigest(),
                parent_shared_picai_prostatex_patients=shared_patients, analysis_cohorts_patient_disjoint=True, toronto_selection='Additional 3722 = archived 4105 minus manual 383; current exclusions not substituted',
                bland_altman=dict(xscale='log', xticks=[10, 30, 100, 300], xlim=BA_XLIM, ylim=BA_YLIM, relative_ylim=BA_REL_YLIM, layout='6x2', page_inches=[8.5, 9.75], common_plot_limits=COMMON_LIMITS, common_criteria_applied_to_analysis=True, highlight=dict(color='#D55E00', edgecolor='white', alpha=1., size=4, criterion='strictly outside both common intervals'), sign='clinical ellipsoid minus multicenter model', statistics_use_all_cases=True),
                tests='two-sided exact binomial p=0.5, non-ties only',
                correction='Holm across six cohorts separately for volume and ordinal PSAD closeness; undefined slots use p=1 then remain undefined',
                error_change='automated minus manual', validation='All assertions passed')
(OUTPUT_DIR / 'run_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
if UPDATE_MANUSCRIPT:
    manuscript_path = REPO_ROOT / 'paper_new.md'
    manuscript = manuscript_path.read_text()
    for number, frame in tables.items():
        start, end = f'<!-- TABLE_{number}_START -->', f'<!-- TABLE_{number}_END -->'
        assert manuscript.count(start) == manuscript.count(end) == 1, f'Missing/duplicate manuscript table markers: {number}'
        before, rest = manuscript.split(start)
        _, after = rest.split(end)
        manuscript = before + start + '\n\n' + markdown(frame) + '\n\n' + end + after
    manuscript_path.write_text(manuscript)
print('Validation passed; outputs:', OUTPUT_DIR)
