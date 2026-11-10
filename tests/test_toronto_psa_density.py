"""Validate Toronto notebook PSAD cells with synthetic cohorts.

Statistical cases mirror the linked PROSTATEx tests with Toronto field names.
"""
import ast
from contextlib import redirect_stdout
from io import StringIO
from itertools import combinations
from pathlib import Path
from textwrap import fill
import tempfile
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
import pandas as pd

NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/toronto_jdmi_clinical_volume.py'
SOURCE = NOTEBOOK.read_text()
CALCULATIONS_SOURCE = SOURCE[SOURCE.index("PSAD_UNIT = 'ng/mL²'"):SOURCE.index('# %% [markdown]\n# ## PSA density: agreement plots')]
AGREEMENT_SOURCE = SOURCE[SOURCE.index('# %% [markdown]\n# ## PSA density: agreement plots'):SOURCE.index('# %% [markdown]\n# ## PSA density: strata and threshold statistics')]
STRATA_SOURCE = SOURCE[SOURCE.index('# %% [markdown]\n# ## PSA density: strata and threshold statistics'):SOURCE.index('# %%\nfig, axes = plt.subplots(\n    1, len(SELECTED_COMPARATORS), figsize=(3.4')]
CONFUSION_SOURCE = SOURCE[SOURCE.index('# %%\nfig, axes = plt.subplots(\n    1, len(SELECTED_COMPARATORS), figsize=(3.4'):SOURCE.index('# %% [markdown]\n# ## PSA density: paired stratum-accuracy comparisons')]
STRATUM_INFERENCE_SOURCE = SOURCE[SOURCE.index('# %% [markdown]\n# ## PSA density: paired stratum-accuracy comparisons'):SOURCE.index('# %% [markdown]\n# ## PSA density: paired absolute-error comparisons')]
INFERENCE_SOURCE = SOURCE[SOURCE.index('# %% [markdown]\n# ## PSA density: paired absolute-error comparisons'):SOURCE.index('# %% [markdown]\n# ## PI-RADS 3: biopsy decisions')]
BIOPSY_SOURCE = SOURCE[SOURCE.index('# %% [markdown]\n# ## PI-RADS 3: biopsy decisions'):SOURCE.index('# %% [markdown]\n# ### Biopsy decisions: confusion matrices')]
BIOPSY_PLOT_SOURCE = SOURCE[SOURCE.index('# %% [markdown]\n# ### Biopsy decisions: confusion matrices'):SOURCE.index('# %% [markdown]\n# ## Local reproducibility')]


class TorontoPSADensityTests(unittest.TestCase):

    def test_continuous_summary_and_units(self):
        ns = self.namespace([10., 20., 30.], [100.] * 3, [50., 100., 200.])
        self.execute(CALCULATIONS_SOURCE, ns)
        error = np.array([.1, 0., -.15])
        percent = 100 * error / np.array([.1, .2, .3])
        row = ns['psad_summary'].iloc[0]
        self.assertAlmostEqual(row.bias_psad, error.mean())
        self.assertAlmostEqual(row.mae_psad, np.abs(error).mean())
        self.assertAlmostEqual(row.rmse_psad, np.sqrt(np.mean(error**2)))
        self.assertAlmostEqual(row.rmspe_percent, np.sqrt(np.mean(percent**2)))
        self.assertAlmostEqual(row.mean_absolute_percentage_error, np.abs(percent).mean())
        self.assertAlmostEqual(row.loa_lower_psad, error.mean() - 1.96 * error.std(ddof=1))
        exported = pd.read_csv(self.output / 'psa_density_summary.csv')
        self.assertEqual(exported.density_unit.iloc[0], 'ng/mL²')

    def biopsy_namespace(self, pirads, outcomes, density):
        n = len(outcomes)
        ns = self.namespace([10.] * n, [100.] * n, [100.] * n)
        audit = ns['volumes'].copy()
        audit['pirads'], audit['cscancer'] = pirads, outcomes
        audit['gt_psad'], audit['clinical_psad'] = density, density
        audit['psad_evaluable'] = True
        ns.update(psad_audit=audit, PSAD_UNIT='ng/mL²')
        return ns

    def test_biopsy_selection_unknown_outcomes_and_immutable_audit(self):
        ns = self.biopsy_namespace([3, 3, 2, 4, np.nan, 3],
                                   ['Y', 'N', 'Y', 'Y', 'N', 'unknown'], [.1] * 6)
        original = ns['psad_audit'].copy(deep=True)
        self.execute(BIOPSY_SOURCE, ns)
        self.assertEqual(ns['biopsy_cases'].ID.tolist(), ['case-0', 'case-1'])
        self.assertEqual(ns['biopsy_cases'].case_cancer_positive.tolist(), [True, False])
        self.assertEqual(ns['biopsy_audit'].biopsy_analysis_status.iloc[-1], 'unknown_cscancer')
        self.assertEqual(len(pd.read_csv(self.output / 'psa_density_biopsy_audit.csv')), 6)
        pd.testing.assert_frame_equal(original, ns['psad_audit'])

    def test_biopsy_threshold_equality_orientation_and_denominators(self):
        ns = self.biopsy_namespace([3] * 4, ['Y', 'Y', 'N', 'N'], [.15, .1, .2, .1])
        ns['psad_audit']['clinical_psad'] = [.1, .15, .1, .15]
        ns['SELECTED_COMPARATORS'].append('model')
        ns['COMPARATORS']['model'] = ('Model', None)
        ns['psad_audit']['model_psad'] = [.2, .2, .1, .1]
        self.execute(BIOPSY_SOURCE, ns)
        row = ns['biopsy_summary'].set_index('method').loc['gt']
        self.assertEqual([row.tp, row.fp, row.fn, row.tn], [1, 1, 1, 1])
        for metric in ['sensitivity', 'specificity', 'ppv', 'npv', 'accuracy',
                       'biopsies_recommended_fraction', 'biopsies_avoided_fraction',
                       'cancer_positive_missed_fraction']:
            self.assertEqual(row[metric], .5)
        matrix = ns['biopsy_confusion'].query('method == "gt"')
        self.assertEqual(matrix.outcome.tolist(), ['TN', 'FP', 'FN', 'TP'])
        self.assertEqual(ns['biopsy_cases'].gt_biopsy_recommended.tolist(), [True, False, True, False])
        model = ns['biopsy_summary'].set_index('method').loc['model']
        self.assertEqual([model.tp, model.fp, model.fn, model.tn], [2, 0, 0, 2])
        self.assertEqual(len(pd.read_csv(self.output / 'psa_density_biopsy_per_case.csv')), 4)

    def plot_namespace(self, ns):
        ns.update(plt=plt, fill=fill, NullFormatter=NullFormatter, FuncFormatter=FuncFormatter,
                  PLOT_STYLES={key: dict(color=f'C{i}', marker='o', linestyle='-')
                               for i, key in enumerate(ns['SELECTED_COMPARATORS'])},
                  scatter_options=dict(s=16), legend_options=dict(fontsize=7))
        return ns

    def test_all_psad_plots_write_png_and_pdf(self):
        ns = self.namespace([5., 10., 15., 20.], [100.] * 4, [90., 110., 80., 120.],
                            selected=['clinical', 'model'], model_volume_ml=[100., 80., 110., 140.])
        self.plot_namespace(ns)
        self.execute(CALCULATIONS_SOURCE + AGREEMENT_SOURCE + STRATA_SOURCE + CONFUSION_SOURCE, ns)
        for stem in ['psa_density_agreement', 'psa_density_confusion']:
            for ext in ['png', 'pdf']:
                self.assertGreater((self.output / f'{stem}.{ext}').stat().st_size, 0)
        plt.close('all')

    def test_biopsy_empty_single_outcome_and_figures_refresh(self):
        for pirads, outcomes in [([2, 4], ['Y', 'N']), ([np.nan, np.nan], ['Y', 'N']),
                                 ([3, 3], ['unknown', 'unknown']), ([3, 3], ['N', 'N']),
                                 ([3, 3], ['Y', 'Y'])]:
            ns = self.plot_namespace(self.biopsy_namespace(pirads, outcomes, [.1, .2]))
            self.execute(BIOPSY_SOURCE + BIOPSY_PLOT_SOURCE, ns)
            result = ns['biopsy_summary']
            if ns['biopsy_n'] == 0:
                self.assertTrue(result[['tp', 'fp', 'fn', 'tn']].eq(0).all().all())
                self.assertTrue(result.accuracy.isna().all())
                self.assertTrue(pd.read_csv(self.output / 'psa_density_biopsy_per_case.csv').empty)
            elif outcomes == ['N', 'N']:
                self.assertTrue(result.sensitivity.isna().all())
                self.assertTrue(result.cancer_positive_missed_fraction.isna().all())
            else:
                self.assertTrue(result.specificity.isna().all())
            for ext in ['png', 'pdf']:
                self.assertGreater((self.output / f'psa_density_biopsy_confusion.{ext}').stat().st_size, 0)
            plt.close('all')

    def test_analysis_sections_execute_in_cell_order(self):
        psa = np.linspace(5., 40., 8)
        ns = self.plot_namespace(self.namespace(
            psa, np.full(8, 100.), np.linspace(70., 130., 8),
            selected=['clinical', 'model'], model_volume_ml=np.linspace(90., 115., 8),
        ))
        ns['volumes']['pirads'] = [3, 3, 3, 3, 2, 4, np.nan, 3]
        ns['volumes']['cscancer'] = ['Y', 'N', 'Y', 'N', 'Y', 'N', 'unknown', 'unknown']
        ns['ALPHA'] = .05  # Established by the preceding volume inference cells.
        source = SOURCE[SOURCE.index("PSAD_UNIT = 'ng/mL²'"):SOURCE.index('# %% [markdown]\n# ## Local reproducibility')]
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == 'N_BOOTSTRAP':
                    node.value = ast.Constant(1999)
        ast.fix_missing_locations(tree)
        with redirect_stdout(StringIO()):
            exec(compile(tree, str(NOTEBOOK), 'exec'), ns)
        self.assertEqual(len(ns['psad_cases']), 8)
        self.assertEqual(ns['biopsy_n'], 4)
        self.assertEqual(len(ns['psad_pairwise_statistics']), 1)
        self.assertEqual(len(ns['biopsy_summary']), 3)
        plt.close('all')

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)


    def namespace(self, psa, gt_volume, clinical_volume, selected=None, **columns):
        n = len(psa)
        cases = pd.DataFrame(dict(
            ID=[f'case-{i}' for i in range(n)], amrn=np.arange(n), aano=np.arange(n),
            psa_consolidated=psa, pirads=np.full(n, 3.), cscancer=np.full(n, "unknown"), gt_volume_ml=gt_volume,
            clinical_volume_ml=clinical_volume,
        ))
        for name, values in columns.items():
            cases[name] = values
        selected = ['clinical'] if selected is None else selected
        return dict(
            np=np, pd=pd, combinations=combinations, volumes=cases, volumes_all=cases.copy(),
            SELECTED_COMPARATORS=selected, COMPARATORS={k: (k, None, [1]) for k in selected},
            OUTPUT_DIR=self.output, PSAD_REPORTING_TOLERANCE=.005,
            QC_NUMERICAL_TOLERANCE=1e-12, display=lambda value: None,
        )


    def execute(self, source, namespace):
        with redirect_stdout(StringIO()):
            exec(compile(source, str(NOTEBOOK), 'exec'), namespace)
        return namespace


    def test_stratum_boundaries_matrix_orientation_and_thresholds(self):
        psa = np.array([5, 10, 15, 20, 25.])
        estimated = np.array([.12, .05, .16, .10, .25])
        namespace = self.namespace(psa, np.full(5, 100.), psa / estimated)
        self.execute(CALCULATIONS_SOURCE + STRATA_SOURCE, namespace)
        self.assertEqual(namespace['psad_cases']['gt_psad_stratum'].cat.codes.tolist(), [0, 1, 2, 3, 3])
        matrix = namespace['psad_confusion_matrices']['clinical']
        np.testing.assert_array_equal(matrix, [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1]])
        summary = namespace['psad_classification'].iloc[0]
        self.assertEqual(summary.n, 5)
        self.assertEqual(summary.accuracy, .4)
        self.assertEqual(summary.lower_stratum_n, 2)
        self.assertEqual(summary.higher_stratum_n, 1)
        self.assertEqual(summary.balanced_accuracy, .375)
        self.assertAlmostEqual(summary.macro_f1, (1 + 2/3) / 4)
        self.assertAlmostEqual(summary.quadratic_weighted_kappa, .53125)
        threshold = namespace['psad_thresholds'].set_index('cutoff_psad').loc[.15]
        np.testing.assert_array_equal(threshold[['tp', 'fp', 'fn', 'tn']], [2, 0, 1, 2])
        self.assertAlmostEqual(threshold.sensitivity, 2/3)
        self.assertEqual(threshold.specificity, 1)
        self.assertAlmostEqual(threshold.npv, 2/3)
        confusion = pd.read_csv(self.output / 'psa_density_confusion.csv')
        self.assertEqual(len(confusion), 16)
        self.assertEqual(confusion.n.sum(), 5)
        self.assertEqual(len(pd.read_csv(self.output / 'psa_density_per_case.csv')), 5)


    def test_perfect_agreement_empty_strata_and_undefined_ratios(self):
        namespace = self.namespace([5, 5, 5], [100, 100, 100], [100, 100, 100])
        self.execute(CALCULATIONS_SOURCE + STRATA_SOURCE, namespace)
        summary = namespace['psad_classification'].iloc[0]
        self.assertEqual(summary.accuracy, 1)
        self.assertEqual(summary.balanced_accuracy, 1)
        self.assertEqual(summary.macro_f1, 1)
        self.assertTrue(np.isnan(summary.quadratic_weighted_kappa))
        per_stratum = namespace['psad_per_stratum']
        self.assertTrue(per_stratum.iloc[1:]['recall'].isna().all())
        self.assertTrue(per_stratum.iloc[1:]['precision'].isna().all())
        self.assertTrue(np.isnan(per_stratum.iloc[0].specificity))
        self.assertTrue(namespace['psad_thresholds'].sensitivity.isna().all())
        self.assertTrue(namespace['psad_thresholds'].ppv.isna().all())
        self.assertTrue(namespace['psad_thresholds'].specificity.eq(1).all())
        self.assertTrue(namespace['psad_confusion'].query('reference_support == 0').row_percent.isna().all())


    def run_stratum_inference(self, correctness, selected=None):
        selected = list(correctness) if selected is None else selected
        n = len(next(iter(correctness.values())))
        reference_codes = np.arange(n) % 4
        densities = np.array([.05, .12, .17, .25])
        method_volumes = {}
        for key, correct in correctness.items():
            codes = np.where(correct, reference_codes, (reference_codes + 1) % 4)
            method_volumes[f'{key}_volume_ml'] = 10 / densities[codes]
        namespace = self.namespace(
            np.full(n, 10.), 10 / densities[reference_codes], np.full(n, 100.),
            selected=selected, **method_volumes,
        )
        namespace['ALPHA'] = .05
        self.execute(CALCULATIONS_SOURCE + STRATA_SOURCE + STRATUM_INFERENCE_SOURCE, namespace)
        result = namespace['psad_stratum_pairwise_statistics']
        exported = pd.read_csv(self.output / 'psa_density_stratum_pairwise_statistics.csv')
        pd.testing.assert_frame_equal(result, exported, check_dtype=False)
        for row in result.itertuples():
            self.assertEqual(row.n, n)
            self.assertEqual(
                row.both_correct + row.a_only_correct + row.b_only_correct + row.both_incorrect, n,
            )
            self.assertEqual(
                row.both_correct + row.a_only_correct,
                np.trace(namespace['psad_confusion_matrices'][row.method_a]),
            )
            self.assertEqual(
                row.both_correct + row.b_only_correct,
                np.trace(namespace['psad_confusion_matrices'][row.method_b]),
            )
        return result, namespace


    def test_stratum_exact_known_pvalue_and_reversed_methods(self):
        correctness = {'a': np.ones(10, dtype=bool), 'b': np.zeros(10, dtype=bool)}
        result, _ = self.run_stratum_inference(correctness)
        row = result.iloc[0]
        self.assertEqual(row.mcnemar_exact_p, .001953125)
        self.assertEqual(row.mcnemar_exact_p_holm, row.mcnemar_exact_p)
        self.assertEqual(row.holm_family_size, 1)
        self.assertEqual(row.alpha, .05)
        self.assertEqual(row.accuracy_a, 1)
        self.assertEqual(row.accuracy_b, 0)
        self.assertEqual(row.accuracy_difference_pp, 100)
        self.assertEqual(row.conclusion, 'A better')
        reverse = self.run_stratum_inference(correctness, selected=['b', 'a'])[0].iloc[0]
        self.assertEqual(reverse.mcnemar_exact_p, row.mcnemar_exact_p)
        self.assertEqual(reverse.accuracy_difference_pp, -row.accuracy_difference_pp)
        self.assertEqual(reverse.a_only_correct, row.b_only_correct)
        self.assertEqual(reverse.b_only_correct, row.a_only_correct)
        self.assertEqual(reverse.conclusion, 'B better')


    def test_stratum_paired_counts_and_equal_accuracy_with_discordance(self):
        result, _ = self.run_stratum_inference({
            'a': [True, True, True, False, False, False, True, False, True, False],
            'b': [True, False, True, True, False, False, False, True, True, False],
        })
        row = result.iloc[0]
        self.assertEqual(
            [row.both_correct, row.a_only_correct, row.b_only_correct, row.both_incorrect],
            [3, 2, 2, 3],
        )
        self.assertEqual(row.accuracy_a, .5)
        self.assertEqual(row.accuracy_b, .5)
        self.assertEqual(row.accuracy_difference_pp, 0)
        self.assertEqual(row.mcnemar_exact_p, 1)
        self.assertEqual(row.conclusion, 'no significant difference')


    def test_stratum_identical_correctness_has_no_discordance(self):
        flags = [True, False, True, False]
        row = self.run_stratum_inference({'a': flags, 'b': flags})[0].iloc[0]
        self.assertEqual(row.a_only_correct + row.b_only_correct, 0)
        self.assertEqual(row.mcnemar_exact_p, 1)
        self.assertEqual(row.mcnemar_exact_p_holm, 1)
        self.assertEqual(row.accuracy_difference_pp, 0)
        self.assertEqual(row.conclusion, 'no significant difference')


    def test_stratum_three_method_holm_family(self):
        result, _ = self.run_stratum_inference({
            'a': np.ones(10, dtype=bool), 'b': np.zeros(10, dtype=bool),
            'c': [True] * 5 + [False] * 5,
        })
        self.assertEqual(
            list(zip(result.method_a, result.method_b)), [('a', 'b'), ('a', 'c'), ('b', 'c')],
        )
        np.testing.assert_allclose(result.mcnemar_exact_p, [.001953125, .0625, .0625])
        np.testing.assert_allclose(result.mcnemar_exact_p_holm, [.005859375, .125, .125])
        self.assertTrue(result.holm_family_size.eq(3).all())
        self.assertEqual(
            result.conclusion.tolist(),
            ['A better', 'no significant difference', 'no significant difference'],
        )


    def test_stratum_single_method_replaces_stale_output_with_same_schema(self):
        correctness = {'a': [True, False], 'b': [False, True]}
        previous, _ = self.run_stratum_inference(correctness)
        result, _ = self.run_stratum_inference(correctness, selected=['a'])
        self.assertTrue(result.empty)
        self.assertEqual(result.columns.tolist(), previous.columns.tolist())
        self.assertTrue(pd.read_csv(self.output / 'psa_density_stratum_pairwise_statistics.csv').empty)


    def test_stratum_missing_assignment_is_rejected(self):
        _, namespace = self.run_stratum_inference({'a': [True, False], 'b': [False, True]})
        namespace['psad_cases'].loc[0, 'a_psad_stratum'] = np.nan
        with self.assertRaisesRegex(AssertionError, 'Missing PSA-density stratum'):
            self.execute(STRATUM_INFERENCE_SOURCE, namespace)


    def run_inference(self, errors, selected=None, references=None):
        selected = list(errors) if selected is None else selected
        n = len(next(iter(errors.values())))
        namespace = self.namespace(np.full(n, 10.), np.full(n, 100.), np.full(n, 100.), selected=selected)
        cases = namespace['volumes'][['ID', 'amrn', 'aano']].copy()
        cases['gt_psad'] = .1 if references is None else references
        for key, error in errors.items():
            cases[f'{key}_psad'] = cases['gt_psad'] + np.asarray(error)
            cases[f'{key}_absolute_error_psad'] = np.abs(error)
        namespace.update(psad_cases=cases, PSAD_UNIT='ng/mL²')
        tree = ast.parse(INFERENCE_SOURCE)
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == 'N_BOOTSTRAP':
                    node.value = ast.Constant(1999)
                elif node.targets[0].id == 'BOOTSTRAP_BATCH':
                    node.value = ast.Constant(113)
        ast.fix_missing_locations(tree)
        with redirect_stdout(StringIO()):
            exec(compile(tree, str(NOTEBOOK), 'exec'), namespace)
        result = namespace['psad_pairwise_statistics']
        exported = pd.read_csv(self.output / 'psa_density_pairwise_statistics.csv')
        self.assertEqual(result.shape, exported.shape)
        if not result.empty:
            np.testing.assert_allclose(exported.rmspe_improvement_pp, result.rmspe_improvement_pp)
            np.testing.assert_allclose(exported.rmspe_bootstrap_p_holm, result.rmspe_bootstrap_p_holm)
        return result


    def test_density_inference_reproducibility_pairing_and_reversal(self):
        errors = {'a': np.linspace(.001, .02, 30), 'b': np.linspace(.03, .08, 30)}
        result = self.run_inference(errors)
        pd.testing.assert_frame_equal(result, self.run_inference(errors))
        row = result.iloc[0]
        reverse = self.run_inference(errors, ['b', 'a']).iloc[0]
        self.assertGreater(row.rmse_improvement_ci_lower_psad, 0)
        self.assertEqual(row.rmse_conclusion, 'A better')
        self.assertEqual(row.closer_conclusion, 'A better')
        self.assertEqual(row.wins_a, 30)
        self.assertEqual(row.density_unit, 'ng/mL²')
        self.assertAlmostEqual(row.rmse_improvement_psad, -reverse.rmse_improvement_psad)
        self.assertAlmostEqual(row.rmse_improvement_ci_lower_psad, -reverse.rmse_improvement_ci_upper_psad)
        self.assertEqual(row.rmse_bootstrap_p, reverse.rmse_bootstrap_p)
        self.assertAlmostEqual(row.probability_a_closer_given_no_tie, 1 - reverse.probability_a_closer_given_no_tie)


    def test_density_inference_identical_single_and_nonfinite(self):
        errors = np.linspace(.001, .02, 30)
        row = self.run_inference({'a': errors, 'b': errors}).iloc[0]
        self.assertEqual(row.rmse_improvement_psad, 0)
        self.assertEqual(row.rmse_bootstrap_p, 1)
        self.assertEqual(row.ties, 30)
        self.assertEqual(row.closer_conclusion, 'uninformative_all_ties')
        self.assertTrue(self.run_inference({'a': errors}).empty)
        self.assertTrue(pd.read_csv(self.output / 'psa_density_pairwise_counts.csv').empty)
        self.assertTrue(pd.read_csv(self.output / 'psa_density_pairwise_per_case.csv').empty)
        with self.assertRaises(AssertionError):
            self.run_inference({'a': errors, 'b': np.full(30, np.nan)})


    def test_rmspe_variable_references_and_pairing(self):
        from scipy import stats

        references = np.linspace(20., 200., 30)
        errors = np.linspace(.5, 4.5, 30)
        row = self.run_inference(dict(a=errors, b=2 * errors), references=references).iloc[0]
        percent = 100 * errors / references
        expected = stats.bootstrap(
            (percent,), lambda x, axis: np.sqrt(np.mean(x**2, axis=axis)),
            vectorized=True, n_resamples=1999, batch=113,
            rng=np.random.default_rng(20261102), method='BCa',
        ).confidence_interval
        self.assertAlmostEqual(row.rmspe_a_percent, np.sqrt(np.mean(percent**2)))
        self.assertAlmostEqual(row.rmspe_b_percent, 2 * row.rmspe_a_percent)
        self.assertAlmostEqual(row.rmspe_improvement_pp, row.rmspe_a_percent)
        np.testing.assert_allclose(
            [row.rmspe_improvement_ci_lower_pp, row.rmspe_improvement_ci_upper_pp], expected,
        )
        reverse = self.run_inference(dict(b=2 * errors, a=errors), references=references).iloc[0]
        self.assertAlmostEqual(row.rmspe_improvement_pp, -reverse.rmspe_improvement_pp)
        self.assertAlmostEqual(row.rmspe_improvement_ci_lower_pp, -reverse.rmspe_improvement_ci_upper_pp)
        self.assertEqual(row.rmspe_bootstrap_p, reverse.rmspe_bootstrap_p)


    def test_rmspe_and_rmse_can_favor_different_methods(self):
        references = np.tile([10., 100.], 20)
        row = self.run_inference(
            dict(a=np.tile([5., 0.], 20), b=np.tile([0., 10.], 20)), references=references,
        ).iloc[0]
        self.assertGreater(row.rmse_improvement_psad, 0)
        self.assertLess(row.rmspe_improvement_pp, 0)


    def test_rmspe_separate_holm_families_and_reproducibility(self):
        from statsmodels.stats.multitest import multipletests

        errors = dict(a=np.linspace(.2, 2, 30), b=np.linspace(3, 15, 30), c=np.linspace(5, 20, 30))
        result = self.run_inference(errors)
        pd.testing.assert_frame_equal(result, self.run_inference(errors))
        self.assertTrue(result.rmspe_holm_family_size.eq(3).all())
        self.assertTrue(result.holm_family_size.eq(6).all())
        np.testing.assert_allclose(result.rmspe_bootstrap_p_holm,
                                   multipletests(result.rmspe_bootstrap_p, method='holm')[1])
        np.testing.assert_allclose(result.rmspe_t_sensitivity_p_holm,
                                   multipletests(result.rmspe_t_sensitivity_p, method='holm')[1])
        expected = multipletests(result[['rmse_bootstrap_p', 'closer_exact_p']].to_numpy().ravel(),
                                method='holm')[1].reshape(-1, 2)
        np.testing.assert_allclose(result[['rmse_bootstrap_p_holm', 'closer_exact_p_holm']], expected)


    def test_rmspe_degenerate_errors_and_invalid_references(self):
        row = self.run_inference(dict(a=np.ones(20), b=np.full(20, 2)), references=np.full(20, 100.)).iloc[0]
        self.assertEqual(row.rmspe_improvement_pp, 1)
        self.assertEqual(row.rmspe_improvement_ci_lower_pp, 1)
        self.assertEqual(row.rmspe_improvement_ci_upper_pp, 1)
        self.assertEqual(row.rmspe_ci_method, 'deterministic')
        self.assertAlmostEqual(row.rmspe_bootstrap_p, 1 / 2000)
        self.assertTrue(np.isnan(row.rmspe_t_sensitivity_p_holm))
        self.assertEqual(row.rmspe_t_sensitivity_conclusion, 'undefined')
        tied = self.run_inference(dict(a=np.ones(20), b=np.ones(20))).iloc[0]
        self.assertEqual(tied.rmspe_improvement_pp, 0)
        self.assertEqual(tied.rmspe_bootstrap_p, 1)
        self.assertTrue(self.run_inference(dict(a=np.ones(20))).empty)
        for invalid in [0., -1., np.nan, np.inf]:
            with self.subTest(reference=invalid), self.assertRaises(AssertionError):
                self.run_inference(dict(a=np.ones(20), b=np.full(20, 2)), references=np.full(20, invalid))


