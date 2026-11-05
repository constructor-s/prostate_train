"""Exercise actual PSA-density notebook cells on small, synthetic cohorts."""
from __future__ import annotations

import ast
from contextlib import redirect_stdout
from io import StringIO
from itertools import combinations
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/prostatex_linked_clinical_volume.py'
SOURCE = NOTEBOOK.read_text()
CALCULATIONS_SOURCE = SOURCE[
    SOURCE.index('# %% [markdown]\n# ## PSA density: calculations'):
    SOURCE.index('# %% [markdown]\n# ## PSA density: agreement plots')
]
STRATA_SOURCE = SOURCE[
    SOURCE.index('# %% [markdown]\n# ## PSA density: strata and threshold statistics'):
    SOURCE.index('# %%\nfig, axes = plt.subplots(\n    1, len(SELECTED_COMPARATORS), figsize=(3.4')
]
STRATUM_INFERENCE_SOURCE = SOURCE[
    SOURCE.index('# %% [markdown]\n# ## PSA density: paired stratum-accuracy comparisons'):
    SOURCE.index('# %% [markdown]\n# ## PSA density: paired absolute-error comparisons')
]
INFERENCE_SOURCE = SOURCE[
    SOURCE.index('# %% [markdown]\n# ## PSA density: paired absolute-error comparisons'):
    SOURCE.index('# %% [markdown]\n# ## Reported marksheet PSA density:')
]
REPORTING_SOURCE = SOURCE[
    SOURCE.index('# %% [markdown]\n# ## Reported marksheet PSA density:'):
    SOURCE.index('# %% [markdown]\n# ### Reported-density agreement')
]
BIOPSY_SOURCE = SOURCE[
    SOURCE.index('# %% [markdown]\n# ## PI-RADS 3: biopsy decisions'):
    SOURCE.index('# %% [markdown]\n# ### Biopsy decisions: confusion matrices')
]
BIOPSY_PLOT_SOURCE = SOURCE[SOURCE.index('# %% [markdown]\n# ### Biopsy decisions: confusion matrices'):]


class LinkedPSADensityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)

    def namespace(self, psa, gt_volume, clinical_volume, selected=None, **columns):
        n = len(psa)
        cases = pd.DataFrame(dict(
            ID=[f'case-{i}' for i in range(n)], patient_id=np.arange(n), study_id=np.arange(n),
            psa=psa, psad=np.full(n, np.nan), gt_volume_ml=gt_volume,
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

    def test_calculation_units_and_psa_only_exclusion(self):
        namespace = self.namespace(
            [5, 10, np.nan, np.inf, 'bad', 60], np.full(6, 50.),
            np.full(6, 50.), selected=['clinical', 'model'],
            model_volume_ml=[25, 25, 0, np.nan, -1, 25],
            psad=[np.nan, 99, np.nan, np.nan, np.nan, .1],
            beyond_reporting_tolerance=[pd.NA, True, pd.NA, pd.NA, pd.NA, True],
        )
        self.execute(CALCULATIONS_SOURCE, namespace)
        audit, cases = namespace['psad_audit'], namespace['psad_cases']
        self.assertEqual(cases.index.tolist(), [0, 1, 5])
        self.assertAlmostEqual(cases.loc[0, 'gt_psad'], .1)
        self.assertAlmostEqual(cases.loc[0, 'model_psad'], .2)
        self.assertAlmostEqual(cases.loc[0, 'model_minus_gt_psad'], .1)
        self.assertAlmostEqual(cases.loc[0, 'model_absolute_percentage_error_psad'], 100)
        self.assertTrue(audit.loc[[2, 3, 4], 'gt_psad'].isna().all())
        self.assertTrue(audit.loc[[2, 3, 4], 'psa_density_status'].eq('missing_or_nonfinite_psa').all())
        self.assertTrue(pd.isna(cases.loc[0, 'psad']))
        self.assertTrue(cases.loc[1, 'beyond_reporting_tolerance'])
        self.assertEqual(len(namespace['volumes']), 6)
        self.assertTrue(namespace['psad_summary']['n'].eq(3).all())
        exported = pd.read_csv(self.output / 'psa_density_cohort_audit.csv')
        self.assertEqual(len(exported), 6)
        self.assertFalse(np.isinf(exported[['gt_psad', 'clinical_psad', 'model_psad']]).any().any())

    def test_invalid_density_inputs_stop_instead_of_silently_dropping_cases(self):
        for psa_value, gt_value, clinical_value, model_value, message, status in [
            (-1, 50, 50, 25, 'Nonpositive PSA', 'negative_psa'),
            (0, 50, 50, 25, 'Nonpositive PSA', 'zero_psa'),
            (10, 0, 50, 25, 'Invalid gt volume', 'invalid_gt_density_or_volume'),
            (10, 50, np.inf, 25, 'Invalid clinical volume', 'invalid_clinical_density_or_volume'),
            (10, 50, 50, -1, 'Invalid model volume', 'invalid_model_density_or_volume'),
            (10, 50, 50, np.nan, 'Invalid model volume', 'invalid_model_density_or_volume'),
            (10, 50, 50, 0, 'Invalid model volume', 'invalid_model_density_or_volume'),
        ]:
            with self.subTest(psa=psa_value, gt=gt_value, clinical=clinical_value, model=model_value):
                namespace = self.namespace(
                    [5, 10, psa_value], [50, 50, gt_value], [50, 50, clinical_value],
                    selected=['clinical', 'model'], model_volume_ml=[25, 25, model_value],
                )
                with self.assertRaisesRegex(AssertionError, message):
                    self.execute(CALCULATIONS_SOURCE, namespace)
                audit = pd.read_csv(self.output / 'psa_density_cohort_audit.csv')
                self.assertEqual(len(audit), 3)
                self.assertEqual(audit.psa_density_status.iloc[2], status)
                self.assertEqual(len(namespace['volumes']), 3)

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

    def test_insufficient_or_repeated_patient_cohort_is_rejected(self):
        namespace = self.namespace([10, np.nan], [100, 100], [100, 100])
        with self.assertRaisesRegex(AssertionError, 'Fewer than two'):
            self.execute(CALCULATIONS_SOURCE, namespace)
        self.assertEqual(len(pd.read_csv(self.output / 'psa_density_cohort_audit.csv')), 2)
        namespace = self.namespace([10, 20], [100, 100], [100, 100])
        namespace['volumes']['patient_id'] = 1
        with self.assertRaisesRegex(AssertionError, 'unique patients'):
            self.execute(CALCULATIONS_SOURCE, namespace)

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

    def run_inference(self, errors, selected=None):
        selected = list(errors) if selected is None else selected
        n = len(next(iter(errors.values())))
        namespace = self.namespace(np.full(n, 10.), np.full(n, 100.), np.full(n, 100.), selected=selected)
        cases = namespace['volumes'][['ID', 'patient_id', 'study_id']].copy()
        cases['gt_psad'] = .1
        for key, error in errors.items():
            cases[f'{key}_psad'] = .1 + np.asarray(error)
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

    def test_reported_density_residuals_decomposition_and_rounding(self):
        namespace = self.namespace(
            [10.] * 8, [125.] * 8, [100.] * 8, selected=['model'],
            model_volume_ml=[0.] * 8, psad=[.1, .104, .105, .106, 0, np.nan, -.1, np.inf],
        )
        namespace.update(PSAD_UNIT='ng/mL²', PSAD_BINS=[0, .1, .15, .2, np.inf],
                         PSAD_STRATA=['0–<0.10', '0.10–<0.15', '0.15–<0.20', '≥0.20'])
        self.execute(REPORTING_SOURCE, namespace)
        cases = namespace['reported_psad_cases']
        self.assertEqual(len(cases), 5)  # Independent of invalid model volumes.
        np.testing.assert_allclose(cases.reported_minus_recalculated_psad, [0, .004, .005, .006, -.1])
        np.testing.assert_allclose(cases.recalculated_minus_manual_mask_psad, .02)
        np.testing.assert_allclose(
            cases.reported_minus_manual_mask_psad,
            cases.reported_minus_recalculated_psad + cases.recalculated_minus_manual_mask_psad,
        )
        np.testing.assert_allclose(
            cases.reporting_change_in_absolute_manual_mask_error_psad, [0, .004, .005, .006, .06],
        )
        self.assertEqual(cases.beyond_reporting_tolerance.tolist(), [False, False, False, True, True])
        self.assertTrue(np.isnan(cases.iloc[4].volume_implied_by_reported_psad_ml))
        self.assertEqual(cases.iloc[0].volume_implied_by_reported_psad_ml, 100)
        self.assertEqual(cases.reporting_changes_stratum.tolist(), [False, False, False, False, True])
        self.assertTrue(cases.clinical_volume_changes_stratum.all())
        self.assertTrue(namespace['reported_psad_summary']['n'].eq(5).all())
        self.assertEqual(namespace['reported_psad_summary'].iloc[0].stratum_accuracy, .8)
        confusion = namespace['reported_psad_confusion']
        self.assertEqual(len(confusion), 48)
        self.assertTrue(confusion.groupby('comparison').n.sum().eq(5).all())
        self.assertEqual(len(pd.read_csv(self.output / 'psa_density_reported_audit.csv')), 8)
        self.assertEqual(len(pd.read_csv(self.output / 'psa_density_reported_per_case.csv')), 5)

    def test_missing_reported_density_produces_explicit_empty_results(self):
        namespace = self.namespace([10, 20], [100, 100], [100, 100])
        namespace.update(PSAD_UNIT='ng/mL²', PSAD_BINS=[0, .1, .15, .2, np.inf],
                         PSAD_STRATA=['0–<0.10', '0.10–<0.15', '0.15–<0.20', '≥0.20'])
        self.execute(REPORTING_SOURCE, namespace)
        self.assertTrue(namespace['reported_psad_cases'].empty)
        self.assertTrue(namespace['reported_psad_summary']['n'].eq(0).all())
        self.assertTrue(namespace['reported_psad_confusion']['n'].eq(0).all())
        self.assertTrue(namespace['reported_psad_audit'].beyond_reporting_tolerance.isna().all())

    def test_reporting_diagnostics_are_independent_of_recalculated_density_subset(self):
        namespace = self.namespace([10, 20, 30], [100] * 3, [100] * 3, psad=[.1, .8, .3])
        # The final diagnostic uses the complete volume cohort, not a modified view.
        namespace['volumes'] = namespace['volumes'].iloc[[0, 2]].copy()
        namespace.update(PSAD_UNIT='ng/mL²', PSAD_BINS=[0, .1, .15, .2, np.inf],
                         PSAD_STRATA=['0–<0.10', '0.10–<0.15', '0.15–<0.20', '≥0.20'])
        self.execute(REPORTING_SOURCE, namespace)
        self.assertEqual(len(namespace['volumes']), 2)
        self.assertEqual(namespace['reported_psad_cases'].ID.tolist(), ['case-0', 'case-1', 'case-2'])
        self.assertTrue(namespace['reported_psad_summary']['n'].eq(3).all())


    def biopsy_namespace(self, pirads, isup, density):
        n = len(isup)
        namespace = self.namespace([10.] * n, [100.] * n, [100.] * n)
        audit = namespace['volumes'].copy()
        audit['case_ISUP'] = isup
        audit['gt_psad'] = density
        audit['clinical_psad'] = density
        audit['psa_density_status'] = 'included'
        audit['beyond_reporting_tolerance'] = True
        if pirads is not None:
            audit['lesion_PIRADS'] = pirads
        namespace.update(psad_audit=audit, PSAD_UNIT='ng/mL²')
        return namespace

    def test_biopsy_uses_maximum_pirads_and_retains_review_flags(self):
        namespace = self.biopsy_namespace(
            ['3', '2, 3', '3,3,N/A', '3,4', '5,3', np.nan, 'N/A', '', '1,3', '3.0'],
            [0, 1, 2, 2, 3, 0, 0, 0, 3, 5], [.1] * 10,
        )
        original = namespace['psad_audit'].copy(deep=True)
        self.execute(BIOPSY_SOURCE, namespace)
        cases = namespace['biopsy_cases']
        self.assertEqual(cases.ID.tolist(), ['case-0', 'case-1', 'case-2', 'case-8', 'case-9'])
        self.assertTrue(cases.beyond_reporting_tolerance.all())
        self.assertTrue(cases.psad.isna().all())  # Missing reported PSAD does not exclude.
        self.assertEqual(cases.case_cspca.tolist(), [False, False, True, True, True])
        self.assertTrue(namespace['biopsy_audit'].loc[5:7, 'biopsy_pirads_status'].eq('no_numeric_pirads').all())
        pd.testing.assert_frame_equal(namespace['psad_audit'], original)

    def test_biopsy_threshold_equality_confusion_orientation_and_denominators(self):
        namespace = self.biopsy_namespace(
            ['3'] * 4, [2, 5, 0, 1], [.15, np.nextafter(.15, 0), .2, .1],
        )
        namespace['psad_audit']['clinical_psad'] = [.1, .15, .1, .15]
        namespace['psad_audit']['model_psad'] = .2
        namespace['SELECTED_COMPARATORS'].append('model')
        namespace['COMPARATORS']['model'] = ('Model', None, [1])
        self.execute(BIOPSY_SOURCE, namespace)
        cases = namespace['biopsy_cases']
        self.assertEqual(cases.gt_biopsy_recommended.tolist(), [True, False, True, False])
        self.assertEqual(cases.gt_biopsy_outcome.tolist(), ['TP', 'FN', 'FP', 'TN'])
        summary = namespace['biopsy_summary'].set_index('method')
        self.assertEqual(summary.index.tolist(), ['gt', 'clinical', 'model'])
        row = summary.loc['gt']
        np.testing.assert_array_equal(row[['tp', 'fp', 'fn', 'tn']], [1, 1, 1, 1])
        np.testing.assert_allclose(row[['sensitivity', 'specificity', 'ppv', 'npv', 'accuracy']].astype(float), .5)
        self.assertEqual(row.n_cspca, 2)
        self.assertEqual(row.n_non_cspca, 2)
        self.assertEqual(row.biopsies_recommended_n, 2)
        self.assertEqual(row.biopsies_avoided_n, 2)
        self.assertEqual(row.cspca_detected_n, 1)
        self.assertEqual(row.cspca_missed_n, 1)
        self.assertEqual(row.cspca_missed_fraction, .5)
        confusion = namespace['biopsy_confusion'].query('method == "model"')
        np.testing.assert_array_equal(confusion['count'].to_numpy().reshape(2, 2), [[0, 2], [0, 2]])
        self.assertEqual(confusion.outcome.tolist(), ['TN', 'FP', 'FN', 'TP'])
        self.assertTrue(namespace['biopsy_confusion'].groupby('method')['count'].sum().eq(4).all())
        self.assertEqual(pd.read_csv(self.output / 'psa_density_biopsy_summary.csv').shape, namespace['biopsy_summary'].shape)
        self.assertEqual(len(pd.read_csv(self.output / 'psa_density_biopsy_per_case.csv')), 4)

    def test_biopsy_falls_back_only_when_pirads_column_is_absent(self):
        namespace = self.biopsy_namespace(None, [0, 1, 5], [.1, .2, .3])
        self.execute(BIOPSY_SOURCE, namespace)
        self.assertEqual(len(namespace['biopsy_cases']), 3)
        self.assertEqual(namespace['biopsy_cohort'], 'entire_linked_cohort_PI-RADS_unavailable')
        self.assertTrue(namespace['biopsy_audit'].biopsy_pirads_status.eq('pirads_column_absent').all())
        for pirads in [[np.nan, 'N/A'], ['2', '3,4']]:
            namespace = self.biopsy_namespace(pirads, [0, 2], [.1, .2])
            self.execute(BIOPSY_SOURCE, namespace)
            self.assertEqual(namespace['biopsy_cohort'], 'maximum_PI-RADS_3')
            self.assertTrue(namespace['biopsy_cases'].empty)
            self.assertTrue(namespace['biopsy_summary']['n'].eq(0).all())
            self.assertTrue(namespace['biopsy_summary'][['sensitivity', 'specificity', 'ppv', 'npv', 'accuracy']].isna().all().all())
            self.assertTrue(namespace['biopsy_confusion']['count'].eq(0).all())
            self.assertTrue(pd.read_csv(self.output / 'psa_density_biopsy_per_case.csv').empty)

    def test_biopsy_audits_missing_invalid_outcomes_and_missing_psa(self):
        namespace = self.biopsy_namespace(
            ['3'] * 10, [np.nan, 'bad', np.inf, -1, 1.5, 6, 'N/A', 0, 2, 3], [.1] * 10,
        )
        namespace['psad_audit'].loc[9, 'psa_density_status'] = 'missing_or_nonfinite_psa'
        self.execute(BIOPSY_SOURCE, namespace)
        audit = namespace['biopsy_audit']
        self.assertEqual(audit.biopsy_outcome_status.tolist(), [
            'missing_case_isup', *(['invalid_case_isup'] * 5), 'missing_case_isup',
            'valid_case_isup', 'valid_case_isup', 'valid_case_isup',
        ])
        self.assertTrue(audit.case_cspca.iloc[:7].isna().all())
        self.assertEqual(audit.biopsy_analysis_status.iloc[9], 'missing_or_nonfinite_psa')
        self.assertEqual(namespace['biopsy_cases'].ID.tolist(), ['case-7', 'case-8'])
        self.assertEqual(len(pd.read_csv(self.output / 'psa_density_biopsy_audit.csv')), 10)

    def test_biopsy_rejects_invalid_pirads_scores(self):
        for score in ['0', '6', '3.5', 'inf', 'nan', 'bad', '3,4.5']:
            with self.subTest(score=score):
                namespace = self.biopsy_namespace([score], [0], [.1])
                with self.assertRaisesRegex(ValueError, 'case-0: .*lesion_PIRADS'):
                    self.execute(BIOPSY_SOURCE, namespace)

    def test_biopsy_undefined_ratios_and_figures(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        self.addCleanup(plt.close, 'all')
        for isup, density in [([0, 1], [.1, .1]), ([2, 5], [.2, .2])]:
            namespace = self.biopsy_namespace(['3', '3'], isup, density)
            namespace.update(plt=plt)
            self.execute(BIOPSY_SOURCE + BIOPSY_PLOT_SOURCE, namespace)
            row = namespace['biopsy_summary'].iloc[0]
            if isup[0] == 0:
                self.assertTrue(np.isnan(row.sensitivity))
                self.assertTrue(np.isnan(row.ppv))
                self.assertEqual(row.specificity, 1)
                self.assertEqual(row.npv, 1)
            else:
                self.assertTrue(np.isnan(row.specificity))
                self.assertTrue(np.isnan(row.npv))
                self.assertEqual(row.sensitivity, 1)
                self.assertEqual(row.ppv, 1)
            plt.close('all')
        # Empty results refresh an earlier nonempty figure; exercise one-panel layout.
        namespace = self.biopsy_namespace(['4'], [0], [.1])
        namespace['SELECTED_COMPARATORS'] = []
        namespace.update(plt=plt)
        self.execute(BIOPSY_SOURCE + BIOPSY_PLOT_SOURCE, namespace)
        self.assertTrue(namespace['biopsy_cases'].empty)
        for extension in ['png', 'pdf']:
            self.assertGreater((self.output / f'psa_density_biopsy_confusion.{extension}').stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()
