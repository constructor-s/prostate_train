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
INFERENCE_SOURCE = SOURCE[
    SOURCE.index('# %% [markdown]\n# ## PSA density: paired absolute-error comparisons'):
    SOURCE.index('# %% [markdown]\n# ## Reported marksheet PSA density:')
]
REPORTING_SOURCE = SOURCE[
    SOURCE.index('# %% [markdown]\n# ## Reported marksheet PSA density:'):
    SOURCE.index('# %% [markdown]\n# ### Reported-density agreement')
]


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
            np=np, pd=pd, combinations=combinations, volumes=cases,
            SELECTED_COMPARATORS=selected, COMPARATORS={k: (k, None, [1]) for k in selected},
            OUTPUT_DIR=self.output, display=lambda value: None,
        )

    def execute(self, source, namespace):
        with redirect_stdout(StringIO()):
            exec(compile(source, str(NOTEBOOK), 'exec'), namespace)
        return namespace

    def test_calculation_units_and_common_exclusions(self):
        namespace = self.namespace(
            [5, 10, np.nan, -1, 0, 20, 30, 40, 50, 60], np.full(10, 50.),
            [50, 50, 50, 50, 50, 0, np.inf, 50, 50, 50], selected=['clinical', 'model'],
            model_volume_ml=[25, 25, 25, 25, 25, 25, 25, -1, np.nan, 25],
        )
        self.execute(CALCULATIONS_SOURCE, namespace)
        audit, cases = namespace['psad_audit'], namespace['psad_cases']
        self.assertEqual(cases.index.tolist(), [0, 1, 9])
        self.assertAlmostEqual(cases.loc[0, 'gt_psad'], .1)
        self.assertAlmostEqual(cases.loc[0, 'model_psad'], .2)
        self.assertAlmostEqual(cases.loc[0, 'model_minus_gt_psad'], .1)
        self.assertAlmostEqual(cases.loc[0, 'model_absolute_percentage_error_psad'], 100)
        self.assertEqual(audit.loc[4, 'gt_psad'], 0)
        self.assertTrue(np.isnan(audit.loc[5, 'clinical_psad']))
        self.assertEqual(audit.psa_density_status.tolist()[2:9], [
            'missing_or_nonfinite_psa', 'negative_psa', 'zero_psa',
            'invalid_clinical_density_or_volume', 'invalid_clinical_density_or_volume',
            'invalid_model_density_or_volume', 'invalid_model_density_or_volume',
        ])
        self.assertEqual(len(namespace['volumes']), 10)
        self.assertTrue(namespace['psad_summary']['n'].eq(3).all())
        exported = pd.read_csv(self.output / 'psa_density_cohort_audit.csv')
        self.assertEqual(len(exported), 10)
        self.assertFalse(np.isinf(exported[['gt_psad', 'clinical_psad', 'model_psad']]).any().any())

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


if __name__ == '__main__':
    unittest.main()
