"""Exercise the notebook's actual inference cells without loading image datasets."""
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
STATISTICS_SOURCE = SOURCE[
    SOURCE.index('# %% [markdown]\n# ## Paired post hoc inference:'):
    SOURCE.index('# %% [markdown]\n# ## PSA density: calculations')
]
CORRECTION_SOURCE = STATISTICS_SOURCE[STATISTICS_SOURCE.index('pairwise_statistics = pd.DataFrame(statistics_rows)'):]


class LinkedVolumeStatisticsTests(unittest.TestCase):
    def run_statistics(self, errors, selected=None, patient_ids=None):
        selected = list(errors) if selected is None else selected
        n = len(next(iter(errors.values())))
        volumes = pd.DataFrame({
            'patient_id': np.arange(n) if patient_ids is None else patient_ids,
            'gt_volume_ml': np.full(n, 50.0),
        })
        for key, error in errors.items():
            volumes[f'{key}_volume_ml'] = 50.0 + np.asarray(error, dtype=float)
        tree = ast.parse(STATISTICS_SOURCE)
        # Exercise production code with a smaller, reproducible validation budget.
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == 'N_BOOTSTRAP':
                    node.value = ast.Constant(1999)
                elif node.targets[0].id == 'BOOTSTRAP_BATCH':
                    node.value = ast.Constant(113)
        ast.fix_missing_locations(tree)
        with tempfile.TemporaryDirectory() as directory:
            namespace = dict(
                np=np, pd=pd, combinations=combinations, volumes=volumes,
                SELECTED_COMPARATORS=selected, COMPARATORS={k: (k, None, [1]) for k in errors},
                OUTPUT_DIR=Path(directory), display=lambda value: None,
            )
            with redirect_stdout(StringIO()):
                exec(compile(tree, str(NOTEBOOK), 'exec'), namespace)
            result = namespace['pairwise_statistics']
            exported = pd.read_csv(Path(directory) / 'comparator_pairwise_statistics.csv')
            self.assertEqual(result.shape, exported.shape)
            return result

    def test_identical_methods_are_uninformative(self):
        errors = np.arange(1, 31, dtype=float)
        row = self.run_statistics({'a': errors, 'b': -errors}).iloc[0]
        self.assertEqual(row.rmse_improvement_ml, 0)
        self.assertEqual(row.rmse_improvement_ci_lower_ml, 0)
        self.assertEqual(row.rmse_improvement_ci_upper_ml, 0)
        self.assertEqual(row.rmse_bootstrap_p, 1)
        self.assertEqual(row.ties, 30)
        self.assertTrue(np.isnan(row.probability_a_closer_given_no_tie))
        self.assertTrue(np.isnan(row.probability_ci_lower))
        self.assertEqual(row.closer_exact_p, 1)
        self.assertEqual(row.closer_conclusion, 'uninformative_all_ties')
        self.assertEqual(row.rmse_t_sensitivity_conclusion, 'undefined')

    def test_clear_improvement_and_reproducibility(self):
        errors = {'a': np.linspace(0.2, 2, 40), 'b': np.linspace(3, 15, 40)}
        result = self.run_statistics(errors)
        pd.testing.assert_frame_equal(result, self.run_statistics(errors))
        row = result.iloc[0]
        self.assertGreater(row.rmse_improvement_ci_lower_ml, 0)
        self.assertGreaterEqual(row.rmse_bootstrap_p, row.monte_carlo_p_resolution)
        self.assertLess(row.rmse_bootstrap_p_holm, 0.05)
        self.assertEqual(row.rmse_conclusion, 'A better')
        self.assertEqual(row.closer_conclusion, 'A better')
        self.assertEqual(row.wins_a, 40)
        self.assertEqual(row.probability_a_closer_given_no_tie, 1)
        self.assertLessEqual(row.probability_ci_upper, 1)

    def test_reversal_mixed_outcomes_and_ties(self):
        errors = {
            'a': np.array([0.5, 3, 2, 5, 1, 6, 1, 4, 2, 5, 1, 3]),
            'b': np.array([2, 1, 2, 2, 4, 3, 3, 2, 4, 1, 3, 3]),
        }
        forward = self.run_statistics(errors).iloc[0]
        reverse = self.run_statistics(errors, ['b', 'a']).iloc[0]
        self.assertEqual(forward.ties, 2)
        self.assertEqual(forward.n_non_ties, 10)
        self.assertEqual(forward.wins_a + forward.losses_a + forward.ties, 12)
        self.assertAlmostEqual(forward.win_proportion_all + forward.loss_proportion_all + forward.tie_proportion_all, 1)
        self.assertAlmostEqual(forward.rmse_improvement_ml, -reverse.rmse_improvement_ml)
        self.assertAlmostEqual(forward.rmse_improvement_ci_lower_ml, -reverse.rmse_improvement_ci_upper_ml)
        self.assertAlmostEqual(forward.rmse_improvement_ci_upper_ml, -reverse.rmse_improvement_ci_lower_ml)
        self.assertEqual(forward.rmse_bootstrap_p, reverse.rmse_bootstrap_p)
        self.assertAlmostEqual(forward.probability_a_closer_given_no_tie, 1-reverse.probability_a_closer_given_no_tie)
        self.assertAlmostEqual(forward.probability_ci_lower, 1-reverse.probability_ci_upper)
        self.assertEqual(forward.closer_exact_p, reverse.closer_exact_p)

    def test_constant_nonzero_difference_uses_fallback(self):
        row = self.run_statistics({'a': np.ones(30), 'b': np.full(30, 2)}).iloc[0]
        self.assertEqual(row.rmse_improvement_ml, 1)
        self.assertEqual(row.rmse_improvement_ci_lower_ml, 1)
        self.assertEqual(row.rmse_improvement_ci_upper_ml, 1)
        self.assertEqual(row.rmse_bootstrap_p, 1/2000)
        self.assertEqual(row.rmse_test_method, 'null_centered_bootstrap_mean_zero_variance')
        self.assertEqual(row.rmse_t_sensitivity_status, 'undefined_zero_variance')
        self.assertTrue(np.isnan(row.rmse_t_sensitivity_p_holm))

    def test_confidence_interval_preserves_patient_pairing(self):
        from scipy import stats

        errors = np.linspace(0.5, 4.5, 30)
        row = self.run_statistics({'a': errors, 'b': 2*errors}).iloc[0]
        # For B=2A on each patient, RMSE(B)-RMSE(A) is exactly RMSE(A).
        expected = stats.bootstrap(
            (errors,), lambda x, axis: np.sqrt(np.mean(x**2, axis=axis)),
            vectorized=True, n_resamples=1999, batch=113,
            rng=np.random.default_rng(20261102), method='BCa',
        ).confidence_interval
        self.assertEqual(row.rmse_ci_method, 'paired_BCa')
        np.testing.assert_allclose(
            [row.rmse_improvement_ci_lower_ml, row.rmse_improvement_ci_upper_ml], expected,
        )

    def test_small_nonzero_difference_and_single_comparator(self):
        row = self.run_statistics({'a': np.ones(20), 'b': np.full(20, 1+0.5e-6)}).iloc[0]
        self.assertEqual(row.wins_a, 20)
        self.assertEqual(row.losses_a, 0)
        self.assertEqual(row.ties, 0)
        self.assertEqual(row.probability_a_closer_given_no_tie, 1)
        self.assertEqual(row.closer_conclusion, 'A better')
        self.assertNotIn('tie_tolerance_ml', row.index)
        self.assertTrue(self.run_statistics({'a': np.arange(20)}).empty)

    def test_holm_joint_family_against_known_values(self):
        from statsmodels.stats.multitest import multipletests

        rows = []
        for bootstrap_p, sign_p, t_p in [(0.01, 0.04, 0.01), (0.03, 0.001, np.nan), (0.2, 0.9, 0.2)]:
            rows.append(dict(method_a='a', method_b='b', n=20,
                             rmse_bootstrap_p=bootstrap_p, closer_exact_p=sign_p,
                             rmse_t_sensitivity_p=t_p, rmse_improvement_ml=1,
                             rmse_improvement_ci_lower_ml=0, rmse_improvement_ci_upper_ml=2,
                             probability_a_closer_given_no_tie=0.7, n_non_ties=20,
                             wins_a=14, losses_a=6, ties=0,
                             probability_ci_lower=0.4, probability_ci_upper=0.9))
        with tempfile.TemporaryDirectory() as directory:
            namespace = dict(pd=pd, np=np, multipletests=multipletests, statistics_rows=rows,
                             ALPHA=0.05, OUTPUT_DIR=Path(directory), display=lambda value: None)
            with redirect_stdout(StringIO()):
                exec(compile(CORRECTION_SOURCE, str(NOTEBOOK), 'exec'), namespace)
            result = namespace['pairwise_statistics']
        np.testing.assert_allclose(result.rmse_bootstrap_p_holm, [0.05, 0.12, 0.4])
        np.testing.assert_allclose(result.closer_exact_p_holm, [0.12, 0.006, 0.9])
        self.assertTrue(result.holm_family_size.eq(6).all())
        self.assertEqual(result.rmse_conclusion.iloc[0], 'A better')
        self.assertTrue(np.isnan(result.rmse_t_sensitivity_p_holm.iloc[1]))
        self.assertAlmostEqual(result.rmse_t_sensitivity_p_holm.iloc[2], 0.6)

    def test_reject_repeated_patients_and_nonfinite_errors(self):
        with self.assertRaises(AssertionError):
            self.run_statistics({'a': np.ones(10), 'b': np.full(10, 2)}, patient_ids=np.zeros(10))
        with self.assertRaises(AssertionError):
            self.run_statistics({'a': np.ones(10), 'b': np.full(10, np.nan)})


if __name__ == '__main__':
    unittest.main()
