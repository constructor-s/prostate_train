"""Exercise the analysis cells on synthetic cohorts, without images or inference."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import hashlib
import json
import textwrap
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests


NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/volume_disagreement_quality_improvement.py'
SOURCE = NOTEBOOK.read_text()
FLAGS = SOURCE[SOURCE.index('threshold_rows = []'):SOURCE.index('plot_flag_path =')]
PSAD = SOURCE[SOURCE.index('psad_rows, transition_rows'):SOURCE.index('# %% [markdown]\n# ## PI-RADS 3:')]
BIOPSY = SOURCE[SOURCE.index('decision_rows, outcome_rows'):SOURCE.index('# %% [markdown]\n# ## Paired volume errors')]
INFERENCE = SOURCE[SOURCE.index('# SciPy requires'):SOURCE.index('# %% [markdown]\n# ## PSAD category changes')]


class DisagreementQualityImprovementTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.namespace = dict(
            np=np, pd=pd, stats=stats, mcnemar=mcnemar, multipletests=multipletests,
            OUTPUT_DIR=Path(self.directory.name), METHODS=['clinical', 'nnunet274'],
            SCREENS=['both', 'absolute', 'relative', 'either'],
            PSAD_CUTOFFS=np.array([.10, .15, .20]), display=lambda value: None,
        )

    def execute(self, source):
        with redirect_stdout(StringIO()):
            exec(compile(source, str(NOTEBOOK), 'exec'), self.namespace)

    def test_prostatex_manifest_rejects_stale_exports_and_retains_exclusions(self):
        root = Path(self.directory.name)
        (root / 'configs').mkdir()
        path = root / 'configs/rumcnotprostatexnoclinical_folds.json'
        manifest = dict(testing=[dict(id='retained')],
                        excluded_due_to_boundary_contact=[dict(id='ProstateX-0031')],
                        excluded_due_to_ineligibility=[dict(id='ProstateX-0042')])
        path.write_text(json.dumps(manifest))
        section = textwrap.dedent(SOURCE[SOURCE.index('        manifest_path ='):
                                         SOURCE.index('    else:\n        metadata_path =')])
        self.namespace.update(REPO_ROOT=root, json=json, hashlib=hashlib, input_hashes={},
                              volume=pd.DataFrame({'patient_id': [1]}, index=['retained']))
        self.execute(section)
        self.assertEqual(self.namespace['excluded_ids'], ['ProstateX-0031', 'ProstateX-0042'])
        self.namespace['volume'] = pd.DataFrame({'patient_id': [1, 2]},
                                               index=['retained', 'ProstateX-0042'])
        with self.assertRaisesRegex(AssertionError, 'rerun prostatex_linked_clinical_volume.py'):
            self.execute(section)
        self.namespace['volume'] = pd.DataFrame({'patient_id': [1]}, index=['retained'])
        manifest['testing'].append(dict(id='retained'))
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(AssertionError, 'duplicate testing IDs'):
            self.execute(section)
        manifest['testing'] = [dict(id='ProstateX-0042')]
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(AssertionError, 'Testing/excluded manifest overlap'):
            self.execute(section)

    def test_flags_do_not_depend_on_reference_and_include_both_tails(self):
        n = 40
        volume = pd.DataFrame(dict(
            ID=[f'case-{i}' for i in range(n)], gt_volume_ml=np.full(n, 50.),
            clinical_volume_ml=[200., 5.] + [50.] * (n - 2),
            nnunet274_volume_ml=[50., 200.] + [50.] * (n - 2),
            psad_evaluable=True, pirads_3=False, outcome_known=False,
        ))
        self.namespace['cohorts'] = {'synthetic': volume}
        self.execute(FLAGS)
        expected = volume.flag_both.copy()
        self.assertEqual(expected.sum(), 2)
        self.assertTrue(expected.iloc[:2].all())
        self.assertTrue((volume.flag_both <= volume.flag_either).all())
        volume.gt_volume_ml *= 100
        self.execute(FLAGS)
        pd.testing.assert_series_equal(expected, volume.flag_both)

    def test_zero_variance_equality_does_not_flag(self):
        volume = pd.DataFrame(dict(
            ID=['a', 'b', 'c'], gt_volume_ml=[50.] * 3,
            clinical_volume_ml=[60.] * 3, nnunet274_volume_ml=[50.] * 3,
            psad_evaluable=True, pirads_3=False, outcome_known=False,
        ))
        self.namespace['cohorts'] = {'synthetic': volume}
        self.execute(FLAGS)
        self.assertFalse(volume.flag_either.any())

    def downstream_cohort(self):
        volume = pd.DataFrame(dict(
            ID=['a', 'b', 'c', 'missing'],
            gt_psad=[.10, .15, .20, np.nan],
            clinical_psad=[.09, .15, .25, np.nan],
            nnunet274_psad=[.10, .14, .20, np.nan],
            psad_evaluable=[True, True, True, False],
            pirads_3=True, outcome_known=[True, False, True, False],
            cancer_positive=[False, False, True, False],
        ))
        for screen in self.namespace['SCREENS']:
            volume[f'flag_{screen}'] = True
        return volume

    def test_cutoff_categories_capture_benefit_and_harm(self):
        self.namespace['cohorts'] = {'synthetic': self.downstream_cohort()}
        self.execute(PSAD)
        row = self.namespace['psad_summary'].iloc[0]
        self.assertEqual((row.n, row.excluded_psad, row.category_changed), (3, 1, 2))
        self.assertEqual((row.corrected, row.newly_incorrect, row.both_correct), (1, 1, 1))
        self.assertEqual((row.ordinal_closer, row.ordinal_farther, row.ordinal_equal), (1, 1, 1))
        self.assertEqual(row.mcnemar_exact_p, 1)

    def test_unknown_outcome_retained_for_decisions_only(self):
        self.namespace['cohorts'] = {'synthetic': self.downstream_cohort()}
        self.execute(BIOPSY)
        row = self.namespace['decision_summary'].iloc[0]
        self.assertEqual((row.n, row.decisions_changed, row.newly_incorrect), (3, 1, 1))
        eligibility = self.namespace['eligibility_summary'].iloc[0]
        self.assertEqual((eligibility.excluded_psad, eligibility.excluded_unknown_outcome,
                          eligibility.n_known_outcomes), (1, 1, 2))
        outcome = self.namespace['biopsy_summary'].iloc[2]
        self.assertEqual((outcome.n, outcome.tp, outcome.tn, outcome.fn, outcome.fp), (2, 1, 1, 0, 0))
        exported = pd.read_csv(Path(self.directory.name) / 'biopsy_per_case.csv')
        self.assertTrue(exported.loc[~exported.outcome_known, 'cancer_positive'].isna().all())

    def test_empty_downstream_groups_have_zero_counts_and_undefined_rates(self):
        self.namespace['cohorts'] = {'synthetic': self.downstream_cohort().iloc[:0]}
        self.execute(PSAD)
        self.execute(BIOPSY)
        self.assertTrue(self.namespace['psad_summary'].n.eq(0).all())
        self.assertTrue(self.namespace['psad_summary'].mcnemar_exact_p.isna().all())
        self.assertTrue(self.namespace['biopsy_summary'].n.eq(0).all())
        self.assertTrue(self.namespace['biopsy_summary'].sensitivity.isna().all())
        self.assertTrue(self.namespace['biopsy_summary'].avoidance_fraction.isna().all())

    def test_primary_inference_handles_ties_and_insufficient_cases(self):
        import warnings

        volume = pd.DataFrame(dict(
            gt_volume_ml=[50.] * 4, clinical_volume_ml=[51., 52., 53., 54.],
            nnunet274_volume_ml=[49., 48., 47., 46.], flag_both=True,
        ))
        self.namespace.update(
            cohorts={'tied': volume, 'empty': volume.iloc[:0]}, warnings=warnings,
            paired_summary=pd.DataFrame(dict(cohort=['tied', 'empty'], screen=['both'] * 2,
                                              closer_exact_p=[1., np.nan])),
            N_BOOTSTRAP=1999, RANDOM_SEED=20261102, BOOTSTRAP_BATCH=113,
            CONFIDENCE_LEVEL=.95, ALPHA=.05,
        )
        self.execute(INFERENCE)
        result = self.namespace['volume_statistics'].set_index('cohort')
        self.assertEqual(result.loc['tied', 'ties'], 4)
        self.assertEqual(result.loc['tied', 'rmse_bootstrap_p_holm'], 1)
        self.assertEqual(result.loc['tied', 'rmse_improvement_ci_lower_ml'], 0)
        self.assertEqual(result.loc['tied', 'rmse_improvement_ci_upper_ml'], 0)
        self.assertTrue(np.isnan(result.loc['tied', 'probability_a_closer_given_no_tie']))
        self.assertEqual(result.loc['empty', 'inference_status'], 'unavailable_fewer_than_two')
        self.assertTrue(np.isnan(result.loc['empty', 'rmse_bootstrap_p_holm']))


if __name__ == '__main__':
    unittest.main()
