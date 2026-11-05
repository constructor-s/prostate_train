"""Exercise clinical review flags, linkage flow, and retention in the notebook."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/prostatex_linked_clinical_volume.py'
SOURCE = NOTEBOOK.read_text()
QC_SOURCE = SOURCE[
    SOURCE.index('# %%\nclinical_qc = linked.copy()'):
    SOURCE.index('# %%\ncohort = cohort.merge(')
]
LINKAGE_SOURCE = SOURCE[
    SOURCE.index('# %%\ncases = pd.read_csv(INDEX_PATH)'):
    SOURCE.index('# %% [markdown]\n# ## Clinical consistency flags')
]
SPLIT_SOURCE = SOURCE[
    SOURCE.index('# %%\nvolumes_all = volumes.copy()'):
    SOURCE.index('# %% [markdown]\n# ## Volume accuracy')
]


class LinkedClinicalConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)

    def assess(self, psa, volume, psad, ids=None):
        n = len(psa)
        linked = pd.DataFrame(dict(
            ID=ids if ids is not None else [f'case-{i}' for i in range(n)],
            patient_id=np.arange(n), study_id=np.arange(n), psa=psa, psad=psad,
            prostate_volume=volume, clinical_volume_ml=volume,
            patient_age=np.full(n, 65), label_wg=['mask.nii.gz'] * n,
        ))
        namespace = dict(
            np=np, pd=pd, linked=linked, OUTPUT_DIR=self.output,
            PSAD_REPORTING_TOLERANCE=.01,
            QC_NUMERICAL_TOLERANCE=1e-12, display=lambda value: None,
        )
        with redirect_stdout(StringIO()):
            exec(compile(QC_SOURCE, str(NOTEBOOK), 'exec'), namespace)
        return namespace

    def test_signed_residuals_threshold_boundaries_and_roundoff(self):
        ns = self.assess(
            [10.] * 9, [100.] * 9,
            [.1, .09, .11, .09 - 1e-7, .11 + 1e-7,
             .09 - 1e-13, .11 + 1e-13, .095, .105],
        )
        qc = ns['clinical_qc']
        self.assertEqual(qc.beyond_reporting_tolerance.tolist(),
                         [False, False, False, True, True, False, False, False, False])
        np.testing.assert_allclose(qc.recalculated_clinical_psad, .1)
        np.testing.assert_allclose(qc.reported_minus_recalculated_psad.iloc[:3], [0, -.01, .01], atol=1e-15)
        self.assertTrue(qc.loc[[3, 4], 'clinical_qc_status'].eq('suspicious_reporting_residual').all())
        self.assertEqual(qc.clinical_qc_status.iloc[0], 'within_reporting_threshold')
        self.assertNotIn('primary_included', qc)

    def test_missing_invalid_and_zero_reported_density(self):
        ns = self.assess(
            [np.nan, np.inf, 'bad', -1, 0, 10, 10, 10, 10, .1, 10, 10],
            [100, 100, 100, 100, 100, 0, -1, 100, np.inf, 1, 5, 100],
            [.1, .1, .1, .1, .1, .1, .1, -.1, .1, 0, 100, np.nan],
        )
        qc = ns['clinical_qc']
        self.assertEqual(qc.clinical_qc_status.iloc[:3].tolist(), ['missing_or_nonfinite_clinical_values'] * 3)
        self.assertTrue(qc.clinical_qc_status.iloc[3:8].eq('invalid_clinical_values').all())
        self.assertTrue(qc.beyond_reporting_tolerance.iloc[:9].isna().all())
        self.assertTrue(qc.beyond_reporting_tolerance.iloc[9:11].all())
        self.assertTrue(np.isnan(qc.volume_implied_by_reported_psad_ml.iloc[9]))
        self.assertTrue(pd.isna(qc.beyond_reporting_tolerance.iloc[11]))
        self.assertAlmostEqual(qc.recalculated_clinical_psad.iloc[11], .1)
        self.assertTrue(np.isfinite(qc.recalculated_clinical_psad.iloc[7]))
        self.assertFalse(np.isinf(qc.recalculated_clinical_psad).any())

    def test_flags_and_missing_values_do_not_remove_volume_cases(self):
        ns = self.assess(
            [5.1, 21, 12, 4.9, 10, np.nan], [97, 32, 201, 104, 100, 50],
            [.07, .78, .05, .21, .1, np.nan],
            ids=['ProstateX-0016', 'ProstateX-0037', 'ProstateX-0058',
                 'ProstateX-0142', 'consistent', 'missing'],
        )
        self.assertEqual(ns['clinical_qc'].beyond_reporting_tolerance.iloc[:5].tolist(),
                         [True, True, False, True, False])
        measured = ns['clinical_qc'].copy()
        measured['gt_volume_ml'] = [100, 37, 190, 88, 100, 50]
        ns.update(
            volumes=measured, MASK_ROOT=self.output, INDEX_PATH=self.output / 'index.csv',
            MARKSHEET_PATH=self.output / 'marksheet.csv', MAPPING_PATH=self.output / 'mapping.json',
            SELECTED_COMPARATORS=['clinical'],
        )
        with redirect_stdout(StringIO()):
            exec(compile(SPLIT_SOURCE, str(NOTEBOOK), 'exec'), ns)
        self.assertEqual(ns['volumes'].ID.tolist(), measured.ID.tolist())
        self.assertEqual(len(ns['volumes_all']), 6)
        self.assertEqual(len(ns['clinical_case_review']), 3)
        report = pd.read_csv(self.output / 'clinical_case_review.csv')
        for column in ['patient_age', 'psa', 'psad', 'prostate_volume', 'manual_mask_path',
                       'clinical_minus_gt_ml', 'review_reason', 'absolute_reporting_residual_psad']:
            self.assertIn(column, report.columns)
        self.assertEqual(report.ID.tolist(), ['ProstateX-0142', 'ProstateX-0037', 'ProstateX-0016'])
        compact = pd.read_csv(self.output / 'clinical_suspicious_cases.csv')
        self.assertEqual(compact.ID.tolist(), report.ID.tolist())
        self.assertTrue(compact.absolute_reporting_residual_psad.is_monotonic_decreasing)
        self.assertFalse((self.output / 'clinical_excluded_cases.csv').exists())
        self.assertFalse((self.output / 'clinical_consistency_failed_cases.csv').exists())

    def test_cohort_flow_starts_at_mapping_and_anticipates_psa_without_filtering(self):
        mapping = {
            'ambiguous_date1': '1_1', 'ambiguous_date2': '1_2',
            'valid_date': '2_1', 'missingpsa_date': '3_1',
            'missingvolume_date': '4_1', 'unmatched_date': '5_1', 'nomask_date': '6_1',
        }
        index_path = self.output / 'index.csv'
        mapping_path = self.output / 'mapping.json'
        marksheet_path = self.output / 'marksheet.csv'
        mapping_path.write_text(json.dumps(mapping))
        pd.DataFrame({'ID': ['ambiguous', 'valid', 'missingpsa', 'missingvolume', 'unmatched', 'unmapped']}).to_csv(
            index_path, index=False,
        )
        pd.DataFrame(dict(patient_id=[2, 3, 4], study_id=[1, 1, 1],
                          prostate_volume=[50, 60, np.nan], psa=[5, np.nan, 3],
                          psad=[.1, np.nan, .1])).to_csv(marksheet_path, index=False)
        displays = []
        ns = dict(np=np, pd=pd, json=json, INDEX_PATH=index_path, MAPPING_PATH=mapping_path,
                  MARKSHEET_PATH=marksheet_path, OUTPUT_DIR=self.output, EXPECTED_CASES=2,
                  display=lambda value: displays.append(value.copy()))
        with redirect_stdout(StringIO()):
            exec(compile(LINKAGE_SOURCE, str(NOTEBOOK), 'exec'), ns)
        flow = pd.read_csv(self.output / 'cohort_flow.csv')
        self.assertEqual(flow.n_examinations.tolist(), [7, 6, 4, 3, 2, 1])
        self.assertEqual(flow.n_patients.tolist(), [6, 5, 4, 3, 2, 1])
        self.assertEqual(flow.excluded_examinations.iloc[1:].tolist(), [1, 2, 1, 1, 1])
        self.assertEqual(flow.excluded_patients.iloc[1:].tolist(), [1, 1, 1, 1, 1])
        self.assertEqual(ns['linked'].ID.tolist(), ['valid', 'missingpsa'])
        pd.testing.assert_frame_equal(displays[0], ns['cohort_flow'])


if __name__ == '__main__':
    unittest.main()
