"""Exercise the RUMC generator's actual selection cells with synthetic visits."""
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/_picai_rumc_incomplete_not_prostatex.py'
SOURCE = NOTEBOOK.read_text()
SELECTION_SOURCE = SOURCE[
    SOURCE.index('audit = rumc.copy()'):
    SOURCE.index('# %% [markdown]\n# ## Add manual PROSTATEx testing')
]
TESTING_SOURCE = SOURCE[
    SOURCE.index("testing = prostatex.sort_values('ID').copy()"):
    SOURCE.index('# %% [markdown]\n# ## Review cohort flow')
]


class RumcIncompleteSelectionTests(unittest.TestCase):
    def select(self, visits, mapping=None):
        rows = []
        for i, visit in enumerate(visits):
            row = dict(
                patient_id=i + 1, study_id=100 + i, center='RUMC',
                mri_date='2020-01-01', prostate_volume=50., psa=5., psad=0.1,
                lesion_PIRADS='3', histopath_type='MRBx',
                lesion_GS='0+0', lesion_ISUP='0', case_ISUP=0,
            )
            row.update(visit)
            rows.append(row)
        namespace = dict(
            rumc=pd.DataFrame(rows), mapping=mapping or {}, np=np, pd=pd,
            display=lambda *args: None,
        )
        exec(compile(SELECTION_SOURCE, str(NOTEBOOK), 'exec'), namespace)
        return namespace

    def test_recorded_psad_is_required_and_must_be_finite_nonnegative(self):
        result = self.select([
            dict(psad=value)
            for value in [None, np.nan, np.inf, -np.inf, -0.1, 'unavailable', 0, 0.2]
        ])
        self.assertEqual(result['audit'].has_valid_psad.tolist(), [False] * 6 + [True, True])
        self.assertEqual(result['audit'].complete_study_information.tolist(), [False] * 6 + [True, True])
        self.assertEqual(result['training'].patient_id.tolist(), list(range(1, 7)))

    def test_all_fields_must_be_on_one_row_and_any_complete_visit_excludes_patient(self):
        result = self.select([
            dict(patient_id=1, psad=np.nan, mri_date='2019-01-01'),
            dict(patient_id=1, psa=np.nan),
            dict(patient_id=2, psad=np.nan, mri_date='2019-01-01'),
            dict(patient_id=2, mri_date='2021-01-01'),
        ])
        self.assertEqual(result['audit'].complete_study_information.tolist(), [False, False, False, True])
        self.assertEqual(result['audit'].patient_has_complete_exam.tolist(), [False, False, True, True])
        self.assertEqual(result['training'].study_id.tolist(), [100])

    def test_negative_biopsy_and_gleason_fallback_qualify(self):
        result = self.select([
            dict(lesion_PIRADS='N/A,3', lesion_ISUP='N/A,0', lesion_GS=np.nan),
            dict(lesion_ISUP='N/A', lesion_GS='0+0,N/A'),
            dict(lesion_ISUP=np.nan, lesion_GS='3+4,N/A', case_ISUP=2),
            dict(histopath_type='RP'),
            dict(lesion_ISUP='N/A', lesion_GS='invalid'),
            dict(case_ISUP=6),
        ])
        self.assertEqual(result['audit'].complete_study_information.tolist(), [True, True, True, False, False, False])
        self.assertEqual(result['training'].patient_id.tolist(), [4, 5, 6])

    def test_volume_psa_and_pirads_are_still_required(self):
        result = self.select([
            dict(prostate_volume=0), dict(prostate_volume=np.inf),
            dict(psa=0), dict(psa=np.nan),
            dict(lesion_PIRADS='N/A'), dict(lesion_PIRADS='3,6'),
        ])
        self.assertFalse(result['audit'].complete_study_information.any())
        self.assertEqual(len(result['training']), 6)

    def test_all_mapped_patients_are_excluded_including_ambiguous_links(self):
        result = self.select(
            [dict(patient_id=p, psad=np.nan) for p in [1, 2, 3, 4]],
            mapping={
                'ProstateX-0001_a': '1_100',
                'ProstateX-0001_b': '2_101',
                'ProstateX-0002_a': '3_999',
            },
        )
        self.assertEqual(result['training'].patient_id.tolist(), [4])

    def test_earliest_date_then_original_row_selects_one_exam(self):
        result = self.select([
            dict(patient_id=1, study_id=100, psad=np.nan, mri_date='2021-01-01'),
            dict(patient_id=1, study_id=900, psad=np.nan, mri_date='2019-01-01'),
            dict(patient_id=1, study_id=200, psad=np.nan, mri_date='2019-01-01'),
        ])
        self.assertEqual(result['training'].study_id.tolist(), [900])
        self.assertEqual(result['audit'].assignment.tolist(), ['excluded', 'training', 'excluded'])

    def test_testing_excludes_unmapped_cases_and_checks_all_ambiguous_links(self):
        namespace = dict(
            prostatex=pd.DataFrame({'ID': ['ProstateX-0002', 'ProstateX-0001']}),
            mapping={'ProstateX-0001_a': '1_100', 'ProstateX-0001_b': '2_200'},
            training=pd.DataFrame({'patient_id': [3], 'ID': ['3_300']}),
            np=np, pd=pd, display=lambda *args: None,
        )
        exec(compile(TESTING_SOURCE, str(NOTEBOOK), 'exec'), namespace)
        testing = namespace['testing']
        self.assertEqual(testing.ID.tolist(), ['ProstateX-0001'])
        self.assertEqual(testing.mapping_count.tolist(), [2])
        self.assertEqual(testing.picai_keys.tolist(), ['1_100,2_200'])
        audit = namespace['testing_audit']
        self.assertEqual(audit.assignment.tolist(), ['testing', 'excluded'])
        self.assertEqual(audit.linkage_status.iloc[1], 'excluded_no_mapping')
        # A different examination of either ambiguous mapped patient must fail.
        namespace['training'] = pd.DataFrame({'patient_id': [2], 'ID': ['2_999']})
        with self.assertRaisesRegex(AssertionError, 'Mapped patient overlap'):
            exec(compile(TESTING_SOURCE, str(NOTEBOOK), 'exec'), namespace)


if __name__ == '__main__':
    unittest.main()
