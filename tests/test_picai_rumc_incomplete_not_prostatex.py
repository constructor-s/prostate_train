"""Exercise the RUMC generator's actual selection cells with synthetic visits."""
from pathlib import Path
import unittest
import tempfile

import nibabel as nib

import numpy as np
import pandas as pd


NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/_picai_train_rumc_incomplete_not_prostatex_test_prostatex.py'
SOURCE = NOTEBOOK.read_text()
SELECTION_SOURCE = SOURCE[
    SOURCE.index('audit = rumc.copy()'):
    SOURCE.index('# %% [markdown]\n# ## Add manual PROSTATEx testing cases and check mapped identities')
]
TESTING_SOURCE = SOURCE[
    SOURCE.index("testing = prostatex.sort_values('ID').copy()"):
    SOURCE.index('# %% [markdown]\n# ## Review cohort flow')
]


class RumcIncompleteSelectionTests(unittest.TestCase):
    def prepare_masks(self, namespace, boundary_faces=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        cases = namespace['prostatex'].copy()
        cases['t2'] = cases.ID + '_image.nii.gz'
        cases['label_wg'] = cases.ID + '_mask.nii.gz'
        boundary_faces = boundary_faces or {}
        for row in cases.itertuples(index=False):
            data = np.zeros((5, 5, 5), dtype=np.uint8)
            data[2, 2, 2] = 1
            for axis, index in boundary_faces.get(row.ID, []):
                coord = [2, 2, 2]
                coord[axis] = index
                data[tuple(coord)] = 1
            image = nib.Nifti1Image(data, np.eye(4))
            image.header.set_xyzt_units('mm')
            nib.save(image, root / row.t2)
            nib.save(image, root / row.label_wg)
        namespace.update(prostatex=cases, PROSTATEX_ROOT=root, nib=nib)

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

    def test_source_quality_exclusion_retains_mapping_and_training_selection(self):
        namespace = self.select(
            [dict(patient_id=10804, study_id=1000820, psad=np.nan),
             dict(patient_id=10180, study_id=1000183, psad=np.nan),
             dict(patient_id=2, study_id=200, psad=np.nan)],
            mapping={'ProstateX-0031_date': '10804_1000820',
                     'ProstateX-0042_date': '10180_1000183',
                     'ProstateX-0001_date': '3_300'})
        training_before = namespace['training'].copy(deep=True)
        namespace.update(
            prostatex=pd.DataFrame({'ID': ['ProstateX-0031', 'ProstateX-0042', 'ProstateX-0001']}),
            marksheet=pd.DataFrame({'patient_id': [10804, 10180, 3],
                                    'study_id': [1000820, 1000183, 300],
                                    'prostate_volume': [50., 196., 60.]}))
        self.prepare_masks(namespace, {'ProstateX-0031': [(2, -1)], 'ProstateX-0042': [(2, 0)]})
        exec(compile(TESTING_SOURCE, str(NOTEBOOK), 'exec'), namespace)
        self.assertEqual(namespace['testing'].ID.tolist(), ['ProstateX-0001'])
        row = namespace['testing_audit'].set_index('ID').loc['ProstateX-0031']
        self.assertEqual(row.assignment, 'excluded_due_to_boundary_contact')
        self.assertEqual(row.exclusion_reason, 'manual_mask_boundary_contact')
        self.assertEqual(row.picai_key, '10804_1000820')
        self.assertTrue(row.has_valid_clinical_volume)
        self.assertTrue(row.has_boundary_contact)
        second = namespace['testing_audit'].set_index('ID').loc['ProstateX-0042']
        self.assertEqual(second.assignment, 'excluded_due_to_boundary_contact')
        self.assertEqual(second.exclusion_reason, 'manual_mask_boundary_contact')
        self.assertEqual(second.picai_key, '10180_1000183')
        self.assertTrue(second.has_boundary_contact)
        self.assertTrue(second.has_valid_clinical_volume)
        pd.testing.assert_frame_equal(namespace['training'], training_before)
        self.assertEqual(namespace['training'].patient_id.tolist(), [2])

    def test_only_single_superior_or_inferior_voxel_excludes_case(self):
        ids = [f'case{i}' for i in range(7)]
        namespace = dict(prostatex=pd.DataFrame({'ID': ids}),
                         mapping={f'{case}_date': f'{i+1}_{i+100}' for i, case in enumerate(ids)},
                         marksheet=pd.DataFrame({'patient_id': range(1, 8),
                                                 'study_id': range(100, 107), 'prostate_volume': 50.}),
                         training=pd.DataFrame({'patient_id': [99], 'ID': ['99_999']}),
                         np=np, pd=pd, display=lambda *args: None)
        faces = [(axis, side) for axis in range(3) for side in (0, -1)]
        self.prepare_masks(namespace, {case: [face] for case, face in zip(ids, faces)})
        exec(compile(TESTING_SOURCE, str(NOTEBOOK), 'exec'), namespace)
        self.assertEqual(namespace['testing'].ID.tolist(), ids[:4] + ['case6'])
        self.assertEqual(namespace['boundary_excluded'].ID.tolist(), ids[4:6])
        self.assertTrue(namespace['boundary_excluded'].boundary_contact_faces.ne('').all())
        self.assertTrue(namespace['boundary_excluded'].exclusion_reason.eq('manual_mask_boundary_contact').all())
        audit = namespace['testing_audit'].set_index('ID')
        self.assertEqual(audit.loc['case4', 'boundary_contact_faces'], 'I')
        self.assertEqual(audit.loc['case5', 'boundary_contact_faces'], 'S')
        # The anatomical rule survives axis permutation and a reversed S/I axis.
        root = namespace['PROSTATEX_ROOT']
        for row in namespace['prostatex'].itertuples(index=False):
            for name in (row.t2, row.label_wg):
                image = nib.load(root / name)
                image = image.as_reoriented(np.array([[2, -1], [1, 1], [0, 1]]))
                nib.save(image, root / name)
        exec(compile(TESTING_SOURCE, str(NOTEBOOK), 'exec'), namespace)
        self.assertEqual(namespace['boundary_excluded'].ID.tolist(), ids[4:6])
        audit = namespace['testing_audit'].set_index('ID')
        self.assertEqual(audit.loc['case4', 'boundary_contact_faces'], 'I')
        self.assertEqual(audit.loc['case5', 'boundary_contact_faces'], 'S')
        # Ineligibility takes precedence; retain the boundary-contact audit flag.
        namespace['marksheet'].loc[4, 'prostate_volume'] = np.nan
        exec(compile(TESTING_SOURCE, str(NOTEBOOK), 'exec'), namespace)
        self.assertEqual(len(namespace['boundary_excluded']), 1)
        self.assertEqual(namespace['ineligible'].ID.tolist(), ['case4'])
        self.assertTrue(namespace['ineligible'].has_boundary_contact.iloc[0])
        self.assertEqual(namespace['ineligible'].exclusion_reason.iloc[0],
                         'missing_nonnumeric_nonfinite_or_nonpositive_clinical_volume')
        self.assertEqual(namespace['testing'].ID.tolist(), ids[:4] + ['case6'])

    def test_testing_excludes_unmapped_cases_and_checks_all_ambiguous_links(self):
        namespace = dict(
            prostatex=pd.DataFrame({'ID': ['ProstateX-0002', 'ProstateX-0001', 'ProstateX-0003']}),
            marksheet=pd.DataFrame({'patient_id': [1, 2, 4], 'study_id': [100, 200, 400],
                                    'prostate_volume': [50., 60., 70.]}),
            mapping={'ProstateX-0001_a': '1_100', 'ProstateX-0001_b': '2_200',
                     'ProstateX-0003_a': '4_400'},
            training=pd.DataFrame({'patient_id': [3], 'ID': ['3_300']}),
            np=np, pd=pd, display=lambda *args: None,
        )
        self.prepare_masks(namespace)
        exec(compile(TESTING_SOURCE, str(NOTEBOOK), 'exec'), namespace)
        testing = namespace['testing']
        self.assertEqual(testing.ID.tolist(), ['ProstateX-0003'])
        self.assertEqual(testing.mapping_count.tolist(), [1])
        self.assertEqual(testing.picai_keys.tolist(), ['4_400'])
        audit = namespace['testing_audit']
        self.assertEqual(audit.assignment.tolist(), ['excluded_due_to_ineligibility',
                                                          'excluded_due_to_ineligibility', 'testing'])
        self.assertEqual(audit.picai_keys.iloc[0], '1_100,2_200')
        self.assertEqual(audit.exclusion_reason.iloc[0], 'ambiguous_mapping')
        self.assertEqual(audit.linkage_status.iloc[1], 'excluded_no_mapping')
        # A different examination of either ambiguous mapped patient must fail.
        namespace['training'] = pd.DataFrame({'patient_id': [2], 'ID': ['2_999']})
        with self.assertRaisesRegex(AssertionError, 'Mapped patient overlap'):
            exec(compile(TESTING_SOURCE, str(NOTEBOOK), 'exec'), namespace)


if __name__ == '__main__':
    unittest.main()
