"""Validate actual PI-CAI notebook cells using small synthetic inputs."""
from __future__ import annotations

import ast
from contextlib import redirect_stdout
from io import StringIO
from itertools import combinations
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import nibabel as nib
import numpy as np
import pandas as pd


NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/picai_clinical_volume.py'
SOURCE = NOTEBOOK.read_text()
SELECTION_SOURCE = SOURCE[
    SOURCE.index('marksheet = pd.read_csv(MARKSHEET_PATH)'):
    SOURCE.index('# %% [markdown]\n# ## Cohort flow')
]
# Supply synthetic marksheets instead of reading the real clinical data.
SELECTION_TREE = ast.parse(SELECTION_SOURCE)
SELECTION_TREE.body = SELECTION_TREE.body[1:]
MEASUREMENT_SOURCE = SOURCE[
    SOURCE.index("assert json.loads((MODEL_ROOT / 'plans.json')"):
    SOURCE.index('# %% [markdown]\n# ## Suspicious clinical cases')
]
STATISTICS_SOURCE = SOURCE[
    SOURCE.index('# %% [markdown]\n# ## Paired post hoc inference:'):
    SOURCE.index('# %% [markdown]\n# ## PSA density: calculations')
]
BIOPSY_STATISTICS_SOURCE = SOURCE[
    SOURCE.index('# %% [markdown]\n# ### Biopsy decisions: paired agreement comparisons'):
]


class PicaiClinicalVolumeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def select(self, visits):
        rows = []
        for i, visit in enumerate(visits):
            row = dict(
                patient_id=1, study_id=100 + i, mri_date='2020-01-01',
                prostate_volume=50., psa=5., psad=np.nan,
                lesion_PIRADS='3', histopath_type='MRBx',
                lesion_GS='0+0', lesion_ISUP='0', case_ISUP=0,
            )
            row.update(visit)
            rows.append(row)
        namespace = dict(
            marksheet=pd.DataFrame(rows), np=np, pd=pd, OUTPUT_DIR=self.root,
        )
        exec(compile(SELECTION_TREE, str(NOTEBOOK), 'exec'), namespace)
        return namespace

    def test_later_complete_visit_wins_and_missing_psad_is_optional(self):
        result = self.select([
            dict(mri_date='2019-01-01', histopath_type=np.nan, lesion_GS=np.nan, lesion_ISUP=np.nan),
            dict(mri_date='2021-01-01', psad=np.nan),
        ])
        self.assertEqual(result['cohort'].study_id.tolist(), [101])
        audit = pd.read_csv(self.root / 'patient_selection_audit.csv')
        self.assertEqual(len(audit), 2)
        self.assertEqual(audit.complete_study_information.tolist(), [False, True])
        self.assertEqual(audit.selected_study_id.tolist(), [101, 101])
        self.assertEqual(audit.selection_reason.tolist(), [
            'not_selected_incomplete_visit', 'selected_earliest_complete_visit',
        ])

    def test_multiple_complete_visits_use_date_then_original_row(self):
        result = self.select([
            dict(study_id=500, mri_date='2021-01-01'),
            dict(study_id=900, mri_date='2019-01-01'),
            dict(study_id=100, mri_date='2019-01-01'),
        ])
        self.assertEqual(result['cohort'].study_id.tolist(), [900])
        self.assertEqual(result['selection'].selected.sum(), 1)

    def test_no_complete_visit_falls_back_without_measurement_substitution(self):
        result = self.select([
            dict(mri_date='2021-01-01', histopath_type=np.nan),
            dict(mri_date='2019-01-01', prostate_volume=np.nan),
        ])
        chosen = result['cohort'].iloc[0]
        self.assertEqual(chosen.study_id, 101)
        self.assertFalse(chosen.has_valid_clinical_volume)
        self.assertEqual(chosen.selection_status, 'invalid_clinical_volume')
        self.assertEqual(chosen.selection_reason, 'selected_earliest_available_visit')

    def test_negative_results_and_missing_lesion_tokens(self):
        result = self.select([
            dict(patient_id=1, lesion_PIRADS='N/A,3', lesion_ISUP='N/A,0', lesion_GS=np.nan),
            dict(patient_id=2, lesion_ISUP=np.nan, lesion_GS='0+0,N/A'),
            dict(patient_id=3, lesion_ISUP='N/A', lesion_GS=np.nan),
            dict(patient_id=4, lesion_PIRADS='N/A', lesion_ISUP='0'),
            dict(patient_id=5, histopath_type=np.nan, case_ISUP=0),
            dict(patient_id=6, case_ISUP=7),
        ])
        self.assertEqual(result['selection'].complete_study_information.tolist(), [
            True, True, False, False, False, False,
        ])
        self.assertEqual(len(result['cohort']), 6)
        self.assertTrue(result['cohort'].patient_id.is_unique)

    def test_invalid_dates_and_duplicate_examination_keys_stop_selection(self):
        with self.assertRaises((ValueError, AssertionError)):
            self.select([dict(mri_date='not-a-date')])
        with self.assertRaises(AssertionError):
            self.select([dict(mri_date=np.nan)])
        with self.assertRaisesRegex(AssertionError, 'Duplicate marksheet keys'):
            self.select([dict(study_id=1), dict(study_id=1)])

    def measurement_namespace(self):
        reference_root = self.root / 'reference'
        model_root = self.root / 'predictions'
        bundle_root = self.root / 'bundles/Dataset274_raw_3d_fullres_5fold/models'
        for folder in [reference_root, model_root, bundle_root]:
            folder.mkdir(parents=True, exist_ok=True)
        labels = {'labels': {'background': 0, 'class1': 1}}
        (model_root / 'plans.json').write_text(json.dumps({'dataset_name': 'Dataset274_raw'}))
        for folder in [model_root, bundle_root]:
            (folder / 'dataset.json').write_text(json.dumps(labels))
        for folder, shape, spacing, count in [
            (reference_root, (2, 2, 2), (2, 3, 4), 2),
            (model_root, (3, 2, 2), (1, 2, 5), 3),
        ]:
            mask = np.zeros(shape, dtype=np.uint8)
            mask.flat[:count] = 1
            image = nib.Nifti1Image(mask, np.diag([*spacing, 1]))
            image.header.set_xyzt_units('mm')
            nib.save(image, folder / '1_100.nii.gz')
        return dict(
            REPO_ROOT=self.root, REFERENCE_ROOT=reference_root, MODEL_ROOT=model_root,
            json=json, np=np, pd=pd, nib=nib,
            linked=pd.DataFrame({'ID': ['1_100'], 'patient_id': [1], 'study_id': [100]}),
        )

    def measure(self, namespace):
        with redirect_stdout(StringIO()):
            exec(compile(MEASUREMENT_SOURCE, str(NOTEBOOK), 'exec'), namespace)
        return namespace['volumes'].iloc[0]

    def test_native_grid_volume_units_and_geometry_audit(self):
        row = self.measure(self.measurement_namespace())
        self.assertAlmostEqual(row.reference_volume_ml, 2 * 24 / 1000)
        self.assertAlmostEqual(row.nnunet274_volume_ml, 3 * 10 / 1000)
        self.assertEqual(row.reference_voxel_count, 2)
        self.assertEqual(row.reference_method, 'bosma22b')
        self.assertEqual(row.reference_role, 'automated_surrogate')
        self.assertFalse(row.nnunet274_reference_shape_match)
        self.assertEqual(row.nnunet274_reference_affine_max_abs_difference, 1)

    def test_threaded_measurements_match_serial_results_and_patient_order(self):
        namespace = self.measurement_namespace()
        ids = ['3_300', '1_100', '2_200']
        for root in [namespace['REFERENCE_ROOT'], namespace['MODEL_ROOT']]:
            for case_id in ['3_300', '2_200']:
                shutil.copyfile(root / '1_100.nii.gz', root / f'{case_id}.nii.gz')
        namespace['linked'] = pd.DataFrame({
            'ID': ids, 'patient_id': [3, 1, 2], 'study_id': [300, 100, 200],
        })
        self.measure(namespace)
        serial = [namespace['measure_case_volumes'](case)
                  for case in namespace['linked'].itertuples(index=False)]
        pd.testing.assert_frame_equal(
            pd.DataFrame(namespace['measurement_rows']), pd.DataFrame(serial),
        )
        self.assertEqual(namespace['volumes'].ID.tolist(), ids)

    def test_missing_empty_unexpected_labels_and_units_stop_measurement(self):
        for failure in ['missing', 'empty', 'labels', 'units']:
            with self.subTest(failure=failure):
                namespace = self.measurement_namespace()
                path = namespace['MODEL_ROOT'] / '1_100.nii.gz'
                if failure == 'missing':
                    path.unlink()
                    expected = FileNotFoundError
                else:
                    image = nib.load(path)
                    mask = np.asanyarray(image.dataobj).copy()
                    header = image.header.copy()
                    if failure == 'empty':
                        mask[:] = 0
                    elif failure == 'labels':
                        mask.flat[0] = 2
                    else:
                        header.set_xyzt_units('meter')
                    nib.save(nib.Nifti1Image(mask, image.affine, header), path)
                    expected = AssertionError
                with self.assertRaises(expected):
                    self.measure(namespace)

    def test_implied_volume_review_uses_recorded_volume_and_strict_50_percent_boundary(self):
        cases = pd.DataFrame({
            'psa': [10.] * 7, 'clinical_volume_ml': [100.] * 7,
            'reference_volume_ml': [100.] * 7,
            'psad': [.2, 10 / 150, .21, .06, .1, 0., np.nan],
        })
        namespace = dict(
            np=np, pd=pd, linked=cases, VOLUME_REPORTING_TOLERANCE=.5,
            QC_NUMERICAL_TOLERANCE=1e-12, OUTPUT_DIR=self.root, display=lambda value: None,
        )
        qc_source = SOURCE[SOURCE.index('clinical_qc = linked.copy()'):SOURCE.index('# %%\ncohort = cohort.merge')]
        exec(compile(qc_source, str(NOTEBOOK), 'exec'), namespace)
        qc = namespace['clinical_qc']
        self.assertEqual(qc.beyond_reporting_tolerance.iloc[:5].tolist(), [False, False, True, True, False])
        self.assertTrue(qc.beyond_reporting_tolerance.iloc[5:].isna().all())
        self.assertEqual(qc.clinical_qc_status.iloc[5], 'unavailable_implied_volume')
        np.testing.assert_allclose(qc.implied_volume_relative_absolute_error.iloc[:2], [.5, .5])
        namespace['volumes'] = qc
        reported_source = SOURCE[SOURCE.index('reported_psad_audit = volumes.copy()'):
                                 SOURCE.index('reported_psad_audit.to_csv')]
        exec(compile(reported_source, str(NOTEBOOK), 'exec'), namespace)
        reported = namespace['reported_psad_audit']
        pd.testing.assert_series_equal(reported.beyond_reporting_tolerance, qc.beyond_reporting_tolerance)
        self.assertEqual(reported.reported_psad_status.iloc[5], 'included')
        self.assertEqual(len(qc), len(cases))

    def test_suspicious_reference_comparison_counts_both_directions_and_handles_empty(self):
        comparison_source = SOURCE[SOURCE.index('suspicious_volume_comparison = clinical_suspicious_cases['):
                                   SOURCE.index('# %%\nvolumes = volumes_all.loc')]
        cases = pd.DataFrame({
            'ID': ['low', 'lower_boundary', 'upper_boundary', 'high'],
            'reference_volume_ml': [100.] * 4, 'prostate_volume': [49., 50., 150., 151.],
        })
        for subset in [cases, cases.iloc[:0]]:
            namespace = dict(
                np=np, pd=pd, clinical_suspicious_cases=subset,
                VOLUME_REPORTING_TOLERANCE=.5, QC_NUMERICAL_TOLERANCE=1e-12,
                OUTPUT_DIR=self.root, display=lambda value: None,
            )
            with redirect_stdout(StringIO()):
                exec(compile(comparison_source, str(NOTEBOOK), 'exec'), namespace)
            result = namespace['suspicious_volume_comparison']
            if len(subset):
                self.assertEqual(result.severe_volume_disagreement.tolist(), [True, False, False, True])
                self.assertEqual(result.volume_error_percent.tolist(), [-51., -50., 50., 51.])
                self.assertEqual(len(namespace['severe_volume_cases']), 2)
            else:
                self.assertTrue(result.empty)
            saved = pd.read_csv(self.root / 'clinical_suspicious_reference_volume.csv')
            self.assertEqual(len(saved), len(subset))

    def statistics(self, patient_ids):
        n = len(patient_ids)
        volumes = pd.DataFrame({
            'patient_id': patient_ids, 'reference_volume_ml': np.full(n, 50.),
            'clinical_volume_ml': 50 + np.arange(1, n + 1),
            'nnunet274_volume_ml': 50 - np.arange(1, n + 1),
        })
        tree = ast.parse(STATISTICS_SOURCE)
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == 'N_BOOTSTRAP':
                    node.value = ast.Constant(199)
        ast.fix_missing_locations(tree)
        namespace = dict(
            np=np, pd=pd, combinations=combinations, volumes=volumes,
            SELECTED_COMPARATORS=['clinical', 'nnunet274'],
            COMPARATORS={'clinical': ('Clinical',), 'nnunet274': ('nnUNet274',)},
            OUTPUT_DIR=self.root, display=lambda value: None,
        )
        with redirect_stdout(StringIO()):
            exec(compile(tree, str(NOTEBOOK), 'exec'), namespace)
        return namespace['pairwise_statistics'].iloc[0]

    def test_paired_inference_uses_surrogate_reference_and_unique_patients(self):
        row = self.statistics(np.arange(12))
        self.assertEqual(row.n, 12)
        self.assertEqual(row.ties, 12)
        self.assertEqual(row.rmse_improvement_ml, 0)
        self.assertEqual(row.rmse_bootstrap_p, 1)
        self.assertEqual(row.holm_family_size, 2)
        self.assertEqual(row.closer_conclusion, 'uninformative_all_ties')
        with self.assertRaisesRegex(AssertionError, 'unique patients'):
            self.statistics(np.zeros(12))


class PicaiBiopsyStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def biopsy_namespace(self, truth, correctness, patient_ids=None):
        truth = np.asarray(truth, dtype=bool)
        cases = pd.DataFrame({
            'patient_id': np.arange(len(truth)) if patient_ids is None else patient_ids,
            'case_cspca': truth,
        }, index=np.arange(len(truth)) + 10)
        for method, correct in correctness.items():
            cases[f'{method}_biopsy_recommended'] = np.where(correct, truth, ~truth)
        return dict(
            np=np, pd=pd, combinations=combinations, biopsy_cases=cases,
            biopsy_truth=cases['case_cspca'].copy(), biopsy_methods=list(correctness),
            biopsy_names={method: method.upper() for method in correctness},
            biopsy_cohort='synthetic_PI-RADS_3', BIOPSY_PSAD_THRESHOLD=.15,
            PSAD_UNIT='ng/mL²', OUTPUT_DIR=self.root, display=lambda value: None,
        )

    def execute_statistics(self, namespace):
        tree = ast.parse(BIOPSY_STATISTICS_SOURCE)
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == 'BIOPSY_N_BOOTSTRAP':
                    node.value = ast.Constant(1999)
        ast.fix_missing_locations(tree)
        with redirect_stdout(StringIO()):
            exec(compile(tree, str(NOTEBOOK), 'exec'), namespace)
        return namespace['biopsy_pairwise_statistics']

    def test_discordant_counts_denominators_and_effect_direction(self):
        namespace = self.biopsy_namespace([True, True, False, False], {
            'a': [True, True, False, True],
            'b': [False, True, True, False],
            'c': [False, False, True, True],
        })
        result = self.execute_statistics(namespace)
        self.assertEqual(len(result), 9)
        ab = result.query('method_a == "a" and method_b == "b"').set_index('endpoint')
        self.assertEqual(ab.n.tolist(), [4, 2, 2])
        row = ab.loc['accuracy']
        self.assertEqual((row.value_a, row.value_b), (.75, .5))
        self.assertEqual((row.a_only_correct, row.b_only_correct), (2, 1))
        self.assertEqual((row.both_correct, row.both_incorrect), (1, 0))
        self.assertEqual(row.n_discordant, 3)
        self.assertEqual(row.difference_a_minus_b_pp, 25)
        self.assertEqual(ab.loc['sensitivity'].difference_a_minus_b_pp, 50)
        self.assertEqual(ab.loc['specificity'].difference_a_minus_b_pp, 0)
        saved = pd.read_csv(self.root / 'psa_density_biopsy_pairwise_statistics.csv')
        pd.testing.assert_frame_equal(saved, result, check_dtype=False)
        self.assertTrue(result.query('family == "primary"').holm_family_size.eq(3).all())
        self.assertTrue(result.query('family == "secondary"').holm_family_size.eq(6).all())

    def test_exact_tests_holm_and_overall_winner(self):
        namespace = self.biopsy_namespace([True] * 6 + [False] * 6, {
            'a': [True] * 12, 'b': [True] * 4 + [False] * 8, 'c': [False] * 12,
        })
        result = self.execute_statistics(namespace)
        primary = result.query('endpoint == "accuracy"').set_index(['method_a', 'method_b'])
        self.assertEqual(primary.loc[('a', 'b')].exact_p, .0078125)
        self.assertEqual(primary.loc[('a', 'c')].exact_p, .00048828125)
        self.assertEqual(primary.loc[('b', 'c')].exact_p, .125)
        self.assertEqual(primary.loc[('a', 'b')].exact_p_holm, .015625)
        self.assertEqual(primary.loc[('a', 'c')].exact_p_holm, .00146484375)
        self.assertEqual(primary.loc[('b', 'c')].exact_p_holm, .125)
        self.assertEqual(namespace['biopsy_accuracy_winners'], ['a'])

        namespace['biopsy_methods'] = ['c', 'b', 'a']
        result = self.execute_statistics(namespace)
        self.assertEqual(namespace['biopsy_accuracy_winners'], ['a'])
        self.assertEqual(result.query('method_a == "c" and method_b == "a" and endpoint == "accuracy"').iloc[0].conclusion,
                         'b_superior')

    def test_highest_observed_accuracy_requires_all_pairwise_wins(self):
        namespace = self.biopsy_namespace([True] * 6 + [False] * 6, {
            'a': [True] * 12, 'b': [True] * 11 + [False], 'c': [False] * 12,
        })
        result = self.execute_statistics(namespace)
        self.assertEqual(result.query('method_a == "a" and method_b == "c" and endpoint == "accuracy"').iloc[0].conclusion,
                         'a_superior')
        self.assertEqual(namespace['biopsy_accuracy_winners'], [])

    def test_bootstrap_intervals_are_reproducible_and_paired(self):
        namespace = self.biopsy_namespace([True] * 4 + [False] * 4, {
            'a': [True] * 8, 'b': [False, True] * 4, 'c': [True, False] * 4,
        })
        first = self.execute_statistics(namespace).copy()
        second = self.execute_statistics(namespace)
        pd.testing.assert_frame_equal(first, second)
        ab = first.query('method_a == "a" and method_b == "b"')
        self.assertTrue(ab.ci_lower_pp.ge(0).all())
        self.assertTrue(ab.ci_upper_pp.le(100).all())
        self.assertTrue(ab.ci_lower_pp.lt(ab.ci_upper_pp).all())

    def test_identical_correctness_and_single_patient_groups(self):
        namespace = self.biopsy_namespace([True, False], {
            'a': [True, False], 'b': [True, False], 'c': [True, False],
        })
        result = self.execute_statistics(namespace)
        self.assertTrue(result[['difference_a_minus_b_pp', 'ci_lower_pp', 'ci_upper_pp']].eq(0).all().all())
        self.assertTrue(result[['exact_p', 'exact_p_holm']].eq(1).all().all())
        self.assertTrue(result.conclusion.eq('no_significant_difference').all())
        self.assertEqual(namespace['biopsy_accuracy_winners'], [])

    def test_empty_endpoint_retains_planned_holm_slots(self):
        for truth_value, empty_endpoint in [(True, 'specificity'), (False, 'sensitivity')]:
            with self.subTest(empty_endpoint=empty_endpoint):
                namespace = self.biopsy_namespace([truth_value] * 8, {
                    'a': [True] * 8, 'b': [False] * 8, 'c': [False] * 8,
                })
                result = self.execute_statistics(namespace)
                empty = result.query('endpoint == @empty_endpoint')
                self.assertTrue(empty.n.eq(0).all())
                self.assertTrue(empty[['value_a', 'value_b', 'difference_a_minus_b_pp', 'ci_lower_pp',
                                       'ci_upper_pp', 'exact_p', 'exact_p_holm']].isna().all().all())
                self.assertTrue(empty.status.eq('no_eligible_patients').all())
                self.assertTrue(empty.conclusion.eq('unavailable').all())
                available = result.query('family == "secondary" and n > 0 and method_a == "a"')
                self.assertTrue(available.holm_family_size.eq(6).all())
                np.testing.assert_allclose(available.exact_p_holm, .046875)

    def test_empty_cohort_exports_unavailable_comparisons(self):
        namespace = self.biopsy_namespace([], {'a': [], 'b': [], 'c': []})
        result = self.execute_statistics(namespace)
        self.assertEqual(len(result), 9)
        self.assertTrue(result.status.eq('no_eligible_patients').all())
        self.assertTrue(result.exact_p_holm.isna().all())
        self.assertEqual(namespace['biopsy_accuracy_winners'], [])
        self.assertEqual(len(pd.read_csv(self.root / 'psa_density_biopsy_pairwise_statistics.csv')), 9)

    def test_rejects_duplicate_patients_misalignment_and_missing_decisions(self):
        namespace = self.biopsy_namespace([True, False], {'a': [True, True], 'b': [False, True]}, [1, 1])
        with self.assertRaisesRegex(AssertionError, 'unique patients'):
            self.execute_statistics(namespace)
        namespace['biopsy_cases']['patient_id'] = [1, 2]
        namespace['biopsy_truth'] = namespace['biopsy_truth'].iloc[::-1]
        with self.assertRaisesRegex(AssertionError, 'align with patients'):
            self.execute_statistics(namespace)
        namespace['biopsy_truth'] = namespace['biopsy_cases']['case_cspca'].copy()
        namespace['biopsy_cases']['a_biopsy_recommended'] = pd.Series([True, pd.NA], index=[10, 11], dtype='boolean')
        with self.assertRaisesRegex(AssertionError, 'Invalid a biopsy decisions'):
            self.execute_statistics(namespace)


if __name__ == '__main__':
    unittest.main()
