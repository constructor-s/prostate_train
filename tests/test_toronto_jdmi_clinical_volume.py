"""Exercise Toronto notebook cells using synthetic private-data substitutes."""
import ast
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import itertools
from itertools import combinations
import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import pandas as pd


NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/toronto_jdmi_clinical_volume.py'
SOURCE = NOTEBOOK.read_text()
SELECTION = SOURCE[SOURCE.index('listing = pd.read_csv('):SOURCE.index('# %% [markdown]\n# ## Direct measurements')]
MEASUREMENT = SOURCE[SOURCE.index('input_hashes = {}'):SOURCE.index('# %% [markdown]\n# ## Volume accuracy')]
SUMMARY = SOURCE[SOURCE.index('gt = volumes['):SOURCE.index('# %% [markdown]\n# ## Agreement plots')]
STATISTICS = SOURCE[SOURCE.index('# %% [markdown]\n# ## Paired post hoc inference:'):SOURCE.index('# %% [markdown]\n# ## Local reproducibility')]


class TorontoVolumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.data = root / 'data_internal/hlab20260928'
        self.data.mkdir(parents=True)
        self.out = root / 'results'
        self.out.mkdir()
        self.csv = self.data / 'clinical.csv'
        self.exclude = self.data / 'exclude.json'
        self.exclude.write_text('[]')
        self.ns = dict(
            pd=pd, np=np, nib=nib, json=json, hashlib=hashlib, Path=Path,
            itertools=itertools, combinations=combinations, REPO_ROOT=root,
            DATA_ROOT=self.data, CLINICAL_CSV=self.csv, EXCLUDE_JSON=self.exclude,
            OUTPUT_DIR=self.out, EXPECTED_CASES=2, GEOMETRY_TOLERANCE_MM=.5,
            GEOMETRY_WARNING_THRESHOLD_MM=.001,
            SELECTED_COMPARATORS=['clinical', 'nnunet274', 'rumcnotprostatexnoclinical236'],
            COMPARATORS=dict(clinical=('Manual ellipsoid', None),
                             nnunet274=('Multicenter', 'Dataset274_raw_3d_fullres_5fold'),
                             rumcnotprostatexnoclinical236=('RUMC', 'Dataset236_raw_3d_fullres_5fold')),
            display=lambda *args: None, tqdm=lambda items, **kwargs: items,
        )
        self.rows = [dict(ID=f'AIPR_p{i}_20260101_s{i}', t2=f'image{i}.nii.gz',
                          ground_truth_volume_ml=.2, vol_report_ref=.25) for i in range(2)]
        self.save_listing()
        for i in range(2):
            self.image(self.data / f'image{i}.nii.gz')
        for model in ('Dataset274_raw_3d_fullres_5fold', 'Dataset236_raw_3d_fullres_5fold'):
            folder = self.data / f'seg_out_{model}'
            folder.mkdir()
            (folder / 'plans.json').write_text(json.dumps(dict(dataset_name=model.removesuffix('_3d_fullres_5fold'))))
            (folder / 'dataset.json').write_text(json.dumps(dict(labels={'background': 0, 'class1': 1})))
            for row in self.rows:
                self.image(folder / f"{row['ID']}.nii.gz")
        self.prediction = self.data / 'seg_out_Dataset274_raw_3d_fullres_5fold' / f"{self.rows[0]['ID']}.nii.gz"

    def save_listing(self):
        pd.DataFrame(self.rows).to_csv(self.csv, index=False)

    def image(self, path, data=None, affine=None, units='mm'):
        img = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8) if data is None else data,
                             np.diag([2., 3., 4., 1.]) if affine is None else affine)
        img.header.set_xyzt_units(units)
        nib.save(img, path)

    def execute(self, source):
        with redirect_stdout(StringIO()):
            exec(compile(source, str(NOTEBOOK), 'exec'), self.ns)

    def test_accession_exclusions_and_invalid_reference_are_audited(self):
        self.rows.extend([
            dict(ID='AIPR_other_20260101_exclude', t2='', ground_truth_volume_ml=.3, vol_report_ref=.4),
            dict(ID='AIPR_missing_20260101_missing', t2='', ground_truth_volume_ml=None, vol_report_ref=.4),
        ])
        self.exclude.write_text(json.dumps([dict(aano='exclude', reason='synthetic exclusion')]))
        self.save_listing()
        self.execute(SELECTION)
        self.assertEqual(len(self.ns['cases']), 2)
        audit = pd.read_csv(self.out / 'cohort_audit.csv')
        self.assertEqual(audit.accession_excluded.sum(), 1)
        self.assertEqual(audit.invalid_reference_volume.sum(), 1)

    def test_duplicate_patients_invalid_ids_and_count_changes_stop_with_audit(self):
        for bad_id in ('AIPR_p0_20260201_s1', '../invalid', 'AIPR_bad/path_20260101_s1'):
            with self.subTest(bad_id=bad_id):
                self.rows[1]['ID'] = bad_id
                self.save_listing()
                with self.assertRaises(ValueError):
                    self.execute(SELECTION)
                self.assertTrue((self.out / 'cohort_audit.csv').is_file())
        self.rows[1]['ID'] = 'AIPR_p1_20260101_s1'
        self.save_listing()
        self.ns['EXPECTED_CASES'] = 3
        with self.assertRaisesRegex(ValueError, 'count changed'):
            self.execute(SELECTION)

    def test_same_size_membership_change_stops(self):
        self.execute(SELECTION)
        (self.out / 'run_metadata.json').write_text(json.dumps(dict(cohort_sha256=self.ns['cohort_sha256'])))
        self.rows[1]['ID'] = 'AIPR_replacement_20260101_s1'
        self.save_listing()
        with self.assertRaisesRegex(ValueError, 'membership changed'):
            self.execute(SELECTION)

    def test_native_volume_empty_prediction_and_summary(self):
        self.image(self.prediction, data=np.zeros((2, 2, 2), dtype=np.uint8))
        self.execute(SELECTION)
        self.execute(MEASUREMENT)
        volumes = self.ns['volumes']
        np.testing.assert_allclose(volumes.nnunet274_volume_ml, [0., .192])
        np.testing.assert_allclose(volumes.rumcnotprostatexnoclinical236_volume_ml, [.192, .192])
        self.execute(SUMMARY)
        result = self.ns['summary'].set_index('method').loc['nnunet274']
        self.assertEqual(result.n, 2)
        self.assertAlmostEqual(result.mae_ml, .104)
        self.assertAlmostEqual(result.rmse_ml, np.sqrt((.2**2 + .008**2) / 2))
        self.assertAlmostEqual(result.rmspe_percent, np.sqrt((100**2 + 4**2) / 2))
        self.assertIn('data_internal/hlab20260928/clinical.csv', self.ns['input_hashes'])

    def test_missing_mask_labels_units_and_geometry_stop_with_measurement_audit(self):
        self.execute(SELECTION)
        for problem in ('missing', 'labels', 'units', 'geometry', 'shape'):
            with self.subTest(problem=problem):
                self.image(self.prediction)
                if problem == 'missing':
                    self.prediction.unlink()
                elif problem == 'labels':
                    self.image(self.prediction, data=np.full((2, 2, 2), 2, dtype=np.uint8))
                elif problem == 'units':
                    self.image(self.prediction, units='meter')
                elif problem == 'geometry':
                    affine = np.diag([2., 3., 4., 1.])
                    affine[0, 3] = .6
                    self.image(self.prediction, affine=affine)
                else:
                    self.image(self.prediction, data=np.ones((3, 2, 2), dtype=np.uint8))
                with self.assertRaisesRegex(ValueError, 'Incomplete or invalid measurements'):
                    self.execute(MEASUREMENT)
                audit = pd.read_csv(self.out / 'measurement_audit.csv')
                self.assertEqual(audit.status.eq('invalid').sum(), 1)
                self.assertFalse((self.out / 'comparator_summary.csv').exists())

    def test_tolerated_displacement_is_recorded_and_bad_identity_rejected(self):
        affine = np.diag([2., 3., 4., 1.])
        affine[0, 3] = .5
        self.image(self.prediction, affine=affine)
        self.execute(SELECTION)
        self.execute(MEASUREMENT)
        self.assertAlmostEqual(self.ns['measurement_audit'].max_corner_displacement_mm.max(), .5)
        (self.prediction.parent / 'plans.json').write_text(json.dumps(dict(dataset_name='wrong')))
        with self.assertRaisesRegex(ValueError, 'dataset identity'):
            self.execute(MEASUREMENT)


class TorontoStatisticsTests(unittest.TestCase):
    def run_statistics(self, errors, references=None):
        n = len(next(iter(errors.values())))
        references = np.full(n, 50.) if references is None else np.asarray(references)
        volumes = pd.DataFrame(dict(patient_id=np.arange(n), gt_volume_ml=references))
        for key, values in errors.items():
            volumes[f'{key}_volume_ml'] = references + np.asarray(values)
        tree = ast.parse(STATISTICS)
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == 'N_BOOTSTRAP':
                    node.value = ast.Constant(1999)
                elif node.targets[0].id == 'BOOTSTRAP_BATCH':
                    node.value = ast.Constant(113)
        ast.fix_missing_locations(tree)
        with tempfile.TemporaryDirectory() as directory:
            ns = dict(np=np, pd=pd, combinations=combinations, volumes=volumes,
                      SELECTED_COMPARATORS=list(errors), COMPARATORS={k: (k, None) for k in errors},
                      OUTPUT_DIR=Path(directory), display=lambda value: None)
            with redirect_stdout(StringIO()):
                exec(compile(tree, str(NOTEBOOK), 'exec'), ns)
            result = ns['pairwise_statistics']
            exported = pd.read_csv(Path(directory) / 'comparator_pairwise_statistics.csv')
            self.assertEqual(result.shape, exported.shape)
            if not result.empty:
                np.testing.assert_allclose(exported.rmspe_improvement_pp, result.rmspe_improvement_pp)
                np.testing.assert_allclose(exported.rmspe_bootstrap_p_holm, result.rmspe_bootstrap_p_holm)
            return result

    def test_reproducibility_pair_reversal_and_joint_holm_family(self):
        from statsmodels.stats.multitest import multipletests

        errors = dict(a=np.linspace(.2, 2, 40), b=np.linspace(3, 15, 40), c=np.linspace(5, 20, 40))
        result = self.run_statistics(errors)
        pd.testing.assert_frame_equal(result, self.run_statistics(errors))
        self.assertTrue(result.holm_family_size.eq(6).all())
        raw = result[['rmse_bootstrap_p', 'closer_exact_p']].to_numpy()
        expected = multipletests(raw.ravel(), method='holm')[1].reshape(-1, 2)
        np.testing.assert_allclose(result[['rmse_bootstrap_p_holm', 'closer_exact_p_holm']], expected)
        forward = self.run_statistics(dict(a=errors['a'], b=errors['b'])).iloc[0]
        reverse = self.run_statistics(dict(b=errors['b'], a=errors['a'])).iloc[0]
        self.assertAlmostEqual(forward.rmse_improvement_ml, -reverse.rmse_improvement_ml)
        self.assertAlmostEqual(forward.rmse_improvement_ci_lower_ml, -reverse.rmse_improvement_ci_upper_ml)
        self.assertEqual(forward.rmse_bootstrap_p, reverse.rmse_bootstrap_p)

    def test_ties_and_constant_squared_errors(self):
        tied = self.run_statistics(dict(a=np.arange(1, 21), b=-np.arange(1, 21))).iloc[0]
        self.assertEqual(tied.ties, 20)
        self.assertEqual(tied.rmse_bootstrap_p, 1)
        self.assertEqual(tied.closer_conclusion, 'uninformative_all_ties')
        constant = self.run_statistics(dict(a=np.ones(20), b=np.full(20, 2))).iloc[0]
        self.assertEqual(constant.rmse_ci_method, 'deterministic')
        self.assertEqual(constant.rmse_t_sensitivity_status, 'undefined_zero_variance')
        self.assertAlmostEqual(constant.rmse_bootstrap_p, 1 / 2000)



    def test_rmspe_variable_references_and_pairing(self):
        from scipy import stats

        references = np.linspace(20., 200., 30)
        errors = np.linspace(.5, 4.5, 30)
        row = self.run_statistics(dict(a=errors, b=2 * errors), references=references).iloc[0]
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
        reverse = self.run_statistics(dict(b=2 * errors, a=errors), references=references).iloc[0]
        self.assertAlmostEqual(row.rmspe_improvement_pp, -reverse.rmspe_improvement_pp)
        self.assertAlmostEqual(row.rmspe_improvement_ci_lower_pp, -reverse.rmspe_improvement_ci_upper_pp)
        self.assertEqual(row.rmspe_bootstrap_p, reverse.rmspe_bootstrap_p)

    def test_rmspe_and_rmse_can_favor_different_methods(self):
        references = np.tile([10., 100.], 20)
        row = self.run_statistics(
            dict(a=np.tile([5., 0.], 20), b=np.tile([0., 10.], 20)), references=references,
        ).iloc[0]
        self.assertGreater(row.rmse_improvement_ml, 0)
        self.assertLess(row.rmspe_improvement_pp, 0)

    def test_rmspe_separate_holm_families_and_reproducibility(self):
        from statsmodels.stats.multitest import multipletests

        errors = dict(a=np.linspace(.2, 2, 30), b=np.linspace(3, 15, 30), c=np.linspace(5, 20, 30))
        result = self.run_statistics(errors)
        pd.testing.assert_frame_equal(result, self.run_statistics(errors))
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
        row = self.run_statistics(dict(a=np.ones(20), b=np.full(20, 2)), references=np.full(20, 100.)).iloc[0]
        self.assertEqual(row.rmspe_improvement_pp, 1)
        self.assertEqual(row.rmspe_improvement_ci_lower_pp, 1)
        self.assertEqual(row.rmspe_improvement_ci_upper_pp, 1)
        self.assertEqual(row.rmspe_ci_method, 'deterministic')
        self.assertAlmostEqual(row.rmspe_bootstrap_p, 1 / 2000)
        self.assertTrue(np.isnan(row.rmspe_t_sensitivity_p_holm))
        self.assertEqual(row.rmspe_t_sensitivity_conclusion, 'undefined')
        tied = self.run_statistics(dict(a=np.ones(20), b=np.ones(20))).iloc[0]
        self.assertEqual(tied.rmspe_improvement_pp, 0)
        self.assertEqual(tied.rmspe_bootstrap_p, 1)
        self.assertTrue(self.run_statistics(dict(a=np.ones(20))).empty)
        for invalid in [0., -1., np.nan, np.inf]:
            with self.subTest(reference=invalid), self.assertRaises(AssertionError):
                self.run_statistics(dict(a=np.ones(20), b=np.full(20, 2)), references=np.full(20, invalid))


if __name__ == '__main__':
    unittest.main()
