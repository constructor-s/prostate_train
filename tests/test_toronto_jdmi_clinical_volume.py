"""Exercise Toronto notebook cells using synthetic private-data substitutes."""
import ast
from contextlib import redirect_stdout
import hashlib
from io import BytesIO, StringIO
import itertools
from itertools import combinations
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import msoffcrypto
import nibabel as nib
import numpy as np
import pandas as pd


NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/toronto_jdmi_clinical_volume.py'
SOURCE = NOTEBOOK.read_text()
SELECTION = SOURCE[SOURCE.index('listing = pd.read_csv('):SOURCE.index('# %% [markdown]\n# ## Workbook loading')]
LOADING = SOURCE[SOURCE.index("with WORKBOOK.open('rb')"):SOURCE.index('# %% [markdown]\n# ## Link clinical fields')]
LINKAGE = SOURCE[SOURCE.index('READER_FIELDS = ['):SOURCE.index('# %% [markdown]\n# ## Direct measurements')]
MEASUREMENT = SOURCE[SOURCE.index('input_hashes = {}'):SOURCE.index('# %% [markdown]\n# ## Volume accuracy')]
SUMMARY = SOURCE[SOURCE.index('gt = volumes['):SOURCE.index('# %% [markdown]\n# ## Agreement plots')]
STATISTICS = SOURCE[SOURCE.index('# %% [markdown]\n# ## Paired post hoc inference:'):SOURCE.index('# %% [markdown]\n# ## PSA density inputs')]
PSAD = SOURCE[SOURCE.index("PSAD_UNIT = 'ng/mL²'"):SOURCE.index("psad_gt = psad_cases['gt_psad']")]


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
        workbook_path = self.data / 'workbook.xlsx'
        workbook_path.write_bytes(b'synthetic workbook checksum input')
        self.ns = dict(
            pd=pd, np=np, nib=nib, json=json, hashlib=hashlib, Path=Path,
            itertools=itertools, combinations=combinations, REPO_ROOT=root,
            DATA_ROOT=self.data, CLINICAL_CSV=self.csv, EXCLUDE_JSON=self.exclude,
            WORKBOOK=workbook_path,
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
        self.ns['workbook'] = pd.DataFrame(dict(
            aano=['s0', 's1'], amrn=['p0', 'p1'], psa_report_ref=[5., 12.],
            psa=['', ''], reportmaxpirads_ref=[3, 3], CSCANCER=['N', 'Y'],
            p_axt2_ch_vol=[.2, .2], p_axt2_ah_vol=['', ''], p_axt2_dd_vol=['', ''],
            p_axt2_ao_vol=['', ''], p_axt2_chepg_vol=['', ''],
            p_axt2_epgch_vol=['', ''], p_axt2_epg_vol=['', ''],
            dim1_report_ref=[3., 3.], dim2_report_ref=[4., 4.], dim3_report_ref=[5., 5.],
        ))
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
        self.assertEqual(audit.aano_excluded.sum(), 1)
        self.assertEqual(audit.invalid_reference_volume.sum(), 1)

    def test_selection_uses_reported_volume_and_preserves_it_after_linkage(self):
        self.execute(SELECTION)
        np.testing.assert_allclose(self.ns['cases'].clinical_volume_ml, .25)
        self.execute(LINKAGE)
        np.testing.assert_allclose(self.ns['cases'].clinical_volume_ml, .25)
        np.testing.assert_allclose(self.ns['cases'].dimension_ellipsoid_volume_ml, 10 * np.pi)

    def test_missing_or_invalid_reported_volume_is_rejected(self):
        for row in self.rows:
            del row['vol_report_ref']
        self.save_listing()
        with self.assertRaisesRegex(AssertionError, 'lacks required columns'):
            self.execute(SELECTION)
        self.rows[1]['vol_report_ref'] = .25
        for value in ['', None, 'invalid', 0, -1, np.inf, -np.inf]:
            with self.subTest(volume=value):
                self.rows[0]['vol_report_ref'] = value
                self.save_listing()
                with self.assertRaisesRegex(AssertionError, 'Invalid reported clinical volumes'):
                    self.execute(SELECTION)
                audit = pd.read_csv(self.out / 'cohort_audit.csv')
                self.assertTrue(audit.included.all())
                self.assertEqual(len(self.ns['cases']), 2)

    def test_duplicate_patients_invalid_ids_and_count_changes_stop_with_audit(self):
        for bad_id in ('AIPR_p0_20260201_s1', '../invalid', 'AIPR_bad/path_20260101_s1'):
            with self.subTest(bad_id=bad_id):
                self.rows[1]['ID'] = bad_id
                self.save_listing()
                with self.assertRaises(AssertionError):
                    self.execute(SELECTION)
                self.assertTrue((self.out / 'cohort_audit.csv').is_file())
        self.rows[1]['ID'] = 'AIPR_p1_20260101_s1'
        self.save_listing()
        self.ns['EXPECTED_CASES'] = 3
        with self.assertRaisesRegex(AssertionError, 'count changed'):
            self.execute(SELECTION)

    def test_same_size_membership_change_stops(self):
        self.execute(SELECTION)
        (self.out / 'run_metadata.json').write_text(json.dumps(dict(cohort_sha256=self.ns['cohort_sha256'])))
        self.rows[1]['ID'] = 'AIPR_replacement_20260101_s1'
        self.save_listing()
        with self.assertRaisesRegex(AssertionError, 'membership changed'):
            self.execute(SELECTION)

    def test_native_volume_empty_prediction_and_summary(self):
        self.image(self.prediction, data=np.zeros((2, 2, 2), dtype=np.uint8))
        self.execute(SELECTION)
        self.execute(LINKAGE)
        self.execute(MEASUREMENT)
        volumes = self.ns['volumes']
        np.testing.assert_allclose(volumes.clinical_volume_ml, .25)
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
        self.execute(LINKAGE)
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
        self.execute(LINKAGE)
        self.execute(MEASUREMENT)
        self.assertAlmostEqual(self.ns['measurement_audit'].max_corner_displacement_mm.max(), .5)
        (self.prediction.parent / 'plans.json').write_text(json.dumps(dict(dataset_name='wrong')))
        with self.assertRaisesRegex(ValueError, 'dataset identity'):
            self.execute(MEASUREMENT)


class TorontoClinicalFieldsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.cases = pd.DataFrame(dict(
            ID=['AIPR_p0_20260101_s0', 'AIPR_p1_20260101_s1'],
            amrn=['p0', 'p1'], aano=['s0', 's1'], gt_volume_ml=[50., 60.],
            clinical_volume_ml=[40., 80.],
            vol_report_ref=[40., 80.],
        ))
        self.workbook = pd.DataFrame(dict(
            aano=[' s1 ', 's0'], amrn=[' p1 ', 'p0'], psa_report_ref=['', '>5,0'],
            psa=['12', '99'], reportmaxpirads_ref=['3', '6'], CSCANCER=['Yes', 'NO BIOPSY'],
            p_axt2_ch_vol=[0, 0], p_axt2_ah_vol=[50, 40], p_axt2_dd_vol=[70, 60],
            p_axt2_ao_vol=['', ''], p_axt2_chepg_vol=['', ''],
            p_axt2_epgch_vol=['', ''], p_axt2_epg_vol=['', ''],
            dim1_report_ref=[3., 3.], dim2_report_ref=[4., 4.], dim3_report_ref=[5., 5.],
        ))

    def namespace(self):
        return dict(pd=pd, np=np, cases=self.cases.copy(), workbook=self.workbook.copy(),
                    OUTPUT_DIR=self.out, display=lambda *args: None)

    def execute(self, source, namespace):
        exec(compile(source, str(NOTEBOOK), 'exec'), namespace)

    def test_linkage_order_raw_fields_reference_and_outcomes(self):
        ns = self.namespace()
        self.execute(LINKAGE, ns)
        cases = ns['cases']
        self.assertEqual(cases.ID.tolist(), self.cases.ID.tolist())
        self.assertEqual(cases.source_excel_row.tolist(), [3, 2])
        self.assertEqual(cases.psa_report_ref.tolist(), ['>5,0', ''])
        self.assertEqual(cases.psa_consolidated.tolist(), [5., 12.])
        self.assertEqual(cases.psa_source.tolist(), ['psa_report_ref', 'psa'])
        self.assertEqual(cases.psa_selected_censored.tolist(), [True, False])
        self.assertEqual(cases.cscancer.tolist(), ['unknown', 'Y'])
        self.assertTrue(pd.isna(cases.pirads.iloc[0]))
        self.assertEqual(cases.pirads.iloc[1], 3)
        np.testing.assert_allclose(cases.workbook_reference_volume_ml, [50, 60])
        np.testing.assert_allclose(cases.dimension_ellipsoid_volume_ml, np.pi / 6 * 60)
        np.testing.assert_allclose(cases.clinical_volume_ml, cases.vol_report_ref)
        np.testing.assert_allclose(cases.vol_report_ref, [40., 80.])

    def test_ellipsoid_parses_decimal_commas_and_preserves_raw_dimensions(self):
        for field, raw in [('dim1_report_ref', '3,5'), ('dim2_report_ref', '4,0'),
                           ('dim3_report_ref', '5,0')]:
            self.workbook[field] = self.workbook[field].astype(object)
            self.workbook.loc[1, field] = raw
        ns = self.namespace()
        self.execute(LINKAGE, ns)
        row = ns['cases'].iloc[0]
        self.assertEqual(row.dim1_report_ref, '3,5')
        self.assertAlmostEqual(row.dimension_ellipsoid_volume_ml, np.pi / 6 * 3.5 * 4 * 5)
        self.assertEqual(row.clinical_volume_ml, 40.)

    def test_invalid_dimensions_remain_diagnostic_without_excluding_cases(self):
        original = self.workbook.copy()
        fields = ['dim1_report_ref', 'dim2_report_ref', 'dim3_report_ref']
        invalid_dimensions = [
            tuple(value if index == axis else [3, 4, 5][index] for index in range(3))
            for axis in range(3) for value in ['', None, 'bad', 0, -1, 'inf', '-inf']
        ] + [(-3, -4, 5), (1e200, 1e200, 1e200), (1e-200, 1e-200, 1e-200)]
        for values in invalid_dimensions:
            with self.subTest(dimensions=values):
                self.workbook = original.copy()
                self.workbook[fields] = self.workbook[fields].astype(object)
                self.workbook.loc[1, fields] = list(values)
                ns = self.namespace()
                self.execute(LINKAGE, ns)
                audit = pd.read_csv(self.out / 'clinical_fields_audit.csv')
                self.assertEqual(audit.ID.tolist(), self.cases.ID.tolist())
                self.assertTrue(pd.isna(audit.dimension_ellipsoid_volume_ml.iloc[0]))
                np.testing.assert_allclose(audit.clinical_volume_ml, self.cases.vol_report_ref)
                np.testing.assert_allclose(ns['cases'].clinical_volume_ml, self.cases.vol_report_ref)

    def test_extreme_dimensions_do_not_override_reported_volume(self):
        self.workbook.loc[1, ['dim1_report_ref', 'dim2_report_ref', 'dim3_report_ref']] = [8., 96.5, 6.3]
        ns = self.namespace()
        self.execute(LINKAGE, ns)
        self.assertAlmostEqual(ns['cases'].dimension_ellipsoid_volume_ml.iloc[0], np.pi / 6 * 8 * 96.5 * 6.3)
        np.testing.assert_allclose(ns['cases'].clinical_volume_ml, self.cases.vol_report_ref)

    def test_workbook_amrn_prefix_whitespace_and_leading_zeros(self):
        self.cases.loc[0, 'amrn'] = '00123456'
        self.cases.loc[0, 'ID'] = 'AIPR_00123456_20260101_s0'
        for raw in ['AIPR_00123456', '  AIPR_00123456  ', '00123456', '  00123456  ']:
            with self.subTest(raw=raw):
                self.workbook.loc[1, 'amrn'] = raw
                ns = self.namespace()
                self.execute(LINKAGE, ns)
                row = ns['cases'].iloc[0]
                self.assertEqual(row.workbook_amrn_raw, raw)
                self.assertEqual(row.workbook_amrn, '00123456')
                self.assertTrue(row.amrn_matches)
        for raw in ['AIPR_123456', 'AIPR_AIPR_00123456', 'other_00123456', 'AIPR_00123457']:
            with self.subTest(raw=raw):
                self.workbook.loc[1, 'amrn'] = raw
                with self.assertRaisesRegex(AssertionError, 'amrn mismatch'):
                    self.execute(LINKAGE, self.namespace())

    def test_positive_psa_fallback_and_invalid_inputs_are_audited(self):
        for report in ['', 'bad', '0', '-1', 'inf', '-inf']:
            with self.subTest(report=report):
                self.workbook.loc[1, 'psa_report_ref'] = report
                ns = self.namespace()
                self.execute(LINKAGE, ns)
                self.assertEqual(ns['cases'].psa_consolidated.iloc[0], 99)
                self.assertEqual(ns['cases'].psa_source.iloc[0], 'psa')
        self.workbook.loc[1, 'psa'] = '-2'
        with self.assertRaisesRegex(AssertionError, 'positive finite PSA'):
            self.execute(LINKAGE, self.namespace())
        audit = pd.read_csv(self.out / 'clinical_fields_audit.csv')
        self.assertEqual(audit.psa_source.iloc[0], 'unavailable')

    def test_haider_correction_keeps_raw_value_and_report_precedence(self):
        self.cases.loc[0, 'aano'] = '333abfe652a8889'
        self.workbook.loc[1, 'aano'] = '333abfe652a8889'
        self.workbook.loc[1, 'psa'] = 'erroneous raw value'
        for report, expected, source in [('', 28.9, 'psa'), ('7', 7., 'psa_report_ref')]:
            self.workbook.loc[1, 'psa_report_ref'] = report
            ns = self.namespace()
            self.execute(LINKAGE, ns)
            row = ns['cases'].iloc[0]
            self.assertEqual(row.psa, 'erroneous raw value')
            self.assertTrue(row.psa_correction_applied)
            self.assertEqual(row.psa_consolidated, expected)
            self.assertEqual(row.psa_source, source)

    def test_bad_keys_missing_links_patient_mismatch_and_reference_stop(self):
        original = self.workbook.copy()
        for field, value, message, audit_file in [
            ('aano', 's0', 'duplicate workbook aano', 'workbook_key_audit.csv'),
            ('aano', '', 'Invalid or duplicate', 'workbook_key_audit.csv'),
            ('aano', 'unmatched', 'linkage or amrn mismatch', 'clinical_fields_audit.csv'),
            ('amrn', 'wrong', 'linkage or amrn mismatch', 'clinical_fields_audit.csv'),
            ('p_axt2_ah_vol', 999, 'reference mismatch', 'clinical_fields_audit.csv'),
            ('p_axt2_ah_vol', -1, 'invalid reader volumes', 'clinical_fields_audit.csv'),
        ]:
            with self.subTest(field=field, value=value):
                self.workbook = original.copy()
                self.workbook.loc[0, field] = value
                with self.assertRaisesRegex(AssertionError, message):
                    self.execute(LINKAGE, self.namespace())
                self.assertTrue((self.out / audit_file).is_file())

    def test_outcomes_remain_optional_and_unknown_is_not_negative(self):
        for raw, expected in [('No', 'N'), ('Y confirmed', 'Y'), ('NOT KNOWN', 'unknown'),
                              ('NOT AVAILABLE', 'unknown'), ('N/A', 'unknown'), ('?', 'unknown')]:
            self.workbook.loc[0, 'CSCANCER'] = raw
            self.workbook.loc[0, 'reportmaxpirads_ref'] = '2.5'
            ns = self.namespace()
            self.execute(LINKAGE, ns)
            self.assertEqual(ns['cases'].cscancer.iloc[1], expected)
            self.assertTrue(pd.isna(ns['cases'].pirads.iloc[1]))
            self.assertEqual(len(ns['cases']), 2)

    def test_density_units_same_numerator_and_zero_volume_audit(self):
        ns = self.namespace()
        self.execute(LINKAGE, ns)
        ns.update(volumes=ns['cases'].assign(model_volume_ml=[25., 120.]),
                  SELECTED_COMPARATORS=['clinical', 'model'])
        self.execute(PSAD, ns)
        density = ns['psad_cases']
        np.testing.assert_allclose(density.gt_psad, [.1, .2])
        np.testing.assert_allclose(density.model_psad, [.2, .1])
        np.testing.assert_allclose(density.clinical_psad, [.125, .15])
        ns['volumes'].loc[0, 'model_volume_ml'] = 0
        with self.assertRaisesRegex(AssertionError, 'Invalid PSA-density inputs'):
            self.execute(PSAD, ns)
        audit = pd.read_csv(self.out / 'psa_density_cohort_audit.csv')
        self.assertFalse(audit.psad_evaluable.iloc[0])
        self.assertTrue(pd.isna(audit.model_psad.iloc[0]))
        self.assertEqual(len(ns['volumes']), 2)

    def test_runtime_loading_unencrypted_and_mock_encrypted_workbook(self):
        path = self.out / 'source.xlsx'
        self.workbook.to_excel(path, index=False)
        ns = dict(WORKBOOK=path, pd=pd, msoffcrypto=msoffcrypto, BytesIO=BytesIO,
                  getpass=Mock(return_value='synthetic-password'))
        self.execute(LOADING, ns)
        ns['getpass'].assert_not_called()
        self.assertEqual(ns['workbook'].psa_report_ref.iloc[0], '')
        payload = path.read_bytes()
        office = Mock()
        office.is_encrypted.return_value = True
        buffers = []
        def decrypt(buffer):
            buffers.append(buffer)
            buffer.write(payload)
        office.decrypt.side_effect = decrypt
        with patch.object(msoffcrypto, 'OfficeFile', return_value=office):
            self.execute(LOADING, ns)
        ns['getpass'].assert_called_once_with('Excel password: ')
        office.load_key.assert_called_once_with(password='synthetic-password', verify_password=True)
        self.assertNotIn('password', ns)
        self.assertNotIn('office', ns)
        self.assertTrue(buffers[0].closed)
        office.load_key.side_effect = ValueError('Invalid synthetic password')
        with patch.object(msoffcrypto, 'OfficeFile', return_value=office):
            with self.assertRaisesRegex(ValueError, 'Invalid synthetic password'):
                self.execute(LOADING, ns)
        self.assertNotIn('password', ns)
        self.assertNotIn('office', ns)


class TorontoStatisticsTests(unittest.TestCase):
    def run_statistics(self, errors, references=None):
        n = len(next(iter(errors.values())))
        references = np.full(n, 50.) if references is None else np.asarray(references)
        volumes = pd.DataFrame(dict(amrn=np.arange(n), gt_volume_ml=references))
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
