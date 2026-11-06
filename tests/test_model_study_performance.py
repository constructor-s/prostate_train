"""Exercise notebook calculations with synthetic masks; no inference is run."""
import ast
from functools import partial
import contextlib
import io
import itertools
import json
import os
import signal
import sys
import time
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import thread_map

NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/model_study_performance.py'
SOURCE = NOTEBOOK.read_text()
MEASUREMENTS = SOURCE[SOURCE.index('def measure_reported_volume('):SOURCE.index('# %% [markdown]\n# ## Evaluate saved predictions')]
COMMANDS = SOURCE[SOURCE.index('COHORTS = []'):SOURCE.index('# %% [markdown]\n# Clinical references')]
EVALUATION = SOURCE[SOURCE.index('def evaluate_cohorts('):SOURCE.index('# %% [markdown]\n# ## Approve and run pending GPU inference')]
RUNNER = SOURCE[SOURCE.index('def prediction_signatures('):SOURCE.index('# %%\nper_case, cohort_audit = evaluate_cohorts')]
WORKFLOW = '''
DATASETS = list({(c['dataset'], c['split']): c for c in COHORTS}.values())
MODELS = [dict(name=name) for name in dict.fromkeys(c['model'] for c in COHORTS)]
''' + SOURCE[SOURCE.index('per_case, cohort_audit = evaluate_cohorts(COHORTS)'):]


class QuietProgress:
    bars = []

    def __init__(self, total, **kwargs):
        self.total, self.n = total, 0
        self.desc, self.unit = kwargs.get('desc'), kwargs.get('unit')
        self.values, self.labels = [], []
        self.bars.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_postfix_str(self, label):
        self.labels.append(label)

    def update(self, amount):
        self.n += amount
        self.values.append(self.n)

    def refresh(self):
        self.values.append(self.n)


class ModelStudyPerformanceTests(unittest.TestCase):
    def setUp(self):
        QuietProgress.bars.clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bundle = self.root / 'bundles/synthetic'
        (self.bundle / 'models').mkdir(parents=True)
        (self.bundle / 'models/dataset.json').write_text(json.dumps(
            {'labels': {'background': 0, 'class1': 1}}))
        self.tables = self.root / 'tables'
        self.tables.mkdir()
        self.ns = dict(nib=nib, np=np, pd=pd, Path=Path, json=json, shlex=shlex,
                       itertools=itertools, yaml=yaml, REPO_ROOT=self.root,
                       GEOMETRY_TOLERANCE_MM=0.5, BUNDLE=self.bundle,
                       GEOMETRY_WARNING_THRESHOLD_MM=1e-3,
                       MODEL='synthetic', TABLE_DIR=self.tables, display=lambda *args: None,
                       os=os, signal=signal, subprocess=subprocess, time=time,
                       tqdm=QuietProgress, input=Mock(return_value='n'),
                       thread_map=partial(thread_map, disable=True))
        for source in (MEASUREMENTS, EVALUATION, RUNNER):
            exec(compile(source, str(NOTEBOOK), 'exec'), self.ns)

    def mask(self, name, data=None, affine=None):
        data = np.ones((2, 2, 2), dtype=np.uint8) if data is None else data
        image = nib.Nifti1Image(data, np.diag([2., 3., 4., 1.]) if affine is None else affine)
        image.header.set_xyzt_units('mm')
        path = self.root / name
        nib.save(image, path)
        return path

    def cohort(self, predictions=2):
        out = self.root / 'predictions'
        out.mkdir()
        rows = []
        for i in range(2):
            image, ref = self.mask(f'image{i}.nii.gz'), self.mask(f'ref{i}.nii.gz')
            rows.append(dict(id=f'case{i}', image=str(image), label=str(ref)))
            if i < predictions:
                self.mask(f'predictions/case{i}.nii.gz')
        return dict(key='test', title='Test', model='synthetic', dataset='test', split='testing',
                    run_id=('synthetic', 'test'), rows=rows, output=out, data_root=self.root,
                    command='.venv/bin/python bundles/synthetic/scripts/infer.py --input manifest.json '
                            '--split testing --output_folder predictions --tile_step_size 0.8 --no-use_mirroring --device cuda --overwrite')

    def evaluate(self, cohort):
        self.ns['COHORTS'] = [cohort]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(WORKFLOW, str(NOTEBOOK), 'exec'), self.ns)
        self.output = output.getvalue()
        return self.ns['cohort_audit'].iloc[0]

    def test_clinical_rmse_coverage_and_invalid_masks(self):
        clinical = self.root / 'data_internal/hlab20260928'
        clinical.mkdir(parents=True)
        (clinical / 'exclude.json').write_text('[]')
        predictions = clinical / 'seg_out_synthetic'
        predictions.mkdir()
        image = self.mask('image.nii.gz')
        pd.DataFrame([
            dict(ID='full', t2=str(image), ground_truth_volume_ml=.1),
            dict(ID='empty', t2=str(image), ground_truth_volume_ml=.2),
            dict(ID='excluded', t2=str(image), ground_truth_volume_ml=None),
        ]).to_csv(clinical / 'hlab20260928_t2_image_listing_full_clinical.csv', index=False)
        # Distinct paths are required for distinct scans.
        second = self.mask('image2.nii.gz')
        listing = clinical / 'hlab20260928_t2_image_listing_full_clinical.csv'
        frame = pd.read_csv(listing)
        frame.loc[1, 't2'] = str(second)
        frame.to_csv(listing, index=False)
        self.ns['MODELS'] = [dict(name='synthetic')]
        section = SOURCE[SOURCE.index('CLINICAL_CSV ='):SOURCE.index('# %% [markdown]\n# ## Clinical manual ellipsoid comparator')]

        def run():
            with contextlib.redirect_stdout(io.StringIO()):
                self.ns.update(COHORTS=[], DATASETS=[], MODELS=[dict(name='synthetic')])
                exec(compile(section, str(NOTEBOOK), 'exec'), self.ns)
                exec(compile(WORKFLOW, str(NOTEBOOK), 'exec'), self.ns)
            audit = self.ns['cohort_audit'].iloc[0].copy()
            audit['volume_rmse_ml'] = self.ns['rmse_table'].iloc[0, 0]
            self.assertTrue(self.ns['dsc_table'].isna().all().all())
            return audit

        self.assertFalse(run().complete)
        manifest = json.loads((self.tables / 'clinical_missing_synthetic.json').read_text())
        self.assertEqual([r['id'] for r in manifest['testing']], ['full', 'empty'])
        self.assertTrue(all(Path(r['image']).is_absolute() for r in manifest['testing']))
        self.mask('data_internal/hlab20260928/seg_out_synthetic/full.nii.gz')
        self.mask('data_internal/hlab20260928/seg_out_synthetic/empty.nii.gz',
                  np.zeros((2, 2, 2), dtype=np.uint8))
        audit = run()
        self.assertTrue(audit.complete)
        self.assertEqual(audit.expected, 2)
        self.assertAlmostEqual(audit.volume_rmse_ml, np.sqrt(((.192 - .1)**2 + .2**2) / 2))
        self.assertEqual(len(self.ns['clinical_exclusions']), 1)
        self.mask('data_internal/hlab20260928/seg_out_synthetic/full.nii.gz',
                  np.full((2, 2, 2), 2, dtype=np.uint8))
        audit = run()
        self.assertEqual(audit.invalid, 1)
        self.assertTrue(pd.isna(audit.volume_rmse_ml))

    def test_clinical_accession_exclusions_are_shared_and_audited(self):
        clinical = self.root / 'data_internal/hlab20260928'
        clinical.mkdir(parents=True)
        exclusion_path = clinical / 'exclude.json'
        exclusion_path.write_text(json.dumps([
            dict(aano='remove', reason='Manual review'),
            dict(aano='overlap', description='Reported PSA error'),
            dict(aano='unmatched', reason='Absent from listing'),
        ]))
        image = self.mask('retained.nii.gz')
        pd.DataFrame([
            dict(ID='scan_date_keep', t2=str(image), ground_truth_volume_ml=.1,
                 vol_report_ref=.12),
            dict(ID='scan_date_remove', t2='', ground_truth_volume_ml=.2),
            dict(ID='scan_date_overlap', t2='', ground_truth_volume_ml=None),
            # Substring matches must not exclude a different accession.
            dict(ID='remove_date_other', t2=str(self.mask('other.nii.gz')),
                 ground_truth_volume_ml=.3, vol_report_ref=.32),
        ]).to_csv(clinical / 'hlab20260928_t2_image_listing_full_clinical.csv', index=False)
        self.ns.update(COHORTS=[], DATASETS=[], MODELS=[
            dict(name='synthetic'), dict(name='second')])
        section = SOURCE[SOURCE.index('CLINICAL_CSV ='):SOURCE.index(
            '# %% [markdown]\n# ## Clinical manual ellipsoid comparator')]
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(section, str(NOTEBOOK), 'exec'), self.ns)
        retained = ['scan_date_keep', 'remove_date_other']
        self.assertEqual(self.ns['clinical_cases'].ID.tolist(), retained)
        for cohort in self.ns['COHORTS']:
            self.assertEqual([row['id'] for row in cohort['rows']], retained)
        audit = pd.read_csv(self.tables / 'clinical_exclusions.csv').set_index('ID')
        self.assertEqual(len(audit), 2)
        self.assertIn('Manual review', audit.loc['scan_date_remove', 'exclusion_reason'])
        overlap = audit.loc['scan_date_overlap', 'exclusion_reason']
        self.assertIn('Reported PSA error', overlap)
        self.assertIn('nonpositive clinical volume', overlap)

        cohorts = self.ns['COHORTS'].copy()
        self.ns['COHORTS'] = cohorts[:1]
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(WORKFLOW, str(NOTEBOOK), 'exec'), self.ns)
        manifest = json.loads((self.tables / 'clinical_missing_synthetic.json').read_text())
        self.assertEqual([row['id'] for row in manifest['testing']], retained)

        labels = self.root / 'data/raw/zenodo/picai/picai_labels'
        (labels / 'clinical_information').mkdir(parents=True)
        (labels / 'additional_resources').mkdir()
        pd.DataFrame([dict(patient_id=1, study_id=10, prostate_volume=20)]).to_csv(
            labels / 'clinical_information/marksheet.csv', index=False)
        (labels / 'additional_resources/ProstateX-mapping.json').write_text(
            json.dumps({'ProstateX-0000_date': '1_10'}))
        source = self.cohort()
        source.update(key='prostatex_original_train_mapped', rows=[
            dict(id='ProstateX-0000', image=str(image))])
        self.ns['COHORTS'] = [source, *cohorts]
        ellipsoid_section = SOURCE[SOURCE.index("ELLIPSOID_NAME ="):SOURCE.index(
            '# %% [markdown]\n# ## Whole-gland measurements')]
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(ellipsoid_section, str(NOTEBOOK), 'exec'), self.ns)
        self.assertEqual([row['id'] for row in self.ns['COHORTS'][-1]['rows']], retained)

        for invalid in ('{}', '[{}]', 'invalid JSON'):
            exclusion_path.write_text(invalid)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                exec(compile(section, str(NOTEBOOK), 'exec'), self.ns)
        exclusion_path.unlink()
        with self.assertRaises(FileNotFoundError):
            exec(compile(section, str(NOTEBOOK), 'exec'), self.ns)

    def test_saved_multilabel_clinical_comparator(self):
        clinical = self.root / 'data_internal/hlab20260928'
        mask_dir = clinical / 'PROSATEPSAD/scan'
        mask_dir.mkdir(parents=True)
        (clinical / 'exclude.json').write_text('[]')
        image = self.mask('clinical_image.nii.gz')
        pd.DataFrame([dict(ID='scan', t2=str(image), ground_truth_volume_ml=.1)]).to_csv(
            clinical / 'hlab20260928_t2_image_listing_full_clinical.csv', index=False)
        self.ns.update(COHORTS=[], DATASETS=[], MODELS=[])
        section = SOURCE[SOURCE.index('CLINICAL_CSV ='):SOURCE.index('# %% [markdown]\n# ## Clinical manual ellipsoid comparator')]
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(section, str(NOTEBOOK), 'exec'), self.ns)
        comparator = self.ns['COHORTS'][0]
        self.assertEqual(comparator['model'], 'nnunet-702')
        self.assertTrue(comparator['saved_mask_only'])
        audit = self.evaluate(comparator)
        self.assertEqual(audit.missing, 1)
        self.ns['input'].assert_not_called()
        self.assertTrue(np.isnan(self.ns['rmse_table'].iloc[0, 0]))
        mask_path = 'data_internal/hlab20260928/PROSATEPSAD/scan/wp-pz-tz_axt2_nnunet-702.nii.gz'
        data = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.uint8).reshape(2, 2, 2)
        self.mask(mask_path, data)
        audit = self.evaluate(comparator)
        self.assertTrue(audit.complete)
        case = self.ns['per_case'].iloc[0]
        self.assertEqual(case.predicted_voxels, 6)
        self.assertAlmostEqual(case.predicted_volume_ml, .144)
        self.assertTrue(np.isnan(case.dsc))
        self.assertEqual(Path(case.prediction_path), self.root / mask_path)
        self.assertAlmostEqual(self.ns['rmse_table'].loc[
            ('hlab20260928_manual_planimetry', 'clinical'), 'nnunet-702'], .044)

        other = self.cohort()
        self.ns['COHORTS'] = [other, comparator]
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(WORKFLOW, str(NOTEBOOK), 'exec'), self.ns)
        self.assertTrue(np.isnan(self.ns['rmse_table'].loc[('test', 'testing'), 'nnunet-702']))
        for invalid_data, affine in (
            (np.full((2, 2, 2), 4, dtype=np.uint8), None),
            (data, np.diag([4., 3., 4., 1.])),
        ):
            self.mask(mask_path, invalid_data, affine)
            audit = self.evaluate(comparator)
            self.assertEqual(audit.invalid, 1)
            self.assertFalse(audit.complete)
            self.assertTrue(np.isnan(self.ns['rmse_table'].iloc[0, 0]))
        self.ns['input'].assert_not_called()

    def test_volume_rmspe_and_exports(self):
        cohort = self.cohort(predictions=0)
        cohort.update(prediction_type='reported_volume', reference_type='clinical_volume')
        for row, reference in zip(cohort['rows'], (20., 100.)):
            row.update(reference_volume_ml=reference)
        for factor, expected in ((1.1, 10.), (1., 0.), (.9, 10.)):
            with self.subTest(factor=factor):
                for row in cohort['rows']:
                    row['predicted_volume_ml'] = factor * row['reference_volume_ml']
                self.assertTrue(self.evaluate(cohort).complete)
                self.assertAlmostEqual(self.ns['rmspe_table'].iloc[0, 0], expected)
                np.testing.assert_allclose(
                    self.ns['per_case'].relative_volume_error, factor - 1, atol=1e-15)
                self.assertAlmostEqual(self.ns['rmse_table'].iloc[0, 0],
                                       abs(factor - 1) * np.sqrt((20. ** 2 + 100. ** 2) / 2))
                exported = pd.read_csv(self.tables / 'volume_rmspe_pct.csv')
                self.assertAlmostEqual(exported['synthetic'].iloc[0], expected)
                self.assertIn('Whole-gland volume RMSPE (%)',
                              (self.tables / 'volume_rmspe_pct.md').read_text())
                self.assertIn('relative_volume_error',
                              pd.read_csv(self.tables / 'per_case_metrics.csv').columns)
        cohort['rows'][0]['predicted_volume_ml'] = np.nan
        self.assertFalse(self.evaluate(cohort).complete)
        self.assertTrue(np.isnan(self.ns['rmspe_table'].iloc[0, 0]))
        self.assertTrue(np.isnan(self.ns['per_case'].relative_volume_error.iloc[0]))

    def test_volume_mape_and_exports(self):
        cohort = self.cohort(predictions=0)
        cohort.update(prediction_type='reported_volume', reference_type='clinical_volume')
        for row, reference in zip(cohort['rows'], (20., 100.)):
            row['reference_volume_ml'] = reference
        for factors, expected in (((1.1, 1.1), 10.), ((.9, .9), 10.),
                                  ((1., 1.), 0.), ((.5, 1.1), 30.)):
            with self.subTest(factors=factors):
                for row, factor in zip(cohort['rows'], factors):
                    row['predicted_volume_ml'] = factor * row['reference_volume_ml']
                self.assertTrue(self.evaluate(cohort).complete)
                self.assertAlmostEqual(self.ns['mape_table'].iloc[0, 0], expected)
                exported = pd.read_csv(self.tables / 'volume_mape_pct.csv')
                self.assertAlmostEqual(exported['synthetic'].iloc[0], expected)
                self.assertIn('Whole-gland volume MAPE (%)',
                              (self.tables / 'volume_mape_pct.md').read_text())
        cohort['rows'][0]['predicted_volume_ml'] = np.nan
        self.assertFalse(self.evaluate(cohort).complete)
        self.assertTrue(np.isnan(self.ns['mape_table'].iloc[0, 0]))
        exported = pd.read_csv(self.tables / 'volume_mape_pct.csv')
        self.assertTrue(np.isnan(exported['synthetic'].iloc[0]))

    def test_reported_volume_metrics_and_coverage(self):
        cohort = self.cohort(predictions=0)
        cohort.update(prediction_type='reported_volume', model='Clinical manual ellipsoid',
                      run_id=('Clinical manual ellipsoid', 'test'))
        for row in cohort['rows']:
            row.update(predicted_volume_ml=.3, source_file='clinical.csv', source_column='volume')
        audit = self.evaluate(cohort)
        self.assertTrue(audit.complete)
        self.assertAlmostEqual(self.ns['rmse_table'].iloc[0, 0], .108)
        self.assertTrue(self.ns['per_case'].dsc.isna().all())
        self.assertTrue(self.ns['per_case'].predicted_voxels.isna().all())
        self.assertTrue(self.ns['per_case'].prediction_path.eq('').all())
        self.assertTrue(self.ns['per_case'].source_column.eq('volume').all())
        self.ns['input'].assert_not_called()
        for value in (0., -1., np.nan, np.inf):
            cohort['rows'][0]['predicted_volume_ml'] = value
            audit = self.evaluate(cohort)
            self.assertFalse(audit.complete)
            self.assertEqual(audit.invalid, 1)
            self.assertTrue(np.isnan(self.ns['rmse_table'].iloc[0, 0]))
        cohort['rows'][0]['predicted_volume_ml'] = .3
        cohort['rows'][0]['source_error'] = 'Missing or ambiguous PROSTATEx mapping'
        self.assertFalse(self.evaluate(cohort).complete)
        cohort['rows'][0].pop('source_error')
        cohort.update(reference_type='clinical_volume')
        for row in cohort['rows']:
            row.pop('label')
            row['reference_volume_ml'] = .2
        self.assertTrue(self.evaluate(cohort).complete)
        self.assertAlmostEqual(self.ns['rmse_table'].iloc[0, 0], .1)

    def test_extra_predictions_are_audited_without_blocking_evaluation(self):
        cohort = self.cohort()
        extra = self.mask('predictions/excluded_case.nii.gz')
        preserved = extra.read_bytes()
        audit = self.evaluate(cohort)
        self.assertTrue(audit.complete)
        self.assertEqual(audit.evaluated, 2)
        self.assertEqual(audit.unexpected, 1)
        self.assertEqual(audit.unexpected_filenames, 'excluded_case.nii.gz')
        self.assertEqual(audit.cohort_errors, '')
        self.assertEqual(len(self.ns['per_case']), 2)
        self.assertAlmostEqual(self.ns['dsc_table'].iloc[0, 0], 1.)
        self.assertIn('ignoring 1 prediction files outside the manifest', self.output)
        self.ns['input'].assert_not_called()
        self.assertEqual(extra.read_bytes(), preserved)
        (cohort['output'] / 'case0.nii.gz').unlink()
        audit = self.evaluate(cohort)
        self.assertFalse(audit.complete)
        self.assertEqual(audit.missing, 1)
        self.assertTrue(np.isnan(self.ns['rmse_table'].iloc[0, 0]))
        self.mask('predictions/case0.nii.gz', np.full((2, 2, 2), 2, dtype=np.uint8))
        audit = self.evaluate(cohort)
        self.assertFalse(audit.complete)
        self.assertEqual(audit.invalid, 1)
        self.assertTrue(np.isnan(self.ns['rmse_table'].iloc[0, 0]))

    def test_reported_volume_reference_validation(self):
        measure = self.ns['measure_reported_volume']
        path = self.mask('reported_ref.nii.gz')
        self.assertAlmostEqual(measure(.3, path)['reference_volume_ml'], .192)
        for data in (
            np.zeros((2, 2, 2), dtype=np.uint8),
            np.full((2, 2, 2), 2, dtype=np.uint8),
        ):
            self.mask('reported_ref.nii.gz', data)
            with self.assertRaises(ValueError):
                measure(.3, path)
        image = nib.load(path)
        image.header.set_xyzt_units('meter')
        nib.save(image, path)
        with self.assertRaises(ValueError):
            measure(.3, path)

    def test_ellipsoid_mapping_uses_patient_and_study(self):
        labels = self.root / 'data/raw/zenodo/picai/picai_labels'
        (labels / 'clinical_information').mkdir(parents=True)
        (labels / 'additional_resources').mkdir()
        pd.DataFrame([
            dict(patient_id=1, study_id=10, prostate_volume=20),
            dict(patient_id=2, study_id=10, prostate_volume=90),
        ]).to_csv(labels / 'clinical_information/marksheet.csv', index=False)
        (labels / 'additional_resources/ProstateX-mapping.json').write_text(json.dumps({
            'ProstateX-0000_date': '1_10', 'ProstateX-0001_a': '1_10',
            'ProstateX-0001_b': '2_10',
        }))
        cohort = self.cohort()
        cohort['key'] = 'prostatex_original_train_mapped'
        cohort['rows'][0]['id'] = 'ProstateX-0000'
        cohort['rows'][1]['id'] = 'ProstateX-0001'
        self.ns.update(COHORTS=[cohort], CLINICAL_CSV=self.root / 'clinical.csv',
                       clinical_cases=pd.DataFrame(columns=['ID', 'image_path', 'ground_truth_volume_ml', 'vol_report_ref']))
        section = SOURCE[SOURCE.index("ELLIPSOID_NAME ="):SOURCE.index('# %% [markdown]\n# ## Whole-gland measurements')]
        exec(compile(section, str(NOTEBOOK), 'exec'), self.ns)
        rows = self.ns['COHORTS'][1]['rows']
        self.assertEqual(rows[0]['predicted_volume_ml'], 20)
        self.assertTrue(np.isnan(rows[1]['predicted_volume_ml']))
        self.assertIn('ambiguous', rows[1]['source_error'])

    def test_mixed_queue_preserves_clinical_masks_and_refreshes(self):
        clinical = self.cohort(predictions=1)
        clinical.update(reference_type='clinical_volume', key='clinical',
                        run_id=('synthetic', 'clinical'), dataset='clinical')
        for row in clinical['rows']:
            row['reference_volume_ml'] = .2
            row.pop('label')
        preserved = (clinical['output'] / 'case0.nii.gz').read_bytes()
        segmentation = self.runner_cohort('segmentation')
        self.ns['COHORTS'] = [segmentation, clinical]
        self.ns['input'] = Mock(return_value='y')
        evaluator = Mock(wraps=self.ns['evaluate_cohorts'])
        self.ns['evaluate_cohorts'] = evaluator
        launched = []

        def launch(argv, **kwargs):
            launched.append(argv)
            if '--input' in argv:
                self.assertNotIn('--overwrite', argv)
                manifest = json.loads(Path(argv[argv.index('--input') + 1]).read_text())
                self.assertEqual([r['id'] for r in manifest['testing']], ['case1'])
                self.mask('predictions/case1.nii.gz')
            else:
                for row in segmentation['rows']:
                    shutil.copyfile(row['label'], segmentation['output'] / f"{row['id']}.nii.gz")
            return Mock(poll=Mock(return_value=0))

        with contextlib.redirect_stdout(io.StringIO()), patch.object(subprocess, 'Popen', side_effect=launch):
            exec(compile(WORKFLOW, str(NOTEBOOK), 'exec'), self.ns)
        self.ns['input'].assert_called_once()
        self.assertEqual(len(launched), 2)
        self.assertEqual(evaluator.call_count, 2)
        self.assertEqual([bar.total for bar in QuietProgress.bars], [2, 1])
        self.assertTrue(self.ns['cohort_audit'].complete.all())
        self.assertEqual((clinical['output'] / 'case0.nii.gz').read_bytes(), preserved)
        # Invalid existing clinical masks are never queued for replacement.
        self.mask('predictions/case0.nii.gz', np.full((2, 2, 2), 2, dtype=np.uint8))
        self.ns['input'].reset_mock()
        self.assertFalse(self.evaluate(clinical).complete)
        self.ns['input'].assert_not_called()
        shifted = np.diag([2., 3., 4., 1.])
        shifted[0, 3] = 1
        self.mask('predictions/case0.nii.gz', affine=shifted)
        self.assertFalse(self.evaluate(clinical).complete)
        self.assertIn('Geometry displacement', self.ns['per_case'].iloc[0].error)

    def test_threaded_evaluation_preserves_result_and_progress_order(self):
        cohort = self.cohort()
        second_completed = threading.Event()
        completed = []
        progress_cases = []
        progress_threads = []
        measure = self.ns['measure_masks']

        def measure_out_of_order(pred, ref):
            if pred.name == 'case0.nii.gz':
                if not second_completed.wait(timeout=5):
                    raise TimeoutError('Second case did not finish while the first case waited')
            result = measure(pred, ref)
            completed.append(pred.name)
            if pred.name == 'case1.nii.gz':
                second_completed.set()
            return result

        class OrderedProgress(tqdm):
            def __iter__(self):
                for record in super().__iter__():
                    progress_cases.append(record['case_id'])
                    progress_threads.append(threading.get_ident())
                    yield record

        with patch.dict(self.ns, measure_masks=measure_out_of_order,
                        thread_map=partial(thread_map, tqdm_class=OrderedProgress, disable=True)):
            per_case, audit = self.ns['evaluate_cohorts']([cohort])

        self.assertEqual(completed, ['case1.nii.gz', 'case0.nii.gz'])
        self.assertEqual(per_case.case_id.tolist(), ['case0', 'case1'])
        self.assertTrue(audit.iloc[0].complete)
        self.assertEqual(progress_cases, ['case0', 'case1'])
        self.assertEqual(progress_threads, [threading.get_ident()] * 2)

    def test_perfect_empty_and_anisotropic_volume(self):
        ref = self.mask('ref.nii.gz')
        result = self.ns['measure_masks'](ref, ref)
        self.assertEqual(result['dsc'], 1.)
        self.assertAlmostEqual(result['reference_volume_ml'], 8 * 24 / 1000)
        empty = self.mask('empty.nii.gz', np.zeros((2, 2, 2), dtype=np.uint8))
        result = self.ns['measure_masks'](empty, ref)
        self.assertEqual(result['dsc'], 0.)
        self.assertAlmostEqual(result['squared_volume_error_ml2'], .192 ** 2)
        with self.assertRaisesRegex(ValueError, 'Empty reference'):
            self.ns['measure_masks'](ref, empty)

    def test_geometry_and_binary_labels(self):
        ref = self.mask('ref.nii.gz')
        affine = np.diag([2., 3., 4., 1.])
        for offset in (.195, .25, .5):
            affine[0, 3] = offset
            near = self.mask('near.nii.gz', affine=affine)
            self.assertAlmostEqual(
                self.ns['measure_masks'](near, ref)['max_corner_displacement_mm'], offset)
        affine[0, 3] = .501
        far = self.mask('far.nii.gz', affine=affine)
        with self.assertRaisesRegex(ValueError, 'Geometry'):
            self.ns['measure_masks'](far, ref)
        wrong = self.mask('wrong.nii.gz', np.full((2, 2, 2), 2, dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, 'binary'):
            self.ns['measure_masks'](wrong, ref)

    def test_complete_cohort_without_run_records_and_rmse_aggregation(self):
        cohort = self.cohort()
        self.mask('predictions/case1.nii.gz', np.zeros((2, 2, 2), dtype=np.uint8))
        self.assertTrue(self.evaluate(cohort).complete)
        self.assertNotIn('please run', self.output)
        self.assertEqual(self.ns['dsc_table'].iloc[0, 0], .5)
        self.assertAlmostEqual(self.ns['rmse_table'].iloc[0, 0], .192 / np.sqrt(2))

    def test_exact_geometry_has_no_warning(self):
        self.assertTrue(self.evaluate(self.cohort()).complete)
        self.assertNotIn('WARNING:', self.output)

    def test_geometry_warning_filters_small_displacements(self):
        cohort = self.cohort()
        for i, offset in enumerate((1e-9, .5)):
            affine = np.diag([2., 3., 4., 1.])
            affine[0, 3] = offset
            self.mask(f'predictions/case{i}.nii.gz', affine=affine)
        self.assertTrue(self.evaluate(cohort).complete)
        marker = 'WARNING: Evaluated cases with prediction/reference geometry displacement'
        self.assertEqual(self.output.count(marker), 1)
        report = self.output.split(marker)[1]
        self.assertNotIn('case0', report)
        self.assertIn('case1', report)
        self.assertIn('synthetic', report)
        self.assertIn('test', report)
        self.assertNotIn('1e-09', report)
        self.assertIn('> 0.001 mm', report)
        self.assertIn('0.5', report)
        self.assertNotIn('Invalid cases excluded', report)

    def test_rejected_geometry_is_reported_and_withholds_metrics(self):
        cohort = self.cohort()
        affine = np.diag([2., 3., 4., 1.])
        affine[0, 3] = .501
        self.mask('predictions/case0.nii.gz', affine=affine)
        audit = self.evaluate(cohort)
        self.assertFalse(audit.complete)
        self.assertEqual(audit.invalid, 1)
        marker = 'WARNING: Invalid cases excluded from evaluation:'
        self.assertEqual(self.output.count(marker), 1)
        report = self.output.split(marker)[1]
        self.assertIn('case0', report)
        self.assertIn('synthetic', report)
        self.assertIn('Geometry displacement 0.501 mm exceeds tolerance', report)
        self.assertTrue(self.ns['dsc_table'].isna().all().all())
        self.assertTrue(self.ns['rmse_table'].isna().all().all())

    def test_partial_run_shows_direct_command_and_can_be_completed(self):
        cohort = self.cohort(predictions=1)
        audit = self.evaluate(cohort)
        self.assertFalse(audit.complete)
        self.assertEqual(audit.evaluated, 1)
        self.assertTrue(pd.isna(self.ns['dsc_table'].iloc[0, 0]))
        self.assertEqual(self.ns['counts_table'].iloc[0, 0], '1/2')
        self.assertIn('1/2 evaluated (1 missing, 0 invalid)', self.output)
        self.assertIn(cohort['command'], self.output)
        self.assertNotIn('STUDY_ID', self.output)
        self.mask('predictions/case1.nii.gz')
        self.assertTrue(self.evaluate(cohort).complete)

    def test_absent_run_keeps_na_tables_and_counts(self):
        cohort = self.cohort(predictions=0)
        shutil.rmtree(cohort['output'])
        self.assertEqual(self.evaluate(cohort).evaluated, 0)
        self.assertEqual(self.ns['counts_table'].iloc[0, 0], '0/2')
        self.assertIn('NA', (self.tables / 'dsc.csv').read_text())
        self.assertIn(cohort['command'], self.output)
        self.assertNotIn(str(self.root), self.output)

    def test_invalid_mask_and_duplicate_cases_withhold_summary(self):
        cohort = self.cohort()
        self.mask('predictions/case0.nii.gz', np.full((2, 2, 2), 2, dtype=np.uint8))
        audit = self.evaluate(cohort)
        self.assertFalse(audit.complete)
        self.assertEqual(audit.invalid, 1)
        self.assertIn(cohort['command'], self.output)
        self.assertIn('binary', self.ns['per_case'].iloc[0].error)
        self.mask('predictions/case0.nii.gz')
        cohort['rows'][1]['id'] = cohort['rows'][0]['id']
        self.assertFalse(self.evaluate(cohort).complete)
        self.assertTrue(self.ns['dsc_table'].isna().all().all())
        self.assertIn('duplicate case IDs', self.ns['cohort_audit'].iloc[0].cohort_errors)

    def runner_cohort(self, key):
        out = self.root / key
        out.mkdir()
        rows = []
        for i in range(2):
            image = self.mask(f'{key}_image{i}.nii.gz')
            ref = self.mask(f'{key}_ref{i}.nii.gz')
            rows.append(dict(id=f'{key}{i}', image=str(image), label=str(ref)))
        # Stand-in inference uses only the standard library and copies valid masks.
        code = (
            'import os,sys,shutil; from pathlib import Path; '
            'assert os.environ["PYTHONUNBUFFERED"] == "1"; '
            'print("lightweight subprocess completed", flush=True); '
            '[shutil.copyfile(ref, Path(sys.argv[1]) / (case_id + ".nii.gz")) '
            'for case_id, ref in zip(sys.argv[2::2], sys.argv[3::2])]'
        )
        argv = [sys.executable, '-c', code, str(out)]
        for row in rows:
            argv.extend([row['id'], row['label']])
        return dict(key=key, title=key, model='synthetic', dataset=key, split='testing',
                    run_id=('synthetic', key), rows=rows, output=out, data_root=self.root,
                    argv=argv, command=f'visible-command-{key}')

    def test_one_approval_prints_all_commands_and_runs_sequentially_then_refreshes(self):
        complete = self.cohort()
        first, second = self.runner_cohort('first'), self.runner_cohort('second')
        self.ns['COHORTS'] = [complete, first, second]
        output, launched = io.StringIO(), []
        real_popen, real_sleep = subprocess.Popen, time.sleep

        def approve(prompt):
            self.assertEqual(prompt, 'Run all commands above? [y/N] ')
            for cohort in (first, second):
                self.assertIn(cohort['command'], output.getvalue())
            self.assertNotIn(complete['command'], output.getvalue())
            self.assertEqual(launched, [])
            return ' YeS '

        def launch(argv, **kwargs):
            if launched:
                self.assertIsNotNone(launched[-1].poll())
            self.assertEqual(kwargs['cwd'], self.root)
            self.assertEqual(kwargs['env'], dict(os.environ, PYTHONUNBUFFERED='1'))
            self.assertTrue(kwargs['start_new_session'])
            process = real_popen(argv, **kwargs)
            launched.append(process)
            return process

        self.ns['input'] = Mock(side_effect=approve)
        evaluator = Mock(wraps=self.ns['evaluate_cohorts'])
        self.ns['evaluate_cohorts'] = evaluator
        with contextlib.redirect_stdout(output), patch.object(subprocess, 'Popen', side_effect=launch), \
                patch.object(time, 'sleep', side_effect=lambda _: real_sleep(.02)):
            exec(compile(WORKFLOW, str(NOTEBOOK), 'exec'), self.ns)
        self.ns['input'].assert_called_once()
        self.assertEqual(len(launched), 2)
        self.assertEqual(evaluator.call_count, 2)
        self.assertTrue(self.ns['cohort_audit'].complete.all())
        self.assertTrue(self.ns['dsc_table'].eq(1).all().all())
        for cohort in (first, second):
            self.assertIn('lightweight subprocess completed', (cohort['output'] / 'inference.log').read_text())
        file_bars = [bar for bar in QuietProgress.bars if bar.unit == 'files']
        self.assertEqual(len(file_bars), 2)
        for bar in file_bars:
            self.assertEqual(bar.values[-1], 2)
            self.assertIn('waiting for inference/postprocessing', bar.labels)

    def test_decline_eof_or_unavailable_input_never_launches(self):
        cohort = self.runner_cohort('pending')
        _, audit = self.ns['evaluate_cohorts']([cohort])
        for response in ('n', '', 'anything', EOFError(), OSError('no stdin'), NotImplementedError('notebook stdin unavailable'), KeyboardInterrupt()):
            with self.subTest(response=response):
                self.ns['input'] = Mock(side_effect=response) if isinstance(response, BaseException) else Mock(return_value=response)
                with patch.object(subprocess, 'Popen') as launch, contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(self.ns['run_pending']([cohort], audit), ({}, False))
                self.ns['input'].assert_called_once()
                launch.assert_not_called()
        self.assertFalse((cohort['output'] / 'inference.log').exists())

    def test_complete_cohort_never_prompts_or_launches(self):
        cohort = self.cohort()
        _, audit = self.ns['evaluate_cohorts']([cohort])
        with patch.object(subprocess, 'Popen') as launch, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.ns['run_pending']([cohort], audit), ({}, False))
        self.ns['input'].assert_not_called()
        launch.assert_not_called()

    def test_failed_exit_stops_queue_and_withholds_metrics_even_with_all_files(self):
        first, second = self.runner_cohort('failed'), self.runner_cohort('not_started')
        first['argv'][2] += '; sys.exit(7)'
        self.ns.update(COHORTS=[first, second], input=Mock(return_value='Y'))
        real_sleep = time.sleep
        with contextlib.redirect_stdout(io.StringIO()) as output, \
                patch.object(time, 'sleep', side_effect=lambda _: real_sleep(.02)):
            exec(compile(WORKFLOW, str(NOTEBOOK), 'exec'), self.ns)
        audit = self.ns['cohort_audit'].iloc[0]
        self.assertEqual(audit.present, 2)
        self.assertFalse(audit.complete)
        self.assertTrue(self.ns['dsc_table'].isna().all().all())
        self.assertIn('exit code 7', output.getvalue())
        self.assertFalse((second['output'] / 'inference.log').exists())

    def test_interruption_terminates_worker_group_and_stops_queue(self):
        first, second = self.runner_cohort('interrupted'), self.runner_cohort('later')
        _, audit = self.ns['evaluate_cohorts']([first, second])
        self.ns['input'] = Mock(return_value='y')
        process = Mock(pid=123456, poll=Mock(side_effect=KeyboardInterrupt), wait=Mock(return_value=-15))
        with patch.object(subprocess, 'Popen', return_value=process) as launch, \
                patch.object(os, 'killpg') as killpg, contextlib.redirect_stdout(io.StringIO()):
            failures, attempted = self.ns['run_pending']([first, second], audit)
        self.assertTrue(attempted)
        self.assertEqual(failures, {('synthetic', 'interrupted'): 'Inference interrupted'})
        launch.assert_called_once()
        self.assertEqual(killpg.call_args_list[0].args, (123456, signal.SIGTERM))
        self.assertEqual(killpg.call_args_list[-1].args, (123456, signal.SIGKILL))
        self.assertFalse((second['output'] / 'inference.log').exists())

    def test_full_file_count_waits_for_process_exit(self):
        cohort = self.runner_cohort('waiting')
        _, audit = self.ns['evaluate_cohorts']([cohort])
        self.ns['input'] = Mock(return_value='y')
        process = Mock(pid=123456, poll=Mock(side_effect=[None, None, 0]))
        signatures = {'waiting0.nii.gz': (1, 1), 'waiting1.nii.gz': (1, 1)}
        with patch.object(subprocess, 'Popen', return_value=process), \
                patch.object(time, 'sleep') as sleep, \
                patch.dict(self.ns, prediction_signatures=Mock(side_effect=[{}, {}, signatures, signatures])), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.ns['run_pending']([cohort], audit), ({}, True))
        self.assertEqual(process.poll.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(QuietProgress.bars[-1].values, [0, 2, 2])
        self.assertIn('waiting for inference/postprocessing', QuietProgress.bars[-1].labels)

    def test_launch_error_stops_remaining_commands(self):
        first, second = self.runner_cohort('launch_error'), self.runner_cohort('unstarted')
        _, audit = self.ns['evaluate_cohorts']([first, second])
        self.ns['input'] = Mock(return_value='y')
        with patch.object(subprocess, 'Popen', side_effect=OSError('executable unavailable')) as launch, \
                contextlib.redirect_stdout(io.StringIO()):
            failures, attempted = self.ns['run_pending']([first, second], audit)
        self.assertTrue(attempted)
        launch.assert_called_once()
        self.assertIn('executable unavailable', failures[('synthetic', 'launch_error')])
        self.assertFalse((second['output'] / 'inference.log').exists())

    def test_cleanup_force_stops_after_grace_period(self):
        process = Mock(pid=123456)
        process.wait.side_effect = [subprocess.TimeoutExpired('fake', 5), -9]
        with patch.object(os, 'killpg') as killpg:
            self.ns['stop_inference_group'](process)
        self.assertEqual(process.wait.call_args_list[0].kwargs, {'timeout': 5})
        self.assertEqual(killpg.call_args_list[-1].args, (123456, signal.SIGKILL))

    def test_progress_ignores_unrelated_and_unchanged_files(self):
        cohort = self.runner_cohort('progress')
        out = cohort['output']
        shutil.copyfile(cohort['rows'][0]['label'], out / 'progress0.nii.gz')
        (out / 'unrelated.nii.gz').touch()
        (out / 'inference.log').touch()
        baseline = self.ns['prediction_signatures'](cohort)
        self.assertEqual(set(baseline), {'progress0.nii.gz'})
        self.assertEqual(sum(sig != baseline.get(name) for name, sig in self.ns['prediction_signatures'](cohort).items()), 0)
        path = out / 'progress0.nii.gz'
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000000))
        shutil.copyfile(cohort['rows'][1]['label'], out / 'progress1.nii.gz')
        self.assertEqual(sum(sig != baseline.get(name) for name, sig in self.ns['prediction_signatures'](cohort).items()), 2)

    def test_dataset_cohort_names_propagate_to_commands_and_table_labels(self):
        assignments = [node for node in ast.parse(SOURCE).body
                       if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id in ('DATASETS', 'MODELS')
                               for target in node.targets)]
        exec(compile(ast.Module(body=assignments, type_ignores=[]), str(NOTEBOOK), 'exec'), self.ns)
        (self.root / 'configs').mkdir()
        for config in ('pooled2', 'rumcnotprostatexnoclinical'):
            (self.root / f'configs/{config}_input.yaml').write_text(
                f'datalist: configs/{config}_folds.json\ndataroot: data\n')
            (self.root / f'configs/{config}_folds.json').write_text(
                json.dumps({'training': [], 'testing': []}))
        for model in self.ns['MODELS']:
            target = self.root / 'bundles' / model['name'] / 'models'
            target.mkdir(parents=True)
            shutil.copyfile(self.bundle / 'models/dataset.json', target / 'dataset.json')
        self.ns['STUDY_ROOT'] = self.root / 'study'
        exec(compile(COMMANDS, str(NOTEBOOK), 'exec'), self.ns)
        cohorts = self.ns['COHORTS']
        self.assertEqual(len(cohorts), 6)
        self.assertEqual(len({c['output'] for c in cohorts}), 6)
        self.assertEqual([(c['dataset'], c['split']) for c in cohorts[3:]], [
            ('rumcnotprostatexnoclinical', 'training'),
            ('rumcnotprostatexnoclinical', 'testing'), ('pooled2', 'testing')])
        for cohort in cohorts:
            argv = cohort['argv']
            self.assertEqual(argv[1], f"bundles/{cohort['model']}/scripts/infer.py")
            self.assertEqual(argv[argv.index('--output_folder') + 1],
                             f"study/{cohort['model']}/{cohort['key']}")
            self.assertEqual(argv[argv.index('--input') + 1],
                             f"configs/{cohort['dataset']}_folds.json")
            self.assertEqual(argv[argv.index('--split') + 1], cohort['split'])
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(WORKFLOW, str(NOTEBOOK), 'exec'), self.ns)
        table = self.ns['counts_table']
        self.assertEqual(table.index.names, ['Dataset', 'Split'])
        self.assertEqual(table.shape, (4, 2))
        self.assertEqual(table.loc[('rumcnotprostatexnoclinical', 'training')].tolist(), ['NA', '0/0'])
        self.assertEqual(table.loc[('pooled2', 'testing')].tolist(), ['0/0', '0/0'])
        self.assertIn('| Dataset | Split |', (self.tables / 'dsc.md').read_text())

    def test_shared_dataset_is_independent_across_models(self):
        first = self.cohort()
        second = dict(first, model='other', run_id=('other', first['key']),
                      output=self.root / 'missing_other_predictions')
        self.ns['COHORTS'] = [first, second]
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(WORKFLOW, str(NOTEBOOK), 'exec'), self.ns)
        audit = self.ns['cohort_audit']
        self.assertEqual(audit.complete.tolist(), [True, False])
        self.assertEqual(audit.cohort_errors.tolist(), ['', ''])
        self.assertEqual(self.ns['counts_table'].iloc[0].tolist(), ['2/2', '0/2'])
        self.assertEqual(self.ns['dsc_table'].iloc[0, 0], 1.)
        self.assertTrue(pd.isna(self.ns['dsc_table'].iloc[0, 1]))
        _, audit = self.ns['evaluate_cohorts']([first, second], {second['run_id']: 'failed'})
        self.assertEqual(audit.cohort_errors.tolist(), ['', 'failed'])
        # Overlapping cohorts within one model still fail the coverage audit.
        duplicate = dict(first, key='duplicate', run_id=('synthetic', 'duplicate'))
        _, audit = self.ns['evaluate_cohorts']([first, duplicate])
        self.assertFalse(audit.complete.any())
        self.assertTrue(audit.cohort_errors.str.contains('overlap').all())

    def test_evaluation_progress_counts_valid_missing_and_invalid_cases(self):
        bars = []

        class RecordingProgress(tqdm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.values = []
                self.desc = kwargs['desc']
                self.unit = kwargs['unit']
                bars.append(self)

            def __iter__(self):
                for i, record in enumerate(super().__iter__(), start=1):
                    self.values.append(i)
                    yield record

        self.ns['thread_map'] = partial(thread_map, tqdm_class=RecordingProgress, disable=True)
        cohort = self.cohort(predictions=1)
        self.evaluate(cohort)
        bar = bars[-1]
        self.assertEqual((bar.desc, bar.unit, bar.total), ('Evaluating Test', 'cases', 2))
        self.assertEqual(bar.values, [1, 2])
        self.assertEqual(self.ns['per_case'].status.tolist(), ['evaluated', 'missing_prediction'])
        self.mask('predictions/case0.nii.gz', np.full((2, 2, 2), 2, dtype=np.uint8))
        self.evaluate(cohort)
        self.assertEqual(bars[-1].values, [1, 2])
        self.assertEqual(self.ns['per_case'].status.tolist(), ['invalid_mask', 'missing_prediction'])

    def test_declined_or_complete_workflow_evaluates_only_once(self):
        cohort = self.cohort(predictions=1)
        evaluator = self.ns['evaluate_cohorts']
        for complete in (False, True):
            if complete:
                self.mask('predictions/case1.nii.gz')
            spy = Mock(wraps=evaluator)
            with patch.dict(self.ns, evaluate_cohorts=spy):
                self.assertEqual(self.evaluate(cohort).complete, complete)
                self.assertFalse(self.ns['execution_attempted'])
            spy.assert_called_once_with([cohort])

    def test_failed_or_interrupted_attempt_refreshes_evaluation(self):
        cohort = self.cohort()
        evaluator = self.ns['evaluate_cohorts']
        for reason in ('Inference failed with exit code 7', 'Inference interrupted'):
            spy = Mock(wraps=evaluator)
            runner = Mock(return_value=({cohort['run_id']: reason}, True))
            with patch.dict(self.ns, evaluate_cohorts=spy, run_pending=runner):
                self.assertFalse(self.evaluate(cohort).complete)
                self.assertTrue(self.ns['dsc_table'].isna().all().all())
            self.assertEqual(spy.call_count, 2)
            self.assertEqual(spy.call_args.args, ([cohort], {cohort['run_id']: reason}))

    def test_command_uses_original_manifest_and_writes_no_artifacts(self):
        (self.root / 'configs').mkdir()
        (self.root / 'configs/test_input.yaml').write_text('datalist: manifest.json\ndataroot: data\n')
        (self.root / 'manifest.json').write_text(json.dumps({'testing': []}))
        self.ns.update(DATASETS=[dict(key='test', dataset='test', split='testing')],
                       MODELS=[dict(name='synthetic', cohort_keys=['test'])], STUDY_ROOT=self.root / 'study')
        exec(compile(COMMANDS, str(NOTEBOOK), 'exec'), self.ns)
        cohort = self.ns['COHORTS'][0]
        command = cohort['command']
        self.assertTrue(command.startswith('.venv/bin/python '))
        self.assertEqual(shlex.split(command.replace(chr(92) + '\n', '')), cohort['argv'])
        self.assertEqual(cohort['argv'][:2], ['.venv/bin/python', 'bundles/synthetic/scripts/infer.py'])
        for name in ('nnUNet_def_n_proc', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
            self.assertNotIn(name, command)
        self.assertNotIn('environment', cohort)
        self.assertIn('.venv/bin/python bundles/synthetic/scripts/infer.py', command)
        self.assertIn('--input manifest.json', command)
        self.assertIn('--tile_step_size 0.8', command)
        self.assertIn('--no-use_mirroring', command)
        self.assertIn('--device cuda', command)
        self.assertIn('--overwrite', command)
        self.assertNotIn('--use_mirroring', command)
        self.assertNotIn(str(self.root), command)
        self.assertNotIn('&&', command)
        self.assertFalse((self.root / 'study').exists())
        result = subprocess.run(['bash', '-n'], input=command, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        (self.bundle / 'models/dataset.json').write_text(json.dumps(
            {'labels': {'background': 0, 'TZ': 1, 'PZ': 2}}))
        with self.assertRaisesRegex(AssertionError, 'whole-gland'):
            exec(compile(COMMANDS, str(NOTEBOOK), 'exec'), self.ns)


if __name__ == '__main__':
    unittest.main()
