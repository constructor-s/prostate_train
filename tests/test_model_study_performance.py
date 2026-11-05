"""Exercise notebook calculations with synthetic masks; no inference is run."""
import ast
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
import unittest
from unittest.mock import Mock, patch

import nibabel as nib
import numpy as np
import pandas as pd
import yaml

NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/model_study_performance.py'
SOURCE = NOTEBOOK.read_text()
MEASUREMENTS = SOURCE[SOURCE.index('def measure_masks('):SOURCE.index('# %% [markdown]\n# ## Evaluate saved predictions')]
COMMANDS = SOURCE[SOURCE.index('for cohort in COHORTS:'):SOURCE.index('# %% [markdown]\n# ## Whole-gland measurements')]
EVALUATION = SOURCE[SOURCE.index('def evaluate_cohorts('):SOURCE.index('# %% [markdown]\n# ## Approve and run pending GPU inference')]
RUNNER = SOURCE[SOURCE.index('def prediction_signatures('):SOURCE.index('# %%\nper_case, cohort_audit = evaluate_cohorts')]
WORKFLOW = SOURCE[SOURCE.index('per_case, cohort_audit = evaluate_cohorts(COHORTS)'):]


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
                       GEOMETRY_TOLERANCE_MM=0.25, BUNDLE=self.bundle,
                       MODEL='synthetic', TABLE_DIR=self.tables, display=lambda *args: None,
                       os=os, signal=signal, subprocess=subprocess, time=time,
                       tqdm=QuietProgress, input=Mock(return_value='n'))
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
        return dict(key='test', title='Test', rows=rows, output=out, data_root=self.root,
                    command='.venv/bin/python bundles/synthetic/scripts/infer.py --input manifest.json '
                            '--split testing --output_folder predictions --tile_step_size 0.8 --no-use_mirroring --device cuda --overwrite')

    def evaluate(self, cohort):
        self.ns['COHORTS'] = [cohort]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(WORKFLOW, str(NOTEBOOK), 'exec'), self.ns)
        self.output = output.getvalue()
        return self.ns['cohort_audit'].iloc[0]

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
        for offset in (.195, .25):
            affine[0, 3] = offset
            near = self.mask('near.nii.gz', affine=affine)
            self.assertAlmostEqual(
                self.ns['measure_masks'](near, ref)['max_corner_displacement_mm'], offset)
        affine[0, 3] = .251
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
        return dict(key=key, title=key, rows=rows, output=out, data_root=self.root,
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
        self.assertEqual(failures, {'interrupted': 'Inference interrupted'})
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
        self.assertIn('executable unavailable', failures['launch_error'])
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
        assignment = next(node for node in ast.parse(SOURCE).body
                          if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == 'COHORTS'
                                  for target in node.targets))
        namespace = {}
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(NOTEBOOK), 'exec'), namespace)
        cohorts = namespace['COHORTS']
        expected = [
            ('pooled2_train', 'pooled2 train', 'pooled2', 'training'),
            ('pooled2_test', 'pooled2 test', 'pooled2', 'testing'),
            ('prostatex_original_train_mapped', 'PROSTATEx original train (mapped)',
             'rumcnotprostatexnoclinical', 'testing'),
        ]
        (self.root / 'configs').mkdir()
        for config in ('pooled2', 'rumcnotprostatexnoclinical'):
            (self.root / f'configs/{config}_input.yaml').write_text(
                f'datalist: configs/{config}_folds.json\ndataroot: data\n')
            (self.root / f'configs/{config}_folds.json').write_text(
                json.dumps({'training': [], 'testing': []}))
        self.ns.update(COHORTS=cohorts, STUDY_ROOT=self.root / 'study')
        exec(compile(COMMANDS, str(NOTEBOOK), 'exec'), self.ns)
        for cohort, (key, title, config, split) in zip(cohorts, expected):
            self.assertEqual((cohort['key'], cohort['title'], cohort['config'], cohort['split']),
                             (key, title, config, split))
            argv = cohort['argv']
            self.assertEqual(argv[argv.index('--output_folder') + 1], f'study/synthetic/{key}')
            self.assertEqual(argv[argv.index('--input') + 1], f'configs/{config}_folds.json')
            self.assertEqual(argv[argv.index('--split') + 1], split)
        # Empty synthetic cohorts still export the requested table headers and audit keys.
        self.ns['input'] = Mock(return_value='n')
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(WORKFLOW, str(NOTEBOOK), 'exec'), self.ns)
        self.assertEqual(self.ns['counts_table'].columns.tolist(), [item[1] for item in expected])
        self.assertEqual(self.ns['cohort_audit'].cohort.tolist(), [item[0] for item in expected])

    def test_evaluation_progress_counts_valid_missing_and_invalid_cases(self):
        cohort = self.cohort(predictions=1)
        self.evaluate(cohort)
        bar = QuietProgress.bars[-1]
        self.assertEqual((bar.desc, bar.unit, bar.total), ('Evaluating Test', 'cases', 2))
        self.assertEqual(bar.values, [1, 2])
        self.assertEqual(self.ns['per_case'].status.tolist(), ['evaluated', 'missing_prediction'])
        self.mask('predictions/case0.nii.gz', np.full((2, 2, 2), 2, dtype=np.uint8))
        self.evaluate(cohort)
        self.assertEqual(QuietProgress.bars[-1].values, [1, 2])
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
            runner = Mock(return_value=({cohort['key']: reason}, True))
            with patch.dict(self.ns, evaluate_cohorts=spy, run_pending=runner):
                self.assertFalse(self.evaluate(cohort).complete)
                self.assertTrue(self.ns['dsc_table'].isna().all().all())
            self.assertEqual(spy.call_count, 2)
            self.assertEqual(spy.call_args.args, ([cohort], {cohort['key']: reason}))

    def test_command_uses_original_manifest_and_writes_no_artifacts(self):
        (self.root / 'configs').mkdir()
        (self.root / 'configs/test_input.yaml').write_text('datalist: manifest.json\ndataroot: data\n')
        (self.root / 'manifest.json').write_text(json.dumps({'testing': []}))
        cohort = dict(key='test', title='Test', config='test', split='testing')
        self.ns.update(COHORTS=[cohort], STUDY_ROOT=self.root / 'study')
        exec(compile(COMMANDS, str(NOTEBOOK), 'exec'), self.ns)
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
