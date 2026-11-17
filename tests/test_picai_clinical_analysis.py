"""Synthetic end-to-end analysis and downstream contract, without model inference."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / 'notebooks/picai_test_clinical_volume.py').read_text()
QI_SOURCE = (ROOT / 'notebooks/volume_disagreement_quality_improvement.py').read_text()


class PicaiClinicalAnalysisTests(unittest.TestCase):
    def test_synthetic_analysis_and_quality_improvement_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configs = root / 'configs'
            configs.mkdir()
            labels = root / 'data/raw/zenodo/picai/picai_labels'
            masks = labels / 'anatomical_delineations/whole_gland/AI/Bosma22b'
            masks.mkdir(parents=True)
            (labels / 'clinical_information').mkdir()
            (labels / 'additional_resources').mkdir()
            (labels / 'additional_resources/ProstateX-mapping.json').write_text('{}')
            rows, entries = [], []
            model_names = ['Dataset274_raw_3d_fullres_5fold', 'Dataset236_raw_3d_fullres_5fold']
            model_roots = []
            for name in model_names:
                bundle = root / 'bundles' / name
                (bundle / 'models').mkdir(parents=True)
                (bundle / 'models/dataset.json').write_text(json.dumps({'labels': {'background': 0, 'class1': 1}}))
                output = root / 'results/20261105_tile0.8_nomirroring' / name / 'picai_test_clinical'
                output.mkdir(parents=True)
                (output / 'dataset.json').write_text((bundle / 'models/dataset.json').read_text())
                (output / 'plans.json').write_text(json.dumps({'dataset_name': name.split('_3d')[0]}))
                config = dict(folds=list(range(5)), bundle_root=str(bundle), arguments=dict(
                    tile_step_size=.8, use_mirroring=False, use_gaussian=True, checkpoint_name='checkpoint_final.pt'))
                (output / 'inference_config.json').write_text(json.dumps(config))
                model_roots.append(output)
            for i in range(8):
                case = f'picai_{100+i}_{200+i}'
                row = dict(patient_id=100+i, study_id=200+i, center='RUMC', psa=4+i,
                           psad=np.nan if i == 0 else .1, prostate_volume=35+3*i,
                           lesion_PIRADS=3, case_ISUP=i % 4, lesion_ISUP=i % 4,
                           lesion_GS='0+0', histopath_type='MRBx', mri_date='2020-01-01')
                rows.append(row)
                data = np.zeros((12, 12, 12), dtype=np.uint8)
                affine = np.diag([5., 5., 5., 1.])
                data[2:10, 2:10, 2:10] = 1
                # Native voxel volume is 125 mm³; vary foreground counts.
                data[2:2+i//2, 2:10, 2:10] = 0
                image = nib.Nifti1Image(data, affine)
                image.header.set_xyzt_units('mm')
                path = masks / f'{100+i}_{200+i}.nii.gz'
                nib.save(image, path)
                relative = path.relative_to(root / 'data/raw').as_posix()
                entries.append(dict(id=case, image=relative, label=relative))
                for j, output in enumerate(model_roots):
                    pred = data.copy()
                    pred[1, 2:4+j, 2:4+i%3] = 1
                    prediction = nib.Nifti1Image(pred, affine)
                    prediction.header.set_xyzt_units('mm')
                    nib.save(prediction, output / f'{case}.nii.gz')
            pd.DataFrame(rows).to_csv(labels / 'clinical_information/marksheet.csv', index=False)
            train = [dict(id='picai_999_999', image='unused', label='unused')]
            (configs / 'rumcnotprostatexnoclinical_folds.json').write_text(json.dumps(dict(training=train, testing=[])))
            (configs / 'pooled2_folds.json').write_text(json.dumps(dict(training=[])))
            (configs / 'picai_test_clinical_folds.json').write_text(json.dumps(dict(training=train, testing_PCNN=[], testing_RUMC=entries, testing_ZGT=[])))
            code = SOURCE.replace("REPO_ROOT = Path(__file__).resolve().parents[1] if '__file__' in globals() else Path.cwd().resolve()",
                                  f'REPO_ROOT = Path({str(root)!r})')
            code = code.replace("globals().get('SITE', 'PCNN')", "'RUMC'").replace('N_BOOTSTRAP = 100_000', 'N_BOOTSTRAP = 100')
            namespace = {}
            with contextlib.redirect_stdout(io.StringIO()), patch('matplotlib.figure.Figure.savefig'), patch('matplotlib.pyplot.show', side_effect=lambda: plt.close('all')):
                exec(compile(code, '<synthetic PI-CAI analysis>', 'exec'), namespace)
            output = root / 'results/picai_test_clinical_volume/RUMC'
            volumes = pd.read_csv(output / 'comparator_volumes.csv')
            self.assertEqual(len(volumes), 8)
            self.assertAlmostEqual(volumes.iloc[0].gt_volume_ml, 64.)
            self.assertAlmostEqual(volumes.iloc[0].nnunet274_volume_ml, 64.5)
            segmentation = pd.read_csv(output / 'segmentation_metrics_per_case.csv')
            self.assertEqual(len(segmentation), 16)
            first = segmentation.loc[segmentation.ID.eq(entries[0]['id']) & segmentation.method.eq('nnunet274')].iloc[0]
            self.assertAlmostEqual(first.dsc, 1024 / 1028)
            self.assertTrue(pd.read_csv(output / 'segmentation_summary.csv').n.eq(8).all())
            metadata = json.loads((output / 'run_metadata.json').read_text())
            self.assertEqual(metadata['cohort_sha256'], hashlib.sha256('\n'.join(sorted(x['id'] for x in entries)).encode()).hexdigest())
            qi = dict(SITE='RUMC', TESTING_FOLD='testing_RUMC', REPO_ROOT=root, OUTPUT_DIR=output, COHORTS={'picai': ('picai_test_clinical_volume', 'PI-CAI', 'Bosma22b automated surrogate')},
                      METHODS=['clinical', 'nnunet274'], SCREENS=['both', 'absolute', 'relative', 'either'],
                      PSAD_CUTOFFS=np.array([.10, .15, .20]), PSAD_STRATA=['0–<0.10', '0.10–<0.15', '0.15–<0.20', '≥0.20'],
                      np=np, pd=pd, json=json, hashlib=hashlib, display=lambda *args: None)
            block = QI_SOURCE[QI_SOURCE.index('cohorts = {}'):QI_SOURCE.index("assert set(cohorts['toronto_manual']")]
            exec(compile(block, '<QI PI-CAI verification>', 'exec'), qi)
            self.assertEqual(len(qi['cohorts']['picai']), 8)
            screen = QI_SOURCE[QI_SOURCE.index('threshold_rows = []'):QI_SOURCE.index('# %% [markdown]\n# ## Volume accuracy and review burden')]
            screen = screen.replace("for flag_cohort in ['prostatex', 'picai']:", "for flag_cohort in ['picai']:")
            exec(compile(screen, '<QI flags>', 'exec'), qi)
            # Missing predictions must stop preflight and prepare commands, not shrink membership.
            (model_roots[0] / f"{entries[0]['id']}.nii.gz").unlink()
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'Predictions missing or unverified'):
                exec(compile(code[:code.index('# %% [markdown]\n# ## Clinical consistency flags')], '<preflight>', 'exec'), {})
            self.assertIn('--no-overwrite', (output / 'inference_commands.txt').read_text())

            # Native-grid measurement retains empty predictions, rejects labels/geometry.
            measurement = SOURCE[SOURCE.index('for key in SELECTED_COMPARATORS:', SOURCE.index('# ## Selected measurements on native grids')):
                                 SOURCE.index('# %% [markdown]\n# ## Suspicious clinical cases')]
            reference = nib.load(root / 'data/raw' / entries[0]['label'])
            path = model_roots[0] / f"{entries[0]['id']}.nii.gz"
            empty = nib.Nifti1Image(np.zeros(reference.shape, dtype=np.uint8), reference.affine)
            empty.header.set_xyzt_units('mm')
            nib.save(empty, path)
            measure_ns = dict(namespace)
            measure_ns['volumes'] = namespace['linked'].copy()
            exec(compile(measurement, '<empty prediction>', 'exec'), measure_ns)
            self.assertEqual(measure_ns['volumes'].iloc[0].nnunet274_volume_ml, 0.)
            segmentation_cell = SOURCE[SOURCE.index('# %% [markdown]\n# ## Whole-gland DSC against the Bosma22b surrogate'):]
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(segmentation_cell, '<DSC retains empty failures>', 'exec'), namespace)
            segmentation = pd.read_csv(output / 'segmentation_metrics_per_case.csv')
            first = segmentation.loc[segmentation.ID.eq(entries[0]['id']) & segmentation.method.eq('nnunet274')].iloc[0]
            self.assertEqual(first.dsc, 0.)
            summary = pd.read_csv(output / 'segmentation_summary.csv').set_index('method')
            self.assertEqual(summary.loc['nnunet274', 'zero_overlap_cases'], 1)
            self.assertEqual(summary.loc['nnunet274', 'n'], 8)
            invalid = nib.Nifti1Image(np.full(reference.shape, 2, dtype=np.uint8), reference.affine)
            invalid.header.set_xyzt_units('mm')
            nib.save(invalid, path)
            with self.assertRaises(AssertionError):
                exec(compile(measurement, '<invalid labels>', 'exec'), dict(namespace))
            shifted_affine = reference.affine.copy()
            shifted_affine[0, 3] += 1.
            shifted = nib.Nifti1Image(np.zeros(reference.shape, dtype=np.uint8), shifted_affine)
            shifted.header.set_xyzt_units('mm')
            nib.save(shifted, path)
            with self.assertRaisesRegex(AssertionError, 'Geometry mismatch'):
                exec(compile(measurement, '<invalid geometry>', 'exec'), dict(namespace))

    def test_optional_figure_is_disabled_and_reports_missing_site_exports(self):
        optional = QI_SOURCE[QI_SOURCE.index('# %% [markdown]\n# ## Optional six-dataset manuscript figure'):]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            namespace = dict(BUILD_ALL_SITE_FIGURE=False, REPO_ROOT=root)
            exec(compile(optional, '<disabled manuscript figure>', 'exec'), namespace)
            self.assertEqual(list(root.iterdir()), [])
            (root / 'configs').mkdir()
            (root / 'configs/picai_test_clinical_folds.json').write_text('{}')
            namespace['BUILD_ALL_SITE_FIGURE'] = True
            with self.assertRaisesRegex(FileNotFoundError, 'Run quality improvement for all three sites'):
                exec(compile(optional, '<missing manuscript inputs>', 'exec'), namespace)
            self.assertFalse((root / 'results/paper_picai_sites').exists())


if __name__ == '__main__':
    unittest.main()
