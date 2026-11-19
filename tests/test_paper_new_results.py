"""Check display clipping independently of cohort selection and inference."""
import ast
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/paper_new_results.py'
SOURCE = NOTEBOOK.read_text()
COMMON_SETUP = SOURCE[SOURCE.index('def outside_both_common('):SOURCE.index('# %% [markdown]\n# ## Tables 4–6:')]
PLOT = COMMON_SETUP + SOURCE[SOURCE.index('def outside_display('):SOURCE.index('# %% [markdown]\n# ## Arithmetic, boundary cases')]


class BlandAltmanDisplayTests(unittest.TestCase):
    def test_union_counts_once_and_includes_boundaries(self):
        node = next(node for node in ast.parse(SOURCE).body if isinstance(node, ast.FunctionDef) and node.name == 'outside_display')
        namespace = {'np': np}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(NOTEBOOK), 'exec'), namespace)
        hidden = namespace['outside_display']([-1, .5, 2, 0, 1], [.5, 2, 2, 0, 1], (0, 1), (0, 1))
        np.testing.assert_array_equal(hidden, [True, True, True, False, False])
        self.assertEqual(int(hidden.sum()), 3)

    def test_common_intersection_strict_boundaries_and_single_failures(self):
        node = next(node for node in ast.parse(SOURCE).body if isinstance(node, ast.FunctionDef) and node.name == 'outside_both_common')
        namespace = {'np': np}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(NOTEBOOK), 'exec'), namespace)
        mask = namespace['outside_both_common'](
            [-14.28, 25.74, -20, 30, -20, 30, -20, 30],
            [-30, 50, -22.70, 39.60, 0, 0, -30, 50],
            {'ml': (-14.28, 25.74), 'percent': (-22.70, 39.60)})
        np.testing.assert_array_equal(mask, [False, False, False, False, False, False, True, True])
        # Different tails still qualify when both intervals are exceeded.
        self.assertTrue(namespace['outside_both_common']([-20], [50],
                        {'ml': (-14.28, 25.74), 'percent': (-22.70, 39.60)})[0])

    def test_volume_ties_are_conditional_but_psad_ties_remain(self):
        start = SOURCE.index('rows = []\nfor r in volume_summary')
        end = SOURCE.index('rows = []\nfor r in biopsy_cases', start)
        section = SOURCE[start:end]
        row = dict(name='Synthetic', n=2, cohort_n=100, wins=2, ties=0, p_holm=.5)
        for metric in ['mae', 'rmse', 'mape', 'rmspe']:
            row.update({f'manual_{metric}': 10., f'automated_{metric}': 5., f'change_{metric}': -5.})
        psad = dict(name='Synthetic', n=2, n_flagged=2, excluded_psad=0, wins=1, losses=0, ties=1, p_holm=1.)
        namespace = dict(pd=pd, tables={}, volume_summary=pd.DataFrame([row]),
                         psad_summary=pd.DataFrame([psad]), fmt_p=lambda value: f'{value:.3f}')
        exec(compile(section, str(NOTEBOOK), 'exec'), namespace)
        self.assertNotIn('Ties', namespace['tables'][4].columns)
        self.assertEqual(namespace['tables'][4].iloc[0]['Flagged n/N (%)'], '2/100 (2.0%)')
        self.assertEqual(namespace['tables'][5].iloc[0]['Equal distance'], 1)
        namespace['volume_summary'].loc[0, 'ties'] = 1
        exec(compile(section, str(NOTEBOOK), 'exec'), namespace)
        self.assertEqual(namespace['tables'][4].iloc[0]['Ties'], 1)

    def test_cropping_preserves_full_cohort_statistics(self):
        cohorts = {f'cohort_{i}': pd.DataFrame({'clinical_volume_ml': [10., 20., 30., 200.],
                                               'nnunet274_volume_ml': [12., 18., 35., 20.]}) for i in range(6)}
        for values in cohorts.values():
            values['flag_both'] = [False, False, False, True]
        sd = np.std([-2., 2., -5., 180.], ddof=1)
        thresholds = pd.DataFrame([dict(cohort=key, mean_ml=43.75, lower_ml=43.75-1.96*sd,
                                        upper_ml=43.75+1.96*sd) for key in cohorts])
        relative = 100*np.array([-2., 2., -5., 180.])/np.array([11., 19., 32.5, 110.])
        thresholds['mean_percent'] = relative.mean()
        thresholds['lower_percent'] = relative.mean()-1.96*relative.std(ddof=1)
        thresholds['upper_percent'] = relative.mean()+1.96*relative.std(ddof=1)
        with tempfile.TemporaryDirectory() as directory:
            namespace = dict(np=np, pd=pd, plt=plt, cohorts=cohorts, thresholds=thresholds,
                             COHORTS={key: ('unused', key, 4, 'unused') for key in cohorts},
                             OUTPUT_DIR=Path(directory), BA_XLIM=None, BA_YLIM=None, BA_REL_YLIM=None)
            exec(compile(PLOT, str(NOTEBOOK), 'exec'), namespace)
            full = namespace['plot_summary'].copy()
            self.assertTrue(full.hidden_n.eq(0).all())
            self.assertEqual(len(full), 12)
            self.assertEqual(namespace['axes'].shape, (6, 2))
            np.testing.assert_allclose(namespace['fig'].get_size_inches(), [8.5, 9.75])
            from PIL import Image
            with Image.open(Path(directory) / 'bland_altman_clinical_nnunet274.png') as exported:
                self.assertEqual(exported.size, (2550, 2925))
            renderer = namespace['fig'].canvas.get_renderer()
            for ax in namespace['axes'][0]:
                heading = next(text for text in ax.texts if text.get_text() in ['Absolute difference', 'Relative difference'])
                self.assertGreater(heading.get_window_extent(renderer).y0, ax.title.get_window_extent(renderer).y1)
            for index, ax in enumerate(namespace['axes'].flat):
                self.assertFalse(any('Outside displayed limits' in item.get_text() for item in ax.texts))
                self.assertEqual(ax.get_xscale(), 'log')
                np.testing.assert_array_equal(ax.get_xticks(), [10, 30, 100, 300])
                self.assertEqual([label.get_text() for label in ax.get_xticklabels()], ['10', '30', '100', '300'])
                self.assertTrue(all(label.get_visible() for label in ax.get_xticklabels()))
                self.assertTrue(all(not label.get_text() for label in ax.get_xticklabels(minor=True)))
                common = [line for line in ax.lines if line.get_color() == 'black' and line.get_linestyle() == '--']
                self.assertEqual(len(common), 2)
                expected = (-14.28, 25.74) if index % 2 == 0 else (-22.70, 39.60)
                np.testing.assert_array_equal([line.get_ydata()[0] for line in common], expected)
                self.assertTrue(all(line.get_zorder() >= 4 for line in common))
                self.assertEqual(ax.collections[0].get_alpha(), .15)
                self.assertEqual(ax.collections[1].get_alpha(), 1.)
                np.testing.assert_array_equal(ax.collections[1].get_sizes(), [4])
                np.testing.assert_allclose(ax.collections[1].get_facecolors()[0], [213/255, 94/255, 0, 1])
                self.assertEqual(len(ax.collections[1].get_offsets()), 1)
                self.assertIn('Outside both common criteria (n = 1)', ax.get_legend_handles_labels()[1])
            self.assertTrue(full.common_both_n.eq(1).all())
            for left, right in namespace['axes']:
                np.testing.assert_array_equal(left.collections[1].get_offsets()[:, 0], right.collections[1].get_offsets()[:, 0])
                np.testing.assert_array_equal(left.collections[1].get_offsets()[:, 0], [110.])
            import re
            pdf = (Path(directory) / 'bland_altman_clinical_nnunet274.pdf').read_bytes()
            self.assertRegex(pdf, rb'/MediaBox\s*\[\s*0\s+0\s+612\s+702\s*\]')
            namespace.update(BA_XLIM=(10, 40), BA_YLIM=(-10, 10), BA_REL_YLIM=(-5, 5))
            exec(compile(PLOT, str(NOTEBOOK), 'exec'), namespace)
            cropped = namespace['plot_summary']
            for ax in namespace['axes'].flat:
                self.assertEqual(sum('Outside displayed limits' in item.get_text() for item in ax.texts), 1)
            self.assertTrue(cropped.loc[cropped.scale.eq('ml'), 'hidden_n'].eq(1).all())
            self.assertTrue(cropped.loc[cropped.scale.eq('percent'), 'hidden_n'].eq(4).all())
            self.assertTrue(cropped.loc[cropped.scale.eq('ml'), 'y_min'].eq(-10).all())
            self.assertTrue(cropped.loc[cropped.scale.eq('percent'), 'y_min'].eq(-5).all())
            pd.testing.assert_frame_equal(full[['n', 'mean', 'lower', 'upper', 'common_both_n']],
                                          cropped[['n', 'mean', 'lower', 'upper', 'common_both_n']])
            self.assertTrue(cropped.n.eq(4).all())
            self.assertTrue((Path(directory) / 'bland_altman_clinical_nnunet274.pdf').is_file())
            namespace['BA_XLIM'] = (0, 40)
            with self.assertRaisesRegex(AssertionError, 'positive'):
                exec(compile(PLOT, str(NOTEBOOK), 'exec'), namespace)
            plt.close('all')


class SourceCohortTests(unittest.TestCase):
    def selection(self, directory, exclusions):
        root = Path(directory)
        base = root / 'data_internal/hlab20260928'
        base.mkdir(parents=True, exist_ok=True)
        ids = ['AIPR_1_20200101_a', 'AIPR_2_20200101_b', 'AIPR_3_20200101_c']
        pd.DataFrame(dict(ID=ids, t2=['a', 'b', 'c'], ground_truth_volume_ml=[20., np.nan, 0.],
                          vol_report_ref=[20., 30., 40.])).to_csv(base / 'hlab20260928_t2_image_listing_full_clinical.csv', index=False)
        import json
        (base / 'exclude.json').write_text(json.dumps(exclusions))
        namespace = dict(REPO_ROOT=root, DATA_ROOT=root / 'data/raw', OUTPUT_DIR=root, pd=pd, np=np, Path=Path, input_sha256={}, hashlib=__import__('hashlib'), json=json)
        reader = next(n for n in ast.parse(SOURCE).body if isinstance(n, ast.FunctionDef) and n.name == 'read_input')
        exec(compile(ast.Module(body=[reader], type_ignores=[]), str(NOTEBOOK), 'exec'), namespace)
        section = SOURCE[SOURCE.index("TORONTO_ROOT ="):SOURCE.index('marksheet = read_input')]
        exec(compile(section, str(NOTEBOOK), 'exec'), namespace)
        return namespace

    def test_current_exclusions_partition_and_unmatched_duplicate_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            ns = self.selection(directory, [{'aano': 'a', 'reason': 'first'}, {'aano': 'a', 'description': 'second'}, {'aano': 'absent'}])
            listing = ns['listing']
            self.assertEqual(listing.loc[listing.included, 'ID'].tolist(), ['AIPR_2_20200101_b', 'AIPR_3_20200101_c'])
            self.assertTrue(listing.loc[listing.included, 'analysis_cohort'].eq('toronto_surrogate').all())
            self.assertEqual(listing.iloc[0].exclusion_reason, 'first; second')
            self.assertEqual(int(ns['exclusion_audit'].duplicate_accession.sum()), 2)
            self.assertEqual(int((~ns['exclusion_audit'].matched_source).sum()), 1)
            ns = self.selection(directory, [{'aano': 'b'}])
            self.assertEqual(ns['listing'].loc[ns['listing'].included, 'analysis_cohort'].tolist(), ['toronto_manual', 'toronto_surrogate'])

    def test_empty_exclusion_list_keeps_every_source_row(self):
        with tempfile.TemporaryDirectory() as directory:
            ns = self.selection(directory, [])
            self.assertTrue(ns['listing'].included.all())
            self.assertEqual(ns['listing'].analysis_cohort.tolist(), ['toronto_manual', 'toronto_surrogate', 'toronto_surrogate'])

    def test_no_upstream_notebook_or_analysis_table_reads(self):
        for forbidden in ['check_record(', 'qi[', 'comparator_summary.csv', 'psa_density_biopsy_audit.csv',
                          'volume_disagreement_quality_improvement/', 'per_case_metrics.csv', 'workbook_key_audit.csv']:
            self.assertNotIn(forbidden, SOURCE)


class DirectMaskTests(unittest.TestCase):
    def namespace(self, root):
        import hashlib
        import nibabel as nib
        from itertools import product
        namespace = dict(REPO_ROOT=root, np=np, nib=nib, pd=pd, Path=Path, hashlib=hashlib, product=product,
                         input_sha256={}, measurement_rows=[], postprocessed_sha256={}, GEOMETRY_TOLERANCE_MM=.5,
                         UNIT_ANCHOR_TOLERANCE_MM=.01)
        nodes = [n for n in ast.parse(SOURCE).body if isinstance(n, ast.FunctionDef)
                 and n.name in ['record_input', 'grid_displacement', 'measure_source']]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), 'exec'), namespace)
        return namespace

    def fixture(self, root, empty_prediction=False, reference_units='mm', shift=0., invalid_label=False):
        import nibabel as nib
        from types import SimpleNamespace
        shape = (3, 4, 5)
        affine = np.diag([2., 3., 4., 1.])
        paths = {}
        for method in ['image', 'gt', 'model']:
            data = np.zeros(shape, dtype=np.uint8)
            if method == 'gt':
                data[0, 0, :3] = [1, 2, 3]
            if method == 'model' and not empty_prediction:
                data[0, 0, :2] = 1
            if invalid_label and method == 'model':
                data[0, 0, 0] = 9
            matrix = affine.copy()
            if method == 'model':
                matrix[0, 3] += shift
            img = nib.Nifti1Image(data, matrix)
            img.header.set_xyzt_units(reference_units if method == 'gt' else 'mm')
            path = root / ('case.nii.gz' if method == 'model' else f'{method}.nii.gz')
            nib.save(img, path)
            paths[method] = path
        case = SimpleNamespace(ID='case', image_path=paths['image'], reference_path=paths['gt'], manual_reference_ml=42.)
        return case, {'model': root}

    def test_union_anisotropic_volume_and_dsc(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); ns = self.namespace(root)
            case, roots = self.fixture(root)
            row = ns['measure_source'](case, 'toronto', roots)
            self.assertAlmostEqual(row['gt_volume_ml'], .072)
            self.assertAlmostEqual(row['model_volume_ml'], .048)
            self.assertAlmostEqual(row['model_dsc'], .8)

    def test_empty_prediction_retained_and_scalar_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); ns = self.namespace(root)
            case, roots = self.fixture(root, empty_prediction=True)
            row = ns['measure_source'](case, 'toronto', roots)
            self.assertEqual(row['model_volume_ml'], 0.)
            self.assertEqual(row['model_dsc'], 0.)
            row = ns['measure_source'](case, 'toronto', roots, scalar_reference=True)
            self.assertEqual(row['gt_volume_ml'], 42.)
            self.assertTrue(np.isnan(row['model_dsc']))

    def test_unknown_units_require_close_mm_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); ns = self.namespace(root)
            case, roots = self.fixture(root, reference_units='unknown')
            row = ns['measure_source'](case, 'toronto', roots)
            self.assertEqual(row['gt_unit_anchor_path'], 'image.nii.gz')
            with self.assertRaisesRegex(AssertionError, 'Spatial units'):
                ns['measure_source'](case, 'picai', roots)
            import nibabel as nib
            img = nib.load(case.reference_path)
            affine = img.affine.copy(); affine[0, 3] = .02
            nib.save(nib.Nifti1Image(np.asanyarray(img.dataobj), affine, img.header), case.reference_path)
            with self.assertRaisesRegex(AssertionError, 'known-mm anchor'):
                ns['measure_source'](case, 'toronto', roots)

    def test_reject_geometry_labels_and_modified_postprocessed_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); ns = self.namespace(root)
            case, roots = self.fixture(root, shift=1.)
            with self.assertRaisesRegex(AssertionError, 'displacement'):
                ns['measure_source'](case, 'toronto', roots)
            ns = self.namespace(root); case, roots = self.fixture(root, invalid_label=True)
            with self.assertRaisesRegex(AssertionError, 'Unexpected labels'):
                ns['measure_source'](case, 'toronto', roots)
            ns = self.namespace(root); case, roots = self.fixture(root)
            ns['postprocessed_sha256']['case.nii.gz'] = 'wrong'
            with self.assertRaisesRegex(AssertionError, 'Postprocessed mask hash mismatch'):
                ns['measure_source'](case, 'toronto', roots)


class PredictionProvenanceTests(unittest.TestCase):
    def test_folds_and_explicit_historical_limitation(self):
        import json
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = 'Dataset274_raw_3d_fullres_5fold'
            predictions = root / 'predictions'
            bundle = root / 'bundles' / name
            predictions.mkdir(); (bundle / 'models').mkdir(parents=True)
            labels = {'labels': {'background': 0, 'class1': 1}}
            for path in [predictions / 'dataset.json', bundle / 'models/dataset.json']:
                path.write_text(json.dumps(labels))
            (predictions / 'plans.json').write_text(json.dumps({'dataset_name': 'Dataset274_raw'}))
            ns = dict(REPO_ROOT=root, Path=Path, json=json, hashlib=hashlib, pd=pd,
                      input_sha256={}, model_metadata={}, postprocessed_sha256={}, MODEL_NAMES={'nnunet274': name})
            nodes = [n for n in ast.parse(SOURCE).body if isinstance(n, ast.FunctionDef)
                     and n.name in ['read_input', 'record_input', 'validate_model']]
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), 'exec'), ns)
            with self.assertRaisesRegex(AssertionError, 'Missing inference configuration'):
                ns['validate_model'](predictions, 'nnunet274')
            ns['validate_model'](predictions, 'nnunet274', historical=True)
            self.assertFalse(ns['model_metadata']['predictions']['inference_verified'])
            self.assertIn('unverified', ns['model_metadata']['predictions']['inference_limitation'])
            ns['model_metadata'].clear()
            config = dict(folds=[0], bundle_root=str(bundle), arguments=dict(tile_step_size=.8,
                use_mirroring=False, use_gaussian=True, checkpoint_name='checkpoint_final.pt'))
            config_path = predictions / 'inference_config.json'
            config_path.write_text(json.dumps(config))
            with self.assertRaises(AssertionError):
                ns['validate_model'](predictions, 'nnunet274')
            config['folds'] = [0, 1, 2, 3, 4]
            config_path.write_text(json.dumps(config))
            ns['validate_model'](predictions, 'nnunet274')
            self.assertTrue(ns['model_metadata']['predictions']['inference_verified'])


class ClinicalLinkageTests(unittest.TestCase):
    def test_psa_precedence_correction_censoring_and_missing_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            workbook = pd.DataFrame(dict(
                aano=['a', '333abfe652a8889', 'c'], amrn=['AIPR_1', 'AIPR_2', 'AIPR_3'],
                psa_report_ref=['>5,2', '', '0'], psa=['10', '999', ''],
                reportmaxpirads_ref=['3', '3', '5'], CSCANCER=['YES', 'NO BIOPSY', 'N']))
            readers = ['p_axt2_ch_vol', 'p_axt2_ah_vol', 'p_axt2_dd_vol', 'p_axt2_ao_vol',
                       'p_axt2_chepg_vol', 'p_axt2_epgch_vol', 'p_axt2_epg_vol']
            for field in readers:
                workbook[field] = ['20', '', ''] if field == readers[0] else ['', '', '']
            cases = pd.DataFrame(dict(ID=['one', 'two', 'three'], aano=workbook.aano,
                                      amrn=['1', '2', '3'], manual_reference_ml=[20., np.nan, np.nan]))
            ns = dict(np=np, pd=pd, workbook=workbook, measured={'toronto': cases}, OUTPUT_DIR=Path(directory))
            section = SOURCE[SOURCE.index('READER_FIELDS ='):SOURCE.index('# %% [markdown]\n# ## Independent analysis cohorts')]
            exec(compile(section, str(NOTEBOOK), 'exec'), ns)
            clinical = ns['clinical']
            np.testing.assert_allclose(clinical.psa_ng_ml, [5.2, 28.9, np.nan], equal_nan=True)
            self.assertEqual(clinical.psa_source.tolist(), ['psa_report_ref', 'psa', 'unavailable'])
            self.assertEqual(clinical.psa_selected_censored.tolist(), [True, False, False])
            self.assertEqual(clinical.pirads_3.tolist(), [True, True, False])
            self.assertEqual(clinical.outcome_known.tolist(), [True, False, True])
            self.assertTrue(pd.isna(clinical.iloc[1].outcome))
            self.assertEqual(len(clinical), 3)  # Missing PSA/outcome never removes a volume case.


if __name__ == '__main__':
    unittest.main()
