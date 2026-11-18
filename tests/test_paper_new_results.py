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


if __name__ == '__main__':
    unittest.main()
