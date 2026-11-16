"""Validate full-cohort cells with synthetic masks and clinical records."""
from contextlib import redirect_stdout
import hashlib
from io import StringIO
from pathlib import Path
import json
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import nibabel as nib
import numpy as np
import pandas as pd

import test_toronto_jdmi_clinical_volume as original


NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/toronto_jdmi_4111_clinical_volume.py'
SOURCE = NOTEBOOK.read_text()
SELECTION = SOURCE[SOURCE.index('listing = pd.read_csv('):SOURCE.index('# %% [markdown]\n# ## Workbook loading')]
LINKAGE = SOURCE[SOURCE.index('READER_FIELDS = ['):SOURCE.index('# %% [markdown]\n# ## Direct measurements')]
MEASUREMENT = SOURCE[SOURCE.index('input_hashes = {}'):SOURCE.index('# %% [markdown]\n# ## Volume agreement')]
SUMMARY = SOURCE[SOURCE.index('gt = volumes['):SOURCE.index('# %% [markdown]\n# ## Agreement plots')]
PSAD = SOURCE[SOURCE.index("PSAD_UNIT = 'ng/mL²'"):SOURCE.index('# %% [markdown]\n# ## PSA density: agreement plots')]
QUALITY = SOURCE[SOURCE.index('# %% [markdown]\n# ## Clinical report volume quality check'):SOURCE.index('# %% [markdown]\n# ## Local reproducibility record')]
BIOPSY = SOURCE[SOURCE.index('BIOPSY_PSAD_THRESHOLD ='):SOURCE.index('# %% [markdown]\n# ### Biopsy decisions')]
VOLUME_VISIBILITY = SOURCE[SOURCE.index('# Collect outliers from the original data against the final shared volume axes.'):
                           SOURCE.index('# %% [markdown]\n# ## Paired absolute-error comparisons')]
PSAD_VISIBILITY = SOURCE[SOURCE.index('# Collect outliers from the original data against the final shared PSA-density axes.'):
                         SOURCE.index('# %% [markdown]\n# ## PSA density: strata and threshold statistics')]
QUALITY_VISIBILITY = SOURCE[SOURCE.index('# Collect outliers from both panels, including the already labeled clipping.'):
                            SOURCE.index('# %%\nquality_flagged =')]


class FullTorontoVolumeTests(unittest.TestCase):
    # Share only the synthetic input builder, not the original notebook's tests.
    save_listing = original.TorontoVolumeTests.save_listing
    image = original.TorontoVolumeTests.image

    def setUp(self):
        original.TorontoVolumeTests.setUp(self)
        self.ns.update(
            REFERENCE_ROOT=self.data / 'PROSATEPSAD',
            REFERENCE_FILENAME='wp-pz-tz_axt2_nnunet-702.nii.gz',
            REFERENCE_LABELS=(1, 2, 3), REFERENCE_NAME='nnUNet702 automated surrogate',
            UNIT_ANCHOR_TOLERANCE_MM=.01,
        )
        self.reference_paths = []
        for row in self.rows:
            path = self.ns['REFERENCE_ROOT'] / row['ID'] / self.ns['REFERENCE_FILENAME']
            path.parent.mkdir(parents=True)
            self.image(path, data=np.array([0, 1, 1, 2, 2, 3, 3, 3], dtype=np.uint8).reshape(2, 2, 2))
            self.reference_paths.append(path)
        # Model masks cover seven foreground voxels at 24 mm³ per voxel = .168 mL.
        # Optional manual reference is .2 mL and must never replace that value.
        self.rows[1]['ground_truth_volume_ml'] = None
        self.ns['workbook'].loc[1, 'p_axt2_ch_vol'] = np.nan
        self.save_listing()
        self.addCleanup(plt.close, 'all')

    def execute(self, source):
        with redirect_stdout(StringIO()):
            exec(compile(source, str(NOTEBOOK), 'exec'), self.ns)

    def measure(self):
        self.execute(SELECTION)
        self.execute(LINKAGE)
        self.execute(MEASUREMENT)

    def test_missing_manual_reference_retained_and_all_accession_reasons_excluded(self):
        for i in range(2):
            self.rows.append(dict(ID=f'AIPR_excluded{i}_20260101_x{i}', t2='',
                                  ground_truth_volume_ml=None, vol_report_ref=.25))
        self.exclude.write_text(json.dumps([
            dict(aano=f'x{i}', **detail) for i, detail in enumerate(
                [{'reason': 'synthetic exclusion'}, {'description': 'review'}]
            )
        ]))
        self.save_listing()
        self.execute(SELECTION)
        self.execute(LINKAGE)
        self.assertEqual(self.ns['cases'].ID.tolist(), [row['ID'] for row in self.rows[:2]])
        audit = pd.read_csv(self.out / 'cohort_audit.csv')
        self.assertEqual(audit.included.tolist(), [True, True, False, False])
        self.assertFalse(audit.has_manual_reference.iloc[1])
        self.assertTrue(self.ns['cases'].manual_reference_matches.all())
        self.assertNotIn('gt_volume_ml', self.ns['cases'])

    def test_optional_manual_values_still_check_workbook_consistency(self):
        self.rows[0]['ground_truth_volume_ml'] = .3
        self.save_listing()
        self.execute(SELECTION)
        with self.assertRaisesRegex(AssertionError, 'manual-reference mismatch'):
            self.execute(LINKAGE)
        audit = pd.read_csv(self.out / 'clinical_fields_audit.csv')
        self.assertFalse(audit.manual_reference_matches.iloc[0])

    def test_duplicate_identity_count_and_membership_changes_stop_after_audit(self):
        for bad_id in ['AIPR_p0_20260201_s1', '../invalid', self.rows[0]['ID']]:
            with self.subTest(case=bad_id):
                self.rows[1]['ID'] = bad_id
                self.save_listing()
                with self.assertRaisesRegex(AssertionError, 'Invalid or repeated'):
                    self.execute(SELECTION)
                self.assertTrue((self.out / 'cohort_audit.csv').is_file())
        self.rows[1]['ID'] = 'AIPR_p1_20260101_s1'
        self.save_listing()
        self.ns['EXPECTED_CASES'] = 3
        with self.assertRaisesRegex(AssertionError, 'count changed'):
            self.execute(SELECTION)
        self.assertTrue((self.out / 'cohort_flow.csv').is_file())
        self.ns['EXPECTED_CASES'] = 2
        self.execute(SELECTION)
        (self.out / 'run_metadata.json').write_text(json.dumps({'cohort_sha256': self.ns['cohort_sha256']}))
        self.rows[1]['ID'] = 'AIPR_replacement_20260101_s1'
        self.save_listing()
        with self.assertRaisesRegex(AssertionError, 'membership changed'):
            self.execute(SELECTION)

    def test_union_counts_volume_audit_and_provenance(self):
        self.measure()
        volumes = self.ns['volumes']
        np.testing.assert_allclose(volumes.gt_volume_ml, .168)
        self.assertTrue(volumes.gt_volume_ml.equals(self.ns['cases'].gt_volume_ml))
        audit = pd.read_csv(self.out / 'measurement_audit.csv')
        reference = audit.loc[audit.method.eq('gt')]
        self.assertEqual(reference.voxel_count.tolist(), [7, 7])
        self.assertEqual(reference.label_3_voxel_count.tolist(), [3, 3])
        self.assertEqual(json.loads(reference.observed_labels.iloc[0]), [0, 1, 2, 3])
        np.testing.assert_allclose(reference.voxel_volume_mm3, 24)
        for path in [*self.reference_paths, self.data / self.rows[0]['t2']]:
            relative = str(path.relative_to(self.ns['REPO_ROOT']))
            self.assertEqual(self.ns['input_hashes'][relative], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(pd.read_csv(self.out / 'cohort_flow.csv').n.tolist(), [2, 2, 2])

    def test_missing_zone_is_allowed_if_whole_gland_nonempty(self):
        self.image(self.reference_paths[0], data=np.ones((2, 2, 2), dtype=np.uint8))
        self.measure()
        self.assertAlmostEqual(self.ns['volumes'].gt_volume_ml.iloc[0], .192)

    def test_invalid_reference_masks_stop_with_audit(self):
        path = self.reference_paths[0]
        for problem in ['missing', 'unexpected_label', 'fractional', 'nonfinite', 'empty', 'shape', 'units', 'geometry']:
            with self.subTest(problem=problem):
                self.image(path)
                if problem == 'missing':
                    path.unlink()
                elif problem in ['unexpected_label', 'fractional', 'nonfinite', 'empty']:
                    value = {'unexpected_label': 4, 'fractional': 1.5, 'nonfinite': np.nan, 'empty': 0}[problem]
                    self.image(path, data=np.full((2, 2, 2), value, dtype=np.float32))
                elif problem == 'shape':
                    self.image(path, data=np.ones((2, 2, 3), dtype=np.uint8))
                elif problem == 'units':
                    self.image(path, units='meter')
                else:
                    affine = np.diag([2., 3., 4., 1.])
                    affine[0, 3] = .6
                    self.image(path, affine=affine)
                with self.assertRaisesRegex(ValueError, 'Incomplete or invalid measurements'):
                    self.measure()
                audit = pd.read_csv(self.out / 'measurement_audit.csv')
                failed = audit.loc[audit.method.eq('gt') & audit.ID.eq(self.rows[0]['ID'])]
                self.assertEqual(failed.status.tolist(), ['invalid'])
                self.assertTrue(failed.error.notna().all())

    def test_tolerated_reference_geometry_is_recorded(self):
        affine = np.diag([2., 3., 4., 1.])
        affine[0, 3] = .4
        self.image(self.reference_paths[0], affine=affine)
        self.measure()
        self.assertAlmostEqual(self.ns['volumes'].gt_max_corner_displacement_mm.iloc[0], .4, places=6)

    def test_unknown_units_require_an_audited_matching_mm_anchor(self):
        self.image(self.reference_paths[0], units='unknown')
        self.measure()
        reference = self.ns['measurement_audit'].query("method == 'gt'").iloc[0]
        self.assertTrue(reference.units_assumed_mm)
        self.assertEqual(reference.mask_spatial_units, 'unknown')
        self.assertEqual(reference.image_spatial_units, 'mm')
        self.assertEqual(reference.unit_anchor_path, str((self.data / 'image0.nii.gz').relative_to(self.ns['REPO_ROOT'])))
        # When both reference and T2 lack unit codes, the mm comparator anchors them.
        self.image(self.data / 'image0.nii.gz', units='unknown')
        self.measure()
        reference = self.ns['measurement_audit'].query("method == 'gt'").iloc[0]
        self.assertEqual(reference.unit_anchor_path, str(self.prediction.relative_to(self.ns['REPO_ROOT'])))
        self.assertEqual(reference.unit_anchor_max_displacement_mm, 0)
        np.testing.assert_allclose(self.ns['volumes'].gt_volume_ml, [.192, .168])

    def test_unknown_units_do_not_allow_loose_geometry_or_an_unanchored_assumption(self):
        affine = np.diag([2., 3., 4., 1.])
        affine[0, 3] = .1  # Allowed for known-mm geometry, too loose to infer units.
        self.image(self.reference_paths[0], affine=affine, units='unknown')
        with self.assertRaisesRegex(ValueError, 'Incomplete or invalid measurements'):
            self.measure()
        audit = pd.read_csv(self.out / 'measurement_audit.csv')
        self.assertTrue(audit.error.str.contains('Unknown-unit grid differs', na=False).any())
        self.image(self.reference_paths[0], units='unknown')
        self.image(self.data / 'image0.nii.gz', units='unknown')
        for folder in self.data.glob('seg_out_*'):
            self.image(folder / f"{self.rows[0]['ID']}.nii.gz", units='unknown')
        with self.assertRaisesRegex(ValueError, 'Incomplete or invalid measurements'):
            self.measure()
        audit = pd.read_csv(self.out / 'measurement_audit.csv')
        self.assertTrue(audit.error.str.contains('known-mm grid anchor', na=False).any())

    def test_unit_anchor_allows_small_coordinate_precision_differences(self):
        affine = np.diag([2., 3., 4., 1.])
        affine[0, 3] = .003
        self.image(self.reference_paths[0], affine=affine, units='unknown')
        self.measure()
        reference = self.ns['measurement_audit'].query("method == 'gt'").iloc[0]
        self.assertTrue(reference.units_assumed_mm)
        self.assertAlmostEqual(reference.unit_anchor_max_displacement_mm, .003, places=6)

    def test_invalid_comparator_stops_and_zero_prediction_remains_in_volume_results(self):
        self.prediction.unlink()
        with self.assertRaisesRegex(ValueError, 'Incomplete or invalid measurements'):
            self.measure()
        self.image(self.prediction, data=np.zeros((2, 2, 2), dtype=np.uint8))
        self.measure()
        self.execute(SUMMARY)
        self.assertEqual(self.ns['volumes'].nnunet274_volume_ml.iloc[0], 0)
        self.assertEqual(self.ns['summary'].n.tolist(), [2, 2, 2])
        with self.assertRaisesRegex(AssertionError, 'Invalid PSA-density inputs'):
            self.execute(PSAD)
        self.assertTrue((self.out / 'psa_density_cohort_audit.csv').is_file())

    def test_reference_propagates_to_summary_density_and_quality_table(self):
        self.measure()
        self.execute(SUMMARY)
        clinical = self.ns['summary'].set_index('method').loc['clinical']
        self.assertAlmostEqual(clinical.bias_ml, .25 - .168)
        self.execute(PSAD)
        np.testing.assert_allclose(self.ns['psad_cases'].gt_psad, np.array([5., 12.]) / .168)
        self.ns.update(plt=plt, FuncFormatter=FuncFormatter, NullFormatter=NullFormatter,
                       BIOPSY_PSAD_THRESHOLD=.15)
        with patch.object(plt, 'show'):
            self.execute(QUALITY)
        table = pd.read_csv(self.out / 'report_volume_quality_table.csv')
        np.testing.assert_allclose(table.reference_volume_ml, .168)
        np.testing.assert_allclose(table.reference_psad, np.array([5., 12.]) / .168)
        self.assertNotIn('manual_planimetry_volume_ml', table)
        self.assertFalse({'ID', 'amrn', 'aano'} & set(table.columns))

    def test_unrounded_density_boundary_and_optional_biopsy_outcomes(self):
        self.measure()
        volumes = self.ns['volumes']
        volumes['psa_consolidated'] = np.array([.15, .15 - 1e-8]) * volumes.gt_volume_ml
        self.execute(PSAD)
        self.assertEqual(self.ns['psad_cases'].gt_psad.iloc[0], .15)
        self.execute(BIOPSY)
        self.assertEqual(self.ns['biopsy_cases'].gt_biopsy_recommended.tolist(), [True, False])
        self.assertEqual(self.ns['biopsy_cases'].clinical_biopsy_recommended.tolist(), [False, False])
        self.assertEqual(self.ns['biopsy_names']['gt'], self.ns['REFERENCE_NAME'])
        # Unknown cancer outcomes affect only biopsy simulation; no full-cohort fallback.
        self.ns['psad_audit']['cscancer'] = 'unknown'
        self.execute(BIOPSY)
        self.assertEqual(self.ns['biopsy_n'], 0)
        self.assertEqual(len(self.ns['psad_cases']), 2)
        self.assertTrue(self.ns['biopsy_summary'].n.eq(0).all())
        self.ns['psad_audit']['cscancer'] = 'Y'
        self.ns['psad_audit']['pirads'] = 4
        self.execute(BIOPSY)
        self.assertEqual(self.ns['biopsy_n'], 0)

    def test_complete_cell_order_run_and_repeat_statistics(self):
        # Execute the entire notebook with real synthetic CSV/XLSX/NIfTI inputs.
        # Reduce only cohort count and resampling cost; production parameters stay pinned.
        root = self.ns['REPO_ROOT']
        (root / 'pyproject.toml').touch()
        (root / 'notebooks').mkdir()
        notebook_path = root / 'notebooks' / NOTEBOOK.name
        notebook_path.write_text(SOURCE)
        pd.DataFrame(self.rows).to_csv(self.data / 'hlab20260928_t2_image_listing_full_clinical.csv', index=False)
        self.ns['workbook'].to_excel(self.data / 'PROSATELLMVOL20260703-12-WorkingBillShi.xlsx', index=False)
        source = SOURCE.replace('EXPECTED_CASES = 4105', 'EXPECTED_CASES = 2')
        source = source.replace('N_BOOTSTRAP = 100_000', 'N_BOOTSTRAP = 99')
        source = source.replace('BOOTSTRAP_BATCH = 1_000', 'BOOTSTRAP_BATCH = 20')
        with patch('IPython.display.display'), patch.object(plt, 'show'):
            self.ns = {'__file__': str(notebook_path)}
            self.execute(source)
        out = root / 'results/toronto_jdmi_4111_clinical_volume'
        metadata = json.loads((out / 'run_metadata.json').read_text())
        self.assertEqual(metadata['reference_type'], 'automated_surrogate')
        self.assertEqual(metadata['reference_foreground_labels'], [1, 2, 3])
        self.assertEqual(metadata['n_patients'], 2)
        self.assertEqual(metadata['notebook_sha256'], hashlib.sha256(SOURCE.encode()).hexdigest())
        self.assertEqual(self.ns['biopsy_n'], 2)
        np.testing.assert_allclose(self.ns['psad_cases'].gt_psad, np.array([5., 12.]) / .168)
        first = self.ns['pairwise_statistics'].copy()
        start = SOURCE.index('# %% [markdown]\n# ## Paired post hoc inference:')
        end = SOURCE.index('# %% [markdown]\n# ## PSA density inputs')
        statistics = SOURCE[start:end].replace('N_BOOTSTRAP = 100_000', 'N_BOOTSTRAP = 99')
        statistics = statistics.replace('BOOTSTRAP_BATCH = 1_000', 'BOOTSTRAP_BATCH = 20')
        self.execute(statistics)
        pd.testing.assert_frame_equal(first, self.ns['pairwise_statistics'])
        for filename in ['comparator_agreement.pdf', 'psa_density_agreement.pdf',
                         'psa_density_biopsy_summary.csv', 'report_volume_quality_table.csv']:
            self.assertTrue((out / filename).is_file(), filename)


class ScatterVisibilityTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(plt.close, 'all')

    def report(self, source, ns):
        ns.update(fig=ns['axes'].flat[0].figure, plt=plt, OUTPUT_DIR=Path('/unused'))
        output = StringIO()
        with redirect_stdout(output), patch.object(ns['fig'], 'savefig') as savefig, \
                patch.object(plt, 'show', side_effect=lambda: print('PLOT_SHOWN')) as show:
            exec(source, ns)
        self.assertEqual(savefig.call_count, 2)
        show.assert_called_once()
        self.assertTrue(output.getvalue().startswith('PLOT_SHOWN\n'))
        return output.getvalue()

    def agreement_namespace(self):
        _, axes = plt.subplots(3, 2, sharex=True, sharey='row')
        for panel in axes.flat:
            panel.set_xscale('log')
        axes[0, 0].set_xlim(1, 10)
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_ylim(1, 10)
        axes[1, 0].set_ylim(-5, 5)
        axes[2, 0].set_ylim(-100, 100)
        methods = ['clinical', 'nnunet274']
        reference = pd.Series([1., 10.])
        values = pd.DataFrame({key: [1., 10.] for key in methods})
        differences = pd.DataFrame({key: [-5., 5.] for key in methods})
        relative = pd.DataFrame({key: [-100., 100.] for key in methods})
        identifiers = pd.DataFrame(dict(ID=['case0', 'case1'], aano=['accession0', 'accession1']))
        return dict(np=np, axes=axes, SELECTED_COMPARATORS=methods,
                    volumes=identifiers, psad_cases=identifiers,
                    gt=reference, measured_volumes=values, differences_ml=differences,
                    differences_percent=relative, psad_gt=reference, psad_values=values,
                    psad_differences=differences, psad_relative_differences=relative)

    def test_agreement_boundaries_are_inclusive(self):
        for guard in [VOLUME_VISIBILITY, PSAD_VISIBILITY]:
            self.assertEqual(self.report(guard, self.agreement_namespace()), 'PLOT_SHOWN\n')

    def test_agreement_reports_hidden_and_invalid_original_coordinates(self):
        cases = [
            ('gt', 0, .5), ('gt', 1, 11.),
            ('measured_volumes', 0, .5), ('measured_volumes', 1, 11.),
            ('measured_volumes', 0, 0.), ('measured_volumes', 0, -1.),
            ('measured_volumes', 0, np.nan), ('measured_volumes', 0, np.inf),
            ('gt', 0, 0.), ('gt', 0, -1.), ('gt', 0, np.nan), ('gt', 0, np.inf),
            ('differences_ml', 0, -6.), ('differences_ml', 1, 6.),
            ('differences_percent', 0, -101.), ('differences_percent', 1, 101.),
        ]
        for guard in [VOLUME_VISIBILITY, PSAD_VISIBILITY]:
            for field, index, value in cases:
                with self.subTest(figure=guard.splitlines()[0], field=field, value=value):
                    ns = self.agreement_namespace()
                    target = ns[field]
                    if isinstance(target, pd.DataFrame):
                        target.loc[index, 'nnunet274'] = value
                    else:
                        target.iloc[index] = value
                    # NaNs can be masked by Matplotlib; report the original inputs.
                    ns['axes'][0, 1].scatter(ns['gt'], ns['measured_volumes']['nnunet274'])
                    output = self.report(guard, ns)
                    self.assertRegex(output, r'1 / 2 hidden or invalid points; x limits=.*y limits=')
                    self.assertIn(f'case{index}', output)
                    self.assertIn(f'accession{index}', output)
                    self.assertNotIn(f'case{1-index}', output)
                    reports = ns.get('volume_plot_outliers', ns.get('psad_plot_outliers'))
                    for message, outliers in reports:
                        self.assertEqual(outliers.columns.tolist(), ['ID', 'aano', 'x', 'y'])
                        self.assertEqual(len(outliers), 1)
                        coordinate = 'x' if field == 'gt' else 'y'
                        actual = outliers[coordinate].iloc[0]
                        self.assertTrue(pd.isna(actual) if pd.isna(value) else actual == value)
                    plt.close('all')

    def test_agreement_checks_final_shared_limits(self):
        for guard in [VOLUME_VISIBILITY, PSAD_VISIBILITY]:
            ns = self.agreement_namespace()
            # Changing one panel updates the entire row after points were plotted.
            ns['axes'][2, 1].scatter(ns['gt'], ns['differences_percent']['nnunet274'])
            ns['axes'][2, 0].set_ylim(-50, 50)
            output = self.report(guard, ns)
            self.assertIn('relative difference: 2 / 2', output)
            self.assertIn('case0', output)
            self.assertIn('case1', output)

    def quality_namespace(self):
        _, axes = plt.subplots(1, 2, sharex=True, sharey=True)
        axes[0].set_xscale('log')
        axes[0].set_xlim(10, 300)
        axes[0].set_ylim(-40, 40)
        return dict(np=np, axes=axes,
                    cases=pd.DataFrame(dict(ID=['case0', 'case1'], aano=['accession0', 'accession1'])),
                    report_volume_quality=pd.DataFrame(dict(
                        mean_volume_ml=[1., 400.], difference_ml=[-50., 50.])),
                    quality_reference_ml=pd.Series([10., 300.]),
                    quality_reference_difference_ml=pd.Series([-40., 40.]))

    def test_quality_reports_labeled_clipping_and_inclusive_boundaries(self):
        ns = self.quality_namespace()
        output = self.report(QUALITY_VISIBILITY, ns)
        self.assertIn('report dimensions: 2 / 2', output)
        self.assertNotIn('nnUNet702 agreement:', output)
        self.assertEqual(ns['quality_plot_outliers'][0][1]['x'].tolist(), [1., 400.])
        self.assertEqual(ns['quality_plot_outliers'][0][1]['y'].tolist(), [-50., 50.])

    def test_quality_reports_unlabeled_clipping_and_invalid_labeled_data(self):
        for field, value in [
            ('quality_reference_ml', 9.), ('quality_reference_ml', 301.),
            ('quality_reference_difference_ml', -41.),
            ('quality_reference_difference_ml', 41.),
            ('mean_volume_ml', 0.), ('mean_volume_ml', -1.),
            ('mean_volume_ml', np.nan), ('difference_ml', np.inf),
        ]:
            with self.subTest(field=field, value=value):
                ns = self.quality_namespace()
                ns['report_volume_quality']['mean_volume_ml'] = [10., 300.]
                ns['report_volume_quality']['difference_ml'] = [-40., 40.]
                if field in ns:
                    ns[field].iloc[0] = value
                    panel_name = 'nnUNet702 agreement'
                else:
                    ns['report_volume_quality'].loc[0, field] = value
                    panel_name = 'report dimensions'
                output = self.report(QUALITY_VISIBILITY, ns)
                self.assertIn(f'{panel_name}: 1 / 2 hidden or invalid', output)
                self.assertIn('case0', output)
                self.assertIn('accession0', output)
                self.assertNotIn('case1', output)
                plt.close('all')


if __name__ == '__main__':
    unittest.main()
