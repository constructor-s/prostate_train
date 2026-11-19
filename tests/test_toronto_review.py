"""Synthetic review checks; no inference required."""
import csv
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
from PIL import Image
from src import toronto_review as review


class TorontoReviewTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        base = root / 'data_internal/hlab20260928'
        base.mkdir(parents=True)
        listing = base / 'hlab20260928_t2_image_listing_full_clinical.csv'
        with listing.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['ID', 't2', 'ground_truth_volume_ml', 'vol_report_ref'])
            writer.writeheader()
            writer.writerow(dict(ID='one', t2='PROSATEPSAD/one/axt2.nii.gz', vol_report_ref=20))
        case = review.listing_cases(root)[0]
        for i, path in enumerate(case['paths']):
            path.parent.mkdir(parents=True, exist_ok=True)
            data = np.zeros((6, 7, 8), dtype=np.uint8)
            data[2:4, 2:4, 2:4] = 1
            if i == 1:
                data[3, 3, 3] = 3
            image = nib.Nifti1Image(data, np.diag([2., 3., 4., 1.]))
            image.header.set_xyzt_units('mm')
            nib.save(image, path)
        return root, case

    def test_index_reuse_stale_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root, case = self.fixture(directory)
            cases = review.read_cases(root, workers=2)
            self.assertAlmostEqual(float(cases[0]['volumes']['nnunet702_volume_ml']), .192)
            with patch.object(review, 'measure_case', side_effect=AssertionError('must reuse')):
                self.assertEqual(review.read_cases(root)[0]['ID'], 'one')
            image = nib.load(case['paths'][1])
            data = image.get_fdata(); data[0, 0, 0] = 9
            nib.save(nib.Nifti1Image(data, image.affine, image.header), case['paths'][1])
            with self.assertRaisesRegex(ValueError, 'stale'):
                review.read_cases(root)
            cases = review.read_cases(root, rebuild=True)
            self.assertIn('Expected labels', cases[0]['volumes']['nnunet702_error'])

    def test_geometry_and_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            _, case = self.fixture(directory)
            images = review.validated_images(case)
            images[0].header.set_xyzt_units('unknown')
            review.validate_geometry(images[0], images[1], images[2:])
            for image in images:
                image.header.set_xyzt_units('unknown')
            with self.assertRaisesRegex(ValueError, 'anchor'):
                review.validate_geometry(images[0], images[1], images[2:])
            images[0].header.set_xyzt_units('mm'); images[1].header.set_xyzt_units('mm')
            images[1].affine[0, 3] += 1
            with self.assertRaisesRegex(ValueError, 'displacement'):
                review.validate_geometry(images[0], images[1], images[2:])
            image = nib.load(case['paths'][1])
            nib.save(nib.Nifti1Image(np.zeros(image.shape, dtype=np.uint8), image.affine, image.header), case['paths'][1])
            self.assertEqual(review.measure_case(case)['nnunet702_volume_ml'], 0)
            self.assertEqual(review.load_case(case)['center'], [3, 3, 4])

    def test_sorting_and_small_geometry_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            root, case = self.fixture(directory)
            listing = root / 'data_internal/hlab20260928/hlab20260928_t2_image_listing_full_clinical.csv'
            with listing.open('a') as stream:
                stream.write('missing,missing.nii.gz,,\nzero,zero.nii.gz,,\n')
            def measurement(case):
                row = dict(ID=case['ID'])
                for model in review.MODELS:
                    row.update({model + '_volume_ml': {'one': 4, 'zero': 0, 'missing': ''}[case['ID']],
                                model + '_path': '', model + '_error': ''})
                return row
            with patch.object(review, 'measure_case', side_effect=measurement):
                self.assertEqual([c['ID'] for c in review.read_cases(root, workers=2)], ['zero', 'one', 'missing'])
            image = nib.load(case['paths'][2])
            affine = image.affine.copy(); affine[0, 3] += .1
            nib.save(nib.Nifti1Image(np.asanyarray(image.dataobj), affine, image.header), case['paths'][2])
            self.assertEqual(review.load_case(case)['gray'].shape, (6, 7, 8))

    def test_overlay_navigation_and_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root, case = self.fixture(directory)
            data = review.load_case(case)
            png = review.render_panel(data, 2, 2)
            rgb = np.asarray(Image.open(BytesIO(png)))
            blended = np.rint(np.mean(review.COLORS, axis=0)).astype(np.uint8)
            self.assertTrue(np.any(np.all(rgb == blended, axis=2)))
            self.assertEqual(len(review.render_slices(data, data['center'])), 3)
            broken = dict(case, ID='broken', paths=[root / 'missing'] * 4)
            app = review.create_app(root=root, cases=[case, broken])
            try:
                client = app.test_client()
                first = client.post('/api/navigate', json={'target': 0}).json
                self.assertEqual(len(first['panels']), 3)
                self.assertEqual(client.get(first['panels'][0]).mimetype, 'image/png')
                failed = client.post('/api/navigate', json={'target': 1, 'previous': dict(index=0, rating='unsure', note='check')}).json
                self.assertIn('error', failed)
                resumed = client.post('/api/navigate', json={'target': None}).json
                self.assertEqual(resumed['ID'], 'broken')
                self.assertEqual(review.ReviewLog(root / 'results/toronto_manual_review/review.csv', [case, broken]).rows['one']['note'], 'check')
            finally:
                for pool in app.extensions['toronto_executors']:
                    pool.shutdown()
