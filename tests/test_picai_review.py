"""Small synthetic checks; no clinical data or inference required."""
import csv
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
import socket
from unittest.mock import patch
from threading import Thread, Event
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen
from werkzeug.serving import make_server

import nibabel as nib
import numpy as np
from PIL import Image

from src.picai_review import ReviewLog, create_app, load_case, plane, render_panel, render_slices


class ReviewTests(unittest.TestCase):
    def test_metadata_does_not_wait_for_pngs_and_ten_cases_are_prepared(self):
        with tempfile.TemporaryDirectory() as directory:
            case = self.fixture(directory)
            data, release, rendering = load_case(case), Event(), Event()
            cases = [dict(case, ID=f'case{i}') for i in range(12)]
            def slow_render(volume, cursor):
                rendering.set()
                if not release.wait(5):
                    raise TimeoutError('Test did not release PNG rendering')
                return render_slices(volume, cursor)
            with patch('src.picai_review.load_case', return_value=data), patch('src.picai_review.render_slices', side_effect=slow_render):
                app = create_app(cases=cases, log_path=Path(directory) / 'review.csv')
                with ThreadPoolExecutor(max_workers=3) as requests:
                    try:
                        first = requests.submit(lambda: app.test_client().post('/api/navigate', json={'target': 0})).result(2)
                        self.assertEqual(first.status_code, 200)
                        self.assertTrue(rendering.wait(2))
                        png = requests.submit(lambda: app.test_client().get(first.json['panels'][0]))
                        self.assertFalse(png.done())
                        second = requests.submit(lambda: app.test_client().post('/api/navigate', json={'target': 1})).result(2)
                        self.assertEqual(second.json['index'], 1)
                        self.assertEqual(set(app.extensions['picai_cache']), set(range(1, 11)))
                        self.assertFalse(png.done())
                    finally:
                        release.set()
                        for pool in app.extensions['picai_executors']:
                            pool.shutdown(wait=True)
                    self.assertEqual(png.result(2).mimetype, 'image/png')
                    self.assertTrue(all(e['panels'].done() and e['panels'].exception() is None
                                        for e in app.extensions['picai_cache'].values()))

    def test_navigation_saves_departing_case_and_returns_images_with_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            case = self.fixture(directory)
            cases = [case, dict(case, ID='case2')]
            path = Path(directory) / 'review.csv'
            client = create_app(cases=cases, log_path=path).test_client()
            first = client.post('/api/navigate', json=dict(target=0)).json
            self.assertEqual(first['index'], 0)
            self.assertEqual(len(first['panels']), 9)
            self.assertIn('spacing', first)
            np.testing.assert_allclose(first['volumes_ml'], [.192, .192, .192])
            self.assertFalse(path.exists())  # Opening a case does not save a review row.
            previous = dict(index=0, rating='unsure', note='check base')
            unknown = client.post('/api/navigate', json=dict(target_id='missing', previous=previous))
            self.assertEqual(unknown.status_code, 400)
            self.assertIn('Unknown case ID', unknown.json['error'])
            self.assertFalse(path.exists())
            second = client.post('/api/navigate', json=dict(target_id=' case2 ', previous=previous)).json
            self.assertEqual(second['ID'], 'case2')
            self.assertEqual(len(second['panels']), 9)
            log = ReviewLog(path, cases)
            self.assertEqual(list(log.rows), [case['ID']])
            self.assertEqual(log.rows[case['ID']]['note'], 'check base')
            self.assertEqual(log.resume(), 1)
            invalid = client.post('/api/navigate', json=dict(target=99, previous=previous))
            self.assertEqual(invalid.status_code, 400)
            self.assertEqual(len(ReviewLog(path, cases).rows), 1)
            previous = dict(index=1, rating='unreviewed', note='')
            self.assertTrue(client.post('/api/navigate', json=dict(target=2, previous=previous)).json['finished'])
            self.assertEqual(len(ReviewLog(path, cases).rows), 2)

    def test_unreviewed_visits_saved_separately_and_ratings_move_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            cases = [dict(ID='one', center='RUMC'), dict(ID='two', center='ZGT')]
            path = Path(directory) / 'review.csv'
            log = ReviewLog(path, cases)
            log.save(1, 'unreviewed', '')
            self.assertEqual(len(path.read_text().splitlines()), 1)
            with log.unreviewed_path.open() as f:
                self.assertEqual([r['ID'] for r in csv.DictReader(f)], ['two'])
            self.assertEqual(ReviewLog(path, cases).resume(), 1)
            log.save(0, 'incorrect', 'bad outline')
            log.save(1, 'unreviewed', '')
            restored = ReviewLog(path, cases)
            self.assertEqual(set(restored.rows), {'one', 'two'})
            self.assertEqual(restored.resume(), 1)
            log.save(0, 'unreviewed', '')
            self.assertEqual(len(ReviewLog(path, cases).rows), 2)
            self.assertEqual(len(path.read_text().splitlines()), 1)
            log.save(1, 'unsure', 'check apex')
            with log.unreviewed_path.open() as f:
                self.assertEqual([r['ID'] for r in csv.DictReader(f)], ['one'])
            with path.open() as f:
                self.assertEqual([r['ID'] for r in csv.DictReader(f)], ['two'])
            log.save(1, 'unreviewed', 'retained note')
            self.assertEqual(ReviewLog(path, cases).rows['two']['note'], 'retained note')

    def test_pre_render_reused_for_initial_views(self):
        with tempfile.TemporaryDirectory() as directory:
            case = self.fixture(directory)
            client = create_app(cases=[case], log_path=Path(directory) / 'review.csv').test_client()
            with patch('src.picai_review.render_slices', wraps=render_slices) as render:
                first = client.post('/api/navigate', json={'target': 0}).json
                z = first['cursor'][2]
                self.assertEqual(len(first['panels']), 9)
                png = client.get(first['panels'][0])
                self.assertEqual(png.mimetype, 'image/png')
                self.assertTrue(png.data.startswith(b'\x89PNG\r\n\x1a\n'))
                self.assertIn('private, max-age=86400, immutable', png.headers['Cache-Control'])
                self.assertEqual(client.get(first['panels'][0], headers={'If-None-Match': png.headers['ETag']}).status_code, 304)
                moved = first['panels'][0].rsplit('/', 1)[0] + f'/{z-1}.png'
                with patch('src.picai_review.render_panel', wraps=render_panel) as single:
                    client.get(moved)
                    client.get(moved)
                    self.assertEqual(single.call_count, 1)
                self.assertEqual(render.call_count, 1)
                expired = first['panels'][0].replace('/api/image/', '/api/image/expired-')
                self.assertEqual(client.get(expired).status_code, 404)

    def test_queue_moves_without_reloading_retained_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            case = self.fixture(directory)
            data = load_case(case)
            cases = [dict(case, ID=f'case{i}') for i in range(12)]
            with patch('src.picai_review.load_case', return_value=data) as load:
                app = create_app(cases=cases, log_path=Path(directory) / 'review.csv')
                client = app.test_client()
                try:
                    for target, count in [(0, 10), (1, 11), (0, 12)]:
                        self.assertEqual(client.post('/api/navigate', json={'target': target}).status_code, 200)
                        for entry in app.extensions['picai_cache'].values():
                            entry['volume'].result(timeout=2)
                        self.assertEqual(set(app.extensions['picai_cache']), set(range(target, target+10)))
                        self.assertEqual(load.call_count, count)
                finally:
                    for pool in app.extensions['picai_executors']:
                        pool.shutdown(wait=True)

    def test_idle_browser_connection_does_not_block_server(self):
        with tempfile.TemporaryDirectory() as directory:
            case = self.fixture(directory)
            app = create_app(cases=[case], log_path=Path(directory) / 'review.csv')
            server = make_server('127.0.0.1', 0, app, threaded=True)
            worker = Thread(target=server.serve_forever, daemon=True)
            worker.start()
            idle = socket.create_connection(('127.0.0.1', server.server_port))
            try:
                with urlopen(f'http://127.0.0.1:{server.server_port}/', timeout=2) as response:
                    self.assertEqual(response.status, 200)
            finally:
                idle.close()
                server.shutdown()
                server.server_close()
                worker.join(timeout=2)

    def fixture(self, directory):
        root = Path(directory)
        pixels = np.arange(4*5*6, dtype=np.float32).reshape(4, 5, 6)
        mask = np.zeros_like(pixels, dtype=np.uint8)
        mask[1:3, 2:4, 3:5] = 1
        affine = np.diag([-2., 3., 4., 1.])
        paths = []
        for i, array in enumerate([pixels, mask, mask, mask]):
            image = nib.Nifti1Image(array, affine)
            image.header.set_xyzt_units('mm')
            path = root / f'{i}.nii.gz'
            nib.save(image, path)
            paths.append(path)
        return dict(ID='case1', center='RUMC', paths=paths)

    def test_planes_and_physical_geometry(self):
        array = np.arange(4*5*6).reshape(4, 5, 6)
        for axis in [2, 0, 1]:
            other = [a for a in range(3) if a != axis]
            for index in [0, array.shape[axis]-1]:
                result = plane(array, axis, index)
                xyz = [0, 0, 0]
                xyz[axis] = index
                xyz[other[0]] = 1
                xyz[other[1]] = 2
                self.assertEqual(result[array.shape[other[1]]-1-2, 1], array[tuple(xyz)])
        with tempfile.TemporaryDirectory() as directory:
            case = self.fixture(directory)
            data = load_case(case)
            self.assertEqual(data['spacing'], [2, 3, 4])
            np.testing.assert_allclose(data['volumes_ml'], [.192, .192, .192])
            self.assertEqual(data['center'], [2, 2, 4])
            self.assertGreater(data['gray'][0, 0, 0], data['gray'][-1, 0, 0])  # x flipped into RAS
            panels = render_slices(data, data['center'])
            self.assertEqual(len(panels), 9)
            image = Image.open(BytesIO(panels[0]))
            self.assertEqual(image.size, (4, 5))
            self.assertTrue(np.any(np.all(np.asarray(image) == [0, 213, 255], axis=2)))
            empty = nib.load(case['paths'][1])
            nib.save(nib.Nifti1Image(np.zeros(empty.shape, dtype=np.uint8), empty.affine, empty.header), case['paths'][1])
            empty_data = load_case(case)
            self.assertEqual(empty_data['center'], [2, 2, 3])
            np.testing.assert_allclose(empty_data['volumes_ml'], [0, .192, .192])
            bad = nib.load(case['paths'][3])
            bad_affine = bad.affine.copy()
            bad_affine[0, 3] += 10
            nib.save(nib.Nifti1Image(bad.get_fdata(), bad_affine, bad.header), case['paths'][3])
            with self.assertRaisesRegex(ValueError, 'geometry'):
                load_case(case)

    def test_log_and_endpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            case = self.fixture(directory)
            cases = [case, dict(case, ID='case2')]
            path = Path(directory) / 'review.csv'
            log = ReviewLog(path, cases)
            self.assertEqual(log.resume(), 0)
            log.save(1, 'incorrect', 'comma, newline\nquoted "note"')
            log.save(0, 'unsure', '')
            log.save(1, 'acceptable', 'revised')
            restored = ReviewLog(path, cases)
            self.assertEqual(len(restored.rows), 2)
            self.assertEqual(restored.resume(), 1)
            self.assertEqual(restored.rows['case2']['note'], 'revised')
            client = create_app(cases=cases, log_path=path).test_client()
            self.assertEqual(client.get('/').status_code, 200)
            self.assertEqual(client.get('/case/1').status_code, 200)
            self.assertEqual(client.get('/case/2').status_code, 200)
            self.assertEqual(client.get('/case/0').status_code, 400)
            self.assertEqual(client.get('/case/3').status_code, 400)
            with self.assertLogs('src.picai_review', level='INFO') as captured:
                response = client.post('/api/navigate', json={'target': None})
            self.assertEqual(response.headers['Cache-Control'], 'no-store')
            messages = '\n'.join(captured.output)
            for text in ['START POST', 'Cache miss', 'END status=200']:
                self.assertIn(text, messages)
            self.assertEqual(response.json['index'], 1)
            self.assertEqual(response.json['center'], 'RUMC')
            self.assertEqual(len(response.json['panels']), 9)
            url = response.json['panels'][0]
            self.assertEqual(client.get(url.rsplit('/', 1)[0] + '/999.png').status_code, 400)
            invalid = dict(index=1, rating='bad', note='')
            self.assertEqual(client.post('/api/navigate', json=dict(target=0, previous=invalid)).status_code, 400)
            previous = dict(index=1, rating='incorrect', note='check')
            self.assertEqual(client.post('/api/navigate', json=dict(target=0, previous=previous)).status_code, 200)
            self.assertEqual(ReviewLog(path, cases).resume(), 0)
            for pool in client.application.extensions['picai_executors']:
                pool.shutdown(wait=True)
            case['paths'][0].unlink()
            # A new app has no cached case; a missing file still supplies review metadata.
            client = create_app(cases=cases, log_path=path).test_client()
            self.assertIn('error', client.post('/api/navigate', json={'target': 0}).json)


if __name__ == '__main__':
    unittest.main()
