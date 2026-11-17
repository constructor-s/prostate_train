"""One-user PI-CAI mask review. Run: .venv/bin/python src/picai_review.py"""
import csv
from datetime import datetime, timezone
from io import BytesIO
import json
import logging
from pathlib import Path
from time import perf_counter
from itertools import count
from threading import RLock
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from hashlib import sha256
from uuid import uuid4

from flask import Flask, Response, g, jsonify, render_template, request
import nibabel as nib
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['ID', 'center', 'rating', 'note', 'visited_at']
RATINGS = {'unreviewed', 'acceptable', 'incorrect', 'unsure'}
COLORS = [(0, 213, 255), (80, 224, 112), (255, 157, 48)]
AXES = [2, 0, 1]  # axial, sagittal, coronal
LOGGER = logging.getLogger(__name__)


def read_cases(root):
    manifest = json.loads((root / 'configs/picai_test_clinical_folds.json').read_text())
    entries = [entry for centre in ('PCNN', 'RUMC', 'ZGT')
               for entry in manifest[f'testing_{centre}']]
    assert entries and len({e['id'] for e in entries}) == len(entries)
    with (root / 'results/picai_test_clinical_volume/comparator_volumes.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len({r['ID'] for r in rows}) == len(rows), 'Duplicate comparator IDs'
    metadata = {r['ID']: r for r in rows}
    cases = []
    for entry in entries:
        row = metadata[entry['id']]
        cases.append(dict(ID=entry['id'], center=row['center'], paths=[
            root / 'data/raw' / entry['image'], root / 'data/raw' / entry['label'],
            root / row['nnunet274_path'], root / row['rumcnotprostatexnoclinical236_path']]))
    return sorted(cases, key=lambda case: case['center'])  # Stable within each centre.


def load_case(case):
    LOGGER.info('Loading %s: reading four NIfTI files', case['ID'])
    images = [nib.load(p) for p in case['paths']]
    source = images[0]
    if len(source.shape) != 3:
        raise ValueError('Expected a 3D T2 volume')
    for image in images:
        if image.shape != source.shape or not np.allclose(image.affine, source.affine, atol=1e-3, rtol=0):
            raise ValueError('Image/mask geometry mismatch')
        if image.header.get_xyzt_units()[0] != 'mm':
            raise ValueError('Expected millimetre spatial units')
    # Use ONE transform, avoiding near-oblique header roundoff differences.
    orientation = nib.orientations.io_orientation(source.affine)
    images = [image.as_reoriented(orientation) for image in images]
    LOGGER.info('Loading %s: decoding T2 and masks', case['ID'])
    pixels = images[0].get_fdata(dtype=np.float32)
    masks = []
    for image in images[1:]:
        values = np.asanyarray(image.dataobj)
        if not np.isin(values, [0, 1]).all():
            raise ValueError('Expected binary whole-gland label 1')
        masks.append(values == 1)
    finite = pixels[np.isfinite(pixels)]
    LOGGER.info('Loading %s: normalizing contrast, shape=%s', case['ID'], pixels.shape)
    if not finite.size:
        raise ValueError('T2 has no finite intensities')
    low, high = np.percentile(finite, [1, 99])
    pixels = np.nan_to_num((pixels - low) / max(float(high-low), 1e-6), nan=0, posinf=1, neginf=0)
    gray = (np.clip(pixels, 0, 1) * 255).astype(np.uint8)
    coordinates = np.nonzero(masks[0])
    center = [int(np.rint(c.mean())) for c in coordinates] if masks[0].any() else [n//2 for n in gray.shape]
    return dict(gray=gray, masks=masks, center=center,
                spacing=nib.affines.voxel_sizes(images[0].affine).tolist())


def plane(array, axis, index):
    """Horizontal coordinate increases rightward; vertical coordinate increases upward."""
    return np.flipud(np.take(array, index, axis=axis).T)


def render_slices(volume, cursor):
    return [render_panel(volume, axis, cursor[axis], model) for axis in AXES for model in range(3)]


def render_panel(volume, axis, index, model):
    gray = plane(volume['gray'], axis, index)
    layer = plane(volume['masks'][model], axis, index)
    padded = np.pad(layer, 1)
    interior = (padded[:-2, 1:-1] & padded[2:, 1:-1]
                & padded[1:-1, :-2] & padded[1:-1, 2:])
    rgb = np.repeat(gray[..., None], 3, axis=2)
    rgb[layer & ~interior] = COLORS[model]
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format='PNG')
    return buffer.getvalue()


class ReviewLog:
    def __init__(self, path, cases):
        self.path = path
        self.unreviewed_path = path.with_name(path.stem + '_unreviewed.csv')
        self.progress_path = path.with_suffix('.last_case')
        self.cases = cases
        self.rows = {}
        for source in [self.unreviewed_path, path]:
            if source.exists():
                with source.open(newline='') as f:
                    self.rows.update({r['ID']: r for r in csv.DictReader(f)})

    def resume(self):
        if self.progress_path.exists():
            last_id = self.progress_path.read_text().strip()
            for index, case in enumerate(self.cases):
                if case['ID'] == last_id:
                    return index
        return max(range(len(self.cases)), key=lambda i: self.rows.get(self.cases[i]['ID'], {}).get('visited_at', ''))

    def save(self, index, rating, note):
        case = self.cases[index]
        row = dict(ID=case['ID'], center=case['center'], rating=rating, note=note,
                   visited_at=datetime.now(timezone.utc).isoformat())
        rows = {**self.rows, case['ID']: row}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for path, unreviewed in [(self.path, False), (self.unreviewed_path, True)]:
            temporary = path.with_suffix('.tmp')
            with temporary.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(r for r in rows.values() if (r['rating'] == 'unreviewed') == unreviewed)
            temporary.replace(path)
        self.rows = rows
        self.visit(index)

    def visit(self, index):
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.progress_path.with_suffix('.tmp')
        temporary.write_text(self.cases[index]['ID'] + '\n')
        temporary.replace(self.progress_path)


def create_app(root=ROOT, cases=None, log_path=None):
    app = Flask(__name__, template_folder=str(Path(__file__).parent))
    app.logger.setLevel(logging.INFO)
    request_ids = count(1)
    state_lock = RLock()
    image_generation = uuid4().hex

    @app.before_request
    def request_start():
        g.request_id, g.started = next(request_ids), perf_counter()
        app.logger.info('[%s] START %s %s', g.request_id, request.method, request.full_path)

    @app.after_request
    def request_end(response):
        response.headers.setdefault('Cache-Control', 'no-store')
        app.logger.info('[%s] END status=%s elapsed=%.3fs bytes=%s', g.request_id,
                        response.status_code, perf_counter()-g.started, response.content_length)
        return response

    cases = read_cases(root) if cases is None else cases
    case_indices = {case['ID']: index for index, case in enumerate(cases)}
    log = ReviewLog(log_path or root / 'results/picai_manual_review/review.csv', cases)
    loader = ThreadPoolExecutor(max_workers=2, thread_name_prefix='picai-load')
    renderer = ThreadPoolExecutor(max_workers=1, thread_name_prefix='picai-render')
    cache = OrderedDict()
    app.extensions['picai_cache'] = cache
    app.extensions['picai_executors'] = (loader, renderer)

    def prepare_panels(index, volume_future):
        data = volume_future.result()
        started = perf_counter()
        app.logger.info('Pre-rendering case %s (%s) at centroid xyz=%s',
                        index, cases[index]['ID'], data['center'])
        panels = render_slices(data, data['center'])
        app.logger.info('Pre-render complete: case %s, nine panels in %.3fs', index, perf_counter()-started)
        return panels

    def cached_case(index, reason='view'):
        with state_lock:
            app.logger.info('Cache %s: case %s (%s), reason=%s',
                            'hit' if index in cache else 'miss', index, cases[index]['ID'], reason)
            if index not in cache:
                if len(cache) == 10:
                    evicted, old = cache.popitem(last=False)
                    old['panels'].cancel()
                    old['volume'].cancel()
                    app.logger.info('Cache evicted: case %s (%s)', evicted, cases[evicted]['ID'])
                future = loader.submit(load_case, cases[index])
                cache[index] = dict(volume=future, panels=renderer.submit(prepare_panels, index, future),
                                    slices=OrderedDict(), lock=RLock())
            cache.move_to_end(index)
            contents = []
            for key, entry in cache.items():
                states = [f'{name}={"failed" if f.done() and (f.cancelled() or f.exception()) else "ready" if f.done() else "pending"}'
                          for name, f in [("volume", entry['volume']), ("PNGs", entry['panels'])]]
                contents.append(f'{key} (#{key+1}) {cases[key]["ID"]} [{", ".join(states)}]')
            app.logger.info('Cache LRU → MRU (%s/10): %s', len(cache), '; '.join(contents))
            return cache[index]

    def preload_window(index):
        start = min(index, max(0, len(cases)-10))
        window = list(range(start, min(start+10, len(cases))))
        with state_lock:
            for key in list(cache):
                if key not in window:
                    old = cache.pop(key)
                    old['panels'].cancel()
                    old['volume'].cancel()
                    app.logger.info('Cache evicted outside window: case %s', key)
            for upcoming in [index] + [key for key in window if key != index]:
                cached_case(upcoming, reason='preload central views')

    def index_value(value):
        index = int(value)
        if not 0 <= index < len(cases):
            raise ValueError('Case index out of range')
        return index

    def volume(index):
        started = perf_counter()
        entry = cached_case(index)
        data = entry['volume'].result()
        app.logger.info('[%s] Volume metadata ready in %.3fs (does not wait for PNGs)', g.request_id, perf_counter()-started)
        return data

    def panel_urls(index, cursor):
        return [f'/api/image/{image_generation}/{index}/{panel}/{cursor[AXES[panel//3]]}.png'
                for panel in range(9)]

    @app.errorhandler(Exception)
    def error(exc):
        app.logger.exception('[%s] Request failed', g.get('request_id', '?'))
        return jsonify(error=str(exc)), 400

    @app.get('/')
    def home():
        return render_template('picai_review.html')

    @app.get('/case/<int:number>')
    def case_page(number):
        index_value(number - 1)  # URL numbers match the one-based viewer counter.
        return render_template('picai_review.html')

    def case_data(index):
        index = index_value(index)
        case = cases[index]
        with state_lock:
            record = log.rows.get(case['ID'], {})
        info = dict(ID=case['ID'], center=case['center'], rating=record.get('rating', 'unreviewed'),
                    note=record.get('note', ''))
        try:
            preload_window(index)
            data = volume(index)
            info.update(shape=list(data['gray'].shape), spacing=data['spacing'], cursor=data['center'])
            info['panels'] = panel_urls(index, data['center'])
            app.logger.info('[%s] Returning metadata + nine binary image URLs: case %s', g.request_id, index)
        except Exception as exc:
            app.logger.exception('[%s] Case load failed: %s', g.request_id, case['ID'])
            info['error'] = str(exc)
        return info

    def save_record(body):
        index = index_value(body['index'])
        if body['rating'] not in RATINGS or not isinstance(body['note'], str):
            raise ValueError('Invalid review record')
        app.logger.info('[%s] Saving departing case %s rating=%s (note omitted)',
                        g.request_id, cases[index]['ID'], body['rating'])
        with state_lock:
            log.save(index, body['rating'], body['note'])

    @app.post('/api/navigate')
    def navigate():
        body = request.get_json()
        with state_lock:
            if 'target_id' in body:
                case_id = body['target_id'].strip()
                if case_id not in case_indices:
                    raise ValueError(f'Unknown case ID: {case_id}')
                target = case_indices[case_id]
            else:
                target = log.resume() if body.get('target') is None else int(body['target'])
        finished = target == len(cases) and body.get('previous') is not None
        if not finished:
            index_value(target)
        elif index_value(body['previous']['index']) != len(cases) - 1:
            raise ValueError('Finish is available on the last case only')
        if body.get('previous') is not None:
            save_record(body['previous'])
        if finished:
            return jsonify(finished=True)
        info = case_data(target)
        with state_lock:
            log.visit(target)  # Resume pointer only; no review row until navigating away.
        return jsonify(**info, index=target, total=len(cases))

    @app.get('/api/image/<generation>/<int:index>/<int:panel>/<int:slice_index>.png')
    def image(generation, index, panel, slice_index):
        if generation != image_generation or not 0 <= panel < 9:
            return jsonify(error='Invalid or expired image URL; refresh the viewer'), 404
        entry = cached_case(index_value(index))
        data = entry['volume'].result()
        axis, model = AXES[panel//3], panel % 3
        if not 0 <= slice_index < data['gray'].shape[axis]:
            raise ValueError('Slice out of range')
        key = (panel, slice_index)
        if slice_index == data['center'][axis]:
            png = entry['panels'].result()[panel]  # Wait here, never in navigation or under state_lock.
            app.logger.info('[%s] Pre-render cache hit: panel %s', g.request_id, panel)
        else:
            with entry['lock']:
                if key not in entry['slices']:
                    entry['slices'][key] = render_panel(data, axis, slice_index, model)
                entry['slices'].move_to_end(key)
                png = entry['slices'][key]
                if len(entry['slices']) > 27:
                    entry['slices'].popitem(last=False)
        response = Response(png, mimetype='image/png')
        response.headers['Cache-Control'] = 'private, max-age=86400, immutable'
        response.set_etag(sha256(png).hexdigest())
        return response.make_conditional(request)

    return app


if __name__ == '__main__':
    import faulthandler
    import signal
    from gunicorn.app.base import BaseApplication

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    class ReviewServer(BaseApplication):
        def load_config(self):
            for key, value in dict(bind='127.0.0.1:5000', workers=1, worker_class='gthread',
                                   threads=16, keepalive=30, timeout=120, accesslog='-', errorlog='-').items():
                self.cfg.set(key, value)

        def load(self):
            faulthandler.register(signal.SIGUSR1)  # Use the worker PID for stack dumps.
            return create_app()

    ReviewServer().run()
