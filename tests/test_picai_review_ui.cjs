// Run with node tests/test_picai_review_ui.cjs; exercises the actual inline JS.
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const path = require('node:path');
const handlers = {}, elements = {}, requests = [], saved = [], logs = [], imageURLs = [];
let failNavigation = false, holdImages = false, heldImages = [], active = 0, maxActive = 0;
let drawCount = 0;
class Element {
  constructor(tag = 'div') {
    Object.assign(this, {tag, value: '', textContent: '', style: {}, clientWidth: 400,
      clientHeight: 250, children: [], events: {}});
  }
  append(...v) { this.children.push(...v); }
  addEventListener(k, fn) { this.events[k] = fn; }
  closest() { return ['input', 'select', 'button'].includes(this.tag) ? this : null; }
  getContext() { return new Proxy({}, {get: (_, key) => () => { if (key === 'drawImage') drawCount++; }}); }
  getBoundingClientRect() { return {left: 0, top: 0, width: 400, height: 250}; }
}
for (const id of ['title', 'prev', 'next', 'rating', 'note', 'status', 'grid', 'jump', 'case-id', 'go']) {
  elements[id] = new Element(id === 'note' ? 'input' : id === 'rating' ? 'select' : 'div');
}
const response = data => ({ok: true, status: 200, json: async () => data});
const fetch = async (url, options = {}) => {
  requests.push(url); assert.equal(url, '/api/navigate');
  if (failNavigation) return {ok: false, status: 400, json: async () => ({error: 'disk full'})};
  const body = JSON.parse(options.body);
  if (body.previous) saved.push(body.previous);
  if (body.target === 880) return response({finished: true});
  const index = body.target_id !== undefined ? Number(body.target_id) : body.target ?? 10;
  return response({index, total: 880, ID: String(index), center: 'RUMC',
    panels: Array.from({length: 9}, (_, p) => `/api/image/g/${index}/${p}/50.png`),
    rating: 'unreviewed', note: '', shape: [100, 100, 100], spacing: [2, 3, 4], volumes_ml: [index + 42.34, 0, 30.56], cursor: [50, 50, 50]});
};
class Image { set src(value) {
  assert(!value.startsWith('data:')); imageURLs.push(value);
  if (holdImages) {
    active++; maxActive = Math.max(active, maxActive);
    heldImages.push((fail = false) => { active--; fail ? this.onerror() : this.onload(); });
  } else queueMicrotask(() => this.onload());
} }
const context = vm.createContext({
  document: {getElementById: id => elements[id], createElement: tag => new Element(tag),
    addEventListener: (k, fn) => handlers[k] = fn},
  window: {addEventListener: (k, fn) => handlers[k] = fn, location: {pathname: '/case/1'},
    history: {replaceState: (_, __, url) => elements.url = url}},
  fetch, Image, devicePixelRatio: 1, AbortController, performance,
  setTimeout, clearTimeout, console: {log: (...v) => logs.push(v.join(' '))},
});
const html = fs.readFileSync(path.join(__dirname, '../src/picai_review.html'), 'utf8');
assert(!html.includes('id="debug"'));
vm.runInContext(html.split('<script>')[1].split('</script>')[0], context);
const pause = () => new Promise(resolve => setTimeout(resolve, 80));
const completeImages = () => heldImages.splice(0).forEach(complete => complete());
const wheel = deltaY => elements.grid.children[0].children[0].events.wheel({deltaY, preventDefault() {}});
const key = (target, value) => handlers.keydown({target, key: value, preventDefault() {}});
const labels = () => elements.grid.children.map(panel => panel.children[1].textContent);
const view = () => vm.runInContext('JSON.stringify({index, info, displayedCursor})', context);
(async () => {
  await pause(); assert.deepEqual(requests, ['/api/navigate']); assert.equal(saved.length, 0);
  assert.equal(elements.url, '/case/1'); // Explicit URL overrides resume at case 11.
  assert.deepEqual(labels().slice(0, 3), ['Axial · Bosma22b · Vol = 42.3 mL',
    'Axial · Multicenter Dataset274 · Vol = 0.0 mL', 'Axial · RUMC Dataset236 · Vol = 30.6 mL']);
  assert(labels().slice(3).every(label => !label.includes('Vol =')));
  const initial = imageURLs.slice(); holdImages = true;
  const initialView = view(), initialDraws = drawCount;
  wheel(1); await pause(); assert.equal(heldImages.length, 9);
  heldImages.shift()(); await pause();
  assert.equal(drawCount, initialDraws); assert.equal(view(), initialView);
  for (let i = 0; i < 100; i++) wheel(1);
  await pause(); assert.equal(heldImages.length, 8); assert.equal(maxActive, 9);
  completeImages(); await pause();
  assert.equal(drawCount, initialDraws); // Superseded slice results never draw.
  assert(imageURLs.includes('/api/image/g/0/0/99.png'));
  assert.deepEqual(imageURLs.slice(-6), initial.slice(-6)); // Other planes keep their URLs.
  completeImages(); await pause(); assert(logs.some(line => line.includes('drawn')));
  assert.equal(drawCount, initialDraws + 9);
  const before = imageURLs.length;
  for (let i = 0; i < 100; i++) wheel(1);
  await pause(); assert.equal(imageURLs.length, before); assert.equal(requests.length, 1);
  wheel(-1); await pause(); assert.equal(heldImages.length, 9);
  key(elements.note, 'ArrowRight'); await pause(); assert(elements.title.textContent.includes('1 / 880'));
  elements.note.value = 'keep this'; elements.note.oninput();
  elements.rating.value = 'unsure'; elements.rating.onchange();
  assert.equal(saved.length, 0); assert.equal(elements.note.onblur, undefined);
  holdImages = false;
  key(elements.grid, 'ArrowRight'); await pause();
  assert.equal(elements.url, '/case/2'); assert.equal(requests.length, 2);
  assert.deepEqual(saved[0], {index: 0, rating: 'unsure', note: 'keep this'});
  const afterNavigation = imageURLs.length;
  const afterNavigationDraws = drawCount;
  completeImages(); await pause(); assert.equal(imageURLs.length, afterNavigation);
  assert.equal(drawCount, afterNavigationDraws);
  wheel(-1); await pause(); assert(imageURLs.includes('/api/image/g/1/0/49.png'));
  assert.equal(requests.length, 2); // Scroll only loads PNGs, never calls a JSON endpoint.
  failNavigation = true; elements.note.value = 'unsaved'; elements.note.oninput();
  key(elements.grid, 'ArrowRight'); await pause();
  assert.equal(elements.note.value, 'unsaved'); assert.equal(elements.url, '/case/2');
  assert(elements.status.textContent.includes('disk full')); failNavigation = false;
  await context.navigate(879); assert.equal(elements.next.textContent, 'Finish ✓');
  elements.next.onclick(); await pause(); assert.equal(elements.status.textContent, 'Saved. Review finished.');
  assert.equal(saved.at(-1).index, 879); holdImages = true;
  const oldLabels = labels();
  const oldView = view(), oldTitle = elements.title.textContent, oldDraws = drawCount;
  const navigation = context.navigate(2); await pause();
  assert.deepEqual(labels(), oldLabels);
  assert.equal(view(), oldView); assert.equal(elements.title.textContent, oldTitle);
  assert.equal(elements.next.disabled, true); assert.equal(heldImages.length, 9);
  assert.equal(elements.status.textContent, 'Loading…');
  const requestsBefore = requests.length;
  wheel(1); elements.next.onclick();
  assert.equal(requests.length, requestsBefore);
  heldImages.shift()(); await pause();
  assert.equal(view(), oldView); assert.equal(drawCount, oldDraws);
  completeImages(); await navigation;
  assert.equal(elements.url, '/case/3'); assert.equal(drawCount, oldDraws + 9);
  assert.equal(elements.next.disabled, false);
  assert.equal(labels()[0], 'Axial · Bosma22b · Vol = 44.3 mL');
  const readyLabels = labels();
  const readyView = view(), readyDraws = drawCount;
  const failedNavigation = context.navigate(3); await pause();
  heldImages.shift()(true); completeImages(); await failedNavigation;
  assert.deepEqual(labels(), readyLabels);
  assert.equal(view(), readyView); assert.equal(drawCount, readyDraws);
  assert.equal(elements.url, '/case/3'); assert.equal(elements.next.disabled, false);
  assert(elements.status.textContent.includes('PNG loading failed'));
  wheel(1); await pause();
  heldImages.shift()(true); completeImages(); await pause();
  assert.deepEqual(labels(), readyLabels);
  assert.equal(view(), readyView); assert.equal(drawCount, readyDraws);
  assert(elements.status.textContent.includes('Image error'));
  holdImages = false;
  elements['case-id'].value = ' 42 ';
  elements.note.value = 'before jump';
  elements.jump.onsubmit({preventDefault() {}}); await pause();
  assert.equal(elements.url, '/case/43');
  assert.equal(elements['case-id'].value, '42');
  assert.equal(saved.at(-1).note, 'before jump');
  console.log('PASS: atomic panel swaps, delayed and failed loads, stale images, scrolling, save-on-leave, Finish, ID jump');
})().catch(error => { console.error(error); process.exitCode = 1; });
