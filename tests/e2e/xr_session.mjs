// End-to-end test: runs the app in headless Chromium inside an EMULATED Meta Quest 3 WebXR session (Meta's IWER runtime)
// and checks hover / select / grab / model switch / hands. Usage:  cd tests/e2e && npm install && npm test
import chromium from '@sparticuz/chromium';
import puppeteer from 'puppeteer-core';
import http from 'node:http'; import fs from 'node:fs'; import path from 'node:path'; import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const root = path.resolve(new URL('../../docs', import.meta.url).pathname);
const srv = http.createServer((q, r) => {
  let p = decodeURIComponent(q.url.split('?')[0]); if (p.endsWith('/')) p += 'index.html';
  fs.readFile(path.join(root, p), (e, d) => { if (e) { r.writeHead(404); r.end(); return; }
    r.writeHead(200, { 'content-type': p.endsWith('.js') ? 'text/javascript' : p.endsWith('.html') ? 'text/html' : 'application/octet-stream' }); r.end(d); });
}).listen(0);
const port = srv.address().port;
const iwer = fs.readFileSync(path.join(path.dirname(require.resolve('iwer/package.json')), 'build/iwer.js'), 'utf8');
const browser = await puppeteer.launch({ executablePath: await chromium.executablePath(), headless: 'shell',
  args: [...chromium.args, '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--no-sandbox'], defaultViewport: { width: 1280, height: 720 } });
const page = await browser.newPage();
const errors = []; page.on('pageerror', (e) => errors.push(e.message));
await page.evaluateOnNewDocument(iwer + `; const dev = new IWER.XRDevice(IWER.metaQuest3); dev.installRuntime({ forceInstall: true }); window.__xrDevice = dev;`);
await page.goto(`http://localhost:${port}/index.html`, { waitUntil: 'load' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const ev = (f, a) => page.evaluate(f, a);
await sleep(3000);
assert.equal(await ev(() => !document.getElementById('enter').disabled), true, 'ENTER VR enabled');
await page.click('#enter'); await sleep(2500);
assert.equal(await ev(() => window.__tv.renderer.xr.isPresenting), true, 'XR session running');
await ev(() => { window.__xrDevice.primaryInputMode = 'controller'; }); await sleep(800);
const aim = (t) => ev((t) => { const tv = window.__tv, dev = window.__xrDevice, V = tv.THREE.Vector3, Q = tv.THREE.Quaternion, hp = dev.position;
  const o = new V(0.15, -0.25, -0.35), tg = t.kind === 'node' ? tv.nodeWorld(t.id) : t.kind === 'btn' ? tv.buttonWorld(t.id) : new V(t.x, t.y, t.z);
  const q = new Q().setFromUnitVectors(new V(0, 0, -1), tg.clone().sub(o).normalize()), c = dev.controllers.right;
  c.position.set(o.x + hp.x, o.y + hp.y, o.z + hp.z); c.quaternion.set(q.x, q.y, q.z, q.w); }, t);
const trigger = async (v) => { await ev((v) => window.__xrDevice.controllers.right.updateButtonValue('trigger', v), v); await sleep(450); };
const stem = await ev(() => window.__tv.ui.model.nodes.findIndex((n) => n.name === 'stem_conv'));
await aim({ kind: 'node', id: stem }); await sleep(700);
assert.match(await ev(() => window.__tv.ui.hoverLabel), /stem_conv/, 'hover shows the layer');
await trigger(1); await trigger(0);
assert.equal(await ev(() => window.__tv.st.selected), stem, 'pinch/trigger selects the layer');
const x0 = await ev(() => window.__tv.st.graphPos.x);
await aim({ kind: 'point', x: 0.9, y: 1.0, z: -2.4 }); await sleep(400); await trigger(1);
for (let i = 0; i < 10; i++) { await ev(() => { window.__xrDevice.controllers.right.position.x += 0.02; }); await sleep(60); }
await trigger(0);
assert.ok((await ev(() => window.__tv.st.graphPos.x)) - x0 > 0.2, 'grabbing empty space moves the graph');
await aim({ kind: 'btn', id: 1 }); await sleep(500); await trigger(1); await trigger(0); await sleep(1200);
assert.equal(await ev(() => window.__tv.ui.model.nodes.length), 37, 'button switches to the MobileNet model');
assert.equal(await ev(() => window.__tv.st.tutorial), 5, 'onboarding completed by performing the gestures');
await ev(() => { window.__xrDevice.primaryInputMode = 'hand'; }); await sleep(1200);
const hands = await ev(() => window.__tv.xrPointers.map((p) => ({ isHand: p.isHand, vis: Object.values(p.hand.joints).filter((j) => j.userData.vis).length })));
assert.ok(hands.every((h) => h.isHand && h.vis >= 20), 'both hands tracked and rendered');
await ev(() => window.__xrDevice.hands.right.setPinchValueImmediate(1)); await sleep(700);
assert.equal(await ev(() => window.__tv.xrPointers.some((p) => p.selecting)), true, 'hand pinch fires select');
await ev(() => window.__xrDevice.hands.right.setPinchValueImmediate(0)); await sleep(500);
assert.equal(await ev(() => window.__tv.xrPointers.some((p) => p.selecting)), false, 'hand release ends select');
assert.deepEqual(errors, [], 'no page errors');
console.log('E2E PASS: hover, select, grab, model switch, onboarding, hands (emulated Quest 3 WebXR session)');
await browser.close(); srv.close();
