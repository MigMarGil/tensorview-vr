// TensorView VR — app logic running inside an IWSDK World. Hands-first: pinch to select, pinch empty space to grab/zoom.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { parseOnnx } from './onnx.js';
import { newState, advanceTutorial, applyGrab, resetView, heatColor, GRAPH_POS } from './logic.js';
import { drawInspector, drawButton, drawBanner, drawTitle } from './hud.js';
import { createAudio } from './audio.js';

export function createTensorView(world) {
const HUD_Z = -1.30;
const SOURCES = [
  { label: 'RESNET CNN', url: 'models/vision_resnet_pro.onnx', file: 'vision_resnet_pro.onnx' },
  { label: 'MOBILENET', url: 'models/mobilenet_lite.onnx', file: 'mobilenet_lite.onnx' },
  { label: 'TRANSFORMER', url: 'models/tiny_transformer.onnx', file: 'tiny_transformer.onnx' },
];
const customUrl = new URLSearchParams(location.search).get('model');
if (customUrl) SOURCES.push({ label: 'CUSTOM', url: customUrl, file: customUrl.split('/').pop() || 'custom.onnx' });

const st = newState();
const ui = { model: null, error: '', modelIndex: 0, colorMode: 'op', contrast: false, fps: 72, hoverLabel: '', hudDirty: true };
const audio = createAudio();

// ------------------------------------------------------------------ IWSDK world provides renderer / scene / camera
const renderer = world.renderer;
const scene = world.scene;
const camera = world.camera;
scene.background = new THREE.Color(0x010205);
scene.fog = new THREE.Fog(0x010205, 7, 20);
camera.position.set(0, -0.10, 1.0);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, -0.10, -1.45);
controls.minDistance = 0.8; controls.maxDistance = 6;
controls.minAzimuthAngle = -1.1; controls.maxAzimuthAngle = 1.1;
controls.minPolarAngle = 0.5; controls.maxPolarAngle = 2.0;
controls.enableDamping = true;
controls.update();

scene.add(new THREE.HemisphereLight(0xbfd4ff, 0x10131f, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(1.5, 2.5, 1.5); scene.add(sun);

// starfield for depth cues
let stars = null;
{
  const N = 700, pos = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    const r = 6 + Math.random() * 8, a = Math.random() * Math.PI * 2, b = Math.acos(2 * Math.random() - 1);
    pos.set([r * Math.sin(b) * Math.cos(a), r * Math.cos(b), r * Math.sin(b) * Math.sin(a)], i * 3);
  }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  stars = new THREE.Points(g, new THREE.PointsMaterial({ color: 0x6f8fd0, size: 0.025, sizeAttenuation: true, fog: false }));
  scene.add(stars);
}

// ------------------------------------------------------------------ graph
const graph = new THREE.Group();
scene.add(graph);
const unitBox = new THREE.BoxGeometry(1, 1, 1);
const sphereGeo = new THREE.SphereGeometry(1, 10, 10);
const haloSel = new THREE.Mesh(unitBox, new THREE.MeshBasicMaterial({ color: 0xffe14d, transparent: true, opacity: 0.38, side: THREE.BackSide, depthWrite: false }));
const haloHov = new THREE.Mesh(unitBox, new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.28, side: THREE.BackSide, depthWrite: false }));
haloSel.visible = haloHov.visible = false;
graph.add(haloSel, haloHov);

const edgeMat = new THREE.MeshBasicMaterial({ color: 0x6685b3 });
const edgeHotMat = new THREE.MeshBasicMaterial({ color: 0xffe14d });
const pulseMat = new THREE.MeshBasicMaterial({ color: 0x8cdcff });
const pulseHotMat = new THREE.MeshBasicMaterial({ color: 0xfff3b0 });
let nodeMeshes = [], edges = [], dynamic = [];
const Z = new THREE.Vector3(0, 0, 1);

function clearGraph() {
  for (const o of dynamic) { graph.remove(o); if (o.userData.ownMat) o.material.dispose(); }
  dynamic = []; nodeMeshes = []; edges = [];
}

function nodeColor(n) {
  const c = ui.colorMode === 'heat' ? heatColor(n.heat) : n.color;
  return new THREE.Color().setRGB(c[0], c[1], c[2], THREE.SRGBColorSpace);
}

function buildGraph(model) {
  clearGraph();
  for (const n of model.nodes) {
    const mat = new THREE.MeshStandardMaterial({ roughness: 0.45, metalness: 0.15, emissiveIntensity: 0.2 });
    const m = new THREE.Mesh(unitBox, mat);
    m.scale.set(2 * n.hx, 2 * n.hy, 2 * n.hz);
    m.position.set(n.x, n.y, n.z);
    m.userData = { kind: 'node', id: n.id, ownMat: true };
    graph.add(m); dynamic.push(m); nodeMeshes.push(m);
  }
  for (const n of model.nodes) for (const c of n.outputs) {
    const t = model.nodes[c];
    const A = new THREE.Vector3(n.x, n.y, n.z), B = new THREE.Vector3(t.x, t.y, t.z), d = B.clone().sub(A);
    let a, b;
    if (Math.abs(d.x) >= Math.abs(d.y)) { const s = d.x >= 0 ? 1 : -1; a = A.clone().add(new THREE.Vector3(s * n.hx, 0, 0)); b = B.clone().sub(new THREE.Vector3(s * t.hx, 0, 0)); }
    else { const s = d.y >= 0 ? 1 : -1; a = A.clone().add(new THREE.Vector3(0, s * n.hy, 0)); b = B.clone().sub(new THREE.Vector3(0, s * t.hy, 0)); }
    const len = a.distanceTo(b);
    const beam = new THREE.Mesh(unitBox, edgeMat);
    beam.position.copy(a).add(b).multiplyScalar(0.5);
    beam.quaternion.setFromUnitVectors(Z, b.clone().sub(a).normalize());
    beam.scale.set(0.004, 0.004, len);
    const pulse = new THREE.Mesh(sphereGeo, pulseMat);
    pulse.scale.setScalar(0.007);
    graph.add(beam, pulse); dynamic.push(beam, pulse);
    edges.push({ from: n.id, to: c, a, b, beam, pulse, phase: (n.id * 5 + c) * 0.137 });
  }
  applyColorMode();
  refreshSelection();
  if (ui.contrast) applyContrast();
}

function applyColorMode() {
  if (!ui.model) return;
  ui.model.nodes.forEach((n, i) => { const c = nodeColor(n); nodeMeshes[i].material.color.copy(c); nodeMeshes[i].material.emissive.copy(c); });
}

function applyContrast() {
  const hc = ui.contrast;
  scene.background.set(hc ? 0x000000 : 0x010205);
  if (stars) stars.visible = !hc;
  edgeMat.color.set(hc ? 0xffffff : 0x6685b3);
  haloSel.material.opacity = hc ? 0.7 : 0.38;
  nodeMeshes.forEach((m) => { m.material.emissiveIntensity = hc ? 0.65 : 0.2; });
  for (const e of edges) e.pulse.scale.setScalar(hc ? 0.011 : 0.007);
  redrawButtons(); banner.redraw(); ui.hudDirty = true;
}

function refreshSelection() {
  const sel = st.selected >= 0 && ui.model ? ui.model.nodes[st.selected] : null;
  haloSel.visible = !!sel;
  if (sel) { haloSel.position.set(sel.x, sel.y, sel.z); haloSel.scale.set(2 * sel.hx + 0.018, 2 * sel.hy + 0.018, 2 * sel.hz + 0.018); }
  for (const e of edges) {
    const hot = e.from === st.selected || e.to === st.selected;
    e.beam.material = hot ? edgeHotMat : edgeMat;
    e.beam.scale.x = e.beam.scale.y = hot ? 0.008 : 0.004;
    e.pulse.material = hot ? pulseHotMat : pulseMat;
  }
  ui.hudDirty = true;
}

// ------------------------------------------------------------------ HUD panels
function makePanel(wm, hm, pw, ph, draw) {
  const canvas = document.createElement('canvas'); canvas.width = pw; canvas.height = ph;
  const ctx = canvas.getContext('2d');
  const tex = new THREE.CanvasTexture(canvas); tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 4;
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(wm, hm), new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, toneMapped: false }));
  const p = { mesh, redraw() { draw(ctx, pw, ph); tex.needsUpdate = true; } };
  p.redraw();
  scene.add(mesh);
  return p;
}
const tilt = (y) => -Math.atan2(-y, -HUD_Z) * 0.85; // lower panels lean toward the seated user

const titlePanel = makePanel(1.5, 0.12, 1500, 120, (c, w, h) => drawTitle(c, w, h));
titlePanel.mesh.position.set(0, GRAPH_POS.y + 0.80, -1.45);
const inspector = makePanel(1.5, 0.42, 1500, 420, (c, w, h) => drawInspector(c, w, h, { model: ui.model, selected: st.selected, hover: ui.hoverLabel, fps: ui.fps, error: ui.error, contrast: ui.contrast }));
inspector.mesh.position.set(0, -0.92, HUD_Z); inspector.mesh.rotation.x = tilt(-0.92);
const banner = makePanel(1.4, 0.10, 1400, 100, (c, w, h) => drawBanner(c, w, h, st.tutorial, ui.contrast));
banner.mesh.position.set(0, -0.645, HUD_Z); banner.mesh.rotation.x = tilt(-0.645);

let buttons = [];
function buttonLabels() { return [...SOURCES.map((s) => s.label), ui.colorMode === 'heat' ? 'COLOR: HEAT' : 'COLOR: OP', ui.contrast ? 'CONTRAST: ON' : 'CONTRAST', 'RESET VIEW']; }
function rebuildButtons() {
  for (const b of buttons) { scene.remove(b.mesh); b.mesh.geometry.dispose(); b.mesh.material.map.dispose(); b.mesh.material.dispose(); }
  const labels = buttonLabels(), n = labels.length, w = 0.27, gap = 0.03, total = n * w + (n - 1) * gap;
  buttons = labels.map((label, i) => {
    const b = { label, hover: false, i };
    const p = makePanel(w, w * 66 / 280, 280, 66, (c, pw, ph) => drawButton(c, pw, ph, b.label, { hover: b.hover, active: (b.i === ui.modelIndex && b.i < SOURCES.length) || (b.i === SOURCES.length + 1 && ui.contrast), contrast: ui.contrast }));
    p.mesh.position.set(-total / 2 + w / 2 + i * (w + gap), -0.55, HUD_Z); p.mesh.rotation.x = tilt(-0.55);
    p.mesh.userData = { kind: 'btn', id: i };
    b.mesh = p.mesh; b.redraw = p.redraw;
    return b;
  });
}
function redrawButtons() { const L = buttonLabels(); buttons.forEach((b, i) => { b.label = L[i]; b.redraw(); }); }
rebuildButtons();

// ------------------------------------------------------------------ model loading
async function loadSource(i) {
  ui.modelIndex = i; ui.error = ''; ui.model = null; ui.hudDirty = true; st.selected = -1;
  clearGraph();
  try {
    const src = SOURCES[i];
    const buf = src.buffer || await (await fetch(src.url)).arrayBuffer();
    setModel(parseOnnx(buf, src.file));
  } catch (e) { ui.error = String(e.message || e); console.error(e); }
  redrawButtons();
}
function setModel(m) { ui.model = m; ui.error = ''; st.selected = -1; buildGraph(m); ui.hudDirty = true; }

// ------------------------------------------------------------------ pointers
const raycaster = new THREE.Raycaster(); raycaster.far = 8;
const V = new THREE.Vector3();
const reticleGeo = new THREE.SphereGeometry(0.01, 12, 12);
const rayGeo = new THREE.BoxGeometry(0.003, 0.003, 1); rayGeo.translate(0, 0, -0.5);

function makePointer(name) {
  const reticle = new THREE.Mesh(reticleGeo, new THREE.MeshBasicMaterial({ color: 0x4de6ff, depthTest: false }));
  reticle.visible = false; reticle.renderOrder = 10; scene.add(reticle);
  return { name, ctrl: null, grip: null, hand: null, ray: null, reticle, connected: false, isHand: false, selecting: false, grabbing: false,
    prev: new THREE.Vector3(), hasPrev: false, hover: { kind: null, id: -1 }, hoverKey: '', origin: new THREE.Vector3(), dir: new THREE.Vector3(0, 0, -1), valid: false };
}
const xrPointers = [0, 1].map((i) => {
  const p = makePointer('xr' + i);
  p.ctrl = renderer.xr.getController(i); p.grip = renderer.xr.getControllerGrip(i); p.hand = renderer.xr.getHand(i);
  p.ray = new THREE.Mesh(rayGeo, new THREE.MeshBasicMaterial({ color: 0x4de6ff })); p.ray.visible = false;
  p.ctrl.add(p.ray); scene.add(p.ctrl, p.grip, p.hand);
  p.ctrl.addEventListener('connected', (e) => { p.connected = true; p.isHand = !!(e.data && e.data.hand); });
  p.ctrl.addEventListener('disconnected', () => { p.connected = false; p.valid = false; p.ray.visible = false; p.reticle.visible = false; p.selecting = false; p.grabbing = false; p.hasPrev = false; });
  p.ctrl.addEventListener('selectstart', () => { p.selecting = true; onSelectStart(p); });
  p.ctrl.addEventListener('selectend', () => { p.selecting = false; onSelectEnd(p); });
  return p;
});
const mouse = makePointer('mouse');
mouse.valid = false;
const ndc = new THREE.Vector2();

const pickables = () => [...nodeMeshes, ...buttons.map((b) => b.mesh)];

function pick(p) {
  raycaster.set(p.origin, p.dir);
  const hits = raycaster.intersectObjects(pickables(), false);
  p.hover = hits.length ? { kind: hits[0].object.userData.kind, id: hits[0].object.userData.id } : { kind: null, id: -1 };
  p.dist = hits.length ? hits[0].distance : 2.5;
  return hits.length ? hits[0].point : null;
}

function onSelectStart(p) {
  audio.unlock();
  if (p.hover.kind === 'btn') pressButton(p.hover.id);
  else if (p.hover.kind === 'node') selectNode(p.hover.id);
  else if (p.name !== 'mouse') { p.grabbing = true; p.hasPrev = false; audio.play('grab'); }
}
function onSelectEnd(p) { if (p.grabbing) { p.grabbing = false; p.hasPrev = false; audio.play('release'); } }

function selectNode(id) { st.selected = id; audio.play('select'); refreshSelection(); }
function pressButton(i) {
  const S = SOURCES.length;
  if (i < S) { if (i !== ui.modelIndex) { st.switched = true; loadSource(i); } }
  else if (i === S) { ui.colorMode = ui.colorMode === 'op' ? 'heat' : 'op'; applyColorMode(); redrawButtons(); }
  else if (i === S + 1) { ui.contrast = !ui.contrast; applyContrast(); }
  else resetView(st);
  audio.play('switch');
}

// desktop pointer (non-XR preview / emulator-free inspection)
let downAt = null;
renderer.domElement.addEventListener('pointermove', (e) => { ndc.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1); mouse.valid = true; });
renderer.domElement.addEventListener('pointerdown', (e) => { downAt = { x: e.clientX, y: e.clientY }; });
renderer.domElement.addEventListener('pointerup', (e) => {
  if (downAt && Math.hypot(e.clientX - downAt.x, e.clientY - downAt.y) < 5) { audio.unlock(); onSelectStart(mouse); }
  downAt = null;
});
window.addEventListener('keydown', (e) => {
  if (e.key >= '1' && e.key <= String(Math.min(9, SOURCES.length))) pressButton(Number(e.key) - 1);
  else if (e.key === 'h' || e.key === 'H') pressButton(SOURCES.length);
  else if (e.key === 'c' || e.key === 'C') pressButton(SOURCES.length + 1);
  else if (e.key === 'r' || e.key === 'R') pressButton(SOURCES.length + 2);
});

// drag & drop an .onnx file (desktop)
const drop = document.getElementById('drop');
window.addEventListener('dragover', (e) => { e.preventDefault(); drop.style.display = 'flex'; });
window.addEventListener('dragleave', () => { drop.style.display = 'none'; });
window.addEventListener('drop', async (e) => {
  e.preventDefault(); drop.style.display = 'none';
  const f = e.dataTransfer.files[0]; if (!f) return;
  const buffer = await f.arrayBuffer();
  const existing = SOURCES.findIndex((s) => s.label === 'MY MODEL');
  const src = { label: 'MY MODEL', file: f.name, buffer };
  if (existing >= 0) SOURCES[existing] = src; else SOURCES.push(src);
  rebuildButtons();
  loadSource(existing >= 0 ? existing : SOURCES.length - 1);
});

// ------------------------------------------------------------------ enter VR (IWSDK launches the session)
const enterBtn = document.getElementById('enter'), hint = document.getElementById('hint');
let wasPresenting = false;
async function setupXR() {
  const desktopHelp = 'Desktop preview: drag to orbit, scroll to zoom, click blocks. Keys 1/2/3 switch model, H heat colors, C high contrast, R reset. Drop an .onnx file to load your own.';
  if (!navigator.xr || !(await navigator.xr.isSessionSupported('immersive-vr').catch(() => false))) {
    enterBtn.textContent = 'WEBXR NOT AVAILABLE';
    hint.innerHTML = '<b>Open this page in the Meta Quest Browser</b> (or Chrome + Meta Immersive Web Emulator) to enter VR. ' + desktopHelp;
    return;
  }
  enterBtn.disabled = false; enterBtn.textContent = 'ENTER VR';
  hint.innerHTML = '<b>Hands-first:</b> point at a block and pinch to inspect it; pinch empty space and drag to move the graph; use both hands to zoom. Seated, no controllers needed. ' + desktopHelp;
  enterBtn.addEventListener('click', () => { audio.unlock(); try { world.launchXR(); } catch (e) { hint.textContent = 'Could not start the VR session: ' + (e.message || e); } });
}

// ------------------------------------------------------------------ frame loop
let hudTimer = 0;
const tmp = new THREE.Vector3();
function jointVisuals(p) {
  if (!p.hand) return;
  for (const j of Object.values(p.hand.joints)) {
    if (!j.userData.vis) { j.userData.vis = new THREE.Mesh(sphereGeo, new THREE.MeshBasicMaterial({ color: 0xa6d9ff })); j.add(j.userData.vis); }
    j.userData.vis.scale.setScalar(Math.max(0.004, j.jointRadius || 0.006));
  }
}
function gripPos(p) { const o = p.grip && p.grip.visible ? p.grip : p.ctrl; return o.getWorldPosition(tmp).clone(); }

function update(dt) {
  const now = performance.now();
  ui.fps = ui.fps * 0.95 + (1 / Math.max(dt, 0.001)) * 0.05;
  const presenting = renderer.xr.isPresenting;
  if (presenting !== wasPresenting) { wasPresenting = presenting; document.body.classList.toggle('in-vr', presenting); }
  controls.enabled = !presenting;
  titlePanel.mesh.visible = presenting; // HTML overlay carries the title on desktop
  if (!presenting) controls.update();

  // grabbing (one hand = move, two hands = zoom)
  const grabbers = [];
  for (const p of xrPointers) {
    if (p.grabbing && p.connected) {
      const pos = gripPos(p);
      if (p.hasPrev) grabbers.push({ pos: { x: pos.x, y: pos.y, z: pos.z }, prev: { x: p.prev.x, y: p.prev.y, z: p.prev.z } });
      p.prev.copy(pos); p.hasPrev = true;
    }
  }
  applyGrab(st, grabbers);
  graph.position.set(st.graphPos.x, st.graphPos.y, st.graphPos.z);
  graph.scale.setScalar(st.graphScale);
  graph.updateMatrixWorld(true);

  // rays + hover
  const pointers = [];
  if (presenting) {
    for (const p of xrPointers) {
      jointVisuals(p);
      if (!p.connected || p.ctrl.visible === false) { p.valid = false; p.ray.visible = false; p.reticle.visible = false; p.hover = { kind: null, id: -1 }; continue; }
      p.ctrl.updateWorldMatrix(true, false);
      p.origin.setFromMatrixPosition(p.ctrl.matrixWorld);
      p.dir.set(0, 0, -1).transformDirection(p.ctrl.matrixWorld);
      p.valid = true; pointers.push(p);
    }
  } else if (mouse.valid) {
    raycaster.setFromCamera(ndc, camera);
    mouse.origin.copy(raycaster.ray.origin); mouse.dir.copy(raycaster.ray.direction);
    mouse.valid = true; pointers.push(mouse);
  }
  let hoverNode = -1;
  for (const b of buttons) b.hoverNow = false;
  for (const p of pointers) {
    const point = pick(p);
    const key = p.hover.kind ? p.hover.kind + p.hover.id : '';
    if (key !== p.hoverKey) { if (key) audio.play('hover'); p.hoverKey = key; }
    if (p.hover.kind === 'btn') buttons[p.hover.id].hoverNow = true;
    if (p.hover.kind === 'node' && hoverNode < 0) hoverNode = p.hover.id;
    if (p !== mouse) {
      const col = p.selecting ? 0xff9a3c : 0x4de6ff;
      p.ray.material.color.setHex(col); p.reticle.material.color.setHex(col);
      p.ray.visible = true; p.ray.scale.z = p.dist;
      p.reticle.visible = true; p.reticle.position.copy(p.origin).addScaledVector(p.dir, p.dist);
      p.reticle.scale.setScalar(p.hover.kind ? 1.3 : 0.8);
    }
  }
  let btnChanged = false;
  for (const b of buttons) if (!!b.hoverNow !== b.hover) { b.hover = !!b.hoverNow; b.redraw(); btnChanged = true; }
  const hovered = hoverNode >= 0 && ui.model ? ui.model.nodes[hoverNode] : null;
  const label = hovered ? `${hovered.name} (${hovered.op})` : '';
  if (label !== ui.hoverLabel) { ui.hoverLabel = label; ui.hudDirty = true; }
  haloHov.visible = !!hovered && hoverNode !== st.selected;
  if (haloHov.visible) { haloHov.position.set(hovered.x, hovered.y, hovered.z); haloHov.scale.set(2 * hovered.hx + 0.012, 2 * hovered.hy + 0.012, 2 * hovered.hz + 0.012); }

  // onboarding
  const before = st.tutorial;
  if (advanceTutorial(st, { anyPointer: pointers.length > 0, anyHover: hoverNode >= 0 })) { audio.play('step'); banner.redraw(); }
  void before;

  // pulses along edges
  const t = now / 1000;
  for (const e of edges) { const ph = (t * 0.3 + e.phase) % 1; e.pulse.position.lerpVectors(e.a, e.b, ph); }

  // HUD refresh (on change, and twice a second for FPS)
  hudTimer += dt;
  if (ui.hudDirty || hudTimer > 0.5) { inspector.redraw(); ui.hudDirty = false; hudTimer = 0; }
  void btnChanged;
}

setupXR();
loadSource(0);
window.__tv = { // debug / test hook
  st, ui, SOURCES, scene, camera, renderer, pressButton, selectNode, xrPointers,
  nodeWorld: (id) => nodeMeshes[id].getWorldPosition(new THREE.Vector3()),
  buttonWorld: (i) => buttons[i].mesh.getWorldPosition(new THREE.Vector3()),
  THREE,
};

return { update, st, ui };
}
