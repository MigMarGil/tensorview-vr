// Pure interaction logic (no DOM / three.js) so it can be unit-tested in Node.

export const GRAPH_POS = { x: 0, y: 0.10, z: -1.55 };
export const TUTORIAL = [
  'RAISE YOUR HANDS AND POINT AT THE GRAPH',
  'POINT AT A BLOCK',
  'PINCH THUMB + INDEX TO SELECT IT',
  'PINCH EMPTY SPACE + DRAG TO MOVE  -  BOTH HANDS TO ZOOM',
  'TAP A MODEL BUTTON TO COMPARE ARCHITECTURES',
];

export function newState() {
  return { graphPos: { ...GRAPH_POS }, graphScale: 1, moved: 0, selected: -1, switched: false, tutorial: 0 };
}

/** Advances the onboarding when the user performs the step's gesture. Returns true if it advanced. */
export function advanceTutorial(st, input) {
  if (st.tutorial >= TUTORIAL.length) return false;
  let ok = false;
  switch (st.tutorial) {
    case 0: ok = input.anyPointer; break;
    case 1: ok = input.anyHover; break;
    case 2: ok = st.selected >= 0; break;
    case 3: ok = st.moved > 0.25 || Math.abs(Math.log(st.graphScale)) > 0.15; break;
    case 4: ok = st.switched; break;
  }
  if (ok) st.tutorial++;
  return ok;
}

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/**
 * Grab gestures. grabbers: [{pos, prev}] (world positions of pinching hands that started on empty space).
 * One hand moves the graph (gain 2.5); two hands move it and scale it by the change of hand distance.
 */
export function applyGrab(st, grabbers) {
  if (grabbers.length === 1) {
    const g = grabbers[0];
    const d = { x: (g.pos.x - g.prev.x) * 2.5, y: (g.pos.y - g.prev.y) * 2.5, z: (g.pos.z - g.prev.z) * 2.5 };
    st.graphPos.x += d.x; st.graphPos.y += d.y; st.graphPos.z += d.z;
    st.moved += Math.hypot(d.x, d.y, d.z);
  } else if (grabbers.length >= 2) {
    const [a, b] = grabbers;
    const d0 = dist(a.prev, b.prev), d1 = dist(a.pos, b.pos);
    if (d0 > 0.03) st.graphScale *= clamp(d1 / d0, 0.9, 1.1);
    const d = {
      x: ((a.pos.x - a.prev.x) + (b.pos.x - b.prev.x)) * 1.25,
      y: ((a.pos.y - a.prev.y) + (b.pos.y - b.prev.y)) * 1.25,
      z: ((a.pos.z - a.prev.z) + (b.pos.z - b.prev.z)) * 1.25,
    };
    st.graphPos.x += d.x; st.graphPos.y += d.y; st.graphPos.z += d.z;
    st.moved += Math.hypot(d.x, d.y, d.z);
  }
  st.graphScale = clamp(st.graphScale, 0.35, 2.5);
  st.graphPos.x = clamp(st.graphPos.x, -1.4, 1.4);
  st.graphPos.y = clamp(st.graphPos.y, -0.6, 1.0);
  st.graphPos.z = clamp(st.graphPos.z, -3.2, -0.7);
}

export function resetView(st) { st.graphPos = { ...GRAPH_POS }; st.graphScale = 1; }

/** blue -> cyan -> yellow -> red heat ramp, t in [0,1] */
export function heatColor(t) {
  t = clamp(t, 0, 1);
  const stops = [[0.12, 0.25, 0.85], [0.10, 0.80, 0.85], [0.98, 0.88, 0.20], [0.95, 0.20, 0.15]];
  const x = t * 3, i = Math.min(2, Math.floor(x)), f = x - i;
  return stops[i].map((c, k) => c + (stops[i + 1][k] - c) * f);
}

export function human(v, unit = '') {
  if (v >= 1e9) return (v / 1e9).toFixed(2) + 'G' + unit;
  if (v >= 1e6) return (v / 1e6).toFixed(2) + 'M' + unit;
  if (v >= 1e3) return (v / 1e3).toFixed(1) + 'K' + unit;
  return Math.round(v) + unit;
}

export function hotspots(model, k = 3) {
  const total = model.totalFlops || 1;
  return [...model.nodes].sort((a, b) => b.flops - a.flops).slice(0, k).map((n) => ({ id: n.id, name: n.name, op: n.op, share: n.flops / total }));
}
