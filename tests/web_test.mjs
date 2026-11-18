// Node tests for the WebXR app's pure logic.  Run:  node tests/web_test.mjs
import fs from 'node:fs';
import assert from 'node:assert/strict';
import { parseOnnx } from '../web/src/onnx.js';
import { newState, advanceTutorial, applyGrab, heatColor, human, hotspots, resetView } from '../web/src/logic.js';

const dir = new URL('../web/public/models/', import.meta.url).pathname;
// Totals produced by the independent native C++ parser (tests/test_onnx.cpp) for the same files.
const expected = {
  'vision_resnet_pro.onnx': { nodes: 40, params: 827048, flops: 763262600, memKb: 11280 },
  'mobilenet_lite.onnx': { nodes: 37, params: 163440, flops: 85634376, memKb: 10720 },
  'tiny_transformer.onnx': { nodes: 31, params: 68234, flops: 2307954, memKb: 117 },
};
for (const [f, e] of Object.entries(expected)) {
  const m = parseOnnx(fs.readFileSync(dir + f), f);
  assert.equal(m.nodes.length, e.nodes, f + ' nodes');
  assert.equal(m.totalParams, e.params, f + ' params');
  assert.equal(m.totalFlops, e.flops, f + ' flops');
  assert.equal(m.totalMemKb, e.memKb, f + ' mem');
  for (const n of m.nodes) assert.ok(Number.isFinite(n.x + n.y + n.z + n.hx + n.hy), 'finite layout');
  console.log('OK parse', f, m.nodes.length, 'nodes', m.totalParams, 'params');
}
// hand-calculated: stem conv 7x7, 3->32, out 112x112 : params 3*32*49+32 = 4736 ; flops 2*32*112*112*147 + 32*112*112
const r = parseOnnx(fs.readFileSync(dir + 'vision_resnet_pro.onnx'), 'r');
const stem = r.nodes.find((n) => n.name === 'stem_conv');
assert.equal(stem.params, 4736); assert.equal(stem.flops, 2 * 32 * 112 * 112 * 147 + 32 * 112 * 112);
assert.equal(r.nodes.find((n) => n.name === 'classifier_fc').params, 129000);
// robustness: garbage and truncated files must throw, not hang
assert.throws(() => parseOnnx(new Uint8Array([1, 2, 3, 4, 5]), 'x'));
assert.throws(() => parseOnnx(fs.readFileSync(dir + 'tiny_transformer.onnx').subarray(0, 500), 'x'));
// layout: no two nodes overlap
for (const f of Object.keys(expected)) {
  const m = parseOnnx(fs.readFileSync(dir + f), f);
  for (let i = 0; i < m.nodes.length; i++) for (let j = i + 1; j < m.nodes.length; j++) {
    const a = m.nodes[i], b = m.nodes[j];
    const ov = Math.abs(a.x - b.x) < a.hx + b.hx && Math.abs(a.y - b.y) < a.hy + b.hy && Math.abs(a.z - b.z) < a.hz + b.hz;
    assert.ok(!ov, `${f}: ${a.name} overlaps ${b.name}`);
  }
}
console.log('OK layout: no overlapping nodes');

// gestures + tutorial
const st = newState();
assert.equal(advanceTutorial(st, { anyPointer: false, anyHover: false }), false);
assert.ok(advanceTutorial(st, { anyPointer: true, anyHover: false })); // step0 -> 1
assert.ok(advanceTutorial(st, { anyPointer: true, anyHover: true }));  // 1 -> 2
st.selected = 5; assert.ok(advanceTutorial(st, {}));                   // 2 -> 3
for (let i = 0; i < 10; i++) applyGrab(st, [{ pos: { x: 0.01 * (i + 1), y: 0, z: 0 }, prev: { x: 0.01 * i, y: 0, z: 0 } }]);
assert.ok(Math.abs(st.moved - 0.25) < 1e-9); assert.ok(!advanceTutorial(st, {}));
applyGrab(st, [{ pos: { x: 0.1, y: 0, z: 0 }, prev: { x: 0.09, y: 0, z: 0 } }]);
assert.ok(advanceTutorial(st, {}));                                    // 3 -> 4
st.switched = true; assert.ok(advanceTutorial(st, {})); assert.equal(st.tutorial, 5);
const z = newState(); let d = 0.2;
for (let i = 0; i < 10; i++) { applyGrab(z, [{ pos: { x: 0, y: 0, z: 0 }, prev: { x: 0, y: 0, z: 0 } }, { pos: { x: d + 0.02, y: 0, z: 0 }, prev: { x: d, y: 0, z: 0 } }]); d += 0.02; }
assert.ok(Math.abs(z.graphScale - 2.0) < 0.05, 'two-hand zoom ~x2, got ' + z.graphScale);
for (let i = 0; i < 200; i++) applyGrab(z, [{ pos: { x: 5, y: 5, z: 5 }, prev: { x: 0, y: 0, z: 0 } }]);
assert.ok(z.graphPos.x <= 1.4 && z.graphPos.y <= 1.0 && z.graphPos.z <= -0.7 && z.graphScale <= 2.5, 'clamped');
resetView(z); assert.equal(z.graphScale, 1);
assert.deepEqual(heatColor(0).map((v) => +v.toFixed(2)), [0.12, 0.25, 0.85]);
assert.equal(human(1_500_000), '1.50M');
assert.equal(hotspots(r, 1)[0].name, 'stem_conv', 'highest estimated FLOPs hotspot');
console.log('OK gestures / tutorial / clamps / heat / format');
