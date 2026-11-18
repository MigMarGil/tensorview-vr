// Dependency-free ONNX (protobuf wire format) reader + cost model + graph layout.
// Mirrors the native C++ implementation; results are cross-checked against it in tests/.

class Reader {
  constructor(u8, p = 0, e = u8.length) { this.u8 = u8; this.p = p; this.e = e; }
  varint() {
    let v = 0, mul = 1;
    while (this.p < this.e) {
      const b = this.u8[this.p++];
      v += (b & 0x7f) * mul;
      if (!(b & 0x80)) return v;
      mul *= 128;
      if (mul > 2 ** 63) break;
    }
    throw new Error('truncated varint');
  }
  tag() { const t = this.varint(); return [Math.floor(t / 8), t % 8]; }
  sub() {
    const n = this.varint();
    if (n > this.e - this.p) throw new Error('truncated field');
    const r = new Reader(this.u8, this.p, this.p + n);
    this.p += n;
    return r;
  }
  str() { const r = this.sub(); return new TextDecoder().decode(this.u8.subarray(r.p, r.e)); }
  skip(w) {
    if (w === 0) this.varint();
    else if (w === 1) { if (this.e - this.p < 8) throw new Error('truncated'); this.p += 8; }
    else if (w === 2) this.sub();
    else if (w === 5) { if (this.e - this.p < 4) throw new Error('truncated'); this.p += 4; }
    else throw new Error('bad wire type ' + w);
  }
  get done() { return this.p >= this.e; }
}

function readInts(r, w, out) {
  if (w === 0) out.push(r.varint());
  else if (w === 2) { const s = r.sub(); while (!s.done) out.push(s.varint()); }
  else r.skip(w);
}

function parseTensor(r) { // TensorProto: dims = 1, name = 8
  const t = { name: '', dims: [] };
  while (!r.done) {
    const [f, w] = r.tag();
    if (f === 1) readInts(r, w, t.dims);
    else if (f === 8 && w === 2) t.name = r.str();
    else r.skip(w);
  }
  return t;
}

function parseDim(r) { // dim_value = 1
  let v = 1;
  while (!r.done) {
    const [f, w] = r.tag();
    if (f === 1 && w === 0) { const x = r.varint(); v = x > 0 ? x : 1; } else r.skip(w);
  }
  return v;
}

function parseValueInfo(r) { // name = 1, type = 2 -> tensor_type = 1 -> shape = 2 -> dim = 1
  const vi = { name: '', dims: [], has: false };
  while (!r.done) {
    const [f, w] = r.tag();
    if (f === 1 && w === 2) vi.name = r.str();
    else if (f === 2 && w === 2) {
      const t = r.sub();
      while (!t.done) {
        const [tf, tw] = t.tag();
        if (tf === 1 && tw === 2) {
          const tt = t.sub();
          while (!tt.done) {
            const [ff, ww] = tt.tag();
            if (ff === 2 && ww === 2) {
              const sh = tt.sub(); vi.has = true;
              while (!sh.done) {
                const [sf, sw] = sh.tag();
                if (sf === 1 && sw === 2) vi.dims.push(parseDim(sh.sub())); else sh.skip(sw);
              }
            } else tt.skip(ww);
          }
        } else t.skip(tw);
      }
    } else r.skip(w);
  }
  return vi;
}

function parseNode(r) { // input = 1, output = 2, name = 3, op_type = 4, attribute = 5
  const n = { name: '', op: '', in: [], out: [], group: 1 };
  while (!r.done) {
    const [f, w] = r.tag();
    if (f === 1 && w === 2) n.in.push(r.str());
    else if (f === 2 && w === 2) n.out.push(r.str());
    else if (f === 3 && w === 2) n.name = r.str();
    else if (f === 4 && w === 2) n.op = r.str();
    else if (f === 5 && w === 2) {
      const a = r.sub(); let an = '', iv = 0;
      while (!a.done) {
        const [af, aw] = a.tag();
        if (af === 1 && aw === 2) an = a.str();
        else if (af === 3 && aw === 0) iv = a.varint();
        else a.skip(aw);
      }
      if (an === 'group' && iv > 0) n.group = iv;
    } else r.skip(w);
  }
  return n;
}

const numel = (d) => (d && d.length ? d.reduce((a, v) => a * (v > 0 ? v : 1), 1) : 0);
const shapeStr = (d) => (d && d.length ? '[' + d.join(',') + ']' : '?');
const PRESERVING = new Set(['Relu', 'LeakyRelu', 'Sigmoid', 'Tanh', 'Clip', 'BatchNormalization', 'Add', 'Sub', 'Mul', 'Div',
  'Softmax', 'Dropout', 'Identity', 'HardSwish', 'HardSigmoid', 'Gelu', 'Erf', 'Sqrt', 'Neg']);

export function colorFor(op) {
  const has = (s) => op.includes(s);
  if (op === 'Input') return [0.20, 0.65, 1.0];
  if (has('Conv')) return [1.0, 0.50, 0.15];
  if (has('Relu') || has('Sigmoid') || has('Tanh') || has('Gelu') || has('Clip') || has('Swish')) return [0.18, 0.85, 0.45];
  if (op === 'Add' || op === 'Sum' || op === 'Concat' || op === 'Mul') return [0.95, 0.80, 0.20];
  if (has('Pool')) return [0.35, 0.55, 1.0];
  if (op === 'Gemm' || op === 'MatMul') return [0.15, 0.90, 0.85];
  if (has('Softmax')) return [0.95, 0.35, 0.85];
  if (has('Norm')) return [0.65, 0.55, 0.95];
  return [0.60, 0.62, 0.68];
}

/** Parses an ONNX file. Throws on malformed input. */
export function parseOnnx(buffer, name) {
  const u8 = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  const top = new Reader(u8);
  let graph = null;
  while (!top.done) {
    const [f, w] = top.tag();
    if (f === 7 && w === 2) graph = top.sub(); else top.skip(w);
  }
  if (!graph) throw new Error('no graph found (not an ONNX file?)');

  const raw = [], init = new Map(), shape = new Map(), graphInputs = [];
  while (!graph.done) {
    const [f, w] = graph.tag();
    if (w !== 2) { graph.skip(w); continue; }
    const s = graph.sub();
    if (f === 1) raw.push(parseNode(s));
    else if (f === 5) { const t = parseTensor(s); init.set(t.name, t.dims); }
    else if (f === 11 || f === 12 || f === 13) {
      const vi = parseValueInfo(s);
      if (vi.has && vi.dims.length) shape.set(vi.name, vi.dims);
      if (f === 11) graphInputs.push(vi);
    }
  }
  if (!raw.length) throw new Error('graph has no nodes');

  const model = { name, nodes: [], totalParams: 0, totalFlops: 0, totalMemKb: 0, width: 1, height: 0.5 };
  const producer = new Map();
  const add = (g) => { g.id = model.nodes.length; g.outputs = []; model.nodes.push(g); return g.id; };

  for (const gi of graphInputs) {
    if (init.has(gi.name)) continue;
    const sh = shape.get(gi.name);
    producer.set(gi.name, add({ name: gi.name, op: 'Input', inShape: shapeStr(sh), outShape: shapeStr(sh),
      params: 0, flops: 0, memKb: Math.floor((numel(sh) * 4 + 1023) / 1024) }));
  }

  for (const rn of raw) {
    if (rn.out.length && !shape.has(rn.out[0]) && PRESERVING.has(rn.op)) {
      for (const i of rn.in) if (!init.has(i) && shape.has(i)) { shape.set(rn.out[0], shape.get(i)); break; }
    }
    let inShape = null;
    for (const i of rn.in) if (!init.has(i) && shape.has(i)) { inShape = shape.get(i); break; }
    const outShape = rn.out.length && shape.has(rn.out[0]) ? shape.get(rn.out[0]) : null;
    const g = { name: rn.name || `${rn.op}_${model.nodes.length}`, op: rn.op, inShape: shapeStr(inShape), outShape: shapeStr(outShape), params: 0, flops: 0 };
    for (const i of rn.in) if (init.has(i)) g.params += numel(init.get(i));
    const outN = outShape ? numel(outShape) : 0;
    g.memKb = Math.floor((outN * 4 + 1023) / 1024);

    if (rn.op === 'Conv' && rn.in.length > 1 && init.has(rn.in[1]) && init.get(rn.in[1]).length >= 3) {
      const wd = init.get(rn.in[1]);
      let kmac = 1; for (let k = 1; k < wd.length; k++) kmac *= wd[k] > 0 ? wd[k] : 1;
      g.flops = 2 * outN * kmac + (rn.in.length > 2 ? outN : 0);
      if (rn.group > 1) g.op = wd[1] === 1 ? 'DepthwiseConv' : 'GroupConv';
    } else if ((rn.op === 'Gemm' || rn.op === 'MatMul') && rn.in.length > 1) {
      let k = 0;
      if (init.has(rn.in[1]) && outShape && outShape.length) {
        const last = outShape[outShape.length - 1];
        k = last ? Math.floor(numel(init.get(rn.in[1])) / last) : 0;
      } else if (inShape && inShape.length) k = inShape[inShape.length - 1];
      g.flops = 2 * outN * k;
    } else if (rn.op === 'Softmax') g.flops = 5 * outN;
    else g.flops = outN;

    const id = add(g);
    for (const i of rn.in) {
      const p = producer.get(i);
      if (p === undefined || p === id) continue;
      if (!model.nodes[p].outputs.includes(id)) model.nodes[p].outputs.push(id);
    }
    for (const o of rn.out) producer.set(o, id);
  }
  for (const n of model.nodes) { model.totalParams += n.params; model.totalFlops += n.flops; model.totalMemKb += n.memKb; }
  layoutModel(model);
  return model;
}

/** Layered DAG layout folded into snake rows (meters, relative to graph center). */
export function layoutModel(m) {
  const n = m.nodes.length;
  if (!n) return;
  const layer = new Array(n).fill(0);
  for (let i = 0; i < n; i++) for (const c of m.nodes[i].outputs) if (c > i) layer[c] = Math.max(layer[c], layer[i] + 1);
  const L = Math.max(1, ...layer.map((v) => v + 1));
  const rows = Math.floor((L + 13) / 14), cols = Math.floor((L + rows - 1) / rows);
  const dx = Math.min(0.22, 2.0 / Math.max(1, cols - 1)), dy = 0.13, rowH = 0.5;
  const count = new Array(L).fill(0), seen = new Array(L).fill(0);
  for (const v of layer) count[v]++;
  const maxCount = Math.max(1, ...count);
  const maxLog = Math.max(...m.nodes.map((q) => Math.log10(Math.max(q.flops, 10))));
  for (let i = 0; i < n; i++) {
    const g = m.nodes[i], l = layer[i], idx = seen[l]++, row = Math.floor(l / cols);
    let col = l % cols;
    if (row % 2 === 1) col = cols - 1 - col;
    const branch = idx - 0.5 * (count[l] - 1);
    g.layer = l;
    g.x = (col - 0.5 * (cols - 1)) * dx;
    g.y = (0.5 * (rows - 1) - row) * rowH + branch * dy;
    g.z = branch * 0.06;
    const s = Math.min(1, Math.max(0, (Math.log10(Math.max(g.flops, 10)) - 1) / 9));
    g.heat = Math.min(1, Math.max(0, (Math.log10(Math.max(g.flops, 10)) - 1) / Math.max(1e-6, maxLog - 1))); // relative to this model
    g.hy = 0.015 + 0.030 * s;
    g.hx = Math.min(dx * 0.34, 0.018 + 0.030 * s);
    g.hz = g.hx;
    g.color = colorFor(g.op);
  }
  m.width = (cols - 1) * dx + 0.12;
  m.height = (rows - 1) * rowH + (maxCount - 1) * dy + 0.12;
}
