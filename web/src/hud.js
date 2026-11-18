// Canvas-2D drawing for the in-VR panels. Pure functions of (ctx, size, data) so they run in Node tests too.
import { human, hotspots, TUTORIAL } from './logic.js';

let HC = false; // high-contrast flag, set by each public draw call
const FONT = '"Segoe UI", system-ui, -apple-system, Roboto, Helvetica, Arial, sans-serif';
const rgb = (c, a = 1) => `rgba(${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)},${a})`;

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function bar(ctx, x, y, w, h, frac, color) {
  ctx.fillStyle = 'rgba(255,255,255,0.10)';
  roundRect(ctx, x, y, w, h, h / 2); ctx.fill();
  ctx.fillStyle = color;
  roundRect(ctx, x, y, Math.max(h, w * Math.min(1, Math.max(0, frac))), h, h / 2); ctx.fill();
}

export function drawTitle(ctx, w, h) {
  HC = false;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#7fd8ff';
  ctx.font = `700 ${h * 0.52}px ${FONT}`;
  ctx.textBaseline = 'middle';
  ctx.fillText('TENSORVIEW VR', 8, h * 0.5);
  const tw = ctx.measureText('TENSORVIEW VR').width;
  ctx.fillStyle = 'rgba(180,200,230,0.8)';
  ctx.font = `400 ${h * 0.30}px ${FONT}`;
  ctx.fillText('spatial neural-network profiler', tw + 36, h * 0.56);
}

export function drawButton(ctx, w, h, label, { hover = false, active = false, contrast = false } = {}) {
  HC = contrast;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = hover ? (HC ? '#ffff00' : 'rgba(90,165,255,1)') : active ? (HC ? '#ffffff' : 'rgba(45,105,190,1)') : (HC ? '#000000' : 'rgba(26,41,72,0.96)');
  roundRect(ctx, 3, 3, w - 6, h - 6, h * 0.28); ctx.fill();
  ctx.lineWidth = HC ? 5 : 3;
  ctx.strokeStyle = HC ? '#ffffff' : hover ? '#ffffff' : 'rgba(140,180,255,0.55)';
  ctx.stroke();
  ctx.fillStyle = HC && (hover || active) ? '#000000' : '#fff';
  ctx.font = `700 ${h * 0.40}px ${FONT}`;
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(label, w / 2, h * 0.53);
  ctx.textAlign = 'left';
}

export function drawBanner(ctx, w, h, step, contrast = false) {
  HC = contrast;
  ctx.clearRect(0, 0, w, h);
  const done = step >= TUTORIAL.length;
  ctx.fillStyle = done ? '#6cffa0' : (HC ? '#ffff00' : '#ffe14d');
  ctx.textBaseline = 'middle';
  const text = done ? 'READY  -  EXPLORE FREELY' : `STEP ${step + 1}/${TUTORIAL.length}   ${TUTORIAL[step]}`;
  let fs = h * 0.42;
  ctx.font = `700 ${fs}px ${FONT}`;
  while (ctx.measureText(text).width > w - 28 && fs > 12) { fs -= 1; ctx.font = `700 ${fs}px ${FONT}`; }
  ctx.fillText(text, 14, h * 0.36);
  bar(ctx, 14, h * 0.74, w - 28, h * 0.12, step / TUTORIAL.length, (HC ? '#ffff00' : '#ffe14d'));
}

/** Main card: model stats + selected layer inspector (or hotspots when nothing is selected). */
export function drawInspector(ctx, w, h, s) {
  HC = !!s.contrast;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = HC ? '#000000' : 'rgba(8,13,28,0.90)';
  roundRect(ctx, 4, 4, w - 8, h - 8, 26); ctx.fill();
  ctx.lineWidth = HC ? 6 : 3; ctx.strokeStyle = HC ? '#ffffff' : 'rgba(120,170,255,0.45)'; ctx.stroke();
  const m = s.model, P = 34;
  ctx.textBaseline = 'alphabetic';

  if (!m || !m.nodes.length) {
    ctx.fillStyle = '#ff7b7b'; ctx.font = `700 40px ${FONT}`;
    ctx.fillText(s.error ? 'MODEL LOAD FAILED' : 'LOADING MODEL...', P, 80);
    ctx.fillStyle = '#b8c6e0'; ctx.font = `400 30px ${FONT}`;
    ctx.fillText(s.error || 'one moment', P, 130);
    return;
  }

  // header chips
  ctx.fillStyle = (HC ? '#ffffff' : '#e8f0ff'); ctx.font = `700 38px ${FONT}`;
  ctx.fillText(m.name, P, 62);
  const chips = [['LAYERS', String(m.nodes.length)], ['PARAMS', human(m.totalParams)], ['FLOPs', human(m.totalFlops)], ['ACT MEM', human(m.totalMemKb * 1024, 'B')], ['FPS', String(Math.round(s.fps))]];
  let x = w - P;
  for (let i = chips.length - 1; i >= 0; i--) {
    const [k, v] = chips[i];
    ctx.font = `700 34px ${FONT}`;
    const vw = ctx.measureText(v).width;
    ctx.font = `400 20px ${FONT}`;
    const kw = ctx.measureText(k).width;
    const cw = Math.max(vw, kw);
    x -= cw;
    ctx.fillStyle = (HC ? '#ffffff' : '#9fb4d8'); ctx.font = `400 20px ${FONT}`; ctx.fillText(k, x, 32);
    ctx.fillStyle = '#ffffff'; ctx.font = `700 34px ${FONT}`; ctx.fillText(v, x, 68);
    x -= 34;
  }
  ctx.fillStyle = 'rgba(120,170,255,0.30)'; ctx.fillRect(P, 88, w - 2 * P, 2);

  const sel = s.selected >= 0 ? m.nodes[s.selected] : null;
  if (sel) {
    ctx.fillStyle = rgb(sel.color); roundRect(ctx, P, 112, 22, 22, 6); ctx.fill();
    ctx.fillStyle = (HC ? '#ffff00' : '#ffe14d'); ctx.font = `700 40px ${FONT}`;
    ctx.fillText(sel.name, P + 36, 134);
    const nameW = ctx.measureText(sel.name).width;
    ctx.fillStyle = (HC ? '#ffffff' : '#9fb4d8'); ctx.font = `400 28px ${FONT}`;
    ctx.fillText('   ' + sel.op, P + 36 + nameW, 134);

    ctx.fillStyle = (HC ? '#ffffff' : '#e8f0ff'); ctx.font = `400 30px ${FONT}`;
    ctx.fillText('IN    ' + sel.inShape, P, 186);
    ctx.fillText('OUT  ' + sel.outShape, P, 228);
    const cols = [['PARAMS', human(sel.params), sel.params / (m.totalParams || 1), '#7fd8ff'],
                  ['FLOPs', human(sel.flops), sel.flops / (m.totalFlops || 1), '#ff9f4d'],
                  ['ACT MEM', human(sel.memKb * 1024, 'B'), sel.memKb / (m.totalMemKb || 1), '#6cffa0']];
    const bx = w * 0.52, bw = (w - bx - P);
    cols.forEach(([k, v, f, c], i) => {
      const y = 172 + i * 62;
      ctx.fillStyle = (HC ? '#ffffff' : '#9fb4d8'); ctx.font = `400 22px ${FONT}`; ctx.fillText(k, bx, y);
      ctx.fillStyle = '#fff'; ctx.font = `700 28px ${FONT}`; ctx.fillText(v, bx + 130, y);
      ctx.fillStyle = (HC ? '#ffffff' : '#9fb4d8'); ctx.font = `400 22px ${FONT}`; ctx.fillText((f * 100).toFixed(f < 0.1 ? 1 : 0) + '% of model', bx + 280, y);
      bar(ctx, bx, y + 10, bw, 12, f, c);
    });
  } else {
    ctx.fillStyle = (HC ? '#ffff00' : '#ffe14d'); ctx.font = `700 34px ${FONT}`;
    ctx.fillText('COMPUTE HOTSPOTS', P, 140);
    const hw = ctx.measureText('COMPUTE HOTSPOTS').width;
    ctx.font = `400 22px ${FONT}`; ctx.fillStyle = (HC ? '#ffffff' : '#9fb4d8');
    ctx.fillText('select any layer to inspect it', P + hw + 28, 140);
    hotspots(m, 3).forEach((hs, i) => {
      const y = 190 + i * 52;
      ctx.fillStyle = (HC ? '#ffffff' : '#e8f0ff'); ctx.font = `700 28px ${FONT}`;
      ctx.fillText(`${i + 1}. ${hs.name}`, P, y);
      ctx.fillStyle = (HC ? '#ffffff' : '#9fb4d8'); ctx.font = `400 24px ${FONT}`;
      ctx.fillText(hs.op, P + 430, y);
      bar(ctx, P + 640, y - 20, w - 2 * P - 640 - 110, 16, hs.share * 2.2, '#ff9f4d');
      ctx.fillStyle = '#fff'; ctx.font = `700 28px ${FONT}`;
      ctx.fillText((hs.share * 100).toFixed(1) + '%', w - P - 96, y);
    });
  }
  if (s.hover) {
    ctx.fillStyle = (HC ? '#ffffff' : '#9fb4d8'); ctx.font = `400 22px ${FONT}`;
    ctx.fillText('HOVER: ' + s.hover, P, h - 22);
  }
}
