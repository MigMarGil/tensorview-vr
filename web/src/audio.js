// Procedural UI sounds with WebAudio (no audio assets).
const VOICES = {
  hover: [1900, 2100, 35, 0.06], select: [880, 1320, 170, 0.16], grab: [320, 380, 90, 0.14],
  release: [260, 200, 90, 0.10], switch: [660, 990, 210, 0.14], step: [1046, 1568, 260, 0.13],
};

export function createAudio() {
  let ctx = null;
  const ensure = () => {
    if (!ctx) {
      const AC = globalThis.AudioContext || globalThis.webkitAudioContext;
      if (!AC) return null;
      try { ctx = new AC(); } catch { return null; }
    }
    if (ctx.state === 'suspended') ctx.resume();
    return ctx;
  };
  return {
    unlock: ensure,
    play(name) {
      const c = ensure();
      const v = VOICES[name];
      if (!c || !v) return;
      const [f0, f1, ms, amp] = v, t = c.currentTime, d = ms / 1000;
      const o = c.createOscillator(), g = c.createGain();
      o.type = 'sine';
      o.frequency.setValueAtTime(f0, t);
      o.frequency.linearRampToValueAtTime(f1, t + d);
      g.gain.setValueAtTime(0.0001, t);
      g.gain.linearRampToValueAtTime(amp, t + 0.004);
      g.gain.exponentialRampToValueAtTime(0.0001, t + d);
      o.connect(g).connect(c.destination);
      o.start(t);
      o.stop(t + d + 0.02);
    },
  };
}
