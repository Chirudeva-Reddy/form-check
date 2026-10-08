// form-check engine: pose → smoothed joints → rep segmentation → per-rep checks.
// Pure JS with no DOM, so the same code runs in the browser and under `node --test`.
// ponytail: every threshold below is a heuristic, not yet calibrated on labelled clips.

export const SIDES = {
  left: { shoulder: 11, elbow: 13, wrist: 15, hip: 23, knee: 25, ankle: 27 },
  right: { shoulder: 12, elbow: 14, wrist: 16, hip: 24, knee: 26, ankle: 28 },
};
const NOSE = 0;
const MIN_VIS = 0.5;
const JOINT_NAMES = { shoulder: 'shoulders', elbow: 'elbows', wrist: 'hands', hip: 'hips', knee: 'knees', ankle: 'feet' };

const deg = (r) => (r * 180) / Math.PI;
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, v: Math.min(a.v, b.v) });
const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
const spread = (xs) => Math.max(...xs) - Math.min(...xs);
const J = (p, side, joint) => p[SIDES[side][joint]];

/** Angle ABC in degrees (2D, image plane). */
export function angle(a, b, c) {
  const d = dist(a, b) * dist(c, b);
  if (!d) return NaN;
  return deg(Math.acos(clamp(((a.x - b.x) * (c.x - b.x) + (a.y - b.y) * (c.y - b.y)) / d, -1, 1)));
}

/** Tilt of the segment lower→upper away from vertical, in degrees. */
export const fromVertical = (lower, upper) => deg(Math.atan2(Math.abs(upper.x - lower.x), Math.abs(upper.y - lower.y)));

const hipMid = (p) => mid(p[23], p[24]);
const shoulderMid = (p) => mid(p[11], p[12]);
const torsoLean = (p) => fromVertical(hipMid(p), shoulderMid(p));
const torsoLen = (p) => dist(hipMid(p), shoulderMid(p));
const knee = (p, s) => angle(J(p, s, 'hip'), J(p, s, 'knee'), J(p, s, 'ankle'));
const elbow = (p, s) => angle(J(p, s, 'shoulder'), J(p, s, 'elbow'), J(p, s, 'wrist'));
const armRaise = (p, s) => angle(J(p, s, 'hip'), J(p, s, 'shoulder'), J(p, s, 'elbow'));

/** How far the hip sits below (+) or above (−) the shoulder→ankle line, as a fraction of body length. */
function hipOffset(p, s) {
  const sh = J(p, s, 'shoulder'), an = J(p, s, 'ankle'), hp = J(p, s, 'hip');
  const k = (hp.x - sh.x) / (an.x - sh.x || 1e-9);
  return (hp.y - (sh.y + k * (an.y - sh.y))) / (dist(sh, an) || 1);
}

/** One-Euro filter (Casiez et al. 2012): smooths jitter when still, stays responsive when moving. */
export class OneEuro {
  constructor({ minCutoff = 1, beta = 0.02, dCutoff = 1 } = {}) {
    Object.assign(this, { minCutoff, beta, dCutoff, x: null, dx: 0, t: null });
  }
  filter(x, tMs) {
    if (this.t === null) return Object.assign(this, { t: tMs, x }).x;
    const dt = Math.max((tMs - this.t) / 1000, 1e-3);
    const alpha = (cutoff) => 1 / (1 + 1 / (2 * Math.PI * cutoff * dt));
    this.dx += alpha(this.dCutoff) * ((x - this.x) / dt - this.dx);
    this.x += alpha(this.minCutoff + this.beta * Math.abs(this.dx)) * (x - this.x);
    this.t = tMs;
    return this.x;
  }
}

/** Largest, most central person wins; `crowded` when someone else is nearly as big. */
export function pickPerson(poses) {
  const scored = (poses ?? []).map((pose) => {
    const pts = pose.filter((l) => (l.visibility ?? 1) >= MIN_VIS);
    const use = pts.length >= 5 ? pts : pose;
    const xs = use.map((l) => l.x), ys = use.map((l) => l.y);
    const area = spread(xs) * spread(ys);
    const off = Math.hypot(mean(xs) - 0.5, mean(ys) - 0.5);
    return { pose, area, score: area * (1 - Math.min(1, off)) };
  }).sort((a, b) => b.score - a.score);
  if (!scored.length) return null;
  return { pose: scored[0].pose, crowded: scored.length > 1 && scored[1].area > 0.4 * scored[0].area };
}

/**
 * Exercise recipes. `signal` is the tracked angle; a rep goes rest → peak → rest.
 * `pick: 'best'` reads the most visible side (side-view lifts); `'all'` reads every visible side.
 * Each check gets { frames, peak, rest } and returns true when the rep is fine.
 */
export const EXERCISES = {
  squat: {
    name: 'Squat', views: ['side'], joints: ['shoulder', 'hip', 'knee', 'ankle'], pick: 'best',
    setup: 'Stand side-on, 2–3 m from the camera, phone at hip height, whole body in frame.',
    signal: (p, [s]) => knee(p, s), rest: 170, peak: 70, partialCue: 'Squat deeper', // hips at knee level ≈ 60–70° at the knee
    checks: [
      { cue: 'Get your hips down to knee level', test: ({ peak: { p, sides: [s] } }) => J(p, s, 'hip').y >= J(p, s, 'knee').y - 0.1 * dist(J(p, s, 'hip'), J(p, s, 'knee')) },
      { cue: 'Keep your chest up', test: ({ peak }) => torsoLean(peak.p) <= 50 },
    ],
  },
  lunge: {
    name: 'Lunge', views: ['side'], joints: ['hip', 'knee', 'ankle'], pick: 'all',
    setup: 'Stand side-on, 2–3 m from the camera, phone at hip height, whole body in frame.',
    signal: (p, sides) => Math.min(...sides.map((s) => knee(p, s))), rest: 170, peak: 90, partialCue: 'Lunge deeper',
    checks: [
      { cue: 'Drop your back knee closer to the floor', test: ({ peak }) => peak.sig <= 105 },
      { cue: 'Keep your torso upright', test: ({ frames }) => Math.max(...frames.map((f) => torsoLean(f.p))) <= 25 },
    ],
  },
  pushup: {
    name: 'Push-up', views: ['side'], joints: ['shoulder', 'elbow', 'wrist', 'hip', 'ankle'], pick: 'best',
    setup: 'Side-on, camera near floor level about 2 m away, whole body in frame.',
    signal: (p, [s]) => elbow(p, s), rest: 165, peak: 85, partialCue: 'Go lower',
    checks: [
      { cue: 'Lower your chest closer to the floor', test: ({ peak }) => peak.sig <= 95 },
      { cue: 'Lock your arms out at the top', test: ({ rest }) => rest >= 155 },
      { cue: "Don't let your hips sag", test: ({ frames }) => Math.max(...frames.map((f) => hipOffset(f.p, f.sides[0]))) <= 0.06 },
      { cue: 'Bring your hips down in line with your body', test: ({ frames }) => Math.min(...frames.map((f) => hipOffset(f.p, f.sides[0]))) >= -0.08 },
    ],
  },
  curl: {
    name: 'Bicep curl', views: ['front', 'side'], joints: ['shoulder', 'elbow', 'wrist', 'hip'], pick: 'all',
    setup: 'Face the camera or stand side-on, 2–3 m away, hips to head in frame.',
    signal: (p, sides) => Math.min(...sides.map((s) => elbow(p, s))), rest: 160, peak: 50, partialCue: 'Curl all the way up',
    checks: [
      { cue: 'Squeeze all the way to the top', test: ({ peak }) => peak.sig <= 60 },
      { cue: 'Lower the weight all the way down', test: ({ rest }) => rest >= 145 },
      { cue: 'Keep your elbows by your sides', test: ({ frames }) => Math.max(...frames.flatMap((f) => f.sides.map((s) => armRaise(f.p, s)))) <= 35 },
      { cue: "Don't swing — keep your body still", views: ['side'], test: ({ frames }) => spread(frames.map((f) => torsoLean(f.p))) <= 12 },
    ],
  },
  press: {
    name: 'Shoulder press', views: ['front', 'side'], joints: ['shoulder', 'elbow', 'wrist'], pick: 'all',
    setup: 'Face the camera, 2–3 m away, hips to above your hands in frame.',
    signal: (p, sides) => Math.min(...sides.map((s) => elbow(p, s))), rest: 85, peak: 165, partialCue: 'Press all the way up',
    checks: [
      { cue: 'Lock out overhead', test: ({ peak }) => peak.sig >= 155 && peak.sides.every((s) => J(peak.p, s, 'wrist').y < peak.p[NOSE].y) },
      { cue: 'Bring the weights down to shoulder height', test: ({ rest }) => rest <= 100 },
      { cue: 'Press both arms evenly', views: ['front'], test: ({ peak: { p, sides } }) => sides.length < 2 || Math.abs(J(p, 'left', 'wrist').y - J(p, 'right', 'wrist').y) <= 0.12 * torsoLen(p) },
      { cue: "Brace — don't lean back", views: ['side'], test: ({ frames }) => spread(frames.map((f) => torsoLean(f.p))) <= 15 },
    ],
  },
  raise: {
    name: 'Lateral raise', views: ['front'], joints: ['shoulder', 'elbow', 'hip'], pick: 'all',
    setup: 'Face the camera, 2–3 m away, hips to above your hands in frame.',
    signal: (p, sides) => mean(sides.map((s) => armRaise(p, s))), rest: 15, peak: 85, partialCue: 'Raise your arms to shoulder height',
    checks: [
      { cue: 'Raise to shoulder height', test: ({ peak }) => peak.sig >= 75 },
      { cue: 'Stop at shoulder height', test: ({ peak }) => peak.sig <= 105 },
      { cue: 'Raise both arms evenly', test: ({ peak: { p, sides } }) => sides.length < 2 || Math.abs(armRaise(p, 'left') - armRaise(p, 'right')) <= 15 },
    ],
  },
};

/** Which sides of the body are usable for this exercise, or a reason why none are. */
function usableSides(ex, p) {
  const ok = (s) => ex.joints.every((j) => J(p, s, j).v >= MIN_VIS);
  const vis = (s) => ex.joints.reduce((a, j) => a + J(p, s, j).v, 0);
  const sides = ['left', 'right'].filter(ok);
  if (ex.pick === 'best' && sides.length) return [sides.sort((a, b) => vis(b) - vis(a))[0]];
  return sides;
}

function missingMessage(ex, p) {
  const gone = ex.joints.filter((j) => J(p, 'left', j).v < MIN_VIS && J(p, 'right', j).v < MIN_VIS);
  return gone.length ? `Step back — I can't see your ${gone.map((j) => JOINT_NAMES[j]).join(' and ')}` : 'Step back so your whole body is in frame';
}

function viewMessage(ex) {
  if (ex.views.length > 1) return 'Face the camera or turn fully side-on';
  return ex.views[0] === 'side' ? 'Turn side-on to the camera' : 'Face the camera';
}

/** One session = one set. Feed it every frame; it returns what to draw and say. */
export function createSession(ex) {
  const filters = [];
  const reps = [];
  let lastT = -Infinity, seenT = -Infinity, ratio = null, rest = ex.rest;
  let phase = 'arming', armed = 0, rep = null;
  const counted = () => reps.filter((r) => r.counted).length;

  function finish(r, t) {
    const duration = t - r.start;
    if (duration < 250 || r.frames.length < 3) return null; // jitter, not a rep
    const peak = r.frames.reduce((a, f) => (f.prog > a.prog ? f : a));
    const ok = peak.prog >= 0.7;
    const reliable = r.gaps <= 0.3 * (r.frames.length + r.gaps);
    const ctx = { frames: r.frames, peak, rest };
    const issues = !ok ? [ex.partialCue]
      : !reliable ? []
      : ex.checks.filter((c) => (!c.views || c.views.includes(peak.view)) && !c.test(ctx)).map((c) => c.cue);
    if (ok && reliable && duration < 600) issues.push('Slow down and control the rep');
    const rec = { n: ok ? counted() + 1 : null, counted: ok, reliable, issues, duration: Math.round(duration), range: Math.round(peak.sig) };
    reps.push(rec);
    return rec;
  }

  function update(poses, t, w, h) {
    if (!(t > lastT)) return null; // MediaPipe VIDEO mode needs strictly increasing timestamps
    lastT = t;
    const out = { person: null, sides: [], status: null, view: null, progress: 0, rep: null, phase, reps: counted() };
    const gap = (msg) => { out.status = msg; if (rep) rep.gaps++; return out; };

    const chosen = pickPerson(poses);
    if (!chosen) return gap('Step into view');
    if (t - seenT > 500) filters.length = 0; // person was lost: restart smoothing
    seenT = t;
    const p = chosen.pose.map((l, i) => {
      const f = (filters[i] ??= [new OneEuro(), new OneEuro()]);
      return { x: f[0].filter(l.x * w, t), y: f[1].filter(l.y * h, t), v: l.visibility ?? 1 };
    });
    out.person = p;
    if (chosen.crowded) out.status = 'More than one person in view — tracking the closest';

    const r = Math.abs(p[11].x - p[12].x) / (torsoLen(p) || 1);
    ratio = ratio === null ? r : ratio + 0.2 * (r - ratio);
    const view = (out.view = ratio > 0.5 ? 'front' : ratio < 0.35 ? 'side' : 'angled');
    if (!ex.views.includes(view)) return gap(viewMessage(ex));
    const sides = (out.sides = usableSides(ex, p));
    if (!sides.length) return gap(missingMessage(ex, p));

    const sig = ex.signal(p, sides);
    const prog = (rest - sig) / (rest - ex.peak);
    out.progress = clamp(prog, 0, 1);
    if (phase !== 'moving' && prog < 0.25) rest = clamp(rest + 0.05 * (sig - rest), ex.rest - 25, ex.rest + 25);

    if (phase === 'arming') {
      armed = prog < 0.25 ? armed + 1 : 0;
      if (armed >= 8) phase = 'ready';
      else out.status ??= 'Get into your start position';
    } else if (phase === 'ready' && prog > 0.3) {
      phase = 'moving';
      rep = { start: t, frames: [], gaps: 0 };
    }
    if (phase === 'moving') {
      rep.frames.push({ t, sig, prog, p, sides, view });
      if (prog < 0.15) {
        out.rep = finish(rep, t);
        rep = null;
        phase = 'ready';
      } else if (t - rep.start > 15000) {
        rep = null; // stalled or walked off mid-rep
        phase = 'arming';
      }
    }
    out.phase = phase;
    out.reps = counted();
    return out;
  }

  return { update, reps, exercise: ex };
}
