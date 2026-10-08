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
const hipHinge = (p, s) => angle(J(p, s, 'shoulder'), J(p, s, 'hip'), J(p, s, 'knee'));

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

/**
 * First pick: the largest, most central person. After that, stay on whoever is nearest the
 * previous position (as long as they're still a decent size), so a passer-by can't steal tracking.
 * `crowded` when someone else is nearly as big.
 */
export function pickPerson(poses, prevCentre = null) {
  const scored = (poses ?? []).map((pose) => {
    const pts = pose.filter((l) => (l.visibility ?? 1) >= MIN_VIS);
    const use = pts.length >= 5 ? pts : pose;
    const xs = use.map((l) => l.x), ys = use.map((l) => l.y);
    const area = spread(xs) * spread(ys);
    const centre = { x: mean(xs), y: mean(ys) };
    return { pose, area, centre, score: area * (1 - Math.min(1, dist(centre, { x: 0.5, y: 0.5 }))) };
  }).sort((a, b) => b.score - a.score);
  if (!scored.length) return null;
  const biggest = Math.max(...scored.map((s) => s.area));
  const best = prevCentre
    ? scored.filter((s) => s.area >= 0.5 * biggest).sort((a, b) => dist(a.centre, prevCentre) - dist(b.centre, prevCentre))[0]
    : scored[0];
  return { pose: best.pose, centre: best.centre, crowded: scored.some((s) => s !== best && s.area > 0.4 * best.area) };
}

/**
 * Exercise recipes. `signal` is the tracked angle; a rep goes rest → peak → rest.
 * `pick: 'best'` reads the most visible side (side-view lifts); `'all'` reads every visible side.
 * Each check gets { frames, peak, rest, start } and returns true when the rep is fine
 * (`start` is the last frame in the start position before the rep began).
 */
export const EXERCISES = {
  squat: {
    joint: knee, name: 'Squat', views: ['side'], joints: ['shoulder', 'hip', 'knee', 'ankle'], pick: 'best',
    setup: 'Stand side-on, 2–3 m from the camera, phone at hip height, whole body in frame.',
    signal: (p, [s]) => knee(p, s), rest: 170, peak: 70, partialCue: 'Squat deeper', // hips at knee level ≈ 60–70° at the knee
    checks: [
      { cue: 'Get your hips down to knee level', test: ({ peak: { p, sides: [s] } }) => J(p, s, 'hip').y >= J(p, s, 'knee').y - 0.1 * dist(J(p, s, 'hip'), J(p, s, 'knee')) },
      { cue: 'Keep your chest up', test: ({ peak }) => torsoLean(peak.p) <= 50 },
    ],
  },
  lunge: {
    joint: knee, name: 'Lunge', views: ['side'], joints: ['hip', 'knee', 'ankle'], pick: 'all',
    setup: 'Stand side-on, 2–3 m from the camera, phone at hip height, whole body in frame.',
    signal: (p, sides) => Math.min(...sides.map((s) => knee(p, s))), rest: 170, peak: 90, partialCue: 'Lunge deeper',
    checks: [
      { cue: 'Drop your back knee closer to the floor', test: ({ peak }) => peak.sig <= 105 },
      { cue: 'Keep your torso upright', test: ({ frames }) => Math.max(...frames.map((f) => torsoLean(f.p))) <= 25 },
    ],
  },
  deadlift: {
    // Each rep starts on the floor, so the floor is the start position and a rep ends when the bar is back down.
    joint: hipHinge, liftFirst: true, name: 'Deadlift', views: ['side'], joints: ['shoulder', 'hip', 'knee', 'ankle'], pick: 'best',
    setup: 'Stand side-on, 2–3 m from the camera, phone at hip height, whole body and the bar in frame. Start with the bar on the floor.',
    signal: (p, [s]) => hipHinge(p, s), rest: 85, peak: 170, partialCue: 'Stand all the way up',
    checks: [
      { cue: 'Lock out at the top: hips through, stand tall', test: ({ peak }) => peak.sig >= 163 },
      { cue: "Hips too low: push them back, don't squat the bar up", test: ({ start: { p, sides: [s] } }) => knee(p, s) >= 100 },
    ],
  },
  rdl: {
    joint: hipHinge, name: 'Romanian deadlift', views: ['side'], joints: ['shoulder', 'hip', 'knee', 'ankle'], pick: 'best',
    setup: 'Stand side-on, 2–3 m from the camera, phone at hip height, whole body in frame. Start standing tall.',
    signal: (p, [s]) => hipHinge(p, s), rest: 170, peak: 95, partialCue: 'Hinge further: push your hips back',
    checks: [
      { cue: 'Keep your knees soft but still: push your hips back instead', test: ({ peak: { p, sides: [s] } }) => knee(p, s) >= 140 },
      { cue: 'Stand all the way up between reps', test: ({ rest }) => rest >= 160 },
    ],
  },
  row: {
    joint: elbow, liftFirst: true, name: 'Bent-over row', views: ['side'], joints: ['shoulder', 'elbow', 'wrist', 'hip'], pick: 'best',
    setup: 'Side-on, 2–3 m from the camera, phone at hip height, whole body in frame. Hinge forward and let the weight hang.',
    signal: (p, [s]) => elbow(p, s), rest: 165, peak: 75, partialCue: 'Pull all the way to your body',
    checks: [
      { cue: 'Pull your elbow past your body', test: ({ peak }) => peak.sig <= 90 },
      { cue: 'Let your arms hang straight between reps', test: ({ rest }) => rest >= 150 },
      { cue: 'Hinge forward more: chest toward the floor', test: ({ frames }) => mean(frames.map((f) => torsoLean(f.p))) >= 30 },
      { cue: "Keep your torso still: don't heave the weight", test: ({ frames }) => spread(frames.map((f) => torsoLean(f.p))) <= 15 },
    ],
  },
  pushup: {
    joint: elbow, name: 'Push-up', views: ['side'], joints: ['shoulder', 'elbow', 'wrist', 'hip', 'ankle'], pick: 'best',
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
    joint: elbow, liftFirst: true, name: 'Bicep curl', views: ['front', 'side'], joints: ['shoulder', 'elbow', 'wrist', 'hip'], pick: 'all',
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
    joint: elbow, liftFirst: true, name: 'Shoulder press', views: ['front', 'side'], joints: ['shoulder', 'elbow', 'wrist'], pick: 'all',
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
    joint: armRaise, liftFirst: true, name: 'Lateral raise', views: ['front'], joints: ['shoulder', 'elbow', 'hip'], pick: 'all',
    setup: 'Face the camera, 2–3 m away, hips to above your hands in frame.',
    signal: (p, sides) => mean(sides.map((s) => armRaise(p, s))), rest: 15, peak: 85, partialCue: 'Raise your arms to shoulder height',
    checks: [
      { cue: 'Raise to shoulder height', test: ({ peak }) => peak.sig >= 75 },
      { cue: 'Stop at shoulder height', test: ({ peak }) => peak.sig <= 105 },
      { cue: 'Raise both arms evenly', test: ({ peak: { p, sides } }) => sides.length < 2 || Math.abs(armRaise(p, 'left') - armRaise(p, 'right')) <= 15 },
    ],
  },
};

/**
 * Which sides of the body are usable for this exercise. For 'best', keep the previous side unless the
 * other one is clearly more visible, so the reading doesn't flip between near and far limbs frame to frame.
 */
function usableSides(ex, p, keep) {
  const ok = (s) => ex.joints.every((j) => J(p, s, j).v >= MIN_VIS);
  const vis = (s) => ex.joints.reduce((a, j) => a + J(p, s, j).v, 0);
  const sides = ['left', 'right'].filter(ok);
  if (ex.pick !== 'best' || !sides.length) return sides;
  const best = sides.sort((a, b) => vis(b) - vis(a))[0];
  return [sides.includes(keep) && vis(keep) >= vis(best) - 0.1 * ex.joints.length ? keep : best];
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
  let phase = 'arming', armedAt = null, rep = null, ready = null;
  let centre = null, prevRaw = null, prevP = null, glitches = 0, side = null;
  const counted = () => reps.filter((r) => r.counted).length;

  function finish(r, t) {
    const duration = t - r.start;
    if (duration < 250 || r.frames.length < 3) return null; // jitter, not a rep
    const peak = r.frames.reduce((a, f) => (f.prog > a.prog ? f : a));
    const ok = peak.prog >= 0.7;
    const reliable = r.gaps <= 0.3 * (r.frames.length + r.gaps);
    const ctx = { frames: r.frames, peak, rest, start: r.startFrame ?? r.frames[0] };
    const issues = !ok ? [ex.partialCue]
      : !reliable ? []
      : ex.checks.filter((c) => (!c.views || c.views.includes(peak.view)) && !c.test(ctx)).map((c) => c.cue);
    if (ok && reliable && duration < 600) issues.push('Slow down and control the rep');
    const rec = { n: ok ? counted() + 1 : null, counted: ok, reliable, issues, duration: Math.round(duration), range: Math.round(peak.sig), metrics: metrics(r, peak, t) };
    reps.push(rec);
    return rec;
  }

  // ponytail: phases run between the start/finish thresholds (30% / 15% of travel), so they read slightly short of true end-to-end times.
  function metrics(r, peak, t) {
    const first = peak.t - r.start, second = t - peak.t;
    const lift = ex.liftFirst ? first : second; // concentric phase: working against the weight
    const liftFrom = ex.liftFirst ? r.frames[0] : r.frames.at(-1);
    const speed = lift > 0 ? Math.round(Math.abs(peak.sig - liftFrom.sig) / (lift / 1000)) : null; // mean angular speed, °/s
    const best = Math.max(speed ?? 0, ...reps.map((x) => x.metrics?.speed ?? 0));
    const twoSides = peak.sides.length === 2 && peak.view === 'front' && ex.pick === 'all' && ex.joint !== knee;
    return {
      lift: Math.round(lift), lower: Math.round(ex.liftFirst ? second : first), liftFirst: !!ex.liftFirst,
      travel: Math.round(Math.abs(rest - peak.sig)),
      speed, speedLoss: speed && best ? Math.round(100 * (1 - speed / best)) : null, // vs fastest rep so far in the set
      lean: peak.view === 'side' ? Math.round(Math.max(...r.frames.map((f) => torsoLean(f.p)))) : null,
      gap: twoSides && ex.joint ? Math.round(Math.abs(ex.joint(peak.p, 'left') - ex.joint(peak.p, 'right'))) : null,
      trace: r.frames.map((f) => [Math.round(f.t - r.start), Math.round(f.sig)]),
      peakAt: Math.round(first), rest: Math.round(rest), target: ex.peak,
    };
  }

  function update(poses, t, w, h) {
    if (!(t > lastT)) return null; // MediaPipe VIDEO mode needs strictly increasing timestamps
    const dt = lastT === -Infinity ? 33 : t - lastT;
    lastT = t;
    const ease = (tauMs) => 1 - Math.exp(-dt / tauMs); // time-based, so 15 fps files and 30 fps cameras behave alike
    const out = { person: null, sides: [], status: null, view: null, progress: 0, rep: null, phase, reps: counted() };
    const gap = (msg) => { out.status = msg; if (rep) rep.gaps++; return out; };

    if (t - seenT > 500) { filters.length = 0; centre = prevRaw = prevP = null; } // person was lost: start fresh
    const chosen = pickPerson(poses, centre);
    if (!chosen) return gap('Step into view');
    seenT = t;
    centre = chosen.centre;
    const raw = chosen.pose.map((l) => ({ x: l.x * w, y: l.y * h, v: l.visibility ?? 1 }));

    // A joint that jumps further than a body can move in one frame is a model glitch: skip the frame.
    // Three in a row means the person (or camera) really moved, so accept it and restart smoothing.
    const limit = Math.max(0.3, (8 * dt) / 1000) * (prevRaw ? torsoLen(prevRaw) || 1 : 1);
    const jumped = prevRaw && raw.some((l, i) => i >= 11 && i <= 28 && l.v >= MIN_VIS && prevRaw[i].v >= MIN_VIS && dist(l, prevRaw[i]) > limit);
    if (jumped && glitches < 3) {
      glitches++;
      out.person = prevP;
      return gap(null);
    }
    if (jumped) filters.length = 0;
    glitches = 0;
    prevRaw = raw;

    const p = raw.map((l, i) => {
      const f = (filters[i] ??= [new OneEuro(), new OneEuro()]);
      return { x: f[0].filter(l.x, t), y: f[1].filter(l.y, t), v: l.v };
    });
    out.person = prevP = p;
    if (chosen.crowded) out.status = 'More than one person in view — tracking the closest';

    const r = Math.abs(p[11].x - p[12].x) / (torsoLen(p) || 1);
    ratio = ratio === null ? r : ratio + ease(300) * (r - ratio);
    const view = (out.view = ratio > 0.5 ? 'front' : ratio < 0.35 ? 'side' : 'angled');
    if (!ex.views.includes(view)) return gap(viewMessage(ex));
    const sides = (out.sides = usableSides(ex, p, side));
    if (!sides.length) return gap(missingMessage(ex, p));
    side = sides[0];

    const sig = ex.signal(p, sides);
    const prog = (rest - sig) / (rest - ex.peak);
    out.progress = clamp(prog, 0, 1);
    if (phase !== 'moving' && prog < 0.25) rest = clamp(rest + ease(1000) * (sig - rest), ex.rest - 25, ex.rest + 25);

    if (phase === 'arming') {
      armedAt = prog < 0.25 ? armedAt ?? t : null;
      if (armedAt !== null && t - armedAt >= 300) phase = 'ready';
      else out.status ??= 'Get into your start position';
    } else if (phase === 'ready' && prog > 0.3) {
      phase = 'moving';
      rep = { start: t, frames: [], gaps: 0, startFrame: ready };
      ready = null;
    }
    // The most start-like frame since the last rep: the bottom of a deadlift, standing tall for a squat.
    if (phase === 'ready' && !(ready?.prog <= prog)) ready = { t, sig, prog, p, sides, view };
    if (phase === 'moving') {
      rep.frames.push({ t, sig, prog, p, sides, view });
      if (prog < 0.15) {
        out.rep = finish(rep, t);
        rep = null;
        phase = 'ready';
      } else if (t - rep.start > 15000) {
        rep = null; // stalled or walked off mid-rep
        phase = 'arming';
        armedAt = null;
      }
    }
    out.phase = phase;
    out.reps = counted();
    return out;
  }

  return { update, reps, exercise: ex };
}
