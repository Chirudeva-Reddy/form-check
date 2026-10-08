// Synthetic stick figures with known joint angles, so the engine is tested without any video.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { angle, OneEuro, pickPerson, createSession, EXERCISES } from './engine.js';

const rad = (d) => (d * Math.PI) / 180;
const rot = ({ x, y }, d) => ({ x: x * Math.cos(rad(d)) - y * Math.sin(rad(d)), y: x * Math.sin(rad(d)) + y * Math.cos(rad(d)) });
const add = (a, v, k = 1) => ({ x: a.x + k * v.x, y: a.y + k * v.y });

function pose(points, vis = 0.95) {
  const p = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, z: 0, visibility: vis }));
  for (const [i, pt, v] of points) p[i] = { x: pt.x, y: pt.y, z: 0, visibility: v ?? vis };
  return p;
}

// Side view, facing +x. Shin tilts forward up to 30°, the thigh takes the rest of the knee bend.
function side({ knee = 175, lean = 5, ankleVis }) {
  const a = Math.min(30, (180 - knee) / 2), b = 180 - knee - a;
  const A = { x: 0.5, y: 0.9 };
  const K = add(A, { x: Math.sin(rad(a)), y: -Math.cos(rad(a)) }, 0.22);
  const Hp = add(K, { x: -Math.sin(rad(b)), y: -Math.cos(rad(b)) }, 0.22);
  const S = add(Hp, { x: Math.sin(rad(lean)), y: -Math.cos(rad(lean)) }, 0.28);
  const pts = [[0, add(S, { x: 0.03, y: -0.08 })]];
  for (const [o, dx] of [[0, 0], [1, 0.01]]) {
    const sh = { x: dx, y: 0 };
    pts.push([11 + o, add(S, sh)], [13 + o, add(S, { x: dx, y: 0.13 })], [15 + o, add(S, { x: dx, y: 0.26 })],
      [23 + o, add(Hp, sh)], [25 + o, add(K, sh)], [27 + o, add(A, sh), ankleVis]);
  }
  return pose(pts);
}

// Front view, facing the camera. Arm = upper-arm raise from vertical + elbow angle.
function front({ left = {}, right = {} } = {}) {
  const pts = [[0, { x: 0.5, y: 0.25 }]];
  for (const [s, o, dir, arm] of [['left', 0, 1, left], ['right', 1, -1, right]]) {
    const { raise = 5, elbow = 170 } = arm;
    const S = { x: 0.5 + dir * 0.1, y: 0.35 };
    const v = { x: dir * Math.sin(rad(raise)), y: Math.cos(rad(raise)) };
    const E = add(S, v, 0.13);
    const W = add(E, rot(v, -dir * (180 - elbow)), 0.12);
    pts.push([11 + o, S], [13 + o, E], [15 + o, W], [23 + o, { x: 0.5 + dir * 0.07, y: 0.63 }],
      [25 + o, { x: 0.5 + dir * 0.07, y: 0.78 }], [27 + o, { x: 0.5 + dir * 0.07, y: 0.92 }]);
  }
  return pose(pts);
}

// Push-up, side view: wrist under elbow, shoulder placed by the elbow angle, hip sag off the shoulder→ankle line.
function pushup({ elbow = 165, sag = 0 }) {
  const W = { x: 0.35, y: 0.85 }, E = { x: 0.35, y: 0.71 }, A = { x: 0.85, y: 0.85 };
  const S = add(E, rot({ x: 0, y: 1 }, elbow), 0.15);
  const Hp = { x: (S.x + A.x) / 2, y: (S.y + A.y) / 2 + sag };
  const pts = [[0, add(S, { x: -0.06, y: 0 })]];
  for (const [o, dx] of [[0, 0], [1, 0.01]]) {
    const sh = { x: dx, y: 0 };
    pts.push([11 + o, add(S, sh)], [13 + o, add(E, sh)], [15 + o, add(W, sh)], [23 + o, add(Hp, sh)],
      [25 + o, add({ x: (Hp.x + A.x) / 2, y: (Hp.y + A.y) / 2 }, sh)], [27 + o, add(A, sh)]);
  }
  return pose(pts);
}

// value goes rest → peak → rest once per period, after a 1 s lead-in, for `reps` reps.
const cycle = (t, rest, peak, reps, period = 2) =>
  t < 1 || t > 1 + reps * period ? rest : rest + ((peak - rest) * (1 - Math.cos((2 * Math.PI * (t - 1)) / period))) / 2;

function run(id, poseAt, seconds, { fps = 30, jitter = 0 } = {}) {
  const s = createSession(EXERCISES[id]);
  let last;
  const events = [];
  for (let i = 0; i < seconds * fps; i++) {
    const t = i / fps;
    const got = poseAt(t); // one pose, or [] for nobody in frame
    const ps = (got.length ? [got] : []).map((p) => p.map((l, j) => ({ ...l, x: l.x + jitter * Math.sin(i * 7.1 + j), y: l.y + jitter * Math.cos(i * 5.3 + j) })));
    const out = s.update(ps, t * 1000, 1000, 1000);
    if (out?.rep) events.push(out.rep);
    last = out;
  }
  return { events, last, counted: events.filter((e) => e.counted) };
}
const issues = (events) => [...new Set(events.flatMap((e) => e.issues))];

test('angle: right angle and straight line', () => {
  assert.equal(Math.round(angle({ x: 0, y: 1 }, { x: 0, y: 0 }, { x: 1, y: 0 })), 90);
  assert.equal(Math.round(angle({ x: -1, y: 0 }, { x: 0, y: 0 }, { x: 1, y: 0 })), 180);
});

test('OneEuro smooths jitter and follows a real move', () => {
  const f = new OneEuro();
  const raw = [], out = [];
  for (let i = 0; i < 90; i++) {
    const x = 500 + 10 * Math.sin(i * 2.7);
    raw.push(x);
    out.push(f.filter(x, (i * 1000) / 30));
  }
  const sd = (xs) => Math.sqrt(xs.reduce((a, x) => a + (x - 500) ** 2, 0) / xs.length);
  assert.ok(sd(out.slice(30)) < 0.7 * sd(raw.slice(30)));
  for (let i = 90; i < 150; i++) f.filter(800, (i * 1000) / 30);
  assert.ok(Math.abs(f.x - 800) < 5);
});

test('pickPerson takes the biggest person and flags a crowd', () => {
  const big = side({}), small = big.map((l) => ({ ...l, x: 0.1 + l.x * 0.2, y: l.y * 0.2 }));
  assert.equal(pickPerson([small, big]).pose, big);
  assert.equal(pickPerson([small, big]).crowded, false);
  assert.equal(pickPerson([big, big.map((l) => ({ ...l }))]).crowded, true);
});

test('squat: 5 full reps counted, no cues', () => {
  const knee = (t) => cycle(t, 175, 62, 5);
  const { counted, events } = run('squat', (t) => side({ knee: knee(t), lean: 5 + (175 - knee(t)) * 0.3 }), 13);
  assert.equal(counted.length, 5);
  assert.deepEqual(issues(events), []);
});

test('squat: survives landmark jitter', () => {
  const knee = (t) => cycle(t, 175, 62, 5);
  const { counted } = run('squat', (t) => side({ knee: knee(t) }), 13, { jitter: 0.004 });
  assert.equal(counted.length, 5);
});

test('squat: above parallel counts but asks for depth; very shallow is a partial rep', () => {
  const half = run('squat', (t) => side({ knee: cycle(t, 175, 95, 3) }), 9);
  assert.equal(half.counted.length, 3);
  assert.deepEqual(issues(half.events), ['Get your hips down to knee level']);
  const shallow = run('squat', (t) => side({ knee: cycle(t, 175, 120, 3) }), 9);
  assert.equal(shallow.counted.length, 0);
  assert.deepEqual(issues(shallow.events), ['Squat deeper']);
});

test('squat: too much forward lean is flagged', () => {
  const r = run('squat', (t) => side({ knee: cycle(t, 175, 62, 3), lean: 5 + (175 - cycle(t, 175, 62, 3)) * 0.55 }), 9);
  assert.equal(r.counted.length, 3);
  assert.ok(issues(r.events).includes('Keep your chest up'));
});

test('squat: wrong view or hidden feet → no reps, a reason instead', () => {
  const wrong = run('squat', () => front(), 4);
  assert.equal(wrong.counted.length, 0);
  assert.equal(wrong.last.status, 'Turn side-on to the camera');
  const hidden = run('squat', (t) => side({ knee: cycle(t, 175, 62, 1), ankleVis: 0.2 }), 4);
  assert.equal(hidden.counted.length, 0);
  assert.match(hidden.last.status, /feet/);
});

test('squat: nobody in frame', () => {
  assert.equal(run('squat', () => [], 1).last.status, 'Step into view');
});

test('lunge: counts reps, flags leaning forward', () => {
  const ok = run('lunge', (t) => side({ knee: cycle(t, 175, 95, 3) }), 9);
  assert.equal(ok.counted.length, 3);
  assert.deepEqual(issues(ok.events), []);
  const leaning = run('lunge', (t) => side({ knee: cycle(t, 175, 95, 3), lean: 35 }), 9);
  assert.ok(issues(leaning.events).includes('Keep your torso upright'));
});

test('push-up: clean reps; sagging hips flagged', () => {
  const ok = run('pushup', (t) => pushup({ elbow: cycle(t, 165, 80, 3) }), 9);
  assert.equal(ok.counted.length, 3);
  assert.deepEqual(issues(ok.events), []);
  const sag = run('pushup', (t) => pushup({ elbow: cycle(t, 165, 80, 3), sag: 0.06 }), 9);
  assert.ok(issues(sag.events).includes("Don't let your hips sag"));
});

test('curl: clean reps; elbows drifting out flagged', () => {
  const arm = (t, raise) => ({ raise, elbow: cycle(t, 165, 45, 4) });
  const ok = run('curl', (t) => front({ left: arm(t, 5), right: arm(t, 5) }), 11);
  assert.equal(ok.counted.length, 4);
  assert.deepEqual(issues(ok.events), []);
  const flare = run('curl', (t) => front({ left: arm(t, 45), right: arm(t, 45) }), 11);
  assert.ok(issues(flare.events).includes('Keep your elbows by your sides'));
});

test('press: starting at the top is not a rep; uneven arms flagged', () => {
  const arm = (t, top = 170) => { const e = cycle(t, 85, 168, 3); return { elbow: e, raise: 90 + ((e - 85) / 83) * (top - 90) }; };
  const startTop = (t) => (t < 0.8 ? front({ left: { elbow: 168, raise: 170 }, right: { elbow: 168, raise: 170 } }) : front({ left: arm(t), right: arm(t) }));
  const ok = run('press', startTop, 9);
  assert.equal(ok.counted.length, 3);
  assert.deepEqual(issues(ok.events), []);
  const uneven = run('press', (t) => front({ left: arm(t), right: arm(t, 125) }), 9);
  assert.ok(issues(uneven.events).includes('Press both arms evenly'));
});

test('lateral raise: clean, too low (partial), too high', () => {
  const raise = (top) => (t) => { const a = { raise: cycle(t, 12, top, 3), elbow: 165 }; return front({ left: a, right: a }); };
  const ok = run('raise', raise(85), 9);
  assert.equal(ok.counted.length, 3);
  assert.deepEqual(issues(ok.events), []);
  assert.equal(run('raise', raise(45), 9).counted.length, 0);
  assert.ok(issues(run('raise', raise(120), 9).events).includes('Stop at shoulder height'));
});

test('timestamps must increase', () => {
  const s = createSession(EXERCISES.squat);
  assert.ok(s.update([side({})], 100, 1000, 1000));
  assert.equal(s.update([side({})], 100, 1000, 1000), null);
});
