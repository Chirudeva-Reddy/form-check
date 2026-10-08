import { PoseLandmarker, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/vision_bundle.mjs';
import { EXERCISES, SIDES, createSession } from './engine.js';

const WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/wasm';
const VARIANT = new URLSearchParams(location.search).has('lite') ? 'lite' : 'full'; // ?lite for older phones
const MODEL = `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_${VARIANT}/float16/1/pose_landmarker_${VARIANT}.task`;

const $ = (id) => document.getElementById(id);
const [exercise, setup, start, file, flip, stop, sound, sessionEl, stage, video, canvas, count, meter, status, cue, summary, dash, sample] =
  ['exercise', 'setup', 'start', 'file', 'flip', 'stop', 'sound', 'session', 'stage', 'video', 'canvas', 'count', 'meter', 'status', 'cue', 'summary', 'dash', 'sample'].map($);
const ctx = canvas.getContext('2d');

let landmarker, session, source, stream, wakeLock, running = false;
let facing = 'user', sent = 0, prevMedia = null;

for (const [id, ex] of Object.entries(EXERCISES)) exercise.add(new Option(ex.name, id));
showSetup();

exercise.onchange = () => { showSetup(); if (running) newSession(); };
start.onclick = startCamera;
flip.onclick = () => { facing = facing === 'user' ? 'environment' : 'user'; startCamera(); };
stop.onclick = () => end(true);
file.onchange = async () => {
  const f = file.files[0];
  file.value = '';
  if (!f) return;
  unlockSpeech();
  end(false);
  video.srcObject = null;
  video.src = URL.createObjectURL(f);
  stage.classList.remove('mirror');
  await begin('file');
};

function showSetup() {
  setup.textContent = `Setup: ${EXERCISES[exercise.value].setup}`;
}

function newSession() {
  session = createSession(EXERCISES[exercise.value]);
  prevMedia = null;
  cue.textContent = '';
  dash.innerHTML = '<p class="empty">Your first rep will appear here.</p>';
  hud(null);
}

async function startCamera() {
  unlockSpeech();
  end(false);
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  } catch (e) {
    setup.textContent = `Couldn't open the camera (${e.name}). Check the browser's camera permission.`;
    return;
  }
  video.removeAttribute('src');
  video.srcObject = stream;
  stage.classList.toggle('mirror', facing === 'user');
  try { wakeLock = await navigator.wakeLock?.request('screen'); } catch { /* not supported: screen may sleep */ }
  await begin('camera');
}

async function begin(kind) {
  source = kind;
  sessionEl.hidden = false;
  sessionEl.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
  summary.hidden = true;
  stop.hidden = false;
  flip.hidden = kind !== 'camera';
  newSession();
  status.textContent = 'Loading the pose model…';
  try {
    await loadModel();
    if (kind === 'camera') await video.play();
    else if (video.readyState < 1) await new Promise((r, j) => { video.onloadedmetadata = r; video.onerror = () => j(new Error("this video format isn't supported")); });
  } catch (e) {
    status.textContent = `Couldn't start: ${e.message}`;
    return;
  }
  running = true;
  if (kind === 'camera') schedule();
  else stepThroughFile();
}

// Files are stepped at a fixed rate instead of played, so slow devices and background tabs see every sampled frame.
const FILE_FPS = 15;
let fileRun = 0;
async function stepThroughFile() {
  const run = ++fileRun; // a newer file or the camera cancels this loop
  const live = () => running && run === fileRun;
  for (let i = 0; live() && i / FILE_FPS <= video.duration; i++) {
    video.currentTime = i / FILE_FPS;
    await new Promise((r) => video.addEventListener('seeked', r, { once: true }));
    if (live()) process(video.currentTime * 1000);
  }
  if (live()) end(true);
}

async function loadModel() {
  if (landmarker) return;
  const fileset = await FilesetResolver.forVisionTasks(WASM);
  const opts = (delegate) => ({ baseOptions: { modelAssetPath: MODEL, delegate }, runningMode: 'VIDEO', numPoses: 2 });
  try {
    landmarker = await PoseLandmarker.createFromOptions(fileset, opts('GPU'));
  } catch {
    landmarker = await PoseLandmarker.createFromOptions(fileset, opts('CPU'));
  }
}

function schedule() {
  if (!running) return;
  if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(tick);
  else requestAnimationFrame(tick);
}

function tick() {
  if (!running) return;
  process(performance.now());
  schedule();
}

function process(media) {
  // One landmarker serves camera and files, so keep its clock strictly increasing across both.
  if (video.readyState >= 2 && (prevMedia === null || media > prevMedia)) {
    sent += prevMedia === null ? 1 : media - prevMedia;
    prevMedia = media;
    const res = landmarker.detectForVideo(video, sent);
    const out = session.update(res.landmarks, sent, video.videoWidth, video.videoHeight);
    if (out) {
      draw(out);
      hud(out);
      if (out.rep) announce(out.rep);
    }
  }
}

function draw(out) {
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const p = out.person;
  if (!p) return;
  const ex = EXERCISES[exercise.value];
  const tracked = new Set(out.sides.flatMap((s) => ex.joints.map((j) => SIDES[s][j])));
  const lw = Math.max(3, canvas.width / 240);
  ctx.lineWidth = lw;
  ctx.lineCap = 'round';
  for (const { start: a, end: b } of PoseLandmarker.POSE_CONNECTIONS) {
    if (a < 11 || b < 11 || p[a].v < 0.5 || p[b].v < 0.5) continue; // no face mesh
    ctx.strokeStyle = tracked.has(a) && tracked.has(b) ? '#4ade80' : 'rgba(255,255,255,.75)';
    ctx.beginPath();
    ctx.moveTo(p[a].x, p[a].y);
    ctx.lineTo(p[b].x, p[b].y);
    ctx.stroke();
  }
  for (let i = 11; i < p.length; i++) {
    if (p[i].v < 0.5) continue;
    ctx.fillStyle = tracked.has(i) ? '#4ade80' : '#fff';
    ctx.beginPath();
    ctx.arc(p[i].x, p[i].y, lw * 1.2, 0, 2 * Math.PI);
    ctx.fill();
  }
}

function hud(out) {
  count.textContent = out?.reps ?? 0;
  meter.firstElementChild.style.height = `${Math.round((out?.progress ?? 0) * 100)}%`;
  status.textContent = out?.status ?? '';
}

function announce(rep) {
  renderDash(dash, rep, session.exercise);
  const issue = rep.issues[0];
  cue.className = issue || !rep.reliable ? 'warn' : 'good';
  if (!rep.counted) cue.textContent = `Not counted: ${issue}`;
  else if (!rep.reliable) cue.textContent = `Rep ${rep.n}: couldn't see you clearly, so no form check`;
  else cue.textContent = `Rep ${rep.n}: ${issue ?? 'good'}`;
  if (sound.checked && 'speechSynthesis' in window) {
    speechSynthesis.cancel();
    speechSynthesis.speak(new SpeechSynthesisUtterance(rep.counted ? `${rep.n}${issue ? `. ${issue}` : ''}` : issue));
  }
}

let speechUnlocked = false;
function unlockSpeech() { // iOS only allows speech that starts from a tap
  if (speechUnlocked || !('speechSynthesis' in window)) return;
  speechSynthesis.speak(new SpeechSynthesisUtterance(''));
  speechUnlocked = true;
}

function end(showSummary) {
  running = false;
  fileRun++;
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  wakeLock?.release().catch(() => {});
  wakeLock = null;
  video.pause();
  stop.hidden = true;
  flip.hidden = true;
  status.textContent = '';
  if (showSummary && session?.reps.length) renderSummary();
}

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const sec = (ms) => (ms / 1000).toFixed(1);
// Left/right gaps above this get flagged: about twice the smallest change 2D video can detect (Munro 2012: 7.5-8.9°).
const GAP_FLAG = 15;

function renderSummary() {
  const reps = session.reps;
  const done = reps.filter((r) => r.counted);
  const tally = {};
  for (const r of reps) for (const i of r.issues) tally[i] = (tally[i] ?? 0) + 1;
  const top = Object.entries(tally).sort((a, b) => b[1] - a[1])[0];
  summary.innerHTML = `
    <h2>${esc(session.exercise.name)}: ${done.length} rep${done.length === 1 ? '' : 's'}${reps.length > done.length ? `, ${reps.length - done.length} not counted` : ''}</h2>
    <p class="muted">${top ? `Most common cue: ${esc(top[0])} (${top[1]}×)` : 'No form cues. Nice set.'}</p>
    <div class="table-scroll"><table>
      <tr><th>Rep</th><th>End range</th><th>Lower / lift</th><th>Lift speed</th><th>Feedback</th></tr>
      ${reps.map((r) => { const m = r.metrics; return `<tr><td class="mono">${r.n ?? '-'}</td><td class="mono">${r.range}°</td>
        <td class="mono">${sec(m.lower)} / ${sec(m.lift)} s</td><td class="mono">${m.speed ?? '-'} °/s</td>
        <td>${esc(!r.counted ? `Not counted: ${r.issues[0]}` : !r.reliable ? 'Not checked (view blocked)' : r.issues.join('; ') || 'Good')}</td></tr>`; }).join('')}
    </table></div>`;
  summary.hidden = false;
}

// Example rep for the hero: a smooth squat, 1.4 s down and 0.9 s up. Not a recording.
const SAMPLE = (() => {
  const trace = [];
  for (let t = 0; t <= 2300; t += 33) {
    const k = t < 1400 ? t / 1400 : 1 - (t - 1400) / 900;
    trace.push([t, Math.round(172 - 108 * (1 - Math.cos(Math.PI * k)) / 2)]);
  }
  return { n: 4, counted: true, reliable: true, issues: [], range: 64,
    metrics: { lower: 1400, lift: 900, liftFirst: false, travel: 108, speed: 120, speedLoss: 6, lean: 34, gap: null, trace, peakAt: 1400, rest: 172, target: 70 } };
})();

function renderDash(el, rep, ex) {
  const m = rep.metrics;
  const up = ex.peak > ex.rest; // press, raise, deadlift: the angle opens toward the end range
  const verdict = !rep.counted ? `Not counted: ${rep.issues[0]}` : !rep.reliable ? 'View blocked, form not checked' : rep.issues[0] ?? 'Good rep';
  const tiles = [
    ['End range', `${rep.range}<small>°</small>`, `Target ${up ? '≥' : '≤'} ${m.target}°`],
    ['Range of motion', `${m.travel}<small>°</small>`, `From ${m.rest}° at the start`],
    ['Lower / lift', `${sec(m.lower)}<small>/</small>${sec(m.lift)}<small>s</small>`, 'Eccentric / concentric'],
    ['Lift speed', `${m.speed ?? '-'}<small>°/s</small>`, m.speedLoss == null ? 'Mean joint speed while lifting'
      : m.speedLoss <= 0 ? 'Fastest rep this set' : `${m.speedLoss}% slower than your best rep`], // informational: 20% and 40% caps both work (Pareja-Blanco 2017)
    m.lean != null && ['Torso lean', `${m.lean}<small>°</small>`, 'Most forward, from vertical'],
    m.gap != null && ['Left / right gap', `${m.gap}<small>°</small>`, 'Joint angle difference at end range', m.gap > GAP_FLAG],
  ].filter(Boolean);
  el.innerHTML = `
    <div class="dash-head"><span class="dash-title">${esc(ex.name)}${rep.n ? `, rep ${rep.n}` : ''}</span>
      <span class="verdict ${rep.issues.length || !rep.reliable ? 'warn' : ''}">${esc(verdict)}</span></div>
    <div class="trace">${traceSvg(m, up)}<div class="tip"></div></div>
    <dl class="tiles">${tiles.map(([k, v, note, warn]) => `<div class="tile"><dt>${k}</dt><dd>${v}<div class="note ${warn ? 'warn' : ''}">${esc(note)}</div></dd></div>`).join('')}</dl>`;
  hoverTrace(el.querySelector('.trace'), m);
}

// Joint angle over the rep: one line, lifting phase in the accent, the target as a dashed reference.
const W = 400, H = 150, PL = 34, PR = 8, PT = 22, PB = 20;
function scales(m) {
  const ys = m.trace.map((p) => p[1]).concat(m.target);
  const lo = Math.floor((Math.min(...ys) - 8) / 10) * 10, hi = Math.ceil((Math.max(...ys) + 8) / 10) * 10;
  const end = m.trace.at(-1)[0] || 1;
  return { x: (t) => PL + (t / end) * (W - PL - PR), y: (a) => PT + (1 - (a - lo) / (hi - lo)) * (H - PT - PB), lo, hi, end };
}
function traceSvg(m, up) {
  const { x, y, lo, hi } = scales(m);
  const path = (pts) => pts.map(([t, a], i) => `${i ? 'L' : 'M'}${x(t).toFixed(1)},${y(a).toFixed(1)}`).join('');
  const k = m.trace.findIndex((p) => p[0] >= m.peakAt);
  const a = m.trace.slice(0, k + 1), b = m.trace.slice(k);
  const [liftPts, lowerPts] = m.liftFirst ? [a, b] : [b, a];
  const peak = m.trace[k];
  const grid = [lo, (lo + hi) / 2, hi].map((v) => `<line x1="${PL}" x2="${W - PR}" y1="${y(v)}" y2="${y(v)}" stroke="var(--line)"/>
    <text x="${PL - 8}" y="${y(v) + 4}" text-anchor="end" font-size="11" fill="var(--muted)" font-family="var(--mono)">${Math.round(v)}°</text>`).join('');
  const mid = (pts) => x((pts[0][0] + pts.at(-1)[0]) / 2);
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Joint angle through the rep: ${m.rest}° at the start, ${peak[1]}° at end range, target ${m.target}°">
    ${grid}
    <line x1="${PL}" x2="${W - PR}" y1="${y(m.target)}" y2="${y(m.target)}" stroke="var(--muted)" stroke-dasharray="3 4"/>
    <text x="${W - PR}" y="${y(m.target) + (up ? 14 : -6)}" text-anchor="end" font-size="11" fill="var(--muted)">target</text>
    <path d="${path(lowerPts)}" fill="none" stroke="var(--faint)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="${path(liftPts)}" fill="none" stroke="var(--accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    <circle cx="${x(peak[0])}" cy="${y(peak[1])}" r="4.5" fill="var(--accent)" stroke="var(--surface)" stroke-width="2"/>
    <text x="${mid(lowerPts)}" y="12" text-anchor="middle" font-size="11" fill="var(--muted)">lower</text>
    <text x="${mid(liftPts)}" y="12" text-anchor="middle" font-size="11" fill="var(--accent)">lift</text>
    <line class="xh" y1="${PT}" y2="${H - PB}" stroke="var(--muted)" visibility="hidden"/>
    <circle class="xd" r="4" fill="var(--ink)" stroke="var(--surface)" stroke-width="2" visibility="hidden"/>
    <rect x="${PL}" y="0" width="${W - PL - PR}" height="${H}" fill="transparent"/>
  </svg>`;
}
function hoverTrace(box, m) {
  const svg = box.querySelector('svg'), tip = box.querySelector('.tip'), xh = svg.querySelector('.xh'), xd = svg.querySelector('.xd');
  const { x, y, end } = scales(m);
  const show = (on) => { xh.setAttribute('visibility', on ? 'visible' : 'hidden'); xd.setAttribute('visibility', on ? 'visible' : 'hidden'); tip.style.opacity = on ? 1 : 0; };
  svg.addEventListener('pointermove', (e) => {
    const r = svg.getBoundingClientRect();
    const t = ((e.clientX - r.left) * (W / r.width) - PL) / (W - PL - PR) * end;
    const [pt, pa] = m.trace.reduce((a, p) => (Math.abs(p[0] - t) < Math.abs(a[0] - t) ? p : a));
    xh.setAttribute('x1', x(pt)); xh.setAttribute('x2', x(pt));
    xd.setAttribute('cx', x(pt)); xd.setAttribute('cy', y(pa));
    tip.textContent = `${sec(pt)} s  ${pa}°`;
    tip.style.left = `${(x(pt) / W) * 100}%`;
    show(true);
  });
  svg.addEventListener('pointerleave', () => show(false));
}

renderDash(sample, SAMPLE, EXERCISES.squat);
