import { PoseLandmarker, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/vision_bundle.mjs';
import { EXERCISES, SIDES, createSession } from './engine.js';

const WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/wasm';
const VARIANT = new URLSearchParams(location.search).has('lite') ? 'lite' : 'full'; // ?lite for older phones
const MODEL = `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_${VARIANT}/float16/1/pose_landmarker_${VARIANT}.task`;

const $ = (id) => document.getElementById(id);
const [exercise, setup, start, file, flip, stop, sound, stage, video, canvas, count, meter, status, cue, summary] =
  ['exercise', 'setup', 'start', 'file', 'flip', 'stop', 'sound', 'stage', 'video', 'canvas', 'count', 'meter', 'status', 'cue', 'summary'].map($);
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
  stage.hidden = false;
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

function renderSummary() {
  const reps = session.reps;
  const done = reps.filter((r) => r.counted);
  const tally = {};
  for (const r of reps) for (const i of r.issues) tally[i] = (tally[i] ?? 0) + 1;
  const top = Object.entries(tally).sort((a, b) => b[1] - a[1])[0];
  const esc = (s) => s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]);
  summary.innerHTML = `
    <h2>${esc(session.exercise.name)}: ${done.length} rep${done.length === 1 ? '' : 's'}${reps.length > done.length ? `, ${reps.length - done.length} not counted` : ''}</h2>
    ${top ? `<p>Most common cue: <b>${esc(top[0])}</b> (${top[1]}×)</p>` : '<p>No form cues. Nice set.</p>'}
    <table>
      <tr><th>Rep</th><th>Range</th><th>Time</th><th>Feedback</th></tr>
      ${reps.map((r) => `<tr><td>${r.n ?? '–'}</td><td>${r.range}°</td><td>${(r.duration / 1000).toFixed(1)} s</td>
        <td>${esc(!r.counted ? `Not counted: ${r.issues[0]}` : !r.reliable ? 'Not checked (view blocked)' : r.issues.join('; ') || 'Good')}</td></tr>`).join('')}
    </table>
    <p class="muted">Range is the tracked joint angle at the deepest point of each rep.</p>`;
  summary.hidden = false;
}
