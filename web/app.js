// 나만의 제스처 웹 - 웹캠 → MediaPipe 손 랜드마크 → 직접 훈련한 MLP → 반응 표시
import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/vision_bundle.mjs";

const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/wasm";
const HAND_MODEL_PATH = "../hand_landmarker.task";
const WEIGHTS_PATH = "gesture_weights.json";
const STABLE_FRAMES = 5;  // 같은 제스처가 이만큼 연속되면 반응 표시
const HIDE_FRAMES = 10;   // 이만큼 연속으로 안 보이면 반응 숨김

// 제스처 이름 → 화면 반응. 라벨 이름은 gesture_studio.py에서 만든 것과 똑같이 써야 함
const NIKE_SWOOSH = `
  <svg viewBox="0 0 220 80" aria-label="Nike">
    <path fill="currentColor" d="M220 2 L62 70 C46 77 32 79 22 77 C9 74 2 66 1 56
      C0 44 8 30 21 17 C17 31 17 43 24 49 C31 55 44 55 59 50 Z"/>
  </svg>`;
const REACTIONS = {
  "나이키": { icon: NIKE_SWOOSH, caption: "JUST DO IT" },
  "오키": { icon: "👌", caption: "OK!" },
  "브이": { icon: "✌️", caption: "V!" },
  "굳": { icon: "👍", caption: "GOOD" },
  "배드": { icon: "👎", caption: "BAD" },
};

// 21개 랜드마크 연결 (엄지, 검지, 중지, 약지, 소지, 손바닥)
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

const $ = (id) => document.getElementById(id);
const video = $("video");
const canvas = $("canvas");
const ctx = canvas.getContext("2d");
const statusEl = $("status");

// ---------------------------------------------------------------- 모델 (파이썬 custom_gesture.py와 동일한 계산)
/** 랜드마크 21개 → 63차원 특징. 파이썬 landmarks_to_features()와 같은 정규화 */
function toFeatures(landmarks, handedness) {
  const w = landmarks[0];
  const pts = landmarks.map((p) => [p.x - w.x, p.y - w.y, p.z - w.z]); // 손목을 원점으로
  if (handedness === "Left") pts.forEach((p) => { p[0] = -p[0]; });   // 왼손은 좌우 반전
  let scale = 0;
  for (const p of pts) scale = Math.max(scale, Math.hypot(p[0], p[1]));
  scale = Math.max(scale, 1e-6);                                        // 가장 먼 점까지 거리를 1로
  return pts.flat().map((v) => v / scale);
}

/** Linear → ReLU → Linear → ReLU → Linear → softmax (Dropout은 추론 때 아무것도 안 함) */
function predict(weights, x) {
  let h = x;
  weights.layers.forEach((layer, li) => {
    const out = layer.b.map((b, o) => {
      let s = b;
      const row = layer.w[o];
      for (let i = 0; i < row.length; i++) s += row[i] * h[i];
      return s;
    });
    h = li < weights.layers.length - 1 ? out.map((v) => Math.max(0, v)) : out;
  });
  const m = Math.max(...h);
  const e = h.map((v) => Math.exp(v - m));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
}

// ---------------------------------------------------------------- 화면
function buildBars(labels) {
  $("bars").innerHTML = labels.map((name, i) => `
    <div class="bar-row" id="bar${i}">
      <span>${name}</span><div class="track"><div class="fill"></div></div><span class="pct">0%</span>
    </div>`).join("");
}

function updateBars(probs) {
  const top = probs ? probs.indexOf(Math.max(...probs)) : -1;
  document.querySelectorAll(".bar-row").forEach((row, i) => {
    const p = probs ? probs[i] : 0;
    row.querySelector(".fill").style.width = `${(p * 100).toFixed(1)}%`;
    row.querySelector(".pct").textContent = `${Math.round(p * 100)}%`;
    row.classList.toggle("top", i === top);
  });
}

let shown = null;
function showReaction(name) {
  if (name === shown) return;
  shown = name;
  const el = $("reaction");
  const r = name && REACTIONS[name];
  if (!r) { el.classList.remove("show"); return; }
  $("reactionIcon").innerHTML = r.icon;
  $("reactionCaption").textContent = r.caption;
  el.classList.remove("show");
  void el.offsetWidth; // 애니메이션 다시 시작
  el.classList.add("show");
}

// 콘솔에서 확인용 (예: gestureDebug.showReaction("나이키"))
window.gestureDebug = { toFeatures, predict, showReaction };

function drawHand(landmarks, color) {
  const pts = landmarks.map((p) => [p.x * canvas.width, p.y * canvas.height]);
  ctx.lineWidth = 3;
  ctx.strokeStyle = color;
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.beginPath(); ctx.moveTo(...pts[a]); ctx.lineTo(...pts[b]); ctx.stroke();
  }
  ctx.fillStyle = "#ff3b3b";
  for (const [x, y] of pts) { ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill(); }
  return pts;
}

// ---------------------------------------------------------------- 메인 루프
async function main() {
  if (location.protocol === "file:") {
    statusEl.textContent = "파일을 직접 열면 동작하지 않습니다. 터미널에서 python web_gesture.py 로 실행하세요.";
    return;
  }
  let weights;
  try {
    weights = await (await fetch(WEIGHTS_PATH)).json();
  } catch {
    statusEl.textContent = "gesture_weights.json이 없습니다. python web_gesture.py 로 실행하세요.";
    return;
  }
  buildBars(weights.labels);

  const fileset = await FilesetResolver.forVisionTasks(WASM_PATH);
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: HAND_MODEL_PATH, delegate },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  let landmarker;
  try {
    landmarker = await HandLandmarker.createFromOptions(fileset, options("GPU"));
  } catch {
    landmarker = await HandLandmarker.createFromOptions(fileset, options("CPU"));
  }

  statusEl.textContent = "카메라 권한을 허용해 주세요…";
  try {
    video.srcObject = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false,
    });
  } catch (e) {
    statusEl.textContent = `웹캠을 열 수 없습니다: ${e.message}`;
    return;
  }
  await video.play();
  statusEl.hidden = true;

  const minScoreInput = $("minScore");
  minScoreInput.addEventListener("input", () => {
    $("minScoreText").textContent = Number(minScoreInput.value).toFixed(2);
  });

  let candidate = null, streak = 0, missing = 0, lastTime = -1;

  function frame() {
    requestAnimationFrame(frame);
    if (video.readyState < 2) return;
    const now = performance.now();
    if (now <= lastTime) return;
    lastTime = now;

    // 파이썬과 같게: 4:3으로 자르고(640x480) 좌우 반전(거울 모드)한 화면에서 인식
    const vw = video.videoWidth, vh = video.videoHeight;
    let sx = 0, sy = 0, sw = vw, sh = vh;
    if (vw / vh > 4 / 3) { sw = vh * 4 / 3; sx = (vw - sw) / 2; } else { sh = vw * 3 / 4; sy = (vh - sh) / 2; }
    ctx.save();
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
    ctx.restore();

    const result = landmarker.detectForVideo(canvas, now);
    const minScore = Number(minScoreInput.value);
    let best = null;
    result.landmarks.forEach((landmarks, i) => {
      const hand = result.handedness[i][0].categoryName;
      const probs = predict(weights, toFeatures(landmarks, hand));
      const idx = probs.indexOf(Math.max(...probs));
      const name = probs[idx] >= minScore ? weights.labels[idx] : "?";
      const pts = drawHand(landmarks, REACTIONS[name] ? "#ff6b2c" : "#22c55e");

      const x = Math.min(...pts.map((p) => p[0]));
      const y = Math.max(Math.min(...pts.map((p) => p[1])) - 12, 24);
      ctx.font = "bold 22px 'Malgun Gothic', sans-serif";
      ctx.lineWidth = 4; ctx.strokeStyle = "#000"; ctx.fillStyle = "#ffe14d";
      const text = `${name} ${probs[idx].toFixed(2)}`;
      ctx.strokeText(text, x, y); ctx.fillText(text, x, y);

      if (!best || probs[idx] > best.score) best = { name, score: probs[idx], probs };
    });

    updateBars(best && best.probs);
    $("current").innerHTML = best ? `${best.name}<small>${Math.round(best.score * 100)}%</small>` : "-";

    // 깜빡임 방지: 같은 제스처가 STABLE_FRAMES 연속일 때만 반응, HIDE_FRAMES 동안 사라지면 숨김
    const name = best && best.name !== "?" ? best.name : null;
    if (name && name === candidate) streak++; else { candidate = name; streak = 1; }
    if (name && streak >= STABLE_FRAMES) { showReaction(name); missing = 0; }
    else if (!name || name !== shown) { if (++missing >= HIDE_FRAMES) showReaction(null); }
  }
  frame();
}

main().catch((e) => {
  console.error(e);
  statusEl.hidden = false;
  statusEl.textContent = `오류: ${e.message}`;
});
