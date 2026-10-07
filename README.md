# MediaPipe 비전 태스크 실습: 손 랜드마크 · 제스처 인식 · 얼굴 랜드마크 · 나만의 제스처 웹

> 웹캠 영상에서 **손의 21개 관절 좌표**를 찾고, 그 손이 **어떤 제스처**를 취하고 있는지 인식하고, **얼굴의 478개 점과 표정**까지 실시간으로 분석하고, **내가 정한 제스처를 직접 학습**시켜 **웹 페이지로 배포**하는 실습 예제입니다.
> Google **MediaPipe Tasks** 파이썬·자바스크립트 API, **OpenCV**, **PyTorch**를 사용합니다.

> 🌐 **웹 데모 (설치 없이 바로 체험)**: https://pakuyama.github.io/mediapipe-hand-gesture-webcam/
> 링크를 열고 카메라를 허용한 뒤 👍 ✌️ 👌 ✋ 같은 손 모양을 보여 주세요.

| 실습 | 파일 | 모델 | 결과 |
|---|---|---|---|
| 1강 | `hand_webcam.py` | `hand_landmarker.task` | 손 관절 21개 + 왼손/오른손 구분 |
| 2강 | `gesture_webcam.py` | `gesture_recognizer.task` | 위 결과 + 제스처 이름(👍, ✌️ …) |
| 3강 | `face_webcam.py` | `face_landmarker.task` | 얼굴 478점 메시 + 홍채 + 표정 점수(블렌드셰이프) |
| 4강 | `gesture_studio.py` (GUI) | `hand_landmarker.task` + 직접 학습한 `gesture_model.pt` | **내가 정한 제스처** 인식 |
| 5강 | `web_gesture.py` + `web/` | `hand_landmarker.task` + `web/gesture_weights.json` | 브라우저에서 인식 → 👌·✌️·나이키 로고 반응, **GitHub Pages 배포** |

---

## 0. 학습 목표

이 강의를 마치면 다음을 할 수 있습니다.

1. MediaPipe Tasks의 **"모델 파일 → 옵션 → 태스크 객체 → 추론"** 흐름을 설명할 수 있다.
2. OpenCV로 웹캠 프레임을 받아 MediaPipe가 요구하는 형식(`mp.Image`, RGB)으로 변환할 수 있다.
3. **정규화 좌표**(0~1)를 화면 픽셀 좌표로 바꿔 랜드마크를 그릴 수 있다.
4. 실행 모드(`IMAGE` / `VIDEO` / `LIVE_STREAM`)의 차이를 알고 상황에 맞게 고를 수 있다.
5. 인식된 제스처 결과를 이용해 간단한 응용 프로그램을 만들 수 있다.
6. 얼굴 랜드마크와 **블렌드셰이프**(표정 점수)를 읽어 눈 깜빡임·미소 같은 표정을 감지할 수 있다.
7. 손 랜드마크 위에 작은 신경망을 얹어 **나만의 제스처를 수집·훈련·인식**할 수 있다.
8. 파이썬에서 훈련한 모델을 **브라우저(JavaScript)로 옮겨** 실행하고 **GitHub Pages로 배포**할 수 있다.

---

## 1. 배경 개념

### 1.1 MediaPipe Tasks란?

MediaPipe는 Google이 만든 온디바이스(on-device) 머신러닝 프레임워크입니다. **Tasks API**는 자주 쓰는 기능(얼굴 검출, 손 추적, 물체 인식 등)을 미리 학습된 모델과 함께 패키지로 제공하므로, 딥러닝 지식 없이도 몇 줄의 코드로 사용할 수 있습니다.

모든 비전 태스크는 같은 4단계 패턴을 따릅니다. **이 패턴만 익히면 다른 태스크도 똑같이 쓸 수 있습니다.**

```
① 모델 파일(.task) 준비
② XxxOptions(...) 로 옵션 설정
③ Xxx.create_from_options(options) 로 태스크 객체 생성
④ detect / recognize 계열 함수로 추론 → 결과 객체 사용
```

### 1.2 손 랜드마크 (Hand Landmarks)

손 하나를 **21개의 점**으로 표현합니다.

```
            8   12  16  20        0      : 손목 (WRIST)
            |   |   |   |         1~4    : 엄지 (THUMB)
            7   11  15  19        5~8    : 검지 (INDEX)
            |   |   |   |         9~12   : 중지 (MIDDLE)
        4   6   10  14  18        13~16  : 약지 (RING)
        |   |   |   |   |         17~20  : 소지 (PINKY)
        3   5---9---13--17
         \  |          /          각 손가락의 끝(TIP) = 4, 8, 12, 16, 20
          2 |         /
           \|        /
            1       /
             \     /
              --0--
```

각 점은 `(x, y, z)` 값을 가집니다.

- `x`, `y`: 이미지 너비·높이에 대한 **정규화 좌표** (0.0 ~ 1.0)
- `z`: 손목을 기준으로 한 상대적 깊이 (값이 작을수록 카메라에 가까움)

### 1.3 두 단계 모델 구조

```
[웹캠 프레임] → (손바닥 검출 모델) → 손 영역 박스 → (랜드마크 모델) → 21개 점
                                                                  │
                                         Gesture Recognizer만 ─────┘
                                                                  ↓
                                                   (제스처 분류 모델) → "Thumb_Up"
```

- **Hand Landmarker**: 손 검출 + 랜드마크까지
- **Gesture Recognizer**: 위 과정에 **제스처 분류기**가 하나 더 붙은 구조. 즉 Hand Landmarker의 상위 호환입니다.

### 1.4 기본 제공 제스처 (8종)

| 카테고리 | 의미 |
|---|---|
| `None` | 알 수 없음 / 해당 없음 |
| `Closed_Fist` | ✊ 주먹 |
| `Open_Palm` | ✋ 손바닥 펴기 |
| `Pointing_Up` | ☝️ 검지 위로 |
| `Thumb_Down` | 👎 엄지 아래 |
| `Thumb_Up` | 👍 엄지 위 |
| `Victory` | ✌️ 브이 |
| `ILoveYou` | 🤟 사랑해 |

---

## 2. 환경 설정

### 2.1 패키지 설치

```bash
pip install mediapipe opencv-python
pip install torch   # 4강(나만의 제스처 학습)에서만 필요
```

> 실습 확인 환경: Python 3.14, mediapipe 1.1.0, opencv-python 5.0.0 (Windows 11)

### 2.2 모델 파일

이 저장소에 세 모델 파일이 포함되어 있습니다. 직접 받으려면 아래 공식 링크를 사용하세요.

| 모델 | 다운로드 |
|---|---|
| `hand_landmarker.task` | https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task |
| `gesture_recognizer.task` | https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task |
| `face_landmarker.task` | https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task |

모델 파일은 `.py` 파일과 **같은 폴더**에 두어야 합니다.

### 2.3 실행

```bash
python hand_webcam.py       # 1강: 손 랜드마크
python gesture_webcam.py    # 2강: 제스처 인식
python face_webcam.py       # 3강: 얼굴 랜드마크
python gesture_studio.py    # 4강: 나만의 제스처 수집·훈련·인식 (GUI)
python web_gesture.py      # 5강: 웹 버전 (브라우저가 열림, 종료는 Ctrl+C)
```

종료: 영상 창에서 `q` 또는 `ESC` (3강은 `m` 키로 메시 표시 켜기/끄기)

---

## 3. 1강: 손 랜드마크 검출 (`hand_webcam.py`)

### 3.1 전체 흐름

```
웹캠 열기 → [반복] 프레임 읽기 → 좌우 반전 → BGR→RGB → mp.Image
         → detect_for_video() → 점·선 그리기 → FPS 표시 → 화면 출력
```

### 3.2 핵심 코드 해설

**① 옵션 설정**

```python
options = vision.HandLandmarkerOptions(
    base_options=mp_tasks.BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
```

| 옵션 | 설명 |
|---|---|
| `model_asset_buffer` | 모델을 **바이트**로 전달. (`model_asset_path`로 경로를 줄 수도 있지만, 경로에 **한글**이 있으면 네이티브 로더가 파일을 못 여는 문제가 있어 바이트로 읽어 넘김) |
| `running_mode` | 실행 모드 (아래 3.3 참고) |
| `num_hands` | 동시에 찾을 최대 손 개수 |
| `min_hand_detection_confidence` | 손바닥 검출 최소 신뢰도 |
| `min_hand_presence_confidence` | 랜드마크 단계에서 "손이 있다"고 볼 최소 신뢰도 |
| `min_tracking_confidence` | 이전 프레임 추적을 이어갈 최소 신뢰도. 낮으면 추적이 끊겨 매 프레임 다시 검출함 |

**② 프레임 전처리**

```python
frame = cv2.flip(frame, 1)                       # 거울 모드
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     # OpenCV는 BGR, MediaPipe는 RGB
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
```

> ⚠️ **자주 하는 실수**: BGR→RGB 변환을 빼먹으면 에러는 안 나지만 인식률이 눈에 띄게 떨어집니다.

**③ 추론**

```python
timestamp_ms = int((time.perf_counter() - start) * 1000)
result = landmarker.detect_for_video(mp_image, timestamp_ms)
```

`VIDEO` 모드에서는 **타임스탬프(ms)가 계속 증가**해야 합니다. 같은 값이나 이전 값을 넣으면 에러가 납니다.

**④ 결과 그리기: 정규화 좌표 → 픽셀**

```python
h, w = frame.shape[:2]
pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
```

결과 객체 구조:

```
result.hand_landmarks[i][j]   # i번째 손의 j번째 점 (x, y, z)
result.handedness[i][0]       # i번째 손의 Left/Right 와 score
```

### 3.3 실행 모드 비교

| 모드 | 호출 함수 | 특징 | 언제 쓰나 |
|---|---|---|---|
| `IMAGE` | `detect()` | 프레임마다 독립 처리 | 사진 한 장 |
| `VIDEO` | `detect_for_video()` | **동기**, 프레임 간 추적 사용 | 동영상 파일, 단순한 웹캠 루프 |
| `LIVE_STREAM` | `detect_async()` + 콜백 | **비동기**, 밀리면 프레임 버림 | 지연이 중요한 실시간 앱 |

이 실습은 코드 흐름을 이해하기 쉽도록 `VIDEO` 모드를 사용했습니다.

---

## 4. 2강: 제스처 인식 (`gesture_webcam.py`)

1강 코드와 **구조가 거의 같습니다.** 바뀐 부분만 보면 됩니다.

### 4.1 1강과 달라진 점

| 항목 | 1강 (Hand Landmarker) | 2강 (Gesture Recognizer) |
|---|---|---|
| 모델 | `hand_landmarker.task` | `gesture_recognizer.task` |
| 옵션 클래스 | `HandLandmarkerOptions` | `GestureRecognizerOptions` |
| 태스크 클래스 | `HandLandmarker` | `GestureRecognizer` |
| 추론 함수 | `detect_for_video()` | `recognize_for_video()` |
| 결과에 추가된 것 | 없음 | `result.gestures` |

### 4.2 제스처 결과 읽기

```python
if i < len(result.gestures) and result.gestures[i]:
    g = result.gestures[i][0]              # 가장 점수가 높은 제스처
    label = f"{hand} {g.category_name} {g.score:.2f}"
```

```
result.gestures[i]       # i번째 손의 제스처 후보 목록 (점수 높은 순)
result.gestures[i][0]    # 1순위 → .category_name, .score
```

화면에는 `Right Thumb_Up 0.87`처럼 표시됩니다.

---

## 5. 3강: 얼굴 랜드마크 & 표정 인식 (`face_webcam.py`)

손에서 배운 4단계 패턴을 **얼굴**에 그대로 적용합니다. 이번에는 점이 훨씬 많고, **표정 점수**라는 새로운 출력이 추가됩니다.

### 5.1 얼굴 랜드마크 개념

| 항목 | 손 (1·2강) | 얼굴 (3강) |
|---|---|---|
| 점 개수 | 21개 | **478개** (얼굴 468 + 홍채 10) |
| 연결선 | 직접 정의한 21쌍 | 라이브러리 제공 `FaceLandmarksConnections` |
| 추가 출력 | Left/Right, 제스처 | **블렌드셰이프 52종**, 3D 변환 행렬 |

478개의 점은 얼굴 표면을 촘촘한 **삼각형 그물(메시)**로 덮습니다. 점을 일일이 연결할 필요 없이 라이브러리가 부위별 연결 목록을 제공합니다.

```python
C = vision.FaceLandmarksConnections
C.FACE_LANDMARKS_TESSELATION   # 전체 삼각형 메시 (회색)
C.FACE_LANDMARKS_CONTOURS      # 얼굴 윤곽·눈·눈썹·입술 (초록)
C.FACE_LANDMARKS_LEFT_IRIS     # 홍채 (주황)
C.FACE_LANDMARKS_LIPS          # 입술만, _NOSE, _LEFT_EYE ... 부위별로도 있음
```

각 연결은 `Connection(start, end)` 형태여서 `c.start`, `c.end`로 두 점의 번호를 꺼냅니다.

### 5.2 블렌드셰이프(Blendshapes)란?

**표정을 52개의 0~1 점수로 표현**한 것입니다. 3D 캐릭터 애니메이션(아바타, 버튜버)에서 쓰는 표준 방식입니다.

| 블렌드셰이프 예 | 의미 |
|---|---|
| `eyeBlinkLeft` / `eyeBlinkRight` | 왼쪽/오른쪽 눈 감기 |
| `mouthSmileLeft` / `mouthSmileRight` | 입꼬리 올리기(미소) |
| `jawOpen` | 입 벌리기 |
| `browInnerUp` | 눈썹 안쪽 올리기(놀람) |
| `cheekPuff` | 볼 부풀리기 |

> 💡 거울 모드에서는 `eyeBlinkLeft`가 **화면상 오른쪽 눈**에 반응할 수 있습니다. 직접 눈을 감아 보며 확인해 보세요.

### 5.3 핵심 코드 해설

**① 옵션: 블렌드셰이프 출력 켜기**

```python
options = vision.FaceLandmarkerOptions(
    base_options=mp_tasks.BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=True,   # ← 기본값 False. 켜야 표정 점수가 나옴
)
```

> ⚠️ `output_face_blendshapes=True`를 빼면 `result.face_blendshapes`가 빈 리스트입니다.

**② 추론**: 1강과 같은 `detect_for_video()`를 사용합니다.

**③ 결과 구조**

```
result.face_landmarks[i][j]      # i번째 얼굴의 j번째 점 (x, y, z), j = 0~477
result.face_blendshapes[i]       # i번째 얼굴의 52개 블렌드셰이프 (.category_name, .score)
```

**④ 상위 표정 막대그래프**

```python
top = sorted(result.face_blendshapes[0], key=lambda b: b.score, reverse=True)
for k, b in enumerate(top[:5]):
    cv2.rectangle(frame, (10, y - 15), (10 + int(b.score * 150), y + 3), ...)  # 점수만큼 막대 길이
```

점수(0~1)에 150을 곱해 **막대 길이(픽셀)**로 바꾸는 것이 핵심입니다.

### 5.4 2강과 달라진 점 (코드 비교)

| 항목 | 2강 (Gesture Recognizer) | 3강 (Face Landmarker) |
|---|---|---|
| 옵션 클래스 | `GestureRecognizerOptions` | `FaceLandmarkerOptions` |
| 태스크 클래스 | `GestureRecognizer` | `FaceLandmarker` |
| 추론 함수 | `recognize_for_video()` | `detect_for_video()` |
| 개수 옵션 | `num_hands` | `num_faces` |
| 그리기 | 직접 정의한 연결선 | 라이브러리 연결선 + 막대그래프 |
| 키 입력 | `q`/`ESC` | `q`/`ESC` + `m`(메시 토글) |

---

## 6. 4강: 나만의 제스처 학습하기 (`gesture_studio.py`)

2강의 기본 제스처 8종 대신 **내가 정한 제스처**(예: 가위·바위·보, 숫자 1~5, 🤙 …)를 인식하도록 직접 학습합니다.

### 6.1 아이디어: 랜드마크 위에 작은 분류기 얹기

```
웹캠 프레임 ─▶ Hand Landmarker(1강) ─▶ 21점 좌표 ─▶ 정규화(63차원) ─▶ 작은 신경망(MLP) ─▶ 제스처 이름
                (이미 학습된 모델)                   custom_gesture.py      내가 학습시키는 부분
```

이미지 전체를 학습하는 대신 **손 관절 좌표만** 학습하므로, 제스처당 수백 장(웹캠 수십 초)이면 충분하고 CPU로 몇 초 만에 훈련됩니다.
2강의 Gesture Recognizer도 내부적으로 같은 구조(랜드마크 → 분류기)입니다.

`custom_gesture.py`의 `landmarks_to_features()`가 좌표를 다음처럼 정규화합니다.

| 처리 | 효과 |
|---|---|
| 손목(0번)을 원점으로 이동 | 손이 화면 어디에 있든 같은 값 |
| 왼손이면 x 좌우 반전 | 오른손으로만 수집해도 왼손까지 인식 |
| 가장 먼 점까지 거리를 1로 | 카메라와 가깝든 멀든 같은 값 |

### 6.2 파일 구성

| 파일 | 역할 | 입력 → 출력 |
|---|---|---|
| **`gesture_studio.py`** | **GUI: 수집 · 훈련 · 실시간 인식을 한 창에서** (추천) | 웹캠 ↔ `gesture_data.csv` ↔ `gesture_model.pt` |
| `custom_gesture.py` | 공통 모듈 (랜드마커 생성, 특징 변환, 신경망, CSV 읽기/쓰기) | — |
| `collect_gesture.py` | (명령줄 버전) ① 데이터 수집 | 웹캠 → `gesture_data.csv` |
| `train_gesture.py` | (명령줄 버전) ② 훈련 — GUI도 이 파일의 `train()` 함수를 사용 | `gesture_data.csv` → `gesture_model.pt` |
| `infer_gesture.py` | (명령줄 버전) ③ 실시간 추론 | 웹캠 + `gesture_model.pt` → 화면 표시 |

### 6.3 사용 방법 (GUI)

```bash
python gesture_studio.py
```

```
┌───────────────────────────────┬──────────────────────────┐
│                               │ 제스처 라벨              │
│                               │  키  라벨      샘플 수   │
│        웹캠 화면              │  1   none        300     │
│   (녹화 중엔 손이 빨간색,     │  2   주먹        300     │
│    인식 중엔 손 위에 결과)    │  [입력칸      ] [추가]   │
│                               │  [선택한 라벨과 데이터 삭제]│
│                               │ ① 수집  목표 개수 [300]  │
│                               │  [● 녹화 시작 (Space)]   │
│                               │ ② 훈련  Epoch [150]      │
│                               │  [훈련 시작] [중단] ▓▓▓░ │
│                               │ ③ 실시간 인식 ☑ 인식 켜기│
│                               │  최소 확신도 ──●── 0.70  │
├───────────────────────────────┴──────────────────────────┤
│ 로그 (훈련 과정, 혼동 행렬 등)                            │
└──────────────────────────────────────────────────────────┘
```

**① 라벨 추가** — 입력칸에 이름을 쓰고 `Enter` 또는 **[추가]**. 한글도 됩니다. 맨 먼저 **`none`** 라벨을 만드세요.

**② 수집** — 표에서 라벨을 고르고(클릭 또는 숫자키 `1`~`9`) **[녹화 시작]** 또는 `Space`.
손이 빨간색으로 바뀌며 샘플이 쌓이고, 목표 개수(기본 300)에 도달하면 자동으로 멈춥니다. 다음 라벨을 골라 반복합니다.

- 녹화 중에 손을 **조금씩 움직이고, 돌리고, 앞뒤로** 옮겨 주세요. 다양할수록 실전에서 잘 됩니다.
- 손이 **하나만** 보일 때만 저장됩니다 (두 손이 보이면 화면에 경고).
- **`none` 라벨을 꼭 만드세요.** 아무 제스처도 아닌 평소 손 모양(반쯤 편 손, 손 내리는 중 등)을 모아 두면, 엉뚱한 손 모양을 억지로 제스처로 분류하는 일이 줄어듭니다.
- 데이터는 `gesture_data.csv`에 **이어서** 저장되므로 프로그램을 껐다 켜도 그대로 남아 있습니다. 잘못 모은 라벨은 **[선택한 라벨과 데이터 삭제]**.

**③ 훈련** — **[훈련 시작]**. 진행 바와 로그에 정확도가 표시되고, 끝나면 **자동으로 새 모델을 불러와 실시간 인식이 켜집니다**.
훈련 중에도 카메라 화면은 계속 움직이며, **[중단]**을 누르면 그때까지의 최고 모델이 저장됩니다.

- 데이터의 20%를 떼어 **검증**에 쓰고, 검증 정확도가 가장 높았던 순간의 모델을 저장합니다.
- 훈련 중에는 데이터에 약간의 회전(±10°)·크기 변화·잡음을 섞어(데이터 증강) 적은 데이터로도 잘 일반화되게 합니다.
- 로그 맨 아래 **혼동 행렬**에서 대각선 밖 숫자가 큰 칸은 모델이 헷갈리는 제스처 쌍입니다 → 그 제스처를 더 수집하거나 더 구분되는 모양으로 바꾸세요.

```
혼동 행렬 (행=정답, 열=예측)
              none    주먹      보
      none      58       1       1
      주먹       0      60       0
        보       0       0      60
```

**④ 인식** — 손 위에 `Right 주먹 0.98`처럼, 오른쪽 패널에 큰 글씨로 결과가 표시됩니다.
확신도가 **최소 확신도** 슬라이더 값보다 낮으면 `?`로 나옵니다.

> **명령줄 버전**도 그대로 쓸 수 있습니다:
> `python collect_gesture.py none rock paper` (라벨은 영어만) → `python train_gesture.py --epochs 300` → `python infer_gesture.py`
> 두 버전은 같은 `gesture_data.csv` / `gesture_model.pt`를 사용하므로 섞어서 써도 됩니다.

### 6.4 코드 핵심 부분

```python
# 수집 (collect_gesture.py) — 손이 보이는 프레임을 63차원 벡터로 바꿔 CSV 한 줄로 저장
feats = landmarks_to_features(landmarks, hand)
writer.writerow([label] + [f"{v:.6f}" for v in feats])

# 추론 (infer_gesture.py) — 같은 변환을 거친 뒤 신경망에 넣고 softmax로 확률화
feats = landmarks_to_features(landmarks, hand)
probs = torch.softmax(model(torch.from_numpy(feats)[None]), dim=1)[0]
score, idx = probs.max(0)
```

> **수집과 추론의 전처리는 반드시 같아야 합니다.** 둘 다 `cv2.flip(frame, 1)`(거울 모드) + `landmarks_to_features()`를 쓰는 이유입니다.

### 6.5 결과가 안 좋을 때

| 증상 | 해결 |
|---|---|
| 검증 정확도는 높은데 실제로는 잘 틀림 | 수집할 때 손을 너무 고정했음 → 각도·거리를 바꿔 가며 추가 수집. 다른 사람 손도 섞으면 더 좋음 |
| 아무 손 모양이나 특정 제스처로 인식 | `none` 라벨 데이터를 늘리거나 **최소 확신도** 슬라이더를 0.8~0.9로 올리기 |
| 두 제스처를 계속 헷갈림 | 혼동 행렬 확인 → 두 제스처를 더 수집하거나, 손 모양이 더 다른 제스처로 교체 |
| 라벨 이름을 잘못 입력함 | GUI에서 라벨 선택 → **[선택한 라벨과 데이터 삭제]** 후 재훈련 |
| 라벨을 추가·삭제함 | 반드시 다시 훈련 (모델에 라벨 목록이 함께 저장됨). 현재 모델의 라벨은 GUI ③ 패널에 표시 |

## 7. 5강: 웹 버전 만들기 & GitHub Pages 배포 (`web_gesture.py`, `web/`)

> 🌐 **배포된 데모**: https://pakuyama.github.io/mediapipe-hand-gesture-webcam/

4강에서 훈련한 모델을 **브라우저에서 실행**하고, 인식된 제스처에 따라 화면에 큰 반응을 띄웁니다.
파이썬 설치 없이 **링크만으로** 누구나 체험할 수 있도록 GitHub Pages에 배포합니다.

| 라벨 | 반응 |
|---|---|
| `나이키` | 나이키 로고 + JUST DO IT |
| `오키` | 👌 OK! |
| `브이` | ✌️ V! |
| `굳` | 👍 GOOD |
| `배드` | 👎 BAD |
| `손바닥` | ✋ HI! |

### 7.1 전체 구조: 파이썬 → 브라우저

```
[파이썬]                                    [브라우저 (web/app.js)]
gesture_studio.py로 훈련                     웹캠 (getUserMedia)
        │                                         │ 4:3으로 자르고 좌우 반전
gesture_model.pt (PyTorch)                  MediaPipe Hand Landmarker (JS판, 같은 .task 모델)
        │ web_gesture.py                          │ 21점 좌표
        ▼                                         ▼
web/gesture_weights.json  ───────────────▶  toFeatures()  ← 파이썬과 같은 정규화
 (라벨 + 층별 가중치)                             │ 63차원
                                            predict()     ← 행렬 곱 + ReLU + softmax 직접 구현
                                                  │ 라벨별 확률
                                            5프레임 연속 같은 제스처 → 반응 표시 (나이키 로고, 👌 …)
```

핵심은 **손 인식은 MediaPipe가, 분류는 내가 만든 작은 신경망이** 한다는 점입니다.
신경망이 63 → 128 → 64 → 6 크기로 아주 작아서 TensorFlow.js 같은 라이브러리 없이 자바스크립트 30줄로 계산할 수 있습니다.

| 파일 | 역할 |
|---|---|
| `web_gesture.py` | `gesture_model.pt` → `web/gesture_weights.json` 변환, 로컬 서버 실행, 브라우저 열기 |
| `web/index.html` | 화면: 웹캠, 반응 오버레이, 라벨별 확률 막대, 최소 확신도 슬라이더 |
| `web/app.js` | 웹캠 → 손 랜드마크 → 정규화 → 신경망 → 반응 표시 |
| `web/gesture_weights.json` | 훈련된 가중치 (`web_gesture.py`가 생성, 배포용으로 커밋됨) |
| `index.html` | GitHub Pages 첫 화면. `web/`으로 자동 이동 |

### 7.2 실행 방법

**방법 1. 배포된 사이트** — https://pakuyama.github.io/mediapipe-hand-gesture-webcam/ 를 열고 **카메라 권한을 허용**하면 끝입니다.

**방법 2. 내 컴퓨터에서 (방금 훈련한 모델로)**

```bash
python web_gesture.py     # 모델 내보내기 + 로컬 서버 + 브라우저 자동 열기 (종료: Ctrl+C)
```

1. 브라우저가 `http://localhost:8000/web/index.html`을 엽니다. **카메라 권한을 허용**하세요.
2. 손을 보여 주면 손 위에 `라벨 확률`이, 오른쪽에 라벨별 확률 막대가 나옵니다.
3. 같은 제스처가 잠깐(5프레임) 유지되면 화면 가운데에 반응이 뜹니다.
4. 엉뚱한 반응이 자주 뜨면 **최소 확신도** 슬라이더를 올리세요.
5. 모델을 다시 훈련했다면 `web_gesture.py`를 다시 실행하기만 하면 됩니다.

> `index.html`을 더블클릭해서 열면 동작하지 않습니다. 브라우저 보안상 웹캠과 파일 읽기는 `http://localhost` 또는 `https://` 주소에서만 허용되기 때문에 로컬 서버를 거칩니다.

### 7.3 핵심 코드 해설

**① 모델 내보내기 (`web_gesture.py`)** — PyTorch 가중치를 JSON으로 저장합니다.

```python
ckpt = torch.load(MODEL_PATH, map_location="cpu")
state = ckpt["state_dict"]
# GestureMLP의 Linear 층: net.0, net.3, net.6 (사이의 ReLU·Dropout은 가중치 없음)
layers = [{"w": state[f"net.{i}.weight"].tolist(), "b": state[f"net.{i}.bias"].tolist()} for i in layer_ids]
```

**② 파이썬과 같은 화면 만들기 (`app.js`)** — 파이썬은 640×480(4:3) 영상을 `cv2.flip`으로 뒤집어 학습했습니다.
웹캠이 16:9여도 결과가 같도록 **가운데를 4:3으로 잘라 좌우 반전**한 캔버스에서 인식합니다.

```js
ctx.translate(canvas.width, 0);
ctx.scale(-1, 1);                                   // 좌우 반전 (거울 모드)
ctx.drawImage(video, sx, sy, sw, sh, 0, 0, 640, 480); // 4:3으로 잘라 그리기
const result = landmarker.detectForVideo(canvas, now);
```

**③ 정규화 (`toFeatures`)** — 파이썬 `landmarks_to_features()`를 한 줄씩 옮긴 것입니다. **이 부분이 다르면 모델이 엉뚱한 답을 냅니다.**

```js
const pts = landmarks.map((p) => [p.x - w.x, p.y - w.y, p.z - w.z]); // 손목을 원점으로
if (handedness === "Left") pts.forEach((p) => { p[0] = -p[0]; });   // 왼손은 좌우 반전
// 가장 먼 점까지 거리를 1로 나눠 크기 통일
```

**④ 신경망 계산 (`predict`)** — `Linear → ReLU → Linear → ReLU → Linear → softmax`. Dropout은 추론 때 아무 일도 하지 않으므로 생략합니다.

```js
weights.layers.forEach((layer, li) => {
  const out = layer.b.map((b, o) => {
    let s = b;
    const row = layer.w[o];
    for (let i = 0; i < row.length; i++) s += row[i] * h[i];        // y = Wx + b
    return s;
  });
  const isLast = li === weights.layers.length - 1;
  h = isLast ? out : out.map((v) => Math.max(0, v));                // 마지막 층 전까지 ReLU
});
// 마지막에 softmax로 확률화
```

> 확인 결과: 같은 입력에 대해 브라우저와 PyTorch의 확률 차이가 **0.0000001 이하**로, 사실상 같은 모델입니다.

**⑤ 깜빡임 방지** — 한 프레임만 보고 반응하면 손을 바꾸는 순간 엉뚱한 반응이 번쩍입니다.
같은 제스처가 `STABLE_FRAMES`(5) 프레임 연속일 때만 띄우고, `HIDE_FRAMES`(10) 프레임 동안 사라지면 숨깁니다.

### 7.4 반응 바꾸기·추가하기

`web/app.js`의 `REACTIONS`만 수정하면 됩니다. 라벨 이름은 **훈련할 때와 글자까지 똑같아야** 합니다.
`icon`에는 이모지나 SVG/HTML을 넣을 수 있습니다 (나이키 로고는 SVG로 그림).

```js
const REACTIONS = {
  "나이키": { icon: NIKE_SWOOSH, caption: "JUST DO IT" },
  "오키": { icon: "👌", caption: "OK!" },
  "브이": { icon: "✌️", caption: "V!" },
  "굳": { icon: "👍", caption: "GOOD" },
  "배드": { icon: "👎", caption: "BAD" },
  "손바닥": { icon: "✋", caption: "HI!" },
  "주먹": { icon: "✊", caption: "ROCK" },   // 새 제스처를 훈련했다면 이렇게 추가
};
```

새 제스처를 추가하는 전체 순서: `gesture_studio.py`에서 수집 → 훈련 → `REACTIONS`에 한 줄 추가 → `python web_gesture.py`.

### 7.5 GitHub Pages 배포

GitHub Pages는 저장소의 HTML·JS 파일을 무료로 `https://<아이디>.github.io/<저장소>/` 주소에 올려 주는 기능입니다.
HTTPS로 제공되므로 웹캠도 바로 사용할 수 있습니다. 이 저장소는 **`gh-pages` 브랜치**를 배포합니다.

| 배포에 필요한 파일 | 이유 |
|---|---|
| `index.html` (루트) | 사이트 주소로 들어오면 `web/`으로 이동 |
| `web/index.html`, `web/app.js` | 페이지 본체 |
| `web/gesture_weights.json` | 훈련된 모델 (그래서 `.gitignore`에서 제외하고 커밋) |
| `hand_landmarker.task` | MediaPipe 손 모델 (`app.js`가 `../hand_landmarker.task`로 읽음) |

MediaPipe 라이브러리(`tasks-vision`)는 CDN(jsDelivr)에서 불러오므로 저장소에 넣을 필요가 없습니다.
원본 데이터(`gesture_data.csv`)와 PyTorch 모델(`gesture_model.pt`)은 배포에 필요 없어서 올리지 않습니다.

**처음 배포할 때** (한 번만)

```bash
git push origin HEAD:gh-pages
```

`gh-pages` 브랜치가 생기면 GitHub가 Pages를 자동으로 켭니다. 자동으로 켜지지 않으면 저장소 **Settings → Pages → Branch: `gh-pages` / `/ (root)`** 로 설정하세요.

**모델을 다시 훈련한 뒤 갱신할 때**

```bash
python -c "import web_gesture; web_gesture.export_weights()"   # web/gesture_weights.json 갱신
git add web/gesture_weights.json
git commit -m "Update gesture model"
git push origin HEAD:gh-pages        # 1~2분 뒤 사이트에 반영
```

> 배포 상태는 저장소의 **Actions** 탭(pages build and deployment)에서 확인할 수 있습니다.

---

## 8. 실습 과제

난이도 순으로 도전해 보세요.

1. **[기초]** 검지 끝(8번 점)에만 큰 원을 그려 보세요.
2. **[기초]** `num_hands=1`로 바꾸고 결과가 어떻게 달라지는지 관찰하세요.
3. **[응용]** 엄지 끝(4)과 검지 끝(8) 사이의 거리를 계산해 화면에 표시하세요. (힌트: `math.dist`)
4. **[응용]** 제스처별로 화면 테두리 색을 바꿔 보세요. (`Thumb_Up` = 초록, `Thumb_Down` = 빨강 …)
5. **[응용]** 펴진 손가락 개수를 세어 숫자로 표시하세요. (힌트: 각 손가락 TIP의 `y`가 PIP 관절보다 위에 있는지 비교)
6. **[심화]** 같은 제스처가 **10프레임 이상 연속**될 때만 "확정"으로 처리해 깜빡임을 없애 보세요.
7. **[심화]** `LIVE_STREAM` 모드 + `result_callback`으로 코드를 바꿔 보고 FPS를 비교하세요.
8. **[심화]** 제스처로 프로그램 제어하기: `Victory` → 스크린샷 저장, `Closed_Fist` → 종료.

**3강 과제**

9. **[기초]** 코끝(1번 점)에 빨간 원을 그려 보세요.
10. **[기초]** `FACE_LANDMARKS_LIPS`만 두껍게 그려 입술을 강조해 보세요.
11. **[응용]** `eyeBlinkLeft`, `eyeBlinkRight`가 0.5를 넘으면 "BLINK" 문구를 띄우고 **깜빡임 횟수**를 세어 보세요.
12. **[응용]** `jawOpen` 점수로 화면 밝기나 원의 크기를 조절해 보세요.
13. **[심화]** 눈을 2초 이상 감고 있으면 경고를 띄우는 **졸음 감지기**를 만들어 보세요.
14. **[심화]** 손(2강)과 얼굴(3강)을 한 프로그램에서 동시에 실행해 보세요. (힌트: 태스크 객체 두 개, 같은 프레임을 둘 다에 전달)

**4강 과제**

15. **[기초]** 가위·바위·보 3종 + `none`을 학습시켜 보세요.
16. **[응용]** 추론 결과로 컴퓨터와 가위바위보 게임을 만들어 보세요. (힌트: 같은 결과가 10프레임 연속일 때 확정)
17. **[심화]** `none` 없이 학습한 모델과 있는 모델의 오인식을 비교해 보세요.

**5강 과제**

18. **[기초]** `REACTIONS`의 `caption`과 이모지를 내 마음대로 바꿔 보세요.
19. **[응용]** 반응이 뜰 때 효과음이 나게 해 보세요. (힌트: `new Audio("파일.mp3").play()`)
20. **[응용]** 제스처마다 화면 배경색이 바뀌게 해 보세요.
21. **[심화]** 새 제스처(예: `주먹`)를 훈련하고 반응까지 추가해 GitHub Pages에 다시 배포해 보세요.

---

## 9. 자주 묻는 질문 / 문제 해결

| 증상 | 원인 및 해결 |
|---|---|
| `웹캠을 열 수 없습니다` | 다른 프로그램(Zoom, 카메라 앱 등)이 웹캠 사용 중 → 종료 후 재시도. 외장 카메라라면 `CAMERA_INDEX = 1`로 변경 |
| 창이 바로 닫힘 | 프레임을 읽지 못해(`cap.read()` 실패) 루프가 끝난 경우. 위와 같은 방법으로 해결 (3강 코드는 이때 콘솔에 메시지를 출력함) |
| `모델 파일이 없습니다` | `.task` 파일이 `.py`와 같은 폴더에 있는지 확인 |
| 모델 로드 실패 (경로 관련) | 경로에 한글이 있으면 `model_asset_path`가 실패할 수 있음 → 이 예제처럼 `model_asset_buffer` 사용 |
| 시작 시 `W0000 ...`, `INFO: ...` 로그 | MediaPipe 내부 로그. 무시해도 됨 |
| 왼손/오른손이 반대로 나옴 | 모델은 입력이 **거울 모드(좌우 반전)** 이미지라고 가정하고 Left/Right를 판단함. `cv2.flip(frame, 1)`을 빼면 라벨이 뒤바뀜 |
| 인식이 불안정함 | 조명을 밝게, 배경을 단순하게, 손을 카메라에서 0.5~1m 거리에 |
| 표정 점수가 안 나옴 (3강) | `output_face_blendshapes=True` 옵션 확인 |
| 얼굴 메시 때문에 화면이 복잡함 (3강) | `m` 키로 메시를 끄고 윤곽선만 보기 |
| 웹 페이지가 "카메라 권한을 허용해 주세요…"에서 멈춤 (5강) | 주소창 왼쪽 자물쇠 아이콘 → 카메라 **허용** 후 새로고침. 다른 프로그램이 웹캠을 쓰고 있지 않은지도 확인 |
| "파일을 직접 열면 동작하지 않습니다" (5강) | `index.html`을 더블클릭하지 말고 `python web_gesture.py`로 실행 |
| 웹에서 반응이 안 뜸 (5강) | 라벨 이름이 `REACTIONS`와 글자까지 같은지, 확률이 **최소 확신도**보다 높은지 확인 |
| 다시 훈련했는데 배포 사이트가 그대로 (5강) | `gesture_weights.json`을 다시 내보내 `gh-pages`에 푸시했는지 확인. 반영까지 1~2분 걸리고, 브라우저 캐시 때문에 `Ctrl+F5`가 필요할 수 있음 |

---

## 10. 더 알아보기

- [Gesture Recognizer 공식 가이드](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer)
- [Hand Landmarker 공식 가이드](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- [Face Landmarker 공식 가이드](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
- [Model Maker로 나만의 제스처 학습시키기](https://ai.google.dev/edge/mediapipe/solutions/customization/gesture_recognizer)
- [Hand Landmarker 웹(JavaScript) 가이드](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/web_js)
- [GitHub Pages 공식 문서](https://docs.github.com/ko/pages)

---

## 저장소 구조

```
.
├── README.md                  # 강의노트 (이 문서)
├── hand_webcam.py             # 1강: 손 랜드마크 검출
├── hand_landmarker.task       # 1강 모델
├── gesture_webcam.py          # 2강: 제스처 인식
├── gesture_recognizer.task    # 2강 모델
├── face_webcam.py             # 3강: 얼굴 랜드마크 & 표정 인식
├── face_landmarker.task       # 3강 모델
├── gesture_studio.py          # 4강: 수집·훈련·인식 GUI (추천)
├── custom_gesture.py          # 4강: 공통 모듈 (특징 변환, 신경망, CSV)
├── collect_gesture.py         # 4강 ①: 데이터 수집 → gesture_data.csv
├── train_gesture.py           # 4강 ②: 훈련 → gesture_model.pt
├── infer_gesture.py           # 4강 ③: 실시간 추론
├── web_gesture.py             # 5강: 모델 내보내기 + 로컬 웹서버
├── index.html                 # 5강: GitHub Pages 첫 화면 (web/으로 이동)
└── web/
    ├── index.html             # 5강: 웹 페이지 화면
    ├── app.js                 # 5강: 손 인식 → 신경망 → 반응 표시
    └── gesture_weights.json   # 5강: 훈련된 모델 (web_gesture.py가 생성)
```

> 모델 파일은 Google MediaPipe에서 제공하며 Apache License 2.0을 따릅니다.
