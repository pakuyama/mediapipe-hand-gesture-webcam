# MediaPipe 비전 태스크 실습: 손 랜드마크 & 제스처 인식

> 웹캠 영상에서 **손의 21개 관절 좌표**를 찾고, 그 손이 **어떤 제스처**를 취하고 있는지 실시간으로 인식하는 두 개의 실습 예제입니다.
> Google **MediaPipe Tasks** 파이썬 API와 **OpenCV**를 사용합니다.

| 실습 | 파일 | 모델 | 결과 |
|---|---|---|---|
| 1강 | `hand_webcam.py` | `hand_landmarker.task` | 손 관절 21개 + 왼손/오른손 구분 |
| 2강 | `gesture_webcam.py` | `gesture_recognizer.task` | 위 결과 + 제스처 이름(👍, ✌️ …) |

---

## 0. 학습 목표

이 강의를 마치면 다음을 할 수 있습니다.

1. MediaPipe Tasks의 **"모델 파일 → 옵션 → 태스크 객체 → 추론"** 흐름을 설명할 수 있다.
2. OpenCV로 웹캠 프레임을 받아 MediaPipe가 요구하는 형식(`mp.Image`, RGB)으로 변환할 수 있다.
3. **정규화 좌표**(0~1)를 화면 픽셀 좌표로 바꿔 랜드마크를 그릴 수 있다.
4. 실행 모드(`IMAGE` / `VIDEO` / `LIVE_STREAM`)의 차이를 알고 상황에 맞게 고를 수 있다.
5. 인식된 제스처 결과를 이용해 간단한 응용 프로그램을 만들 수 있다.

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
```

> 실습 확인 환경: Python 3.14, mediapipe 1.1.0, opencv-python 5.0.0 (Windows 11)

### 2.2 모델 파일

이 저장소에 두 모델 파일이 포함되어 있습니다. 직접 받으려면 아래 공식 링크를 사용하세요.

| 모델 | 다운로드 |
|---|---|
| `hand_landmarker.task` | https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task |
| `gesture_recognizer.task` | https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task |

모델 파일은 `.py` 파일과 **같은 폴더**에 두어야 합니다.

### 2.3 실행

```bash
python hand_webcam.py       # 1강: 손 랜드마크
python gesture_webcam.py    # 2강: 제스처 인식
```

종료: 영상 창에서 `q` 또는 `ESC`

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

## 5. 실습 과제

난이도 순으로 도전해 보세요.

1. **[기초]** 검지 끝(8번 점)에만 큰 원을 그려 보세요.
2. **[기초]** `num_hands=1`로 바꾸고 결과가 어떻게 달라지는지 관찰하세요.
3. **[응용]** 엄지 끝(4)과 검지 끝(8) 사이의 거리를 계산해 화면에 표시하세요. (힌트: `math.dist`)
4. **[응용]** 제스처별로 화면 테두리 색을 바꿔 보세요. (`Thumb_Up` = 초록, `Thumb_Down` = 빨강 …)
5. **[응용]** 펴진 손가락 개수를 세어 숫자로 표시하세요. (힌트: 각 손가락 TIP의 `y`가 PIP 관절보다 위에 있는지 비교)
6. **[심화]** 같은 제스처가 **10프레임 이상 연속**될 때만 "확정"으로 처리해 깜빡임을 없애 보세요.
7. **[심화]** `LIVE_STREAM` 모드 + `result_callback`으로 코드를 바꿔 보고 FPS를 비교하세요.
8. **[심화]** 제스처로 프로그램 제어하기: `Victory` → 스크린샷 저장, `Closed_Fist` → 종료.

---

## 6. 자주 묻는 질문 / 문제 해결

| 증상 | 원인 및 해결 |
|---|---|
| `웹캠을 열 수 없습니다` | 다른 프로그램(Zoom, 카메라 앱 등)이 웹캠 사용 중 → 종료 후 재시도. 외장 카메라라면 `CAMERA_INDEX = 1`로 변경 |
| 창이 바로 닫힘 | 프레임을 읽지 못해(`cap.read()` 실패) 루프가 끝난 경우. 위와 같은 방법으로 해결 |
| `모델 파일이 없습니다` | `.task` 파일이 `.py`와 같은 폴더에 있는지 확인 |
| 모델 로드 실패 (경로 관련) | 경로에 한글이 있으면 `model_asset_path`가 실패할 수 있음 → 이 예제처럼 `model_asset_buffer` 사용 |
| 시작 시 `W0000 ...`, `INFO: ...` 로그 | MediaPipe 내부 로그. 무시해도 됨 |
| 왼손/오른손이 반대로 나옴 | 모델은 입력이 **거울 모드(좌우 반전)** 이미지라고 가정하고 Left/Right를 판단함. `cv2.flip(frame, 1)`을 빼면 라벨이 뒤바뀜 |
| 인식이 불안정함 | 조명을 밝게, 배경을 단순하게, 손을 카메라에서 0.5~1m 거리에 |

---

## 7. 더 알아보기

- [Gesture Recognizer 공식 가이드](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer)
- [Hand Landmarker 공식 가이드](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- [Model Maker로 나만의 제스처 학습시키기](https://ai.google.dev/edge/mediapipe/solutions/customization/gesture_recognizer)

---

## 저장소 구조

```
.
├── README.md                  # 강의노트 (이 문서)
├── hand_webcam.py             # 1강: 손 랜드마크 검출
├── hand_landmarker.task       # 1강 모델
├── gesture_webcam.py          # 2강: 제스처 인식
└── gesture_recognizer.task    # 2강 모델
```

> 모델 파일은 Google MediaPipe에서 제공하며 Apache License 2.0을 따릅니다.
