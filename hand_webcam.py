"""MediaPipe Hand Landmarker - 웹캠 실시간 손 랜드마크 검출

실행: python hand_webcam.py   (종료: q 또는 ESC)
"""
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
CAMERA_INDEX = 0
NUM_HANDS = 2

# 21개 랜드마크 연결 (엄지, 검지, 중지, 약지, 소지, 손바닥)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def draw_hands(frame, result):
    h, w = frame.shape[:2]
    for i, landmarks in enumerate(result.hand_landmarks):
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for p in pts:
            cv2.circle(frame, p, 4, (0, 0, 255), -1)

        # 손 라벨 (Left/Right) + 신뢰도
        if result.handedness and i < len(result.handedness):
            cat = result.handedness[i][0]
            x = min(p[0] for p in pts)
            y = min(p[1] for p in pts) - 10
            cv2.putText(frame, f"{cat.category_name} {cat.score:.2f}", (x, max(y, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")

    options = vision.HandLandmarkerOptions(
        # 경로에 한글이 있으면 네이티브 로더가 못 열어서 바이트로 전달
        base_options=mp_tasks.BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    start = time.perf_counter()
    prev = start
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울 모드

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.perf_counter() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            draw_hands(frame, result)

            now = time.perf_counter()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS {fps:.1f}  Hands {len(result.hand_landmarks)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Hand Landmarker", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
