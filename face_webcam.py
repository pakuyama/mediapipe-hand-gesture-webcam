"""MediaPipe Face Landmarker - 웹캠 실시간 얼굴 랜드마크(478점) 검출

화면: 얼굴 메시 + 윤곽선 + 홍채, 표정 블렌드셰이프 상위 5개
실행: python face_webcam.py   (종료: q 또는 ESC, 메시 표시 토글: m)
"""
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).with_name("face_landmarker.task")
CAMERA_INDEX = 0
NUM_FACES = 1
TOP_BLENDSHAPES = 5

C = vision.FaceLandmarksConnections
CONTOURS = C.FACE_LANDMARKS_CONTOURS
IRISES = C.FACE_LANDMARKS_LEFT_IRIS + C.FACE_LANDMARKS_RIGHT_IRIS
TESSELATION = C.FACE_LANDMARKS_TESSELATION


def draw_lines(frame, pts, connections, color, thickness):
    for c in connections:
        cv2.line(frame, pts[c.start], pts[c.end], color, thickness)


def draw_faces(frame, result, show_mesh):
    h, w = frame.shape[:2]
    for landmarks in result.face_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        if show_mesh:
            draw_lines(frame, pts, TESSELATION, (80, 80, 80), 1)
        draw_lines(frame, pts, CONTOURS, (0, 255, 0), 1)
        draw_lines(frame, pts, IRISES, (0, 200, 255), 1)

    # 첫 번째 얼굴의 표정 블렌드셰이프 상위 N개 (smile, blink 등)
    if result.face_blendshapes:
        top = sorted(result.face_blendshapes[0], key=lambda b: b.score, reverse=True)
        for k, b in enumerate(top[:TOP_BLENDSHAPES]):
            y = 60 + k * 25
            cv2.rectangle(frame, (10, y - 15), (10 + int(b.score * 150), y + 3), (255, 150, 0), -1)
            cv2.putText(frame, f"{b.category_name} {b.score:.2f}", (170, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")

    options = vision.FaceLandmarkerOptions(
        # 경로에 한글이 있으면 네이티브 로더가 못 열어서 바이트로 전달
        base_options=mp_tasks.BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=NUM_FACES,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    show_mesh = True
    start = time.perf_counter()
    prev = start
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("웹캠 프레임을 읽지 못해 종료합니다.")
                break
            frame = cv2.flip(frame, 1)  # 거울 모드

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.perf_counter() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            draw_faces(frame, result, show_mesh)

            now = time.perf_counter()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS {fps:.1f}  Faces {len(result.face_landmarks)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Face Landmarker", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                show_mesh = not show_mesh

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
