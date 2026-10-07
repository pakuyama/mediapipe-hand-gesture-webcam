"""나만의 제스처 실시간 추론 - gesture_model.pt로 웹캠 손 제스처 분류

실행: python infer_gesture.py   (종료: q 또는 ESC)
"""
import time

import cv2
import mediapipe as mp
import torch

from custom_gesture import MODEL_PATH, GestureMLP, create_hand_landmarker, draw_hand, landmarks_to_features

CAMERA_INDEX = 0
NUM_HANDS = 2
MIN_SCORE = 0.7  # 이보다 확신이 낮으면 "?"로 표시


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"학습된 모델이 없습니다: {MODEL_PATH}\n먼저 train_gesture.py를 실행하세요.")
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    labels = ckpt["labels"]
    model = GestureMLP(len(labels))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"라벨: {labels}")

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    start = time.perf_counter()
    prev = start
    with create_hand_landmarker(num_hands=NUM_HANDS) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 수집 때와 동일하게 거울 모드
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.perf_counter() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            for i, landmarks in enumerate(result.hand_landmarks):
                hand = result.handedness[i][0].category_name
                feats = landmarks_to_features(landmarks, hand)
                with torch.no_grad():
                    probs = torch.softmax(model(torch.from_numpy(feats)[None]), dim=1)[0]
                score, idx = probs.max(0)
                name = labels[idx] if score >= MIN_SCORE else "?"

                pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
                draw_hand(frame, pts)
                x = min(p[0] for p in pts)
                y = min(p[1] for p in pts) - 10
                cv2.putText(frame, f"{hand} {name} {score:.2f}", (x, max(y, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            now = time.perf_counter()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS {fps:.1f}  Hands {len(result.hand_landmarks)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Custom Gesture", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
