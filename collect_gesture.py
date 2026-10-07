"""나만의 제스처 데이터 수집 - 웹캠 손 랜드마크를 라벨과 함께 CSV로 저장

실행: python collect_gesture.py 라벨1 라벨2 ...
  예) python collect_gesture.py none rock scissors paper
조작:
  숫자키 1~9  : 수집할 라벨 선택
  SPACE      : 녹화 시작/정지 (녹화 중에는 손이 보이는 프레임마다 1개씩 저장)
  q / ESC    : 종료 (종료할 때 남은 샘플도 저장됨)

라벨 이름은 영어로 쓰세요 (OpenCV 화면 글씨가 한글을 표시하지 못함).
같은 파일(gesture_data.csv)에 계속 이어서 저장되므로 여러 번 나눠 수집해도 됩니다.
"""
import sys
import time

import cv2
import mediapipe as mp

from custom_gesture import DATA_PATH, append_rows, create_hand_landmarker, draw_hand, landmarks_to_features, load_counts

CAMERA_INDEX = 0
TARGET_PER_LABEL = 300  # 라벨당 이만큼 모이면 자동으로 녹화 정지
SAVE_EVERY = 2          # N프레임마다 1개 저장 (연속 프레임은 거의 같아서 간격을 둠)


def main():
    labels = sys.argv[1:]
    if not labels or len(labels) > 9:
        print(__doc__)
        sys.exit("라벨을 1~9개 입력하세요.  예) python collect_gesture.py none rock scissors paper")

    counts = load_counts()
    pending = []  # 아직 파일에 안 쓴 샘플 (녹화 정지·라벨 변경·종료 시 저장)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    current = 0
    recording = False
    frame_idx = 0
    start = time.perf_counter()
    with create_hand_landmarker(num_hands=1) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울 모드 (추론 때도 똑같이 뒤집어야 함)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.perf_counter() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            label = labels[current]
            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                hand = result.handedness[0][0].category_name
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
                draw_hand(frame, pts, (0, 0, 255) if recording else (0, 255, 0))

                frame_idx += 1
                if recording and frame_idx % SAVE_EVERY == 0:
                    feats = landmarks_to_features(landmarks, hand)
                    pending.append((label, feats))
                    counts[label] += 1
                    if counts[label] >= TARGET_PER_LABEL:
                        recording = False
                        append_rows(pending)
                        pending.clear()

            # 상태 표시
            status = "REC" if recording else "PAUSE"
            color = (0, 0, 255) if recording else (200, 200, 200)
            cv2.putText(frame, f"[{status}] {label}  ({counts[label]}/{TARGET_PER_LABEL})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if not result.hand_landmarks:
                cv2.putText(frame, "No hand", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            for i, name in enumerate(labels):
                mark = ">" if i == current else " "
                cv2.putText(frame, f"{mark}{i + 1}: {name} {counts[name]}", (10, 100 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "1-9: label  SPACE: rec  q: quit", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Collect Gesture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                recording = not recording
                append_rows(pending)
                pending.clear()
            if ord("1") <= key <= ord("9") and key - ord("1") < len(labels):
                current = key - ord("1")
                recording = False
                append_rows(pending)
                pending.clear()

    append_rows(pending)
    cap.release()
    cv2.destroyAllWindows()
    print(f"저장 위치: {DATA_PATH}")
    for name, n in sorted(counts.items()):
        print(f"  {name:>12}: {n}개")


if __name__ == "__main__":
    main()
