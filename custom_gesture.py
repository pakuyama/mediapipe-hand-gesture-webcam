"""나만의 제스처 학습용 공통 모듈 (수집 / 훈련 / 추론 스크립트가 함께 사용)

- 손 랜드마커 생성
- 랜드마크 21개 → 63차원 특징 벡터 변환 (위치·크기·왼오른손 영향 제거)
- 분류 신경망(MLP) 정의
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

BASE_DIR = Path(__file__).parent
HAND_MODEL_PATH = BASE_DIR / "hand_landmarker.task"
DATA_PATH = BASE_DIR / "gesture_data.csv"
MODEL_PATH = BASE_DIR / "gesture_model.pt"

NUM_FEATURES = 21 * 3

# 21개 랜드마크 연결 (엄지, 검지, 중지, 약지, 소지, 손바닥)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def create_hand_landmarker(num_hands=1):
    if not HAND_MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {HAND_MODEL_PATH}")
    options = vision.HandLandmarkerOptions(
        # 경로에 한글이 있으면 네이티브 로더가 못 열어서 바이트로 전달
        base_options=mp_tasks.BaseOptions(model_asset_buffer=HAND_MODEL_PATH.read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def landmarks_to_features(landmarks, handedness):
    """랜드마크 21개를 화면 위치·손 크기·왼/오른손과 무관한 63차원 벡터로 변환"""
    pts = np.array([(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32)
    pts -= pts[0]                      # 손목(0번)을 원점으로 → 화면 위치 무관
    if handedness == "Left":
        pts[:, 0] = -pts[:, 0]         # 왼손은 좌우 반전 → 한 손으로 학습해도 양손 인식
    scale = np.linalg.norm(pts[:, :2], axis=1).max()
    pts /= max(scale, 1e-6)            # 가장 먼 점까지 거리를 1로 → 카메라 거리 무관
    return pts.flatten()


class GestureMLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_FEATURES, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def draw_hand(frame, pts, color=(0, 255, 0)):
    import cv2
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)
