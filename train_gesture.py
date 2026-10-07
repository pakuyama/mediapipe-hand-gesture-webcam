"""나만의 제스처 분류기 훈련 - gesture_data.csv → gesture_model.pt

실행: python train_gesture.py   (옵션: --epochs 200 --lr 0.001)
"""
import argparse
import csv

import numpy as np
import torch
from torch import nn

from custom_gesture import DATA_PATH, MODEL_PATH, GestureMLP

VAL_RATIO = 0.2
BATCH_SIZE = 64
SEED = 42


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"데이터가 없습니다: {DATA_PATH}\n먼저 collect_gesture.py로 수집하세요.")
    names, feats = [], []
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row:
                names.append(row[0])
                feats.append([float(v) for v in row[1:]])
    labels = sorted(set(names))
    if len(labels) < 2:
        raise ValueError(f"라벨이 2개 이상 필요합니다. 현재: {labels}")
    X = np.array(feats, dtype=np.float32)
    y = np.array([labels.index(n) for n in names], dtype=np.int64)
    return X, y, labels


def split(X, y, rng):
    """라벨별로 같은 비율(VAL_RATIO)만큼 검증용으로 떼어냄"""
    train_idx, val_idx = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        n_val = max(1, int(len(idx) * VAL_RATIO))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return np.array(train_idx), np.array(val_idx)


def augment(x):
    """약간의 회전·크기·잡음을 더해 데이터가 적어도 잘 일반화되도록 함"""
    n = x.shape[0]
    pts = x.view(n, 21, 3)
    angle = (torch.rand(n) - 0.5) * (2 * np.pi / 18)  # ±10도 회전 (화면 평면)
    cos, sin = torch.cos(angle), torch.sin(angle)
    px, py = pts[..., 0], pts[..., 1]
    rx = cos[:, None] * px - sin[:, None] * py
    ry = sin[:, None] * px + cos[:, None] * py
    pts = torch.stack([rx, ry, pts[..., 2]], dim=-1)
    pts = pts * (1 + (torch.rand(n, 1, 1) - 0.5) * 0.1)  # ±5% 크기
    pts = pts + torch.randn_like(pts) * 0.01            # 작은 잡음
    return pts.view(n, -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    X, y, labels = load_data()
    print(f"데이터 {len(X)}개, 라벨 {len(labels)}개")
    for i, name in enumerate(labels):
        print(f"  {name:>12}: {(y == i).sum()}개")

    train_idx, val_idx = split(X, y, rng)
    X_train, y_train = torch.from_numpy(X[train_idx]), torch.from_numpy(y[train_idx])
    X_val, y_val = torch.from_numpy(X[val_idx]), torch.from_numpy(y[val_idx])
    print(f"훈련 {len(X_train)}개 / 검증 {len(X_val)}개\n")

    model = GestureMLP(len(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(X_train))
        total_loss = 0.0
        for i in range(0, len(perm), BATCH_SIZE):
            b = perm[i:i + BATCH_SIZE]
            loss = loss_fn(model(augment(X_train[b])), y_train[b])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(b)

        model.eval()
        with torch.no_grad():
            pred = model(X_val).argmax(1)
            acc = (pred == y_val).float().mean().item()
        if acc >= best_acc:  # 같은 정확도면 더 오래 학습한 쪽을 저장
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:4d}  loss {total_loss / len(X_train):.4f}  val_acc {acc:.3f}")

    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "labels": labels}, MODEL_PATH)
    print(f"\n최고 검증 정확도 {best_acc:.3f} → 저장: {MODEL_PATH}")

    # 혼동 행렬: 행 = 정답, 열 = 예측. 대각선 밖 숫자가 크면 그 두 제스처를 헷갈리는 것
    model.eval()
    with torch.no_grad():
        pred = model(X_val).argmax(1).numpy()
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_val.numpy(), pred):
        cm[t, p] += 1
    width = max(len(n) for n in labels) + 2
    print("\n혼동 행렬 (행=정답, 열=예측)")
    print(" " * width + "".join(f"{n[:6]:>8}" for n in labels))
    for i, name in enumerate(labels):
        print(f"{name:>{width}}" + "".join(f"{v:8d}" for v in cm[i]))


if __name__ == "__main__":
    main()
