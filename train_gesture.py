"""나만의 제스처 분류기 훈련 - gesture_data.csv → gesture_model.pt

실행: python train_gesture.py   (옵션: --epochs 200 --lr 0.001)
GUI(gesture_studio.py)에서는 train() 함수를 직접 호출합니다.
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
MIN_SAMPLES = 10  # 라벨당 최소 샘플 수


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"데이터가 없습니다: {DATA_PATH}\n먼저 제스처를 수집하세요.")
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
    few = [n for n in labels if names.count(n) < MIN_SAMPLES]
    if few:
        raise ValueError(f"샘플이 {MIN_SAMPLES}개 미만인 라벨이 있습니다: {few}")
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


def confusion_text(cm, labels):
    """혼동 행렬: 행 = 정답, 열 = 예측. 대각선 밖 숫자가 크면 그 두 제스처를 헷갈리는 것"""
    width = max(len(n) for n in labels) + 2
    lines = ["혼동 행렬 (행=정답, 열=예측)",
             " " * width + "".join(f"{n[:6]:>8}" for n in labels)]
    for i, name in enumerate(labels):
        lines.append(f"{name:>{width}}" + "".join(f"{v:8d}" for v in cm[i]))
    return "\n".join(lines)


def train(epochs=150, lr=1e-3, log=print, on_epoch=None, should_stop=None):
    """데이터를 읽어 훈련하고 gesture_model.pt에 저장. (최고 검증 정확도, 라벨 목록)을 반환

    log(str)                       : 진행 메시지 출력 함수
    on_epoch(epoch, epochs, loss, acc): 매 epoch 끝날 때 호출 (진행 바 등)
    should_stop()                  : True를 반환하면 지금까지의 최고 모델로 중단
    """
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    X, y, labels = load_data()
    log(f"데이터 {len(X)}개, 라벨 {len(labels)}개")
    for i, name in enumerate(labels):
        log(f"  {name:>12}: {(y == i).sum()}개")

    train_idx, val_idx = split(X, y, rng)
    X_train, y_train = torch.from_numpy(X[train_idx]), torch.from_numpy(y[train_idx])
    X_val, y_val = torch.from_numpy(X[val_idx]), torch.from_numpy(y[val_idx])
    log(f"훈련 {len(X_train)}개 / 검증 {len(X_val)}개\n")

    model = GestureMLP(len(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
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
        avg_loss = total_loss / len(X_train)
        if epoch % 10 == 0 or epoch == 1:
            log(f"epoch {epoch:4d}  loss {avg_loss:.4f}  val_acc {acc:.3f}")
        if on_epoch:
            on_epoch(epoch, epochs, avg_loss, acc)
        if should_stop and should_stop():
            log(f"epoch {epoch}에서 중단")
            break

    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "labels": labels}, MODEL_PATH)
    log(f"\n최고 검증 정확도 {best_acc:.3f} → 저장: {MODEL_PATH}")

    model.eval()
    with torch.no_grad():
        pred = model(X_val).argmax(1).numpy()
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_val.numpy(), pred):
        cm[t, p] += 1
    log("\n" + confusion_text(cm, labels))
    return best_acc, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args.epochs, args.lr)


if __name__ == "__main__":
    main()
