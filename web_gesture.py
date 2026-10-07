"""나만의 제스처 웹 버전 실행기

1) gesture_model.pt(파이썬에서 훈련한 모델) → web/gesture_weights.json 으로 내보내기
2) 이 폴더를 로컬 웹서버로 열고 브라우저에서 web/index.html 실행

실행: python web_gesture.py          (종료: 이 창에서 Ctrl+C)
모델을 다시 훈련했다면 이 스크립트를 다시 실행하기만 하면 됩니다.
"""
import json
import socket
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import torch

from custom_gesture import BASE_DIR, MODEL_PATH

WEB_DIR = BASE_DIR / "web"
WEIGHTS_PATH = WEB_DIR / "gesture_weights.json"


def export_weights():
    """PyTorch 가중치를 브라우저(JavaScript)가 읽을 수 있는 JSON으로 저장"""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"학습된 모델이 없습니다: {MODEL_PATH}\n먼저 gesture_studio.py에서 훈련하세요.")
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    state = ckpt["state_dict"]
    # GestureMLP의 Linear 층: net.0, net.3, net.6 (사이의 ReLU·Dropout은 가중치 없음)
    layer_ids = sorted({int(k.split(".")[1]) for k in state if k.endswith(".weight")})
    layers = [{"w": state[f"net.{i}.weight"].tolist(), "b": state[f"net.{i}.bias"].tolist()} for i in layer_ids]
    data = {"labels": ckpt["labels"], "layers": layers}
    WEIGHTS_PATH.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"모델 내보내기 완료: {WEIGHTS_PATH.name}  라벨 {ckpt['labels']}")


def free_port(start=8000):
    for port in range(start, start + 50):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("사용 가능한 포트가 없습니다.")


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")  # 재훈련한 모델이 바로 반영되도록
        super().end_headers()


def main():
    export_weights()
    port = free_port()
    # 웹캠(getUserMedia)은 localhost 또는 https에서만 허용되므로 로컬 서버로 엶
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(QuietHandler, directory=str(BASE_DIR)))
    url = f"http://localhost:{port}/web/index.html"
    print(f"웹 페이지: {url}\n종료하려면 Ctrl+C")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n서버 종료")


if __name__ == "__main__":
    main()
