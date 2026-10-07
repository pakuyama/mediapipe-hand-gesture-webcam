"""나만의 제스처 스튜디오 - 수집 · 훈련 · 실시간 인식을 한 창에서 (GUI)

실행: python gesture_studio.py
  1) 오른쪽 위에 라벨 이름 입력 → [추가]   (한글 가능, 'none' 라벨도 꼭 만들기)
  2) 라벨 선택 → [녹화 시작] 또는 Space  → 손 모양을 조금씩 움직이며 수집
  3) [훈련 시작]  → 끝나면 자동으로 실시간 인식이 켜짐
"""
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import mediapipe as mp
import torch
from PIL import Image, ImageDraw, ImageFont, ImageTk

from custom_gesture import (MODEL_PATH, GestureMLP, append_rows, create_hand_landmarker, delete_label,
                            draw_hand, landmarks_to_features, load_counts)
from train_gesture import train

CAMERA_INDEX = 0
NUM_HANDS = 2
SAVE_EVERY = 2   # 녹화 중 N프레임마다 1개 저장 (연속 프레임은 거의 같아서 간격을 둠)
FLUSH_EVERY = 30  # 샘플이 이만큼 쌓이면 파일에 기록
UI_FONT = ("맑은 고딕", 10)


def load_font(size):
    """영상 위에 한글을 쓰기 위한 폰트 (OpenCV putText는 한글 불가)"""
    for name in ("malgun.ttf", "malgunbd.ttf", "gulim.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


class GestureStudio:
    def __init__(self, root):
        self.root = root
        self.counts = load_counts()
        self.extra_labels = set()  # 추가했지만 아직 데이터가 없는 라벨
        self.pending = []          # 아직 파일에 안 쓴 샘플
        self.recording = False
        self.frame_idx = 0
        self.model = None
        self.model_labels = []
        self.training = False
        self.stop_flag = False
        self.events = queue.Queue()  # 훈련 스레드 → 화면 갱신용 메시지
        self.font = load_font(22)
        self.small_font = load_font(18)
        self.prev_time = time.perf_counter()

        self.build_ui()
        self.refresh_labels()
        self.load_model(quiet=True)

        self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.log("웹캠을 열 수 없습니다. 다른 프로그램이 사용 중인지 확인하세요. (훈련은 가능)")
        self.landmarker = create_hand_landmarker(num_hands=NUM_HANDS)
        self.start = time.perf_counter()

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.bind("<space>", self.on_space)
        for d in range(1, 10):
            root.bind(str(d), self.on_digit)
        self.update_frame()
        self.poll_events()

    # ------------------------------------------------------------------ UI 구성
    def build_ui(self):
        root = self.root
        root.title("나만의 제스처 스튜디오")
        style = ttk.Style()
        style.configure(".", font=UI_FONT)
        style.configure("Treeview.Heading", font=(UI_FONT[0], 10, "bold"))
        style.configure("Rec.TButton", foreground="red")

        self.video = ttk.Label(root, text="카메라 준비 중...", anchor="center", width=80)
        self.video.grid(row=0, column=0, padx=8, pady=8, sticky="n")

        side = ttk.Frame(root)
        side.grid(row=0, column=1, padx=(0, 8), pady=8, sticky="ns")

        # 라벨 목록
        box = ttk.LabelFrame(side, text=" 제스처 라벨 ", padding=6)
        box.pack(fill="x")
        self.tree = ttk.Treeview(box, columns=("no", "label", "count"), show="headings", height=7,
                                 selectmode="browse")
        for col, text, w, anchor in (("no", "키", 36, "center"), ("label", "라벨", 150, "w"),
                                     ("count", "샘플 수", 80, "e")):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=w, anchor=anchor, stretch=False)
        self.tree.pack(fill="x")
        self.tree.bind("<<TreeviewSelect>>", lambda e: self.stop_recording())

        row = ttk.Frame(box)
        row.pack(fill="x", pady=(6, 0))
        self.new_label = tk.StringVar()
        entry = ttk.Entry(row, textvariable=self.new_label, width=18)
        entry.pack(side="left", fill="x", expand=True)
        entry.bind("<Return>", lambda e: self.add_label())
        ttk.Button(row, text="추가", width=6, command=self.add_label).pack(side="left", padx=(4, 0))
        ttk.Button(box, text="선택한 라벨과 데이터 삭제", command=self.remove_label).pack(fill="x", pady=(4, 0))

        # ① 수집
        box = ttk.LabelFrame(side, text=" ① 수집 ", padding=6)
        box.pack(fill="x", pady=(8, 0))
        row = ttk.Frame(box)
        row.pack(fill="x")
        ttk.Label(row, text="라벨당 목표 개수").pack(side="left")
        self.target = tk.IntVar(value=300)
        ttk.Spinbox(row, from_=20, to=5000, increment=50, textvariable=self.target, width=7).pack(side="right")
        self.rec_btn = ttk.Button(box, text="● 녹화 시작  (Space)", command=self.toggle_recording)
        self.rec_btn.pack(fill="x", pady=(6, 0))
        ttk.Label(box, text="녹화 중에 손을 조금씩 돌리고 앞뒤로 움직이세요.", foreground="gray").pack(anchor="w")

        # ② 훈련
        box = ttk.LabelFrame(side, text=" ② 훈련 ", padding=6)
        box.pack(fill="x", pady=(8, 0))
        row = ttk.Frame(box)
        row.pack(fill="x")
        ttk.Label(row, text="Epoch").pack(side="left")
        self.epochs = tk.IntVar(value=150)
        ttk.Spinbox(row, from_=10, to=2000, increment=50, textvariable=self.epochs, width=7).pack(side="right")
        row = ttk.Frame(box)
        row.pack(fill="x", pady=(6, 0))
        self.train_btn = ttk.Button(row, text="훈련 시작", command=self.start_training)
        self.train_btn.pack(side="left", fill="x", expand=True)
        self.stop_btn = ttk.Button(row, text="중단", command=self.request_stop, state="disabled", width=6)
        self.stop_btn.pack(side="left", padx=(4, 0))
        self.progress = ttk.Progressbar(box, maximum=100)
        self.progress.pack(fill="x", pady=(6, 0))
        self.train_status = tk.StringVar(value="")
        ttk.Label(box, textvariable=self.train_status).pack(anchor="w")

        # ③ 인식
        box = ttk.LabelFrame(side, text=" ③ 실시간 인식 ", padding=6)
        box.pack(fill="x", pady=(8, 0))
        self.infer_on = tk.BooleanVar(value=False)
        ttk.Checkbutton(box, text="인식 켜기", variable=self.infer_on).pack(anchor="w")
        row = ttk.Frame(box)
        row.pack(fill="x")
        self.min_score = tk.DoubleVar(value=0.7)
        ttk.Label(row, text="최소 확신도").pack(side="left")
        score_text = ttk.Label(row, text="0.70", width=5)
        score_text.pack(side="right")
        ttk.Scale(row, from_=0.3, to=0.99, variable=self.min_score,
                  command=lambda v: score_text.configure(text=f"{float(v):.2f}")).pack(side="right", fill="x",
                                                                                      expand=True, padx=4)
        self.model_status = tk.StringVar(value="모델 없음")
        ttk.Label(box, textvariable=self.model_status, foreground="gray", wraplength=270).pack(anchor="w")
        self.pred_text = tk.StringVar(value="-")
        ttk.Label(box, textvariable=self.pred_text, font=(UI_FONT[0], 16, "bold"),
                  foreground="#1a73e8").pack(anchor="w", pady=(4, 0))

        # 로그
        logf = ttk.Frame(root)
        logf.grid(row=1, column=0, columnspan=2, padx=8, pady=(0, 8), sticky="nsew")
        self.log_box = tk.Text(logf, height=9, font=("Consolas", 9), state="disabled", wrap="none")
        scroll = ttk.Scrollbar(logf, command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=scroll.set)
        self.log_box.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

    def log(self, msg):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    # ------------------------------------------------------------------ 라벨 관리
    def all_labels(self):
        return sorted(set(self.counts) | self.extra_labels)

    def current_label(self):
        sel = self.tree.selection()
        return sel[0] if sel else None

    def refresh_labels(self, select=None):
        select = select or self.current_label()
        self.tree.delete(*self.tree.get_children())
        for i, name in enumerate(self.all_labels()):
            self.tree.insert("", "end", iid=name, values=(i + 1 if i < 9 else "", name, self.counts.get(name, 0)))
        if select and self.tree.exists(select):
            self.tree.selection_set(select)
        elif self.tree.get_children():
            self.tree.selection_set(self.tree.get_children()[0])

    def add_label(self):
        name = self.new_label.get().strip()
        if not name:
            return
        if name in self.all_labels():
            self.log(f"이미 있는 라벨입니다: {name}")
        else:
            self.extra_labels.add(name)
            self.log(f"라벨 추가: {name}")
        self.new_label.set("")
        self.refresh_labels(select=name)
        self.root.focus_set()  # Space가 입력칸이 아닌 녹화 버튼으로 동작하도록

    def remove_label(self):
        name = self.current_label()
        if not name:
            return
        n = self.counts.get(name, 0)
        if n and not messagebox.askyesno("삭제 확인", f"'{name}' 라벨의 데이터 {n}개를 삭제할까요?\n(되돌릴 수 없습니다)"):
            return
        self.stop_recording()
        removed = delete_label(name)
        self.counts.pop(name, None)
        self.extra_labels.discard(name)
        self.tree.selection_remove(name)
        self.refresh_labels()
        self.log(f"라벨 삭제: {name} (데이터 {removed}개). 다시 훈련해야 모델에 반영됩니다.")

    # ------------------------------------------------------------------ 수집
    def toggle_recording(self):
        if self.recording:
            self.stop_recording()
            return
        label = self.current_label()
        if not label:
            messagebox.showinfo("라벨 필요", "먼저 오른쪽 위에서 라벨을 추가하고 선택하세요.")
            return
        if not self.cap.isOpened():
            return
        self.recording = True
        self.frame_idx = 0
        self.rec_btn.configure(text="■ 녹화 정지  (Space)", style="Rec.TButton")
        self.log(f"녹화 시작: {label}")

    def stop_recording(self):
        self.flush_samples()
        if self.recording:
            self.recording = False
            self.rec_btn.configure(text="● 녹화 시작  (Space)", style="TButton")
            label = self.current_label()
            self.log(f"녹화 정지: {label} → {self.counts.get(label, 0)}개")

    def flush_samples(self):
        if self.pending:
            append_rows(self.pending)
            self.pending.clear()

    def target_count(self):
        try:
            return max(1, int(self.target.get()))
        except (tk.TclError, ValueError):
            return 300

    def on_space(self, event):
        if isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Spinbox)):
            return
        self.toggle_recording()

    def on_digit(self, event):
        if isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Spinbox)):
            return
        labels = self.all_labels()
        i = int(event.char) - 1
        if i < len(labels):
            self.tree.selection_set(labels[i])

    # ------------------------------------------------------------------ 훈련
    def start_training(self):
        if self.training:
            return
        self.stop_recording()
        try:
            epochs = max(1, int(self.epochs.get()))
        except (tk.TclError, ValueError):
            epochs = 150
        self.training = True
        self.stop_flag = False
        self.train_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress["value"] = 0
        self.log("\n===== 훈련 시작 =====")

        def run():
            try:
                acc, _ = train(
                    epochs,
                    log=lambda m: self.events.put(("log", m)),
                    on_epoch=lambda e, total, loss, a: self.events.put(
                        ("progress", e / total * 100, f"epoch {e}/{total}   검증 정확도 {a:.3f}")),
                    should_stop=lambda: self.stop_flag,
                )
                self.events.put(("done", acc))
            except Exception as exc:  # 데이터 부족 등은 화면에 알려줌
                self.events.put(("error", str(exc)))

        threading.Thread(target=run, daemon=True).start()

    def request_stop(self):
        self.stop_flag = True
        self.stop_btn.configure(state="disabled")

    def poll_events(self):
        try:
            while True:
                ev = self.events.get_nowait()
                if ev[0] == "log":
                    self.log(ev[1])
                elif ev[0] == "progress":
                    self.progress["value"] = ev[1]
                    self.train_status.set(ev[2])
                else:
                    self.training = False
                    self.train_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                    if ev[0] == "done":
                        self.train_status.set(f"완료! 검증 정확도 {ev[1]:.3f}")
                        self.load_model()
                        self.infer_on.set(True)
                    else:
                        self.train_status.set("실패")
                        self.log(f"훈련 실패: {ev[1]}")
                        messagebox.showwarning("훈련 실패", ev[1])
        except queue.Empty:
            pass
        self.root.after(100, self.poll_events)

    def load_model(self, quiet=False):
        if not MODEL_PATH.exists():
            if not quiet:
                self.log("모델 파일이 없습니다.")
            return
        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        model = GestureMLP(len(ckpt["labels"]))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        self.model, self.model_labels = model, ckpt["labels"]
        self.model_status.set("모델 라벨: " + ", ".join(self.model_labels))
        self.log(f"모델 불러옴: {self.model_labels}")

    def predict(self, feats):
        with torch.no_grad():
            probs = torch.softmax(self.model(torch.from_numpy(feats)[None]), dim=1)[0]
        score, idx = probs.max(0)
        score = score.item()
        name = self.model_labels[idx] if score >= self.min_score.get() else "?"
        return name, score

    # ------------------------------------------------------------------ 카메라 루프
    def update_frame(self):
        ok, frame = self.cap.read() if self.cap.isOpened() else (False, None)
        if not ok:
            self.video.configure(text="웹캠 영상을 받을 수 없습니다.")
            self.root.after(500, self.update_frame)
            return
        frame = cv2.flip(frame, 1)  # 거울 모드 (수집·인식 모두 동일해야 함)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.perf_counter() - self.start) * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        texts = []  # (위치, 글자, 색) — PIL로 한글까지 그림
        preds = []
        hands = result.hand_landmarks
        for i, landmarks in enumerate(hands):
            hand = result.handedness[i][0].category_name
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            draw_hand(frame, pts, (0, 0, 255) if self.recording else (0, 255, 0))
            feats = landmarks_to_features(landmarks, hand)

            if self.recording and i == 0 and len(hands) == 1:
                self.save_sample(feats)
            if self.infer_on.get() and self.model is not None:
                name, score = self.predict(feats)
                x = min(p[0] for p in pts)
                y = max(min(p[1] for p in pts) - 34, 0)
                texts.append(((x, y), f"{hand} {name} {score:.2f}", (255, 255, 0)))
                preds.append(f"{'오른손' if hand == 'Right' else '왼손'}: {name}")
        self.pred_text.set("\n".join(preds) if preds else "-")

        now = time.perf_counter()
        fps = 1.0 / max(now - self.prev_time, 1e-6)
        self.prev_time = now
        texts.append(((10, 8), f"FPS {fps:.0f}  손 {len(hands)}개", (0, 255, 255)))
        if self.recording:
            label = self.current_label()
            texts.append(((10, 40), f"● REC  {label}  {self.counts.get(label, 0)}/{self.target_count()}", (255, 60, 60)))
            if len(hands) > 1:
                texts.append(((10, 72), "손을 하나만 보여주세요", (255, 165, 0)))
            elif not hands:
                texts.append(((10, 72), "손이 보이지 않습니다", (255, 165, 0)))

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        for pos, text, color in texts:
            draw.text(pos, text, font=self.font, fill=color, stroke_width=2, stroke_fill=(0, 0, 0))
        photo = ImageTk.PhotoImage(img)
        self.video.configure(image=photo, text="")
        self.video.image = photo  # 참조를 유지하지 않으면 이미지가 사라짐

        self.root.after(1, self.update_frame)

    def save_sample(self, feats):
        label = self.current_label()
        self.frame_idx += 1
        if self.frame_idx % SAVE_EVERY:
            return
        self.pending.append((label, feats))
        self.counts[label] += 1
        self.tree.set(label, "count", self.counts[label])
        if self.counts[label] >= self.target_count():
            self.stop_recording()
            self.log(f"'{label}' 목표 개수 도달")
        elif len(self.pending) >= FLUSH_EVERY:
            self.flush_samples()

    def on_close(self):
        self.stop_recording()
        self.stop_flag = True
        self.cap.release()
        self.landmarker.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    GestureStudio(root)
    root.mainloop()


if __name__ == "__main__":
    main()
