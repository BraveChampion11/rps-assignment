"""Real-time Rock-Paper-Scissors GUI (Tkinter + OpenCV + PyTorch).

  python gui_app.py --ckpt models/rps_final.pt
Show your hand inside the green square; the prediction appears live.
"""
import argparse, time
from collections import deque

import cv2
import numpy as np
import torch
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

from model import load_checkpoint, eval_transform

EMOJI = {"rock": "✊", "paper": "✋", "scissors": "✌"}
COLORS = {"rock": "#d9534f", "paper": "#337ab7", "scissors": "#5cb85c"}


class App:
    def __init__(self, root, ckpt, cam, roi=0.8):
        self.root = root
        root.title("Rock-Paper-Scissors Classifier")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cam_index = cam
        self.cap = None
        self.model = None
        self.history = deque(maxlen=5)
        self.last_t = time.time()
        self.fps = 0.0
        self.frame = None

        # ---- layout ----
        left = ttk.Frame(root); left.grid(row=0, column=0, padx=8, pady=8)
        right = ttk.Frame(root); right.grid(row=0, column=1, padx=8, pady=8, sticky="n")
        self.video = tk.Label(left); self.video.pack()

        self.big = tk.Label(right, text="—", font=("Segoe UI", 42, "bold"), width=12)
        self.big.pack(pady=(0, 4))
        self.conf_lbl = ttk.Label(right, text="Confidence: –", font=("Segoe UI", 12)); self.conf_lbl.pack()
        self.bars = {}
        self.bar_frame = ttk.LabelFrame(right, text="Class probabilities"); self.bar_frame.pack(fill="x", pady=10)

        ctl = ttk.LabelFrame(right, text="Controls"); ctl.pack(fill="x", pady=6)
        self.thresh = tk.DoubleVar(value=0.60)
        ttk.Label(ctl, text="Min confidence").grid(row=0, column=0, sticky="w")
        ttk.Scale(ctl, from_=0.3, to=0.99, variable=self.thresh, length=160).grid(row=0, column=1)
        self.smooth = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctl, text="Smooth (5-frame vote)", variable=self.smooth).grid(row=1, column=0, columnspan=2, sticky="w")
        self.mirror = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctl, text="Mirror camera", variable=self.mirror).grid(row=2, column=0, columnspan=2, sticky="w")
        self.roi = tk.DoubleVar(value=roi)
        ttk.Label(ctl, text="Box size").grid(row=3, column=0, sticky="w")
        ttk.Scale(ctl, from_=0.4, to=1.0, variable=self.roi, length=160).grid(row=3, column=1)
        ttk.Label(ctl, text="Camera index").grid(row=4, column=0, sticky="w")
        self.cam_var = tk.IntVar(value=cam)
        ttk.Spinbox(ctl, from_=0, to=5, width=4, textvariable=self.cam_var, command=self.open_camera).grid(row=4, column=1, sticky="w")
        ttk.Button(ctl, text="Load model…", command=self.pick_model).grid(row=5, column=0, pady=4)
        ttk.Button(ctl, text="Save snapshot", command=self.snapshot).grid(row=5, column=1, pady=4)
        ttk.Label(right, text="Hold ONE sign inside the box.  Esc = quit", foreground="gray").pack()
        self.info = ttk.Label(right, text="", foreground="gray"); self.info.pack(pady=6)

        self.load_model(ckpt)
        self.open_camera()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Escape>", lambda e: self.close())
        self.loop()

    def load_model(self, path):
        self.model, self.classes, img = load_checkpoint(path, self.device)
        self.tf = eval_transform(img)
        for w in self.bar_frame.winfo_children():
            w.destroy()
        self.bars = {}
        for c in self.classes:
            row = ttk.Frame(self.bar_frame); row.pack(fill="x", padx=4, pady=2)
            ttk.Label(row, text=f"{EMOJI.get(c, '')} {c}", width=11).pack(side="left")
            pb = ttk.Progressbar(row, length=170, maximum=100); pb.pack(side="left")
            lbl = ttk.Label(row, text="0%", width=5); lbl.pack(side="left")
            self.bars[c] = (pb, lbl)
        self.info.config(text=f"Model: {path.split('/')[-1].split(chr(92))[-1]} | device: {self.device}")

    def pick_model(self):
        p = filedialog.askopenfilename(filetypes=[("PyTorch checkpoint", "*.pt *.pth")])
        if p:
            try:
                self.load_model(p)
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def open_camera(self):
        if self.cap is not None:
            self.cap.release()
        idx = self.cam_var.get() if hasattr(self, "cam_var") else self.cam_index
        import os
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if os.name == "nt" else 0)
        if not self.cap.isOpened():
            messagebox.showwarning("Camera", f"Cannot open camera {idx}")

    @torch.no_grad()
    def predict(self, roi_bgr):
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        x = self.tf(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        return torch.softmax(self.model(x), 1)[0].cpu().numpy()

    def snapshot(self):
        if self.frame is not None:
            name = f"snapshot_{int(time.time())}.png"
            cv2.imwrite(name, self.frame)
            self.info.config(text=f"Saved {name}")

    def loop(self):
        if self.cap is not None and self.cap.isOpened():
            ok, frame = self.cap.read()
            if ok:
                if self.mirror.get():
                    frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                s = int(min(h, w) * self.roi.get())
                x0, y0 = (w - s) // 2, (h - s) // 2
                probs = self.predict(frame[y0:y0 + s, x0:x0 + s])
                self.history.append(probs)
                p = np.mean(self.history, 0) if self.smooth.get() else probs
                k = int(p.argmax()); conf = float(p[k]); cls = self.classes[k]
                confident = conf >= self.thresh.get()

                label = cls if confident else "no sign"
                col = COLORS.get(cls, "#444") if confident else "#888888"
                self.big.config(text=f"{EMOJI.get(cls, '') if confident else ''} {label.upper()}", fg=col)
                self.conf_lbl.config(text=f"Confidence: {conf * 100:.1f}%")
                for c, v in zip(self.classes, p):
                    self.bars[c][0]["value"] = v * 100
                    self.bars[c][1].config(text=f"{v * 100:.0f}%")

                now = time.time()
                self.fps = 0.9 * self.fps + 0.1 / max(now - self.last_t, 1e-6); self.last_t = now
                disp = frame.copy()
                bgr = tuple(int(col[i:i + 2], 16) for i in (5, 3, 1))
                cv2.rectangle(disp, (x0, y0), (x0 + s, y0 + s), bgr, 4)
                cv2.putText(disp, f"{label.upper()} {conf * 100:.0f}%", (x0, max(y0 - 12, 28)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, bgr, 3)
                cv2.putText(disp, f"{self.fps:.0f} FPS", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                self.frame = disp
                img = Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)).resize((640, int(640 * h / w)))
                self.imgtk = ImageTk.PhotoImage(img)
                self.video.config(image=self.imgtk)
        self.root.after(15, self.loop)

    def close(self):
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/rps_final.pt")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--roi", type=float, default=0.8, help="box size as a fraction of the short side of the frame")
    a = ap.parse_args()
    root = tk.Tk()
    App(root, a.ckpt, a.cam, a.roi)
    root.mainloop()