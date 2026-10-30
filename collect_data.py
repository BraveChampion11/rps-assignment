"""Webcam dataset collector for Rock-Paper-Scissors (ImageFolder layout).

Place your hand inside the green square. Keys:
  r / p / s : start/stop auto-recording that class (saves ~6 imgs/sec)
  space     : stop recording      q : quit
Run it several times with different --subject (you, friends, family), lighting,
backgrounds, distances and hands. Frames are mirrored, same as the GUI.

  python collect_data.py --out rps --subject me
"""
import argparse, os, time
import cv2

KEYS = {ord("r"): "rock", ord("p"): "paper", ord("s"): "scissors"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rps")
    ap.add_argument("--subject", default="me")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--interval", type=float, default=0.16, help="seconds between saved frames")
    args = ap.parse_args()

    for c in KEYS.values():
        os.makedirs(os.path.join(args.out, c), exist_ok=True)
    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW if os.name == "nt" else 0)
    if not cap.isOpened():
        raise SystemExit("Could not open camera")

    recording, last = None, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        s = int(min(h, w) * 0.7)
        x0, y0 = (w - s) // 2, (h - s) // 2
        roi = frame[y0:y0 + s, x0:x0 + s]

        if recording and time.time() - last >= args.interval:
            name = f"{args.subject}_{int(time.time() * 1000)}.jpg"
            cv2.imwrite(os.path.join(args.out, recording, name), cv2.resize(roi, (224, 224)))
            last = time.time()

        disp = frame.copy()
        cv2.rectangle(disp, (x0, y0), (x0 + s, y0 + s), (0, 255, 0) if not recording else (0, 0, 255), 3)
        counts = {c: len(os.listdir(os.path.join(args.out, c))) for c in KEYS.values()}
        txt = "  ".join(f"{c}:{n}" for c, n in counts.items()) + f"  total:{sum(counts.values())}"
        cv2.putText(disp, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(disp, f"REC {recording}" if recording else "idle (r/p/s to record, q quit)",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if recording else (200, 200, 200), 2)
        cv2.imshow("RPS collector", disp)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == 32:
            recording = None
        elif k in KEYS:
            recording = None if recording == KEYS[k] else KEYS[k]

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()