"""Build the Rock-Paper-Scissors dataset from raw stills + videos.

Input (filenames start with the gesture: Paper*, Rock*, Scissor*; typos like "Papaer2" are fine):
  source/images/Paper1.jpeg ... Rock1.jpeg ... Scissor1.jpeg
  source/videos/Paper.mp4, Paper2.mp4 ... Rock.mp4 ... Scissor.mp4 ...

Output (ImageFolder layout, already split):
  final_dataset/{train,val,test}/{rock,paper,scissors}/*.jpg
  final_dataset/manifest.csv, final_dataset/preview_<split>.jpg

Leakage control: the unit of splitting is the SOURCE FILE (one video or one still).
All frames of a video go to exactly one split, so near-identical neighbouring frames
never appear in both train and test. Stills are split the same way, separately.
No augmentation is baked in: every saved image is a real frame (augmentation happens
on the fly in train.py, on the train split only).

Example:
  python build_dataset.py --images source/images --videos source/videos --out final_dataset \
      --targets rock=700,paper=850,scissors=550
"""
import argparse, csv, os, random, re
from collections import defaultdict

import cv2
import numpy as np

CLASSES = ["rock", "paper", "scissors"]
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
VID_EXT = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


def class_of(filename):
    """Gesture from the first letters of the filename (robust to typos like 'Papaer')."""
    n = filename.lower()
    if n.startswith("r"): return "rock"
    if n.startswith("p"): return "paper"
    if n.startswith("s"): return "scissors"
    return None


def square_resize(img, size, frac=1.0):
    """Centre-crop to a square (no distortion), then resize."""
    h, w = img.shape[:2]
    s = int(min(h, w) * frac)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return cv2.resize(img[y0:y0 + s, x0:x0 + s], (size, size), interpolation=cv2.INTER_AREA)


def sharpness(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def split_groups(groups, rng, ratios=(0.70, 0.15, 0.15)):
    """Assign whole groups to train/val/test; at least 1 group each in val/test when possible."""
    g = sorted(groups)
    rng.shuffle(g)
    n = len(g)
    if n == 0:
        return [], [], []
    if n == 1:
        return g, [], []
    if n == 2:
        return g[:1], g[1:], []
    nv = max(1, round(ratios[1] * n))
    nt = max(1, round(ratios[2] * n))
    return g[:n - nv - nt], g[n - nv - nt:n - nt], g[n - nt:]


def read_video_frames(path, step, size, frac, min_sharp):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  ! cannot open {path}")
        return []
    out, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            img = square_resize(frame, size, frac)
            if sharpness(img) >= min_sharp:
                out.append((i, img))
        i += 1
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="source/images")
    ap.add_argument("--videos", default="source/videos")
    ap.add_argument("--out", default="final_dataset")
    ap.add_argument("--targets", default="rock=700,paper=850,scissors=550",
                    help="total images per class (deliberately unequal => class imbalance)")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--frame_step", type=int, default=5, help="keep every Nth video frame")
    ap.add_argument("--crop_frac", type=float, default=1.0, help="fraction of the short side kept in the centre crop")
    ap.add_argument("--min_sharp", type=float, default=30.0, help="drop blurrier frames (Laplacian variance)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--people_map", default=None,
                    help="optional CSV 'source_file,person' so one person never spans splits")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    targets = {k: int(v) for k, v in (kv.split("=") for kv in a.targets.split(","))}
    people = {}
    if a.people_map:
        with open(a.people_map) as f:
            for r in csv.reader(f):
                if len(r) >= 2 and r[0] != "source_file":
                    people[os.path.splitext(r[0])[0].lower()] = r[1]

    # ---- 1. discover sources, grouped per class ----
    sources = {c: {"image": [], "video": []} for c in CLASSES}
    for folder, kind, exts in [(a.images, "image", IMG_EXT), (a.videos, "video", VID_EXT)]:
        if not os.path.isdir(folder):
            print(f"! folder not found: {folder}")
            continue
        for fn in sorted(os.listdir(folder)):
            if not fn.lower().endswith(exts):
                continue
            c = class_of(fn)
            if c is None:
                print(f"! skipping {fn} (cannot infer class)")
                continue
            sources[c][kind].append(os.path.join(folder, fn))
    for c in CLASSES:
        print(f"{c}: {len(sources[c]['image'])} stills, {len(sources[c]['video'])} videos")

    def group_key(path):
        stem = os.path.splitext(os.path.basename(path))[0].lower()
        return people.get(stem, stem)

    # ---- 2. split source groups -> train/val/test (per class, stills and videos separately) ----
    assign = {c: {"train": [], "val": [], "test": []} for c in CLASSES}
    for c in CLASSES:
        for kind in ("image", "video"):
            by_group = defaultdict(list)
            for p in sources[c][kind]:
                by_group[group_key(p)].append(p)
            tr, va, te = split_groups(by_group.keys(), rng)
            for name, gs in zip(("train", "val", "test"), (tr, va, te)):
                for g in gs:
                    assign[c][name] += [(kind, p) for p in by_group[g]]
    # a person present in several classes must stay in ONE split -> warn if people_map used inconsistently
    if people:
        where = defaultdict(set)
        for c in CLASSES:
            for sp in assign[c]:
                for _, p in assign[c][sp]:
                    where[group_key(p)].add(sp)
        bad = [k for k, v in where.items() if len(v) > 1]
        if bad:
            print("! WARNING: these people appear in more than one split:", bad,
                  "\n  (classes were split independently; make the same person land in one split manually)")

    # ---- 3. build images per split with per-class quotas ----
    ratio = {"train": 0.70, "val": 0.15, "test": 0.15}
    manifest, counts = [], defaultdict(int)
    for c in CLASSES:
        for sp in ("train", "val", "test"):
            quota = round(targets[c] * ratio[sp])
            cand = []  # (source_name, frame_idx, img)
            for kind, p in assign[c][sp]:
                stem = ("img_" if kind == "image" else "vid_") + os.path.splitext(os.path.basename(p))[0]
                if kind == "image":
                    img = cv2.imread(p)
                    if img is None:
                        print(f"  ! unreadable {p}"); continue
                    cand.append((stem, 0, square_resize(img, a.size, a.crop_frac), True))
                else:
                    for i, img in read_video_frames(p, a.frame_step, a.size, a.crop_frac, a.min_sharp):
                        cand.append((stem, i, img, False))
            stills = [x for x in cand if x[3]]
            frames = [x for x in cand if not x[3]]
            # keep every real still; fill the rest with evenly-spread video frames across the videos
            room = max(quota - len(stills), 0)
            if len(frames) > room:
                by_src = defaultdict(list)
                for x in frames:
                    by_src[x[0]].append(x)
                per = max(room // max(len(by_src), 1), 1)
                picked = []
                for src, lst in by_src.items():
                    k = min(per, len(lst))
                    idx = np.linspace(0, len(lst) - 1, k).round().astype(int)
                    picked += [lst[i] for i in sorted(set(idx))]
                rest = [x for x in frames if x not in picked]
                rng.shuffle(rest)
                picked += rest[:max(room - len(picked), 0)]
                frames = picked[:room]
            chosen = stills + frames
            if len(chosen) < quota:
                print(f"! {sp}/{c}: only {len(chosen)} real images available (quota {quota}) - "
                      f"lower --frame_step or record more video")
            d = os.path.join(a.out, sp, c)
            os.makedirs(d, exist_ok=True)
            for stem, i, img, _ in chosen:
                fn = f"{stem}_f{i:05d}.jpg"
                cv2.imwrite(os.path.join(d, fn), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                manifest.append((sp, c, stem, fn))
                counts[(sp, c)] += 1

    with open(os.path.join(a.out, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["split", "class", "source", "file"]); w.writerows(manifest)

    # ---- 4. report + leakage check + preview sheets ----
    print("\nSplit sizes (rows=split, cols=class):")
    for sp in ("train", "val", "test"):
        print(f"  {sp:5s}", {c: counts[(sp, c)] for c in CLASSES}, "total", sum(counts[(sp, c)] for c in CLASSES))
    tot = {c: sum(counts[(sp, c)] for sp in ratio) for c in CLASSES}
    print("Per class:", tot, "| grand total:", sum(tot.values()))
    owner = {}
    for sp, c, src, fn in manifest:
        if owner.setdefault((c, src), sp) != sp:
            raise SystemExit(f"LEAK: source {src} appears in two splits")
    print("Leakage check passed: every source file lives in exactly one split.")
    if min(tot.values()) < 500 or sum(tot.values()) < 2000:
        print("! below the assignment minimum (>=500/class, >=2000 total)")
    if len(set(tot.values())) == 1:
        print("! classes are perfectly balanced - the assignment requires imbalance")

    for sp in ("train", "val", "test"):
        rows = []
        for c in CLASSES:
            files = sorted(os.listdir(os.path.join(a.out, sp, c)))
            sel = [files[i] for i in np.linspace(0, len(files) - 1, min(8, len(files))).astype(int)] if files else []
            ims = [cv2.resize(cv2.imread(os.path.join(a.out, sp, c, f)), (96, 96)) for f in sel]
            ims += [np.zeros((96, 96, 3), np.uint8)] * (8 - len(ims))
            rows.append(np.hstack(ims))
        cv2.imwrite(os.path.join(a.out, f"preview_{sp}.jpg"), np.vstack(rows))
    print("Open preview_train/val/test.jpg and check hands are visible and correctly labelled.")


if __name__ == "__main__":
    main()