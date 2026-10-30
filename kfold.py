"""Grouped K-fold cross-validation for Rock-Paper-Scissors (no leakage between clips).

Every video / still is one GROUP. All frames of a group are always in the same side of a split,
so each fold's test set contains only clips the model has never seen. Across the K folds every
clip is held out exactly once, so the result is a mean +/- std over many unseen clips instead
of the verdict of 3-4 clips.

No model selection is done on the held-out fold: every fold trains for a fixed number of epochs
and is evaluated with the FINAL-epoch weights. (Held-out accuracy per epoch is logged to W&B for
monitoring only.)

  python kfold.py --data final_dataset --k 5 --epochs 30            # cross-validation
  python kfold.py --data final_dataset --skip_cv --final --epochs 30 # model for the GUI (all data)
"""
import argparse, glob, json, os, random, time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

from model import RPSNet, train_transform, eval_transform

CLASSES = ["paper", "rock", "scissors"]   # same alphabetical order as ImageFolder


def collect_items(root):
    """All images from train/val/test (or a flat folder) -> [(path, class_idx, source_group)]."""
    items = []
    for ci, c in enumerate(CLASSES):
        for p in glob.glob(os.path.join(root, "*", c, "*.jpg")) + glob.glob(os.path.join(root, c, "*.jpg")):
            src = os.path.basename(p).rsplit("_f", 1)[0]     # e.g. vid_Scissor4 / img_Paper12
            items.append((p, ci, src))
    return sorted(items)


def make_folds(items, k, seed):
    y = np.array([t[1] for t in items])
    g = np.array([t[2] for t in items])
    skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    for tr, te in skf.split(np.zeros(len(y)), y, g):
        assert not (set(g[tr]) & set(g[te])), "leak: a clip is in both train and test"
        folds.append((tr.tolist(), te.tolist()))
    return folds


class ListDS(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y, _ = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict(model, items, args, device):
    dl = DataLoader(ListDS(items, eval_transform(args.img)), args.bs, num_workers=args.workers)
    model.eval()
    preds, loss_sum = [], 0.0
    ce = nn.CrossEntropyLoss(reduction="sum")
    for x, y in dl:
        out = model(x.to(device))
        loss_sum += ce(out, y.to(device)).item()
        preds += out.argmax(1).cpu().tolist()
    return np.array(preds), loss_sum / len(items)


def train_model(train_items, test_items, args, device, log_prefix=""):
    seed_all(args.seed)
    y = np.array([t[1] for t in train_items])
    counts = np.bincount(y, minlength=len(CLASSES))
    sampler = WeightedRandomSampler(torch.tensor((1.0 / counts)[y], dtype=torch.double),
                                    num_samples=len(y), replacement=True)   # balanced batches
    dl = DataLoader(ListDS(train_items, train_transform(args.img)), args.bs, sampler=sampler,
                    drop_last=True, num_workers=args.workers, pin_memory=device == "cuda")
    model = RPSNet(len(CLASSES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(dl))
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    y_test = np.array([t[1] for t in test_items]) if test_items else None

    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        for x, yb in dl:
            x, yb = x.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                out = model(x)
                loss = crit(out, yb)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == yb).sum().item(); seen += x.size(0)
        log = {f"{log_prefix}epoch": ep, f"{log_prefix}train/loss": loss_sum / seen, f"{log_prefix}train/acc": correct / seen}
        msg = f"  ep {ep:02d}/{args.epochs} train loss {loss_sum / seen:.4f} acc {correct / seen:.4f}"
        if test_items:   # monitoring only - never used to pick epochs or weights
            p, tl = predict(model, test_items, args, device)
            log.update({f"{log_prefix}heldout/loss": tl, f"{log_prefix}heldout/acc": float((p == y_test).mean())})
            msg += f" | held-out loss {tl:.4f} acc {(p == y_test).mean():.4f}"
        wandb.log(log)
        print(msg)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="final_dataset")
    ap.add_argument("--out", default="outputs_kfold")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_cv", action="store_true")
    ap.add_argument("--final", action="store_true", help="also train the deployment model on ALL clips")
    ap.add_argument("--wandb_project", default="rps-applied-cv")
    ap.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--run_name", default="kfold")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    items = collect_items(args.data)
    assert items, f"no images found under {args.data}"
    groups = sorted({t[2] for t in items})
    print(f"{len(items)} images from {len(groups)} source clips | per class:",
          {c: sum(1 for t in items if t[1] == i) for i, c in enumerate(CLASSES)})

    results = {}
    if not args.skip_cv:
        folds = make_folds(items, args.k, args.seed)
        pooled_true, pooled_pred, per_source = [], [], defaultdict(list)
        fold_acc, fold_f1 = [], []
        for f, (tr, te) in enumerate(folds, 1):
            tr_items, te_items = [items[i] for i in tr], [items[i] for i in te]
            te_src = sorted({t[2] for t in te_items})
            print(f"\n=== fold {f}/{args.k}: train {len(tr_items)} imgs | test {len(te_items)} imgs from clips {te_src}")
            run = wandb.init(project=args.wandb_project, mode=args.wandb_mode, group=args.run_name,
                             name=f"{args.run_name}_fold{f}", config={**vars(args), "fold": f, "test_clips": te_src})
            model = train_model(tr_items, te_items, args, device)
            pred, tl = predict(model, te_items, args, device)
            yt = np.array([t[1] for t in te_items])
            acc, f1 = float((pred == yt).mean()), float(f1_score(yt, pred, average="macro"))
            fold_acc.append(acc); fold_f1.append(f1)
            pooled_true += yt.tolist(); pooled_pred += pred.tolist()
            for t, ok in zip(te_items, pred == yt):
                per_source[t[2]].append(bool(ok))
            wandb.log({"fold/test_acc": acc, "fold/test_macro_f1": f1, "fold/test_loss": tl})
            print(f"  FOLD {f} test acc {acc:.4f} macro-F1 {f1:.4f}")
            run.finish()

        cm = confusion_matrix(pooled_true, pooled_pred)
        src_acc = {s: float(np.mean(v)) for s, v in sorted(per_source.items())}
        results = {"k": args.k, "fold_acc": fold_acc, "fold_macro_f1": fold_f1,
                   "mean_acc": float(np.mean(fold_acc)), "std_acc": float(np.std(fold_acc)),
                   "mean_macro_f1": float(np.mean(fold_f1)), "std_macro_f1": float(np.std(fold_f1)),
                   "pooled_acc": float(np.mean(np.array(pooled_true) == np.array(pooled_pred))),
                   "confusion_matrix": cm.tolist(), "per_clip_acc": src_acc, "epochs": args.epochs}
        print("\n===== grouped %d-fold CV =====" % args.k)
        print("fold accuracies:", [round(a, 3) for a in fold_acc])
        print("accuracy %.4f +/- %.4f | macro-F1 %.4f +/- %.4f" % (results["mean_acc"], results["std_acc"],
                                                                   results["mean_macro_f1"], results["std_macro_f1"]))
        print(CLASSES); print(cm)
        print("per-clip accuracy (worst first):")
        for s, a in sorted(src_acc.items(), key=lambda kv: kv[1])[:12]:
            print(f"  {s:25s} {a:.2f}")
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].bar(range(1, len(fold_acc) + 1), fold_acc); ax[0].axhline(0.85, color="r", ls="--", label="85% target")
        ax[0].set_ylim(0, 1); ax[0].set_title("Held-out accuracy per fold"); ax[0].set_xlabel("fold"); ax[0].legend()
        ax[1].imshow(cm, cmap="Blues"); ax[1].set_title("Pooled confusion matrix (all clips held out once)")
        ax[1].set_xticks(range(3), CLASSES); ax[1].set_yticks(range(3), CLASSES)
        for i in range(3):
            for j in range(3):
                ax[1].text(j, i, cm[i, j], ha="center", va="center", color="red" if cm[i, j] > cm.max() / 2 else "black")
        plt.tight_layout(); fig_path = os.path.join(args.out, "kfold_summary.png"); plt.savefig(fig_path, dpi=150); plt.close()
        run = wandb.init(project=args.wandb_project, mode=args.wandb_mode, group=args.run_name, name=f"{args.run_name}_summary",
                         config=vars(args))
        wandb.log({"cv/mean_acc": results["mean_acc"], "cv/std_acc": results["std_acc"],
                   "cv/mean_macro_f1": results["mean_macro_f1"], "cv/pooled_acc": results["pooled_acc"],
                   "cv/summary_figure": wandb.Image(fig_path),
                   "cv/per_clip_acc": wandb.Table(columns=["clip", "acc"], data=[[k, v] for k, v in src_acc.items()])})
        run.finish()
        with open(os.path.join(args.out, "kfold_results.json"), "w") as fh:
            json.dump(results, fh, indent=2)

    if args.final:
        print("\n=== final model on ALL clips (same recipe, final-epoch weights) ===")
        run = wandb.init(project=args.wandb_project, mode=args.wandb_mode, group=args.run_name,
                         name=f"{args.run_name}_final", config=vars(args))
        model = train_model(items, None, args, device)
        path = os.path.join(args.out, "rps_final.pt")
        torch.save({"model": model.state_dict(), "classes": CLASSES, "img_size": args.img}, path)
        art = wandb.Artifact("rps-final", type="model"); art.add_file(path); run.log_artifact(art)
        run.finish()
        print("saved", path)


if __name__ == "__main__":
    main()