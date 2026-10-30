"""Train RPSNet from scratch on an ImageFolder dataset with W&B logging.

Usage:
  python train.py --data rps --epochs 30 --wandb_project rps-cv
  python train.py --data rps --wandb_mode offline      # no login needed
"""
import argparse, json, os, random, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

from model import RPSNet, train_transform, eval_transform, LightPerturb


class FocalLoss(nn.Module):
    """FL = -alpha_t * (1 - p_t)^gamma * log(p_t). gamma=0 and alpha=None reduces to plain cross-entropy."""

    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits, target):
        logp_t = F.log_softmax(logits.float(), dim=1).gather(1, target[:, None]).squeeze(1)
        loss = -((1.0 - logp_t.exp()) ** self.gamma) * logp_t
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean()


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, preds, labels = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += criterion(out, y).item() * x.size(0)
        preds += out.argmax(1).cpu().tolist()
        labels += y.cpu().tolist()
    n = len(labels)
    acc = float(np.mean(np.array(preds) == np.array(labels)))
    f1 = f1_score(labels, preds, average="macro")
    return loss_sum / n, acc, f1, labels, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="final_dataset")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--loss", default="ce", choices=["ce", "focal"])
    ap.add_argument("--gamma", type=float, default=2.0, help="focal-loss focusing parameter")
    ap.add_argument("--imbalance", default="sampler", choices=["sampler", "weights", "none"],
                    help="sampler: balanced batches via oversampling + on-the-fly augmentation; "
                         "weights: inverse-frequency loss weights; none")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--wandb_project", default="rps-applied-cv")
    ap.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    args = ap.parse_args()

    seed_all(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- dataset: pre-split folders (data/train, data/val, data/test) from build_dataset.py,
    #      otherwise fall back to a stratified random 70/15/15 split of a flat ImageFolder ----
    presplit = all(os.path.isdir(os.path.join(args.data, sp)) for sp in ("train", "val", "test"))
    if presplit:
        train_ds = datasets.ImageFolder(os.path.join(args.data, "train"), transform=train_transform(args.img))
        val_ds = datasets.ImageFolder(os.path.join(args.data, "val"), transform=eval_transform(args.img))
        test_ds = datasets.ImageFolder(os.path.join(args.data, "test"), transform=eval_transform(args.img))
        assert train_ds.classes == val_ds.classes == test_ds.classes, "class folders differ across splits"
        classes = train_ds.classes
        tr_targets = np.array(train_ds.targets)
        all_targets = np.concatenate([tr_targets, val_ds.targets, test_ds.targets])
        print("Using pre-split dataset (source-level split, no leakage).")
    else:
        print("WARNING: flat folder -> random image-level split; near-duplicate frames may leak across splits.")
        base = datasets.ImageFolder(args.data)
        classes = base.classes
        all_targets = np.array(base.targets)
        idx = np.arange(len(all_targets))
        tr_idx, tmp_idx = train_test_split(idx, test_size=0.30, stratify=all_targets, random_state=args.seed)
        va_idx, te_idx = train_test_split(tmp_idx, test_size=0.50, stratify=all_targets[tmp_idx], random_state=args.seed)
        train_ds = Subset(datasets.ImageFolder(args.data, transform=train_transform(args.img)), tr_idx)
        val_ds = Subset(datasets.ImageFolder(args.data, transform=eval_transform(args.img)), va_idx)
        test_ds = Subset(datasets.ImageFolder(args.data, transform=eval_transform(args.img)), te_idx)
        tr_targets = all_targets[tr_idx]
    counts = Counter(all_targets.tolist())
    n_tr, n_va, n_te = len(train_ds), len(val_ds), len(test_ds)
    print("Classes:", classes, "| per-class:", {classes[k]: v for k, v in counts.items()},
          "| total:", len(all_targets), "| train/val/test:", n_tr, n_va, n_te)

    kw = dict(num_workers=args.workers, pin_memory=device == "cuda")
    cls_counts = np.bincount(tr_targets, minlength=len(classes))
    if args.imbalance == "sampler":
        # minority classes are drawn more often; every draw gets fresh random augmentation,
        # so this is augmentation-based oversampling (no duplicate files on disk)
        sample_w = (1.0 / cls_counts)[tr_targets]
        sampler = WeightedRandomSampler(torch.tensor(sample_w, dtype=torch.double), num_samples=len(tr_targets), replacement=True)
        train_dl = DataLoader(train_ds, args.bs, sampler=sampler, drop_last=True, **kw)
    else:
        train_dl = DataLoader(train_ds, args.bs, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, args.bs, **kw)
    test_dl = DataLoader(test_ds, args.bs, **kw)

    # ---- class imbalance: inverse-frequency weights from TRAIN split only ----
    tr_counts = np.bincount(tr_targets, minlength=len(classes))
    weights = torch.tensor(tr_counts.sum() / (len(classes) * tr_counts), dtype=torch.float32, device=device)

    model = RPSNet(len(classes)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    # never combine sampler + weights: batches are already balanced when --imbalance sampler
    alpha = weights if args.imbalance == "weights" else None
    if args.loss == "focal":
        criterion = FocalLoss(args.gamma, alpha)
    else:
        criterion = nn.CrossEntropyLoss(weight=alpha, label_smoothing=0.05)
    eval_crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_dl))
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

    run = wandb.init(project=args.wandb_project, mode=args.wandb_mode, name=args.run_name, config={
        **vars(args), "classes": classes, "n_params": n_params, "device": device,
        "split": {"train": n_tr, "val": n_va, "test": n_te},
        "class_counts": {classes[k]: int(v) for k, v in counts.items()},
        "class_weights": weights.cpu().tolist(), "architecture": "RPSNet (custom, from scratch)"})
    wandb.watch(model, log="gradients", log_freq=100)
    wandb.log({"dataset/class_counts": wandb.plot.bar(
        wandb.Table(data=[[classes[k], int(v)] for k, v in counts.items()], columns=["class", "images"]),
        "class", "images", title="Class distribution")})

    hist = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc", "val_f1"]}
    best_f1, best_ep, bad = -1, 0, 0
    ckpt_path = os.path.join(args.out, "rps_best.pt")
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item(); seen += x.size(0)
        tr_loss, tr_acc = loss_sum / seen, correct / seen
        va_loss, va_acc, va_f1, _, _ = evaluate(model, val_dl, eval_crit, device)

        for k, v in zip(hist, [tr_loss, va_loss, tr_acc, va_acc, va_f1]):
            hist[k].append(v)
        wandb.log({"epoch": ep, "train/loss": tr_loss, "train/acc": tr_acc, "val/loss": va_loss,
                   "val/acc": va_acc, "val/macro_f1": va_f1, "lr": opt.param_groups[0]["lr"]})
        print(f"ep {ep:02d}/{args.epochs} | train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f} f1 {va_f1:.4f}")

        if va_f1 > best_f1:
            best_f1, best_ep, bad = va_f1, ep, 0
            torch.save({"model": model.state_dict(), "classes": classes, "img_size": args.img,
                        "epoch": ep, "val_f1": va_f1}, ckpt_path)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping at epoch {ep} (best epoch {best_ep})"); break

    # ---- final test evaluation with best checkpoint ----
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model"])
    te_loss, te_acc, te_f1, y_true, y_pred = evaluate(model, test_dl, eval_crit, device)
    report = classification_report(y_true, y_pred, target_names=classes, digits=4, output_dict=True)
    print("\nTEST  loss %.4f  acc %.4f  macro-F1 %.4f" % (te_loss, te_acc, te_f1))
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

    # ---- lighting-robustness test: same test images under controlled lighting changes ----
    conditions = {"normal": (1, 1, 1), "dark x0.3": (0.3, 1, 1), "dark x0.5": (0.5, 1, 1),
                  "bright x1.6": (1.6, 1, 1), "bright x2.5": (2.5, 1, 1), "low contrast": (1, 0.5, 1),
                  "high contrast": (1, 1.8, 1), "gamma 0.5": (1, 1, 0.5), "gamma 2.0": (1, 1, 2.0)}
    rob = {}
    print("Lighting robustness (test split):")
    for name, (b, c, g) in conditions.items():
        tf = transforms.Compose([LightPerturb(b, c, g), eval_transform(args.img)])
        if presplit:
            ds_c = datasets.ImageFolder(os.path.join(args.data, "test"), transform=tf)
        else:
            ds_c = Subset(datasets.ImageFolder(args.data, transform=tf), te_idx)
        _, acc_c, f1_c, _, _ = evaluate(model, DataLoader(ds_c, args.bs, **kw), eval_crit, device)
        rob[name] = {"acc": acc_c, "macro_f1": f1_c}
        print(f"  {name:14s} acc {acc_c:.4f}  macro-F1 {f1_c:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val"); ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(hist["train_acc"], label="train"); ax[1].plot(hist["val_acc"], label="val"); ax[1].set_title("Accuracy"); ax[1].legend()
    ax[2].imshow(cm, cmap="Blues"); ax[2].set_title("Test confusion matrix")
    ax[2].set_xticks(range(len(classes)), classes); ax[2].set_yticks(range(len(classes)), classes)
    ax[2].set_xlabel("Predicted"); ax[2].set_ylabel("True")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax[2].text(j, i, cm[i, j], ha="center", va="center", color="red" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    fig_path = os.path.join(args.out, "training_summary.png")
    plt.savefig(fig_path, dpi=150); plt.close()

    wandb.log({"test/loss": te_loss, "test/acc": te_acc, "test/macro_f1": te_f1,
               "test/confusion_matrix": wandb.plot.confusion_matrix(y_true=y_true, preds=y_pred, class_names=classes),
               "summary_figure": wandb.Image(fig_path),
               "lighting_robustness": wandb.Table(columns=["condition", "acc", "macro_f1"],
                                                  data=[[k, v["acc"], v["macro_f1"]] for k, v in rob.items()])})
    wandb.summary.update({"best_epoch": best_ep, "best_val_macro_f1": best_f1, "test_acc": te_acc})

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"test_loss": te_loss, "test_acc": te_acc, "test_macro_f1": te_f1, "best_epoch": best_ep,
                   "train_time_s": time.time() - t0, "n_params": n_params, "split_sizes":
                   {"train": n_tr, "val": n_va, "test": n_te},
                   "class_counts": {classes[k]: int(v) for k, v in counts.items()},
                   "per_class": report, "lighting_robustness": rob,
                   "loss": args.loss, "imbalance": args.imbalance}, f, indent=2)
    art = wandb.Artifact("rps-model", type="model"); art.add_file(ckpt_path); run.log_artifact(art)
    run.finish()
    print("Saved:", ckpt_path, "|", fig_path)
    if te_acc < 0.85:
        print("WARNING: test accuracy below the 85% requirement - collect more varied data or train longer.")


if __name__ == "__main__":
    main()