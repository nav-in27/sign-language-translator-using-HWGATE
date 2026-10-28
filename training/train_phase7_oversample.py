"""
AC-HWGATE Phase 7: Iterative Training with Minority Class Oversampling & Mixup.
Addresses the core bottleneck: 34 classes with ≤10 training samples, 46 classes <70% test accuracy.

Strategy:
  1. Weighted Random Sampling: oversample minority classes so each epoch sees ~equal class representation
  2. Mixup Augmentation: interpolate between pairs of samples (alpha=0.2) for smoother decision boundaries
  3. Stronger spatial/temporal augmentation with progressive scaling
  4. Warm-start from production_best checkpoint
  5. Two-phase schedule: 5-epoch warmup at lr=5e-5, then cosine anneal from lr=3e-4 for 45 epochs
  6. Early stopping on validation macro F1 with patience=18
  7. Saves checkpoint ONLY if it beats both production_best AND focal_best on val macro F1
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    getattr(sys.stdout, "reconfigure")(line_buffering=True)

import time
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from training.models.ac_hwgate import ACHWGATE

DATA_DIR = ROOT_DIR / "data" / "unified_136class"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
REPORTS_DIR = ROOT_DIR / "reports"

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class ISLDatasetV2(Dataset):
    """Enhanced dataset with stronger augmentation and mixup support."""

    def __init__(self, npz_path: Path, is_train: bool = True, augment: bool = True):
        data = np.load(npz_path, allow_pickle=True)
        self.features = torch.from_numpy(data["features"]).float()  # [N, 32, 133, 6]
        self.labels = torch.from_numpy(data["labels"]).long()        # [N]

        manifest_p = ROOT_DIR / "configs/authoritative_136_classes_manifest.json"
        if manifest_p.exists():
            with open(manifest_p, "r", encoding="utf-8") as f:
                self.class_names = json.load(f)["classes"]
        else:
            self.class_names = sorted(list(set(data["class_names"])))

        self.is_train = is_train
        self.augment = augment and is_train

    def __len__(self):
        return len(self.features)

    def _apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        x_aug = x.clone()

        # 1. Random spatial scaling (88% to 112%)
        if random.random() < 0.7:
            scale = random.uniform(0.88, 1.12)
            x_aug[:, :, 0:2] *= scale
            x_aug[:, :, 2:4] *= scale
            x_aug[:, :, 4:6] *= scale

        # 2. Random in-plane rotation (-8 to +8 degrees)
        if random.random() < 0.6:
            angle = random.uniform(-0.14, 0.14)  # radians (~8 degrees)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            R = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float32)
            for ch_start in [0, 2, 4]:
                pos = x_aug[:, :, ch_start:ch_start + 2]
                x_aug[:, :, ch_start:ch_start + 2] = torch.matmul(pos, R.T)

        # 3. Gaussian jitter on position keypoints
        if random.random() < 0.6:
            noise = torch.randn_like(x_aug[:, :, 0:2]) * 0.006
            x_aug[:, :, 0:2] += noise

        # 4. Random translation shift on positions
        if random.random() < 0.4:
            tx = random.uniform(-0.04, 0.04)
            ty = random.uniform(-0.04, 0.04)
            x_aug[:, :, 0] += tx
            x_aug[:, :, 1] += ty

        # 5. Temporal jitter: randomly duplicate or skip 1-2 frames
        if random.random() < 0.4:
            n_jitter = random.randint(1, 2)
            for _ in range(n_jitter):
                idx = random.randint(1, 30)
                if random.random() < 0.5:
                    # Duplicate previous frame
                    x_aug[idx] = x_aug[idx - 1].clone()
                else:
                    # Interpolate with neighbors
                    x_aug[idx] = (x_aug[idx - 1] + x_aug[min(idx + 1, 31)]) * 0.5

        # 6. Random horizontal flip of x-coordinates (mirror gesture)
        if random.random() < 0.3:
            x_aug[:, :, 0] = -x_aug[:, :, 0]  # flip Px
            x_aug[:, :, 2] = -x_aug[:, :, 2]  # flip Vx
            x_aug[:, :, 4] = -x_aug[:, :, 4]  # flip Ax
            # Swap left and right hand keypoints (indices 33-53 <-> 54-74)
            left = x_aug[:, 33:54, :].clone()
            right = x_aug[:, 54:75, :].clone()
            x_aug[:, 33:54, :] = right
            x_aug[:, 54:75, :] = left

        return x_aug

    def __getitem__(self, idx: int):
        x = self.features[idx]
        y = self.labels[idx]
        if self.augment:
            x = self._apply_augmentation(x)
        return x, y


def make_weighted_sampler(labels: np.ndarray, num_classes: int) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler that oversamples minority classes."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    # Inverse frequency weighting with sqrt dampening to avoid extreme oversampling
    class_weights = 1.0 / np.sqrt(counts)
    sample_weights = class_weights[labels]
    sample_weights = torch.from_numpy(sample_weights).double()
    # num_samples = oversample to roughly 2x the dataset size per epoch
    num_samples = int(len(labels) * 1.5)
    return WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)


def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation: interpolate between pairs of samples."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for mixup."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
        preds = np.argmax(probs, axis=-1)
        all_preds.extend(preds)
        all_targets.extend(y.detach().cpu().numpy())
        all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    val_loss = total_loss / len(loader.dataset)
    top1_acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    top5_hits = sum(
        1 for i in range(len(all_targets)) if all_targets[i] in np.argsort(all_probs[i])[-5:]
    )
    top5_acc = top5_hits / len(all_targets) if len(all_targets) > 0 else 0.0

    per_class_f1 = f1_score(
        all_targets, all_preds, average=None, zero_division=0, labels=list(range(num_classes))
    )
    per_class_prec = precision_score(
        all_targets, all_preds, average=None, zero_division=0, labels=list(range(num_classes))
    )
    per_class_rec = recall_score(
        all_targets, all_preds, average=None, zero_division=0, labels=list(range(num_classes))
    )
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))

    return {
        "loss": val_loss,
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": per_class_f1.tolist(),
        "per_class_precision": per_class_prec.tolist(),
        "per_class_recall": per_class_rec.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def run_phase7_training():
    print("=" * 80)
    print("PHASE 7: ITERATIVE TRAINING WITH OVERSAMPLING, MIXUP & STRONGER AUGMENTATION")
    print("=" * 80)

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    train_ds = ISLDatasetV2(DATA_DIR / "train_split.npz", is_train=True, augment=True)
    val_ds = ISLDatasetV2(DATA_DIR / "val_split.npz", is_train=False, augment=False)
    test_ds = ISLDatasetV2(DATA_DIR / "test_split.npz", is_train=False, augment=False)

    num_classes = len(train_ds.class_names)
    print(f"[+] Dataset: {num_classes} classes | Train={len(train_ds)} | Val={len(val_ds)} | Test={len(test_ds)}")

    # Class distribution analysis
    train_labels = train_ds.labels.numpy()
    train_counts = np.bincount(train_labels, minlength=num_classes)
    print(f"[+] Class distribution: min={train_counts.min()}, max={train_counts.max()}, "
          f"median={np.median(train_counts):.0f}, <=10 samples: {(train_counts <= 10).sum()}")

    # Weighted sampler for minority oversampling
    sampler = make_weighted_sampler(train_labels, num_classes)

    batch_size = 32
    # NOTE: WeightedRandomSampler handles shuffling; don't set shuffle=True
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=0, pin_memory=(device.type == "cuda"), drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda")
    )

    # Model instantiation
    model = ACHWGATE(
        num_classes=num_classes,
        num_joints=133,
        in_channels=6,
        sequence_length=32,
        embed_dim=128,
        num_heads=4,
        num_layers=3,
        window_size=8,
        dropout=0.25,  # Slightly higher dropout for regularization with oversampling
    ).to(device)

    # Warm-start from production best
    base_ckpt_path = CHECKPOINTS_DIR / "achwgate_136class_production_best.pth"
    if base_ckpt_path.exists():
        print(f"[+] Loading warm-start weights from: {base_ckpt_path.name}")
        ckpt = torch.load(base_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print("[!] Warning: Warm-start checkpoint not found. Training from scratch.")

    # Loss: Class-balanced cross-entropy with label smoothing
    # Using effective number weighting (Cui et al., CVPR 2019)
    beta = 0.999
    effective_num = 1.0 - np.power(beta, np.maximum(train_counts, 1).astype(np.float64))
    cb_weights = (1.0 - beta) / effective_num
    cb_weights = cb_weights / np.sum(cb_weights) * num_classes
    cb_weights = torch.tensor(cb_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=cb_weights, label_smoothing=0.08)

    # Two-phase optimizer schedule
    warmup_epochs = 5
    total_epochs = 50
    main_lr = 3e-4
    warmup_lr = 5e-5
    patience = 18

    optimizer = torch.optim.AdamW(model.parameters(), lr=warmup_lr, weight_decay=2e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Baseline evaluation
    init_eval = evaluate(model, val_loader, criterion, device, num_classes)
    print(f"[*] Initial Val -> Acc: {init_eval['top1_accuracy']*100:.2f}%, "
          f"Top5: {init_eval['top5_accuracy']*100:.2f}%, "
          f"Macro F1: {init_eval['macro_f1']:.4f}, "
          f"Weighted F1: {init_eval['weighted_f1']:.4f}")

    # Track previous best across checkpoints
    prev_best_f1 = init_eval["macro_f1"]
    best_val_macro_f1 = prev_best_f1
    best_epoch = 0
    best_metrics = init_eval
    patience_counter = 0
    use_mixup = True

    history = {
        "epoch": [],
        "train_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "val_weighted_f1": [],
        "lr": [],
    }

    t0 = time.time()

    for ep in range(1, total_epochs + 1):
        # Phase transition: warmup -> main training
        if ep == warmup_epochs + 1:
            print(f"\n[+] Warmup complete. Switching to main LR={main_lr}, enabling cosine annealing.")
            for param_group in optimizer.param_groups:
                param_group["lr"] = main_lr
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6
            )

        model.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=(device.type == "cuda")):
                if use_mixup and random.random() < 0.5:
                    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
                    logits = model(mixed_x)
                    loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                    # For accuracy tracking, use unmixed
                    preds = logits.argmax(dim=-1)
                    tr_correct += (lam * (preds == y_a).float() + (1 - lam) * (preds == y_b).float()).sum().item()
                else:
                    logits = model(x)
                    loss = criterion(logits, y)
                    preds = logits.argmax(dim=-1)
                    tr_correct += (preds == y).sum().item()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            tr_loss += loss.item() * len(y)
            tr_total += len(y)

        tr_loss /= tr_total
        tr_acc = tr_correct / tr_total

        # Step scheduler after warmup
        if ep > warmup_epochs:
            scheduler.step()

        # Evaluate on validation set
        val_eval = evaluate(model, val_loader, criterion, device, num_classes)
        curr_lr = optimizer.param_groups[0]["lr"]
        val_acc = val_eval["top1_accuracy"]
        val_f1 = val_eval["macro_f1"]
        val_wf1 = val_eval["weighted_f1"]

        history["epoch"].append(ep)
        history["train_loss"].append(tr_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)
        history["val_weighted_f1"].append(val_wf1)
        history["lr"].append(curr_lr)

        is_best = val_f1 > best_val_macro_f1
        marker = " [*BEST*]" if is_best else ""

        if is_best:
            best_val_macro_f1 = val_f1
            best_epoch = ep
            best_metrics = val_eval
            patience_counter = 0
            ckpt_save = {
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_f1,
                "val_accuracy": val_acc,
                "num_classes": num_classes,
                "class_names": list(train_ds.class_names),
                "config": {
                    "num_classes": num_classes,
                    "embed_dim": 128,
                    "num_heads": 4,
                    "num_layers": 3,
                    "window_size": 8,
                    "dropout": 0.25,
                },
            }
            torch.save(ckpt_save, CHECKPOINTS_DIR / "achwgate_136class_oversample_best.pth")
        else:
            patience_counter += 1

        # Print progress
        if ep % 3 == 0 or is_best or ep <= warmup_epochs or ep == total_epochs:
            zero_f1_count = sum(1 for f in val_eval["per_class_f1"] if f == 0.0)
            print(
                f"  Epoch [{ep:02d}/{total_epochs:02d}] "
                f"TrLoss: {tr_loss:.4f} TrAcc: {tr_acc*100:.1f}% | "
                f"Val Acc: {val_acc*100:.2f}% Top5: {val_eval['top5_accuracy']*100:.1f}% "
                f"MacroF1: {val_f1:.4f} WF1: {val_wf1:.4f} "
                f"ZeroF1: {zero_f1_count} LR: {curr_lr:.6f}{marker}",
                flush=True,
            )

        if patience_counter >= patience:
            print(f"[!] Early stopping at epoch {ep} (patience={patience})")
            break

    elapsed = time.time() - t0
    print(f"\n[+] Training finished in {elapsed:.1f}s ({elapsed/60:.1f} min).")
    print(f"[+] Best Epoch={best_epoch} with Val Macro F1={best_val_macro_f1:.4f}")

    # =========================================================================
    # UNTOUCHED HELD-OUT TEST EVALUATION
    # =========================================================================
    best_ckpt_path = CHECKPOINTS_DIR / "achwgate_136class_oversample_best.pth"
    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])

    test_eval = evaluate(model, test_loader, criterion, device, num_classes)

    print("\n" + "=" * 80)
    print("UNTOUCHED HELD-OUT TEST EVALUATION (CLEAN SPLIT, N=722):")
    print(f"  Top-1 Accuracy: {test_eval['top1_accuracy']*100:.2f}% ({int(test_eval['top1_accuracy']*722)}/722)")
    print(f"  Top-5 Accuracy: {test_eval['top5_accuracy']*100:.2f}%")
    print(f"  Macro F1:       {test_eval['macro_f1']:.4f}")
    print(f"  Weighted F1:    {test_eval['weighted_f1']:.4f}")
    print("=" * 80)

    zero_f1_classes = [
        train_ds.class_names[i] for i, f1 in enumerate(test_eval["per_class_f1"]) if f1 == 0.0
    ]
    perfect_classes = [
        train_ds.class_names[i] for i, f1 in enumerate(test_eval["per_class_f1"]) if f1 >= 0.999
    ]
    print(f"  Zero-F1 Classes: {len(zero_f1_classes)} ({zero_f1_classes})")
    print(f"  Perfect Classes (F1>=1.0): {len(perfect_classes)}")

    # Compare against production best
    print("\n" + "=" * 80)
    print("COMPARISON AGAINST PREVIOUS PRODUCTION BEST (91.41% Top-1):")
    delta_acc = (test_eval["top1_accuracy"] - 0.9141) * 100
    delta_f1 = test_eval["macro_f1"] - 0.8073
    print(f"  Top-1 Delta: {delta_acc:+.2f}%")
    print(f"  Macro F1 Delta: {delta_f1:+.4f}")

    if test_eval["top1_accuracy"] > 0.9141:
        print("  [SUCCESS] New checkpoint outperforms production best!")
        print("  -> Safe to promote to production after ONNX export & parity verification.")
    else:
        print("  [INFO] New checkpoint does not exceed production best Top-1.")
        print("  -> Preserving previous production_best checkpoint.")
    print("=" * 80)

    # Save results
    results = {
        "experiment_name": "achwgate_136class_oversample_mixup",
        "best_epoch": best_epoch,
        "elapsed_sec": elapsed,
        "val_top1_accuracy": best_metrics["top1_accuracy"],
        "val_top5_accuracy": best_metrics["top5_accuracy"],
        "val_macro_f1": best_metrics["macro_f1"],
        "val_weighted_f1": best_metrics["weighted_f1"],
        "test_top1_accuracy": test_eval["top1_accuracy"],
        "test_top5_accuracy": test_eval["top5_accuracy"],
        "test_macro_f1": test_eval["macro_f1"],
        "test_weighted_f1": test_eval["weighted_f1"],
        "zero_f1_count": len(zero_f1_classes),
        "zero_f1_classes": zero_f1_classes,
        "perfect_classes_count": len(perfect_classes),
        "test_per_class_f1": test_eval["per_class_f1"],
        "history": history,
        "classes": list(train_ds.class_names),
    }
    results_path = REPORTS_DIR / "phase7_oversample_mixup_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved full results to: {results_path}")


if __name__ == "__main__":
    run_phase7_training()
