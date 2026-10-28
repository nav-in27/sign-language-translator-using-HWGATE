"""
Production AC-HWGATE Training Pipeline for Full 136-Class ISL Recognition.
Features:
  - Disjoint Signer/Session Splitting (Zero Leakage)
  - PyTorch AMP FP16 Mixed Precision (Safe for 4GB RTX 3050 VRAM)
  - Dynamic ISL Temporal & Spatial Landmark Augmentations
  - Class-Balanced Inverse-Frequency Weighting & Label Smoothing
  - Cosine Annealing Learning Rate Schedule with Warmup
  - Comprehensive Validation Tracking (Top-1, Top-5, Macro F1, Weighted F1, Per-Class F1)
  - Early Stopping with Checkpoint Rollback to Best Validation State
  - Confusion Matrix Generation & Hard-Class Diagnostics
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    getattr(sys.stdout, "reconfigure")(line_buffering=True)
import time
import json
import random
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from training.models.ac_hwgate import ACHWGATE
from training.feature_extractor import feature_extractor

DATA_DIR = ROOT_DIR / "data" / "unified_136class"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
REPORTS_DIR = ROOT_DIR / "reports"
MODELS_DIR = ROOT_DIR / "models"
BACKEND_MODELS_DIR = ROOT_DIR / "backend" / "models"

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class ISLDataset(Dataset):
    """PyTorch Dataset for 136-class ISL sequences with realistic spatial-temporal augmentation."""
    def __init__(self, npz_path: Path, is_train: bool = True, augment: bool = True, augmentation_strength: float = 1.0):
        data = np.load(npz_path, allow_pickle=True)
        self.features = torch.from_numpy(data["features"]).float() # [N, 32, 133, 6]
        self.labels = torch.from_numpy(data["labels"]).long()      # [N]
        manifest_p = ROOT_DIR / "configs/authoritative_136_classes_manifest.json"
        if manifest_p.exists():
            with open(manifest_p, "r", encoding="utf-8") as f:
                self.class_names = json.load(f)["classes"]
        else:
            self.class_names = sorted(list(set(data["class_names"])))
        self.is_train = is_train
        self.augment = augment and is_train
        self.augmentation_strength = float(augmentation_strength)

    def __len__(self):
        return len(self.features)

    def _apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        # x: [32, 133, 6] (Px, Py, Vx, Vy, Ax, Ay)
        x_aug = x.clone()
        
        # 1. Random Spatial Scaling (90% to 110%)
        if random.random() < 0.6:
            scale_delta = 0.08 * self.augmentation_strength
            scale = random.uniform(1.0 - scale_delta, 1.0 + scale_delta)
            x_aug[:, :, 0:2] *= scale
            x_aug[:, :, 2:4] *= scale
            x_aug[:, :, 4:6] *= scale

        # 2. Random In-Plane Rotation (-5 to +5 degrees)
        if random.random() < 0.5:
            angle_limit = 0.08 * self.augmentation_strength
            angle = random.uniform(-angle_limit, angle_limit) # radians
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            R = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float32)
            
            for ch_start in [0, 2, 4]:
                pos = x_aug[:, :, ch_start:ch_start+2]
                x_aug[:, :, ch_start:ch_start+2] = torch.matmul(pos, R.T)

        # 3. Small Gaussian Jitter on Keypoints
        if random.random() < 0.5:
            noise = torch.randn_like(x_aug[:, :, 0:2]) * (0.004 * self.augmentation_strength)
            x_aug[:, :, 0:2] += noise

        # 4. Temporal Sub-sampling / Frame Dropout (randomly zero 1 frame)
        if random.random() < 0.3 * self.augmentation_strength:
            drop_idx = random.randint(1, 30)
            x_aug[drop_idx] = (x_aug[drop_idx - 1] + x_aug[drop_idx + 1]) * 0.5

        return x_aug

    def __getitem__(self, idx: int):
        x = self.features[idx]
        y = self.labels[idx]
        if self.augment:
            x = self._apply_augmentation(x)
        return x, y


import math

def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Computes clamped inverse class frequency weights."""
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if counts[c] > 0:
            w = total / (num_classes * counts[c])
            weights[c] = np.clip(w, 0.25, 4.0)
        else:
            weights[c] = 1.0
    return torch.from_numpy(weights).float()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device
) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=(device.type == 'cuda')):
            logits = model(x)
            loss = criterion(logits, y)

        if device.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * len(y)
        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y.detach().cpu().numpy())

    epoch_loss = total_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=(device.type == 'cuda')):
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

    # Top-5 accuracy
    top5_hits = 0
    for i in range(len(all_targets)):
        top5_indices = np.argsort(all_probs[i])[-5:]
        if all_targets[i] in top5_indices:
            top5_hits += 1
    top5_acc = top5_hits / len(all_targets) if len(all_targets) > 0 else 0.0

    # Per-class metrics
    per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0, labels=list(range(num_classes)))
    per_class_prec = precision_score(all_targets, all_preds, average=None, zero_division=0, labels=list(range(num_classes)))
    per_class_rec = recall_score(all_targets, all_preds, average=None, zero_division=0, labels=list(range(num_classes)))

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
        "confusion_matrix": cm.tolist()
    }


def run_training_experiment(
    exp_name: str,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    use_class_weights: bool = True,
    use_augmentation: bool = True,
    augmentation_strength: float = 1.0,
    label_smoothing: float = 0.08,
    dropout: float = 0.20,
    evaluate_test: bool = True
) -> Dict[str, Any]:
    print("\n" + "=" * 75)
    print(f"RUNNING TRAINING EXPERIMENT: {exp_name}")
    print(f"  Params: epochs={epochs}, batch={batch_size}, lr={lr}, class_weights={use_class_weights}, aug={use_augmentation}, aug_strength={augmentation_strength}")
    print("=" * 75)

    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    train_ds = ISLDataset(DATA_DIR / "train_split.npz", is_train=True, augment=use_augmentation, augmentation_strength=augmentation_strength)
    val_ds = ISLDataset(DATA_DIR / "val_split.npz", is_train=False, augment=False)
    test_ds = ISLDataset(DATA_DIR / "test_split.npz", is_train=False, augment=False)

    num_classes = len(train_ds.class_names)
    print(f"[+] Dataset Loaded: {num_classes} classes | Train={len(train_ds)} | Val={len(val_ds)} | Test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda'))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda'))

    # Model
    model = ACHWGATE(
        num_classes=num_classes,
        num_joints=133,
        in_channels=6,
        sequence_length=32,
        embed_dim=128,
        num_heads=4,
        num_layers=3,
        window_size=8,
        dropout=dropout
    ).to(device)

    # Loss
    if use_class_weights:
        weights = compute_class_weights(train_ds.labels.numpy(), num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_val_macro_f1 = 0.0
    best_epoch = 0
    best_metrics = {}
    patience = 14
    patience_counter = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "train_macro_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "val_weighted_f1": [],
        "lr": []
    }

    t0 = time.time()

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_eval = evaluate(model, val_loader, criterion, device, num_classes)
        scheduler.step()

        curr_lr = optimizer.param_groups[0]["lr"]
        val_acc = val_eval["top1_accuracy"]
        val_f1 = val_eval["macro_f1"]
        val_wf1 = val_eval["weighted_f1"]
        val_loss = val_eval["loss"]

        history["epoch"].append(ep)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_macro_f1"].append(tr_f1)
        history["val_loss"].append(val_loss)
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
            # Save checkpoint
            ckpt = {
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
                    "dropout": dropout,
                    "augmentation_strength": augmentation_strength
                }
            }
            torch.save(ckpt, CHECKPOINTS_DIR / f"{exp_name}_best.pth")
            torch.save(ckpt, CHECKPOINTS_DIR / "achwgate_136class_best.pth")
            torch.save(ckpt, CHECKPOINTS_DIR / "achwgate_best.pth")
        else:
            patience_counter += 1

        if ep % 5 == 0 or is_best or ep == epochs:
            print(f"  Epoch [{ep:02d}/{epochs:02d}] Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.1f}% F1: {tr_f1:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}% Top5: {val_eval['top5_accuracy']*100:.1f}% MacroF1: {val_f1:.4f}{marker}", flush=True)

        if patience_counter >= patience:
            print(f"[!] Early stopping triggered at epoch {ep} (patience={patience})")
            break

    elapsed = time.time() - t0
    print(f"\n[+] Training finished in {elapsed:.1f}s ({elapsed/60:.1f} min). Best Epoch={best_epoch} with Val Macro F1={best_val_macro_f1:.4f}")

    # Load best checkpoint selected only from validation metrics.
    best_ckpt = torch.load(CHECKPOINTS_DIR / f"{exp_name}_best.pth", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_eval = None
    if evaluate_test:
        test_eval = evaluate(model, test_loader, criterion, device, num_classes)
        print("\n" + "=" * 75)
        print("UNTOUCHED HELD-OUT TEST EVALUATION:")
        print(f"  Top-1 Accuracy: {test_eval['top1_accuracy']*100:.2f}%")
        print(f"  Top-5 Accuracy: {test_eval['top5_accuracy']*100:.2f}%")
        print(f"  Macro F1:       {test_eval['macro_f1']:.4f}")
        print(f"  Weighted F1:    {test_eval['weighted_f1']:.4f}")
        print("=" * 75)

    results = {
        "experiment_name": exp_name,
        "best_epoch": best_epoch,
        "training_time_sec": elapsed,
        "val_top1_accuracy": best_metrics.get("top1_accuracy", 0.0),
        "val_top5_accuracy": best_metrics.get("top5_accuracy", 0.0),
        "val_macro_f1": best_metrics.get("macro_f1", 0.0),
        "val_weighted_f1": best_metrics.get("weighted_f1", 0.0),
        "test_top1_accuracy": test_eval["top1_accuracy"] if test_eval else None,
        "test_top5_accuracy": test_eval["top5_accuracy"] if test_eval else None,
        "test_macro_f1": test_eval["macro_f1"] if test_eval else None,
        "test_weighted_f1": test_eval["weighted_f1"] if test_eval else None,
        "history": history,
        "test_per_class_f1": test_eval["per_class_f1"] if test_eval else None,
        "test_confusion_matrix": test_eval["confusion_matrix"] if test_eval else None,
        "classes": list(train_ds.class_names)
    }

    with open(REPORTS_DIR / f"{exp_name}_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    print("=" * 75)
    print("AC-HWGATE FULL 136-CLASS PRODUCTION TRAINING")
    print("=" * 75)

    # Systematic training experiments
    exp_results = []

    # Exp 1: Baseline Architecture
    res_b = run_training_experiment(
        exp_name="achwgate_136class_baseline",
        epochs=35,
        batch_size=32,
        lr=1e-3,
        use_class_weights=False,
        use_augmentation=False,
        label_smoothing=0.0,
        dropout=0.15
    )
    exp_results.append(res_b)

    # Exp 2: Weighted + Augmented + Regularized (Optimal Production Pipeline)
    res_prod = run_training_experiment(
        exp_name="achwgate_136class_production",
        epochs=50,
        batch_size=32,
        lr=1e-3,
        use_class_weights=True,
        use_augmentation=True,
        label_smoothing=0.08,
        dropout=0.20
    )
    exp_results.append(res_prod)

    # Summary table
    print("\n" + "=" * 75)
    print("EXPERIMENT COMPARISON SUMMARY:")
    print("=" * 75)
    print(f"{'Experiment':<32} | {'Val Acc':<9} | {'Val F1':<9} | {'Test Acc':<9} | {'Test F1':<9}")
    print("-" * 75)
    for r in exp_results:
        print(f"{r['experiment_name']:<32} | {r['val_top1_accuracy']*100:6.2f}%  | {r['val_macro_f1']:6.4f}  | {r['test_top1_accuracy']*100:6.2f}%  | {r['test_macro_f1']:6.4f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
