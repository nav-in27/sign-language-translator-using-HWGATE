"""
Controlled Systematic Training Campaign for AC-HWGATE 136-Class Recognition.
Strict Protocol:
  - Uses strictly train_split.npz for training.
  - Uses strictly val_split.npz for checkpoint selection & hyperparameter comparison.
  - Held-out test_split.npz remains completely UNTOUCHED during model selection.
  - Hardware: RTX 3050 Laptop GPU (PyTorch AMP FP16).
  - Reproducible seed: 42.
  - Preserves existing production checkpoints.
"""

import os
import sys
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

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class ISLDataset(Dataset):
    """Sign-preserving spatial-temporal augmented dataset."""
    def __init__(self, npz_path: Path, is_train: bool = True, augment: bool = True, aug_strength: float = 1.0):
        data = np.load(npz_path, allow_pickle=True)
        self.features = torch.from_numpy(data["features"]).float() # [N, 32, 133, 6]
        self.labels = torch.from_numpy(data["labels"]).long()      # [N]
        manifest_p = ROOT_DIR / "configs" / "authoritative_136_classes_manifest.json"
        with open(manifest_p, "r", encoding="utf-8") as f:
            self.class_names = json.load(f)["classes"]
        self.is_train = is_train
        self.augment = augment and is_train
        self.aug_strength = float(aug_strength)

    def __len__(self):
        return len(self.features)

    def _apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        # x: [32, 133, 6] (Px, Py, Vx, Vy, Ax, Ay)
        x_aug = x.clone()

        # 1. Sign-preserving spatial scale jitter (0.95 to 1.05)
        if random.random() < 0.5:
            delta = 0.05 * self.aug_strength
            scale = random.uniform(1.0 - delta, 1.0 + delta)
            x_aug[:, :, 0:2] *= scale
            x_aug[:, :, 2:4] *= scale
            x_aug[:, :, 4:6] *= scale

        # 2. Minor in-plane rotation (-4 to +4 degrees)
        if random.random() < 0.4:
            angle_limit = 0.07 * self.aug_strength
            angle = random.uniform(-angle_limit, angle_limit) # radians
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            R = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float32)
            for ch_start in [0, 2, 4]:
                pos = x_aug[:, :, ch_start:ch_start+2]
                x_aug[:, :, ch_start:ch_start+2] = torch.matmul(pos, R.T)

        # 3. Small Gaussian keypoint jitter (sigma = 0.003)
        if random.random() < 0.4:
            noise = torch.randn_like(x_aug[:, :, 0:2]) * (0.003 * self.aug_strength)
            x_aug[:, :, 0:2] += noise

        # 4. Temporal jitter: subtle frame interpolation/subsampling
        if random.random() < 0.25 * self.aug_strength:
            drop_idx = random.randint(1, 30)
            x_aug[drop_idx] = (x_aug[drop_idx - 1] + x_aug[drop_idx + 1]) * 0.5

        return x_aug

    def __getitem__(self, idx: int):
        x = self.features[idx]
        y = self.labels[idx]
        if self.augment:
            x = self._apply_augmentation(x)
        return x, y


class FocalLoss(nn.Module):
    """Multi-class Focal Loss with optional label smoothing."""
    def __init__(self, gamma: float = 1.5, weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.04):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        # Compute smoothed targets
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)
        focal_weight = (1.0 - p) ** self.gamma

        loss = -smooth_targets * focal_weight * log_p
        if self.weight is not None:
            loss = loss * self.weight.unsqueeze(0)
        return loss.sum(dim=-1).mean()


def compute_smoothed_weights(labels: np.ndarray, num_classes: int, power: float = 0.25) -> torch.Tensor:
    """Computes power-law smoothed inverse frequency weights normalized to mean 1.0."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    raw_weights = (np.max(counts) / counts) ** power
    weights = raw_weights / np.mean(raw_weights)
    weights = np.clip(weights, 0.6, 2.2)
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

        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=(device.type == 'cuda')):
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
def evaluate_split(
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
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=(device.type == 'cuda')):
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
    macro_recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)

    top5_hits = 0
    for i in range(len(all_targets)):
        top5_cands = np.argsort(all_probs[i])[-5:]
        if all_targets[i] in top5_cands:
            top5_hits += 1
    top5_acc = top5_hits / len(all_targets) if len(all_targets) > 0 else 0.0

    per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0, labels=list(range(num_classes)))
    per_class_rec = recall_score(all_targets, all_preds, average=None, zero_division=0, labels=list(range(num_classes)))

    return {
        "loss": float(val_loss),
        "top1_accuracy": float(top1_acc),
        "top5_accuracy": float(top5_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_recall": float(macro_recall),
        "per_class_f1": per_class_f1.tolist(),
        "per_class_recall": per_class_rec.tolist(),
        "zero_f1_count": int(np.sum(per_class_f1 == 0.0)),
        "perfect_f1_count": int(np.sum(per_class_f1 == 1.0)),
    }



def run_experiment(
    exp_id: str,
    exp_name: str,
    loss_type: str = "ce",               # "ce", "focal", "smoothed_weighted_ce"
    lr: float = 6e-4,
    epochs: int = 50,
    batch_size: int = 32,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 3,
    use_augmentation: bool = True,
    aug_strength: float = 1.0,
    label_smoothing: float = 0.04,
    dropout: float = 0.15,
    gamma: float = 1.5,
    patience: int = 14
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"EXPERIMENT {exp_id}: {exp_name}")
    print(f"  Config: loss={loss_type}, lr={lr}, epochs={epochs}, batch={batch_size}, aug={use_augmentation} (str={aug_strength}), smooth={label_smoothing}, drop={dropout}")
    print("=" * 80)

    # Set seeds for strict reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = ISLDataset(DATA_DIR / "train_split.npz", is_train=True, augment=use_augmentation, aug_strength=aug_strength)
    val_ds   = ISLDataset(DATA_DIR / "val_split.npz", is_train=False, augment=False)
    num_classes = len(train_ds.class_names)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda'))

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

    # Loss selection
    if loss_type == "ce":
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    elif loss_type == "focal":
        criterion = FocalLoss(gamma=gamma, label_smoothing=label_smoothing)
    elif loss_type == "smoothed_weighted_ce":
        w = compute_smoothed_weights(train_ds.labels.numpy(), num_classes, power=0.25).to(device)
        criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)
    elif loss_type == "weighted_focal":
        w = compute_smoothed_weights(train_ds.labels.numpy(), num_classes, power=0.20).to(device)
        criterion = FocalLoss(gamma=gamma, weight=w, label_smoothing=label_smoothing)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate schedule: Warmup + Cosine Annealing
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_val_macro_f1 = 0.0
    best_epoch = 0
    best_metrics = {}
    patience_counter = 0

    history = {
        "epoch": [], "train_loss": [], "train_acc": [], "train_f1": [],
        "val_loss": [], "val_acc": [], "val_top5": [], "val_macro_f1": [],
        "val_weighted_f1": [], "val_macro_recall": [], "lr": []
    }

    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_eval = evaluate_split(model, val_loader, criterion, device, num_classes)
        scheduler.step()

        curr_lr = optimizer.param_groups[0]["lr"]
        val_acc = val_eval["top1_accuracy"]
        val_f1 = val_eval["macro_f1"]
        val_rec = val_eval["macro_recall"]
        val_wf1 = val_eval["weighted_f1"]
        val_loss = val_eval["loss"]

        history["epoch"].append(ep)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_f1"].append(tr_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_top5"].append(val_eval["top5_accuracy"])
        history["val_macro_f1"].append(val_f1)
        history["val_weighted_f1"].append(val_wf1)
        history["val_macro_recall"].append(val_rec)
        history["lr"].append(curr_lr)

        # Checkpoint selection strictly on Validation Macro F1 & Accuracy
        is_best = (val_f1 > best_val_macro_f1) or (abs(val_f1 - best_val_macro_f1) < 0.005 and val_acc > best_metrics.get("top1_accuracy", 0.0))
        marker = " [*BEST*]" if is_best else ""

        if is_best:
            best_val_macro_f1 = val_f1
            best_epoch = ep
            best_metrics = val_eval
            patience_counter = 0
            
            # Save experiment checkpoint (never overwriting production checkpoint)
            ckpt = {
                "experiment_id": exp_id,
                "experiment_name": exp_name,
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "val_macro_f1": val_f1,
                "val_accuracy": val_acc,
                "val_macro_recall": val_rec,
                "val_top5": val_eval["top5_accuracy"],
                "num_classes": num_classes,
                "class_names": list(train_ds.class_names),
                "config": {
                    "loss_type": loss_type,
                    "lr": lr,
                    "batch_size": batch_size,
                    "dropout": dropout,
                    "aug_strength": aug_strength,
                    "label_smoothing": label_smoothing
                }
            }
            torch.save(ckpt, CHECKPOINTS_DIR / f"exp_{exp_id}_best.pth")
        else:
            patience_counter += 1

        if ep % 5 == 0 or is_best or ep == epochs:
            print(f"  Epoch [{ep:02d}/{epochs:02d}] TrLoss: {tr_loss:.4f} TrAcc: {tr_acc*100:5.1f}% | ValLoss: {val_loss:.4f} ValAcc: {val_acc*100:5.2f}% ValF1: {val_f1:.4f} ValRec: {val_rec:.4f} (ZeroF1: {val_eval['zero_f1_count']}){marker}", flush=True)

        if patience_counter >= patience:
            print(f"  [!] Early stopping triggered at epoch {ep} (patience={patience})")
            break

    elapsed = time.time() - t0
    print(f"[+] Experiment {exp_id} completed in {elapsed:.1f}s ({elapsed/60:.1f}m). Best Epoch={best_epoch} | Best Val Macro F1={best_metrics.get('macro_f1', 0.0):.4f} | Best Val Acc={best_metrics.get('top1_accuracy', 0.0)*100:.2f}%")

    res = {
        "experiment_id": exp_id,
        "experiment_name": exp_name,
        "loss_type": loss_type,
        "lr": lr,
        "best_epoch": best_epoch,
        "training_time_sec": elapsed,
        "val_top1_accuracy": best_metrics.get("top1_accuracy", 0.0),
        "val_top5_accuracy": best_metrics.get("top5_accuracy", 0.0),
        "val_macro_f1": best_metrics.get("macro_f1", 0.0),
        "val_weighted_f1": best_metrics.get("weighted_f1", 0.0),
        "val_macro_recall": best_metrics.get("macro_recall", 0.0),
        "val_zero_f1_count": best_metrics.get("zero_f1_count", 0),
        "val_perfect_f1_count": best_metrics.get("perfect_f1_count", 0),
        "history": history
    }

    with open(REPORTS_DIR / f"exp_{exp_id}_results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    return res


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run 136-Class Training Experiments")
    parser.add_argument("--exp", type=str, default="all", help="Experiment ID to run: 1, 2, 3, 4, or all")
    parser.add_argument("--epochs", type=int, default=45, help="Max epochs per experiment")
    args = parser.parse_args()

    experiments = [
        {
            "exp_id": "1",
            "exp_name": "Baseline Unweighted CE + Augmentation (lr=5e-4)",
            "loss_type": "ce",
            "lr": 5e-4,
            "epochs": args.epochs,
            "batch_size": 32,
            "use_augmentation": True,
            "aug_strength": 1.0,
            "label_smoothing": 0.04,
            "dropout": 0.15,
            "patience": 14
        },
        {
            "exp_id": "2",
            "exp_name": "Focal Loss (gamma=1.5) + Label Smoothing (0.04, lr=6e-4)",
            "loss_type": "focal",
            "gamma": 1.5,
            "lr": 6e-4,
            "epochs": args.epochs,
            "batch_size": 32,
            "use_augmentation": True,
            "aug_strength": 1.0,
            "label_smoothing": 0.04,
            "dropout": 0.15,
            "patience": 14
        },
        {
            "exp_id": "3",
            "exp_name": "Smoothed Power-Law Class Weighted CE (power=0.25, lr=5e-4)",
            "loss_type": "smoothed_weighted_ce",
            "lr": 5e-4,
            "epochs": args.epochs,
            "batch_size": 32,
            "use_augmentation": True,
            "aug_strength": 1.0,
            "label_smoothing": 0.04,
            "dropout": 0.15,
            "patience": 14
        },
        {
            "exp_id": "4",
            "exp_name": "Smoothed Weighted Focal Loss (power=0.20, gamma=1.5, lr=6e-4)",
            "loss_type": "weighted_focal",
            "gamma": 1.5,
            "lr": 6e-4,
            "epochs": args.epochs,
            "batch_size": 32,
            "use_augmentation": True,
            "aug_strength": 1.0,
            "label_smoothing": 0.04,
            "dropout": 0.15,
            "patience": 14
        }
    ]

    results = []
    for exp_cfg in experiments:
        if args.exp != "all" and exp_cfg["exp_id"] != args.exp:
            continue
        res = run_experiment(**exp_cfg)
        results.append(res)

    print("\n" + "=" * 90)
    print("ALL RUN EXPERIMENTS SUMMARY (VALIDATION METRICS ONLY)")
    print("=" * 90)
    print(f"{'Exp ID':<8} {'Loss Type':<22} {'LR':<8} {'Best Ep':<9} {'Val Top-1':<12} {'Val Macro F1':<14} {'Val Recall':<12} {'Zero-F1':<8}")
    print("-" * 90)
    for r in results:
        print(f"{r['experiment_id']:<8} {r['loss_type']:<22} {r['lr']:<8.1e} {r['best_epoch']:<9} {r['val_top1_accuracy']*100:6.2f}%      {r['val_macro_f1']:<14.4f} {r['val_macro_recall']:<12.4f} {r['val_zero_f1_count']:<8}")
    print("=" * 90)

    # Save summary report
    with open(REPORTS_DIR / "experiments_comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Summary saved to {REPORTS_DIR / 'experiments_comparison_summary.json'}")


if __name__ == "__main__":
    main()

