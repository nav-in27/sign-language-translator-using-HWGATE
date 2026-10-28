"""
Targeted Fine-Tuning with Class-Balanced Focal Loss for 136-Class AC-HWGATE.
Addresses minority/hard classes by:
  - Effective number of samples weighting (Cui et al., CVPR 2019)
  - Focal loss modulation ((1 - p_t)^gamma) to downweight easy examples
  - Cosine annealing fine-tuning from best production weights
  - Validating on clean, non-leaking held-out test set
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
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from training.models.ac_hwgate import ACHWGATE

DATA_DIR = ROOT_DIR / "data" / "unified_136class"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
REPORTS_DIR = ROOT_DIR / "reports"
MODELS_DIR = ROOT_DIR / "models"
BACKEND_MODELS_DIR = ROOT_DIR / "backend" / "models"


class ISLDataset(Dataset):
    def __init__(self, npz_path: Path, is_train: bool = True, augment: bool = True):
        data = np.load(npz_path, allow_pickle=True)
        self.features = torch.from_numpy(data["features"]).float()
        self.labels = torch.from_numpy(data["labels"]).long()
        
        # Load unique class list
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
        if random.random() < 0.5:
            scale = random.uniform(0.94, 1.06)
            x_aug[:, :, 0:2] *= scale
            x_aug[:, :, 2:4] *= scale
            x_aug[:, :, 4:6] *= scale

        if random.random() < 0.4:
            angle = random.uniform(-0.06, 0.06)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            R = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float32)
            for ch_start in [0, 2, 4]:
                pos = x_aug[:, :, ch_start:ch_start+2]
                x_aug[:, :, ch_start:ch_start+2] = torch.matmul(pos, R.T)

        if random.random() < 0.4:
            noise = torch.randn_like(x_aug[:, :, 0:2]) * 0.003
            x_aug[:, :, 0:2] += noise

        return x_aug

    def __getitem__(self, idx: int):
        x = self.features[idx]
        y = self.labels[idx]
        if self.augment:
            x = self._apply_augmentation(x)
        return x, y


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Loss (CVPR 2019) with Focal Modulation and Label Smoothing.
    Downweights well-classified examples and balances minority class gradients.
    """
    def __init__(
        self,
        samples_per_class: np.ndarray,
        num_classes: int,
        beta: float = 0.999,
        gamma: float = 1.5,
        label_smoothing: float = 0.05
    ):
        super().__init__()
        effective_num = 1.0 - np.power(beta, np.maximum(samples_per_class, 1))
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * num_classes
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        device = logits.device
        w = self.weights.to(device)
        
        # Softmax probabilities
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0 - 1e-6)
        
        # Cross entropy loss with label smoothing and class weights
        ce_loss = F.cross_entropy(logits, targets, weight=w, reduction="none", label_smoothing=self.label_smoothing)
        
        # Focal modulation factor
        focal_weight = torch.pow(1.0 - pt, self.gamma)
        
        loss = focal_weight * ce_loss
        return loss.mean()


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

    top5_hits = sum(1 for i in range(len(all_targets)) if all_targets[i] in np.argsort(all_probs[i])[-5:])
    top5_acc = top5_hits / len(all_targets) if len(all_targets) > 0 else 0.0

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


def finetune():
    print("=" * 75)
    print("AC-HWGATE 136-CLASS FINE-TUNING (CLASS-BALANCED FOCAL LOSS)")
    print("=" * 75)

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    train_ds = ISLDataset(DATA_DIR / "train_split.npz", is_train=True, augment=True)
    val_ds = ISLDataset(DATA_DIR / "val_split.npz", is_train=False, augment=False)
    test_ds = ISLDataset(DATA_DIR / "test_split.npz", is_train=False, augment=False)

    num_classes = len(train_ds.class_names)
    print(f"[+] Dataset: {num_classes} classes | Train={len(train_ds)} | Val={len(val_ds)} | Test={len(test_ds)}")

    batch_size = 32
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda'))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda'))

    # Instantiate model
    model = ACHWGATE(
        num_classes=num_classes,
        num_joints=133,
        in_channels=6,
        sequence_length=32,
        embed_dim=128,
        num_heads=4,
        num_layers=3,
        window_size=8,
        dropout=0.20
    ).to(device)

    # Load warm-start checkpoint
    base_ckpt_path = CHECKPOINTS_DIR / "achwgate_136class_production_best.pth"
    if base_ckpt_path.exists():
        print(f"[+] Loading warm-start weights from: {base_ckpt_path.name}")
        ckpt = torch.load(base_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print("[!] Warning: Warm-start checkpoint not found. Training from scratch.")

    # Class-Balanced Focal Loss setup
    train_counts = np.bincount(train_ds.labels.numpy(), minlength=num_classes)
    criterion = ClassBalancedFocalLoss(
        samples_per_class=train_counts,
        num_classes=num_classes,
        beta=0.999,
        gamma=1.5,
        label_smoothing=0.05
    )

    epochs = 25
    lr = 2e-4  # Gentle fine-tuning learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    # Baseline evaluation before fine-tuning
    init_eval = evaluate(model, val_loader, criterion, device, num_classes)
    print(f"[*] Initial Validation Metrics -> Acc: {init_eval['top1_accuracy']*100:.2f}%, Top5: {init_eval['top5_accuracy']*100:.2f}%, Macro F1: {init_eval['macro_f1']:.4f}")

    best_val_macro_f1 = init_eval["macro_f1"]
    best_epoch = 0
    best_metrics = init_eval

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=(device.type == 'cuda')):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item() * len(y)

        tr_loss /= len(train_loader.dataset)
        val_eval = evaluate(model, val_loader, criterion, device, num_classes)
        scheduler.step()

        val_acc = val_eval["top1_accuracy"]
        val_f1 = val_eval["macro_f1"]
        is_best = val_f1 > best_val_macro_f1
        marker = " [*BEST*]" if is_best else ""

        if is_best:
            best_val_macro_f1 = val_f1
            best_epoch = ep
            best_metrics = val_eval
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
                    "dropout": 0.20
                }
            }
            torch.save(ckpt_save, CHECKPOINTS_DIR / "achwgate_136class_focal_best.pth")

        if ep % 5 == 0 or is_best or ep == epochs:
            print(f"  Epoch [{ep:02d}/{epochs:02d}] Train Loss: {tr_loss:.4f} | Val Acc: {val_acc*100:.2f}% Top5: {val_eval['top5_accuracy']*100:.2f}% MacroF1: {val_f1:.4f}{marker}", flush=True)

    elapsed = time.time() - t0
    print(f"\n[+] Fine-tuning completed in {elapsed:.1f}s ({elapsed/60:.1f} min). Best Epoch={best_epoch} with Val Macro F1={best_val_macro_f1:.4f}")

    # Evaluate best focal model on untouched test set
    if (CHECKPOINTS_DIR / "achwgate_136class_focal_best.pth").exists():
        best_focal = torch.load(CHECKPOINTS_DIR / "achwgate_136class_focal_best.pth", map_location=device, weights_only=False)
        model.load_state_dict(best_focal["model_state_dict"])

    test_eval = evaluate(model, test_loader, criterion, device, num_classes)
    print("\n" + "=" * 75)
    print("UNTOUCHED HELD-OUT TEST EVALUATION (CLEAN SPLIT, N=722):")
    print(f"  Top-1 Accuracy: {test_eval['top1_accuracy']*100:.2f}%")
    print(f"  Top-5 Accuracy: {test_eval['top5_accuracy']*100:.2f}%")
    print(f"  Macro F1:       {test_eval['macro_f1']:.4f}")
    print(f"  Weighted F1:    {test_eval['weighted_f1']:.4f}")
    print("=" * 75)

    zero_f1_classes = [train_ds.class_names[i] for i, f1 in enumerate(test_eval['per_class_f1']) if f1 == 0.0]
    perfect_classes = [train_ds.class_names[i] for i, f1 in enumerate(test_eval['per_class_f1']) if f1 >= 0.999]
    print(f"  Zero-F1 Classes: {len(zero_f1_classes)} ({zero_f1_classes})")
    print(f"  Perfect Classes (F1=1.0): {len(perfect_classes)}")

    # Save results report
    results = {
        "experiment_name": "achwgate_136class_focal_finetune",
        "best_epoch": best_epoch,
        "elapsed_sec": elapsed,
        "val_top1_accuracy": best_metrics["top1_accuracy"],
        "val_top5_accuracy": best_metrics["top5_accuracy"],
        "val_macro_f1": best_metrics["macro_f1"],
        "test_top1_accuracy": test_eval["top1_accuracy"],
        "test_top5_accuracy": test_eval["top5_accuracy"],
        "test_macro_f1": test_eval["macro_f1"],
        "test_weighted_f1": test_eval["weighted_f1"],
        "zero_f1_count": len(zero_f1_classes),
        "zero_f1_classes": zero_f1_classes,
        "perfect_classes_count": len(perfect_classes),
        "classes": list(train_ds.class_names)
    }
    with open(REPORTS_DIR / "focal_finetune_136class_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Saved results to: {REPORTS_DIR / 'focal_finetune_136class_results.json'}")


if __name__ == "__main__":
    finetune()
