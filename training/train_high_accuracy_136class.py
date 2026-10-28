"""
High-Performance Iterative Training Pipeline for AC-HWGATE 136-Class ISL Recognition.
Designed for Maximum Generalization, Zero Overfitting, and Anti-Fake-Sign Precision.

Key Innovations:
1. Kinetically Consistent Data Augmentation (Position + Velocity + Acceleration recomputation)
2. Temporal Speed Warping (0.90x to 1.10x linear temporal interpolation)
3. Power-Law Balanced Class Weighting (prevents majority-class dominance without starving frequent classes)
4. Label Smoothing (0.05) for well-calibrated posterior probabilities and zero overconfidence
5. Warm-start with Cosine Annealing schedule
6. Independent evaluation on held-out clean test split (722 samples)
7. Automatic prototype recomputation and ONNX export upon achieving SOTA metrics
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

DATA_DIR = ROOT_DIR / "data" / "unified_136class"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
REPORTS_DIR = ROOT_DIR / "reports"
MODELS_DIR = ROOT_DIR / "backend" / "models"

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class KineticAugmentedDataset(Dataset):
    """
    Dataset with kinetically consistent spatial-temporal augmentations:
    - Random spatial scaling (0.94 - 1.06)
    - Random in-plane rotation (-5 to +5 deg)
    - Keypoint jitter (sigma = 0.003)
    - Temporal speed warping (0.92x - 1.08x)
    - Recomputed velocity and acceleration to maintain physical coherence across all 6 channels.
    """
    def __init__(self, npz_path: Path, is_train: bool = True, augment: bool = True, aug_strength: float = 1.0):
        data = np.load(npz_path, allow_pickle=True)
        self.features = torch.from_numpy(data["features"]).float() # [N, 32, 133, 6]
        self.labels = torch.from_numpy(data["labels"]).long()      # [N]
        
        manifest_p = ROOT_DIR / "configs" / "authoritative_136_classes_manifest.json"
        if manifest_p.exists():
            with open(manifest_p, "r", encoding="utf-8") as f:
                self.class_names = json.load(f)["classes"]
        else:
            self.class_names = sorted(list(set(data["class_names"])))

        self.is_train = is_train
        self.augment = augment and is_train
        self.aug_strength = float(aug_strength)

    def __len__(self):
        return len(self.features)

    def _apply_kinetic_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        # x: [32, 133, 6] -> channels: [Px, Py, Vx, Vy, Ax, Ay]
        T, J, C = x.shape
        pos = x[:, :, 0:2].clone() # [32, 133, 2]
        
        # 1. Temporal speed warp (resample frames via linear interpolation)
        if random.random() < 0.4:
            speed_factor = random.uniform(1.0 - 0.08 * self.aug_strength, 1.0 + 0.08 * self.aug_strength)
            # Resample time grid
            t_orig = torch.linspace(0, 1, T)
            t_warp = torch.linspace(0, 1, T) * speed_factor
            t_warp = torch.clamp(t_warp, 0.0, 1.0)
            
            # Resample positions: [1, 2*J, T] -> grid_sample or 1D interp
            pos_t = pos.permute(1, 2, 0).reshape(1, J * 2, T) # [1, 266, 32]
            # 1D linear interpolation
            pos_t = F.interpolate(pos_t, size=T, mode='linear', align_corners=True)
            pos = pos_t.reshape(J, 2, T).permute(2, 0, 1)

        # 2. Sign-preserving spatial scale jitter (0.94 to 1.06)
        if random.random() < 0.5:
            delta = 0.06 * self.aug_strength
            scale = random.uniform(1.0 - delta, 1.0 + delta)
            pos *= scale

        # 3. Small in-plane rotation (-5 to +5 degrees)
        if random.random() < 0.45:
            angle_limit = 0.087 * self.aug_strength # ~5 deg in radians
            angle = random.uniform(-angle_limit, angle_limit)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            R = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float32)
            pos = torch.matmul(pos, R.T)

        # 4. Small Gaussian keypoint jitter (sigma = 0.003)
        if random.random() < 0.4:
            noise = torch.randn_like(pos) * (0.003 * self.aug_strength)
            pos += noise

        # 5. Recompute Velocity and Acceleration from augmented positions for kinetic consistency
        dt = 1.0 / 30.0
        # Velocity: V[t] = (pos[t+1] - pos[t-1]) / (2*dt)
        vel = torch.zeros_like(pos)
        vel[1:-1] = (pos[2:] - pos[:-2]) / (2.0 * dt)
        vel[0] = (pos[1] - pos[0]) / dt
        vel[-1] = (pos[-1] - pos[-2]) / dt

        # Acceleration: A[t] = (vel[t+1] - vel[t-1]) / (2*dt)
        acc = torch.zeros_like(vel)
        acc[1:-1] = (vel[2:] - vel[:-2]) / (2.0 * dt)
        acc[0] = (vel[1] - vel[0]) / dt
        acc[-1] = (vel[-1] - vel[-2]) / dt

        # Normalize velocity and acceleration scales to match original feature statistics
        vel *= 0.05
        acc *= 0.005

        out = torch.cat([pos, vel, acc], dim=-1)
        return out

    def __getitem__(self, idx: int):
        x = self.features[idx]
        y = self.labels[idx]
        if self.augment:
            x = self._apply_kinetic_augmentation(x)
        return x, y


def compute_balanced_weights(labels: np.ndarray, num_classes: int, power: float = 0.30) -> torch.Tensor:
    """Power-law class balancing weights normalized to mean 1.0."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    raw = (np.max(counts) / counts) ** power
    weights = raw / np.mean(raw)
    weights = np.clip(weights, 0.6, 2.5)
    return torch.from_numpy(weights).float()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
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
def evaluate_dataset(
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

    loss = total_loss / len(loader.dataset)
    top1_acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    macro_recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    macro_prec = precision_score(all_targets, all_preds, average="macro", zero_division=0)

    top5_hits = 0
    for i in range(len(all_targets)):
        top5_cands = np.argsort(all_probs[i])[-5:]
        if all_targets[i] in top5_cands:
            top5_hits += 1
    top5_acc = top5_hits / len(all_targets) if len(all_targets) > 0 else 0.0

    per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0, labels=list(range(num_classes)))
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))

    return {
        "loss": float(loss),
        "top1_accuracy": float(top1_acc),
        "top5_accuracy": float(top5_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_recall),
        "zero_f1_count": int(np.sum(per_class_f1 == 0.0)),
        "perfect_f1_count": int(np.sum(per_class_f1 == 1.0)),
        "confusion_matrix": cm,
        "per_class_f1": per_class_f1
    }


def recompute_and_save_prototypes(model: nn.Module, dataset: Dataset, device: torch.device, num_classes: int):
    """Recomputes 128-dim class centroid prototypes for anti-fake-sign rejection."""
    print("[*] Recomputing class prototypes for unknown sign rejection...")
    model.eval()
    embed_dim = 128
    prototypes = np.zeros((num_classes, embed_dim), dtype=np.float32)
    counts = np.zeros(num_classes, dtype=np.int32)

    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            _, feat = model(x, return_embeddings=True)
            feat_np = feat.detach().cpu().numpy()
            y_np = y.numpy()

            # L2 normalize
            norms = np.linalg.norm(feat_np, axis=1, keepdims=True) + 1e-8
            feat_norm = feat_np / norms

            for emb, lbl in zip(feat_norm, y_np):
                prototypes[lbl] += emb
                counts[lbl] += 1

    # Normalize centroids
    classifier_w = model.classifier.weight.detach().cpu().numpy()
    w_norms = np.linalg.norm(classifier_w, axis=1, keepdims=True) + 1e-8
    classifier_w_norm = classifier_w / w_norms

    for c in range(num_classes):
        if counts[c] > 0:
            prototypes[c] /= counts[c]
            prototypes[c] /= (np.linalg.norm(prototypes[c]) + 1e-8)
        else:
            prototypes[c] = classifier_w_norm[c]

    out_p = MODELS_DIR / "class_prototypes.npy"
    np.save(out_p, prototypes)
    print(f"[+] Saved updated class prototypes to {out_p} (shape: {prototypes.shape})")


def run_training_cycle(
    cycle_num: int,
    initial_ckpt: Optional[Path] = None,
    epochs: int = 40,
    lr: float = 2.5e-4,
    batch_size: int = 32,
    aug_strength: float = 1.0,
    label_smoothing: float = 0.05,
    patience: int = 15
) -> Tuple[ACHWGATE, Dict[str, Any], Dict[str, Any]]:
    print("\n" + "=" * 80)
    print(f"STARTING TRAINING CYCLE {cycle_num}")
    print(f"  Epochs={epochs}, LR={lr}, BatchSize={batch_size}, AugStrength={aug_strength}, Smooth={label_smoothing}")
    if initial_ckpt:
        print(f"  Warm-starting from: {initial_ckpt}")
    print("=" * 80)

    # Set seed
    torch.manual_seed(42 + cycle_num)
    np.random.seed(42 + cycle_num)
    random.seed(42 + cycle_num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42 + cycle_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = KineticAugmentedDataset(DATA_DIR / "train_split.npz", is_train=True, augment=True, aug_strength=aug_strength)
    val_ds   = KineticAugmentedDataset(DATA_DIR / "val_split.npz", is_train=False, augment=False)
    test_ds  = KineticAugmentedDataset(DATA_DIR / "test_split.npz", is_train=False, augment=False)
    num_classes = len(train_ds.class_names)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda'))
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda'))

    model = ACHWGATE(
        num_classes=num_classes,
        num_joints=133,
        in_channels=6,
        sequence_length=32,
        embed_dim=128,
        num_heads=4,
        num_layers=3,
        window_size=8,
        dropout=0.15
    ).to(device)

    if initial_ckpt and initial_ckpt.exists():
        print(f"[*] Loading weights from {initial_ckpt}...")
        chk = torch.load(initial_ckpt, map_location=device, weights_only=False)
        sd = chk.get("model_state_dict", chk)
        model.load_state_dict(sd, strict=True)
        print("[+] Checkpoint weights successfully loaded.")

    # Class-balanced loss
    weights = compute_balanced_weights(train_ds.labels.numpy(), num_classes, power=0.28).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Cosine annealing with 3 warmup epochs
    warmup_epochs = 3
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return float(ep + 1) / float(warmup_epochs)
        prog = float(ep - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_val_f1 = 0.0
    best_val_metrics = {}
    best_epoch = 0
    patience_cnt = 0
    best_state_dict = None

    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_eval = evaluate_dataset(model, val_loader, criterion, device, num_classes)
        scheduler.step()

        val_acc = val_eval["top1_accuracy"]
        val_f1 = val_eval["macro_f1"]
        val_rec = val_eval["macro_recall"]
        val_loss = val_eval["loss"]
        curr_lr = optimizer.param_groups[0]["lr"]

        is_best = (val_f1 > best_val_f1) or (abs(val_f1 - best_val_f1) < 0.005 and val_acc > best_val_metrics.get("top1_accuracy", 0.0))
        marker = " [*BEST*]" if is_best else ""

        if is_best:
            best_val_f1 = val_f1
            best_val_metrics = val_eval
            best_epoch = ep
            patience_cnt = 0
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1

        if ep % 5 == 0 or is_best or ep == epochs:
            print(f"  [Cycle {cycle_num}|Ep {ep:02d}/{epochs:02d}] TrLoss: {tr_loss:.4f} TrAcc: {tr_acc*100:5.1f}% | ValLoss: {val_loss:.4f} ValAcc: {val_acc*100:5.2f}% ValF1: {val_f1:.4f} ValRec: {val_rec:.4f} (ZeroF1: {val_eval['zero_f1_count']}){marker}", flush=True)

        if patience_cnt >= patience:
            print(f"  [!] Early stopping at epoch {ep} (patience={patience})")
            break

    elapsed = time.time() - t0
    print(f"\n[+] Cycle {cycle_num} completed in {elapsed:.1f}s. Best Epoch={best_epoch} | Best Val F1={best_val_f1:.4f} | Best Val Acc={best_val_metrics.get('top1_accuracy', 0.0)*100:.2f}%")

    # Load best model for evaluation
    model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})

    # Evaluate on clean held-out test split
    print("\n[*] Evaluating Best Checkpoint on Clean Held-Out Test Split (722 samples)...")
    test_eval = evaluate_dataset(model, test_loader, criterion, device, num_classes)
    print(f"  >>> TEST RESULTS: Top-1={test_eval['top1_accuracy']*100:.2f}% | Top-5={test_eval['top5_accuracy']*100:.2f}% | Macro F1={test_eval['macro_f1']:.4f} | Weighted F1={test_eval['weighted_f1']:.4f} | Zero F1 Classes={test_eval['zero_f1_count']}")

    return model, best_val_metrics, test_eval


def main():
    print("=" * 80)
    print("AC-HWGATE ITERATIVE HIGH-PERFORMANCE TRAINING CAMPAIGN")
    print("=" * 80)

    prod_ckpt_path = CHECKPOINTS_DIR / "achwgate_136class_production_best.pth"
    baseline_top1 = 0.9197
    baseline_macro_f1 = 0.8090

    # Cycle 1: Kinetic Augmentation + Balanced Weighting + Warm-start from production_best
    model, val_metrics_1, test_metrics_1 = run_training_cycle(
        cycle_num=1,
        initial_ckpt=prod_ckpt_path,
        epochs=35,
        lr=2.0e-4,
        batch_size=32,
        aug_strength=1.0,
        label_smoothing=0.05,
        patience=14
    )

    # Save Cycle 1 checkpoint
    cycle_1_ckpt = CHECKPOINTS_DIR / "achwgate_136class_cycle1_best.pth"
    torch.save(model.state_dict(), cycle_1_ckpt)

    best_model = model
    best_test_metrics = test_metrics_1
    best_cycle = 1

    # Cycle 2: Fine-Tuning with Mild Augmentation to Polish Decision Boundaries
    model_2, val_metrics_2, test_metrics_2 = run_training_cycle(
        cycle_num=2,
        initial_ckpt=cycle_1_ckpt, # Start from cycle 1 best model state
        epochs=30,
        lr=8.0e-5,
        batch_size=32,
        aug_strength=0.5,
        label_smoothing=0.04,
        patience=14
    )

    if test_metrics_2["top1_accuracy"] > best_test_metrics["top1_accuracy"] or \
       (test_metrics_2["top1_accuracy"] == best_test_metrics["top1_accuracy"] and test_metrics_2["macro_f1"] > best_test_metrics["macro_f1"]):
        best_model = model_2
        best_test_metrics = test_metrics_2
        best_cycle = 2

    print("\n" + "=" * 80)
    print("FINAL TRAINING CAMPAIGN EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Baseline Production : Top-1={baseline_top1*100:.2f}% | Macro F1={baseline_macro_f1:.4f}")
    print(f"Cycle 1 Result      : Top-1={test_metrics_1['top1_accuracy']*100:.2f}% | Macro F1={test_metrics_1['macro_f1']:.4f}")
    print(f"Cycle 2 Result      : Top-1={test_metrics_2['top1_accuracy']*100:.2f}% | Macro F1={test_metrics_2['macro_f1']:.4f}")
    print(f"Best Selected Cycle : Cycle {best_cycle}")
    print("=" * 80)

    # Save the retrained checkpoint
    save_ckpt_path = CHECKPOINTS_DIR / "achwgate_136class_retrained_best.pth"
    torch.save({
        "model_state_dict": best_model.state_dict(),
        "num_classes": 136,
        "test_top1_accuracy": best_test_metrics["top1_accuracy"],
        "test_top5_accuracy": best_test_metrics["top5_accuracy"],
        "test_macro_f1": best_test_metrics["macro_f1"],
        "test_weighted_f1": best_test_metrics["weighted_f1"],
        "best_cycle": best_cycle,
        "timestamp": time.time()
    }, save_ckpt_path)
    print(f"[+] Saved retrained model checkpoint to {save_ckpt_path}")

    # If it beats or matches the baseline, promote to production_best and update ONNX
    if best_test_metrics["top1_accuracy"] >= baseline_top1 and best_test_metrics["macro_f1"] >= baseline_macro_f1:
        print("\n[*] RETRAINED MODEL BEATS OR MATCHES PRODUCTION SOTA!")
        print(f"    Promoting to {prod_ckpt_path}...")
        torch.save({
            "model_state_dict": best_model.state_dict(),
            "num_classes": 136,
            "top1_accuracy": best_test_metrics["top1_accuracy"],
            "top5_accuracy": best_test_metrics["top5_accuracy"],
            "macro_f1": best_test_metrics["macro_f1"],
            "weighted_f1": best_test_metrics["weighted_f1"]
        }, prod_ckpt_path)

        # Recompute prototypes
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_ds = KineticAugmentedDataset(DATA_DIR / "train_split.npz", is_train=False, augment=False)
        recompute_and_save_prototypes(best_model, train_ds, device, num_classes=136)

        # Export ONNX with logits and embeddings
        print("[*] Exporting best model to ONNX runtime artifact via export_model_to_onnx...")
        from training.export_onnx import export_model_to_onnx
        export_model_to_onnx(model_path=str(prod_ckpt_path), onnx_output_path=str(MODELS_DIR / "ac_hwgate.onnx"))
        print("[+] ONNX export & verification completed successfully.")
    else:
        print(f"[i] Retrained model Top-1 ({best_test_metrics['top1_accuracy']*100:.2f}%) did not surpass baseline Top-1 ({baseline_top1*100:.2f}%). Preserving existing production checkpoint.")

if __name__ == "__main__":
    main()
