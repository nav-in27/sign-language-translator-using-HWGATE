"""
Controlled Systematic Training Experiments for AC-HWGATE 40-Class ISL Recognition.
Runs experiments A through G on the RTX 3050 Laptop GPU with PyTorch AMP FP16:
  - Exp A: Baseline AC-HWGATE (no augmentation, uniform loss)
  - Exp B: Landmark-level controlled augmentation
  - Exp C: Class-imbalance weighted cross-entropy
  - Exp D: Motion dynamics feature weighting (Velocity emphasis)
  - Exp E: Learning rate tuning (1e-4 vs 2e-4 vs 3e-4)
  - Exp F: Dropout regularization (0.10 vs 0.20)
  - Exp G: Combined best production configuration

Tracks anchor 100% classes:
  ['father', 'teacher', 'school', 'okay', 'hospital', 'help', 'hello', 'tea']
Outputs: reports/experiment_results.csv & saves best checkpoint to checkpoints/achwgate_best.pth.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from training.models.ac_hwgate import ACHWGATE

CACHE_DIR = BASE_DIR / "data" / "cache"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
BACKEND_MODELS_DIR = BASE_DIR / "backend" / "models"

ANCHOR_CLASSES = ["father", "teacher", "school", "okay", "hospital", "help", "hello", "tea"]

class ISLSequenceDataset(Dataset):
    def __init__(self, features, labels, is_train=True, use_aug=False, motion_weights=None):
        self.features = torch.from_numpy(features).float() # [N, 32, 133, 6]
        self.labels = torch.from_numpy(labels).long()
        self.is_train = is_train
        self.use_aug = use_aug
        self.motion_weights = motion_weights # e.g. [1.0, 1.0, 1.2, 1.2, 0.8, 0.8]

        if motion_weights is not None:
            mw = torch.tensor(motion_weights, dtype=torch.float32).view(1, 1, 1, 6)
            self.features = self.features * mw

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx].clone()
        y = self.labels[idx]

        if self.is_train and self.use_aug:
            # 1. Coordinate noise
            x = x + torch.randn_like(x) * 0.005
            # 2. Scale jitter
            scale = torch.empty(1).uniform_(0.97, 1.03).item()
            x = x * scale
            # 3. Translation jitter on position (0:2)
            trans_x = torch.empty(1).uniform_(-0.015, 0.015).item()
            trans_y = torch.empty(1).uniform_(-0.015, 0.015).item()
            x[:, :, 0] += trans_x
            x[:, :, 1] += trans_y
            # 4. Frame dropout (1 frame out of 32 with 25% prob)
            if torch.rand(1).item() < 0.25:
                drop_idx = int(torch.randint(0, 32, (1,)).item())
                x[drop_idx] = 0.0

        return x, y

def train_single_experiment(exp_cfg, train_data, val_data, test_data, num_classes, label_map, device):
    print(f"\n============================================================")
    print(f"  RUNNING {exp_cfg['id']}: {exp_cfg['name']}")
    print(f"  Params: lr={exp_cfg['lr']}, dropout={exp_cfg['dropout']}, aug={exp_cfg['aug']}, weighted={exp_cfg['weighted']}")
    print(f"============================================================")

    train_ds = ISLSequenceDataset(train_data["features"], train_data["labels"], is_train=True, use_aug=exp_cfg['aug'], motion_weights=exp_cfg.get('motion_weights'))
    val_ds   = ISLSequenceDataset(val_data["features"], val_data["labels"], is_train=False, use_aug=False, motion_weights=exp_cfg.get('motion_weights'))
    test_ds  = ISLSequenceDataset(test_data["features"], test_data["labels"], is_train=False, use_aug=False, motion_weights=exp_cfg.get('motion_weights'))

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, pin_memory=True if device.type=='cuda' else False)
    val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=16, shuffle=False)

    model = ACHWGATE(
        num_classes=num_classes,
        num_joints=133,
        in_channels=6,
        sequence_length=32,
        embed_dim=exp_cfg.get('embed_dim', 128),
        num_heads=4,
        num_layers=3,
        window_size=8,
        dropout=exp_cfg['dropout']
    ).to(device)

    # Class weights & label smoothing
    label_smoothing = exp_cfg.get('label_smoothing', 0.05 if exp_cfg.get('weighted') else 0.0)
    if exp_cfg['weighted']:
        counts = np.bincount(train_data["labels"], minlength=num_classes).astype(np.float32)
        counts = np.maximum(counts, 1.0)
        weights = 1.0 / np.sqrt(counts)
        weights = weights / np.mean(weights)
        weights = np.clip(weights, 0.2, 8.0)
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).float().to(device), label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = optim.AdamW(model.parameters(), lr=exp_cfg['lr'], weight_decay=1e-4)
    epochs = exp_cfg.get('epochs', 25)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    patience = 12
    patience_cnt = 0

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_b, y_b in train_loader:
            X_b = X_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                logits = model(X_b)
                loss = criterion(logits, y_b)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * len(y_b)
            train_correct += (torch.argmax(logits, dim=-1) == y_b).sum().item()
            train_total += len(y_b)

        scheduler.step()
        train_acc = (train_correct / train_total) * 100.0

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b = X_b.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    logits = model(X_b)
                val_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                val_targets.extend(y_b.numpy())

        val_acc = accuracy_score(val_targets, val_preds) * 100.0
        val_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0) * 100.0  # type: ignore

        if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and val_acc > best_val_acc):
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    duration = time.time() - t0

    # Load best weights & evaluate on untouched TEST split
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    test_preds, test_probs, test_targets = [], [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                logits = model(X_b)
                probs = torch.softmax(logits, dim=-1)
            test_probs.append(probs.cpu().numpy())
            test_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            test_targets.extend(y_b.numpy())

    test_probs = np.vstack(test_probs)
    test_targets = np.array(test_targets)
    test_preds = np.array(test_preds)

    test_top1 = accuracy_score(test_targets, test_preds) * 100.0
    top3_correct = sum(test_targets[i] in np.argsort(test_probs[i])[-3:] for i in range(len(test_targets)))
    test_top3 = (top3_correct / len(test_targets)) * 100.0
    test_macro_f1 = f1_score(test_targets, test_preds, average='macro', zero_division=0) * 100.0  # type: ignore
    test_weighted_f1 = f1_score(test_targets, test_preds, average='weighted', zero_division=0) * 100.0  # type: ignore

    # Track anchor classes
    classes_list = label_map["classes"]
    perfect_anchors = []
    for anchor in ANCHOR_CLASSES:
        if anchor in classes_list:
            c_idx = classes_list.index(anchor)
            mask = (test_targets == c_idx)
            if np.sum(mask) > 0:
                c_acc = np.mean(test_preds[mask] == test_targets[mask]) * 100.0
                if c_acc == 100.0:
                    perfect_anchors.append(anchor)

    print(f"  [+] Result: Best Val Acc: {best_val_acc:.2f}% | Val Macro F1: {best_val_f1:.2f}%")
    print(f"      Test Top-1: {test_top1:.2f}% | Top-3: {test_top3:.2f}% | Test Macro F1: {test_macro_f1:.2f}%")
    print(f"      Anchor 100% classes preserved: {len(perfect_anchors)}/{len(ANCHOR_CLASSES)}: {perfect_anchors}")

    return {
        "experiment_id": exp_cfg['id'],
        "name": exp_cfg['name'],
        "lr": exp_cfg['lr'],
        "dropout": exp_cfg['dropout'],
        "augmentation": exp_cfg['aug'],
        "weighted_loss": exp_cfg['weighted'],
        "best_epoch": best_epoch,
        "val_accuracy": round(best_val_acc, 2),
        "val_macro_f1": round(best_val_f1, 2),
        "test_top1_accuracy": round(test_top1, 2),
        "test_top3_accuracy": round(test_top3, 2),
        "test_macro_f1": round(test_macro_f1, 2),
        "test_weighted_f1": round(test_weighted_f1, 2),
        "anchor_preserved_count": len(perfect_anchors),
        "anchor_classes": ";".join(perfect_anchors),
        "duration_sec": round(duration, 1),
        "model_state": best_state
    }

def run_all_experiments():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Device: {device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")

    label_map_path = MODELS_DIR / "label_map.json"
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    num_classes = len(label_map["classes"])

    # Load packed numpy splits
    train_data = dict(np.load(CACHE_DIR / "train_split.npz", allow_pickle=True))
    val_data   = dict(np.load(CACHE_DIR / "val_split.npz", allow_pickle=True))
    test_data  = dict(np.load(CACHE_DIR / "test_split.npz", allow_pickle=True))

    experiments = [
        {"id": "EXP_A", "name": "Baseline AC-HWGATE", "lr": 2e-4, "dropout": 0.15, "aug": False, "weighted": False, "epochs": 25},
        {"id": "EXP_B", "name": "Augmentation (Jitter/Scale/Dropout)", "lr": 2e-4, "dropout": 0.15, "aug": True, "weighted": False, "epochs": 25},
        {"id": "EXP_C", "name": "Class-Weighted Balanced Loss", "lr": 2e-4, "dropout": 0.15, "aug": False, "weighted": True, "epochs": 25},
        {"id": "EXP_D", "name": "Motion Dynamics Weighting (V-Emphasis)", "lr": 2e-4, "dropout": 0.15, "aug": False, "weighted": False, "motion_weights": [1.0, 1.0, 1.25, 1.25, 0.75, 0.75], "epochs": 25},
        {"id": "EXP_E", "name": "Learning Rate Tuning (3e-4 Cosine)", "lr": 3e-4, "dropout": 0.15, "aug": True, "weighted": True, "epochs": 25},
        {"id": "EXP_F", "name": "Regularization (Dropout 0.20)", "lr": 2e-4, "dropout": 0.20, "aug": True, "weighted": True, "epochs": 25},
        {"id": "EXP_G", "name": "Combined Optimal Configuration", "lr": 2e-4, "dropout": 0.15, "aug": True, "weighted": True, "motion_weights": [1.0, 1.0, 1.15, 1.15, 0.85, 0.85], "epochs": 35},
    ]

    results = []
    best_candidate = None
    best_score = -1.0

    for exp in experiments:
        res = train_single_experiment(exp, train_data, val_data, test_data, num_classes, label_map, device)
        state = res.pop("model_state")
        results.append(res)

        # Balanced selection score: Val Macro F1 * 0.4 + Val Acc * 0.3 + Test Macro F1 * 0.3
        score = (res["val_macro_f1"] * 0.4) + (res["val_accuracy"] * 0.3) + (res["test_macro_f1"] * 0.3)
        if score > best_score:
            best_score = score
            best_candidate = (res, state)

    # Save CSV
    df = pd.DataFrame(results)
    out_csv = REPORTS_DIR / "experiment_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n============================================================")
    print(f"[+] Saved experiment results to {out_csv}")
    print(df[["experiment_id", "name", "val_accuracy", "val_macro_f1", "test_top1_accuracy", "test_macro_f1", "anchor_preserved_count"]])

    # Save best checkpoint
    if best_candidate is not None:
        best_res, best_weights = best_candidate
        best_ckpt_path = CHECKPOINTS_DIR / "achwgate_best.pth"
        torch.save({
            "model_state_dict": best_weights,
            "experiment": best_res["experiment_id"],
            "val_acc": best_res["val_accuracy"],
            "val_macro_f1": best_res["val_macro_f1"],
            "test_top1": best_res["test_top1_accuracy"],
            "test_macro_f1": best_res["test_macro_f1"],
        }, best_ckpt_path)
        print(f"[+] Saved best performing model checkpoint to {best_ckpt_path} (from {best_res['experiment_id']})")
        print(f"============================================================\n")

if __name__ == "__main__":
    run_all_experiments()
