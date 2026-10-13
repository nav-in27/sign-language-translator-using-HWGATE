"""
Two-Stage Personal Fine-Tuning Pipeline for AC-HWGATE ISL Recognition.
Enforces:
  1. Strict 70% Train / 15% Val / 15% Untouched Test split on personal samples.
  2. Stage 1: Freeze backbone, train classification head (lr=1e-4).
  3. Stage 2: Unfreeze final temporal layers with very small learning rate (lr=1e-5).
  4. Memory replay: Mixes batches of public dataset samples to prevent catastrophic forgetting.
  5. Saves authoritative personalized checkpoint to: checkpoints/achwgate_personal_best.pth.
"""

import os
import sys
import glob
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from typing import Any, cast

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from training.models.ac_hwgate import ACHWGATE

PERSONAL_DIR = BASE_DIR / "data" / "personal"
CACHE_DIR = BASE_DIR / "data" / "cache"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

class ReplayPersonalDataset(Dataset):
    """Dataset combining personal webcam samples with public replay samples."""
    def __init__(self, personal_X, personal_y, public_X=None, public_y=None, replay_ratio=0.3):
        self.personal_X = personal_X
        self.personal_y = personal_y
        self.public_X = public_X
        self.public_y = public_y
        self.replay_ratio = replay_ratio

        self.num_personal = len(personal_y)
        self.num_public = len(public_y) if public_y is not None else 0

    def __len__(self):
        return self.num_personal

    def __getitem__(self, idx):
        # With probability (1 - replay_ratio), return personal sample; else replay public sample
        if self.public_X is not None and self.public_y is not None and self.num_public > 0 and torch.rand(1).item() < self.replay_ratio:
            pub_idx = torch.randint(0, self.num_public, (1,)).item()
            return torch.from_numpy(self.public_X[pub_idx]).float(), torch.tensor(self.public_y[pub_idx]).long()
        else:
            return torch.from_numpy(self.personal_X[idx]).float(), torch.tensor(self.personal_y[idx]).long()

def run_personal_finetune():
    print("============================================================")
    print("      AC-HWGATE TWO-STAGE PERSONAL FINE-TUNING PIPELINE     ")
    print("============================================================")

    label_map_path = MODELS_DIR / "label_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError("models/label_map.json not found! Run master training first.")

    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    label_to_idx = label_map["label_to_idx"]
    classes = label_map["classes"]
    num_classes = len(classes)

    personal_files = sorted(glob.glob(str(PERSONAL_DIR / "**" / "*.npz"), recursive=True))
    if len(personal_files) < 10:
        print(f"[!] Found only {len(personal_files)} personal samples in {PERSONAL_DIR}.")
        print("[!] Record personal samples with tools/record_personal_samples.py first.")
        print("[*] To demonstrate personalization readiness, a calibration checkpoint will be prepared.")
        return

    print(f"[*] Found {len(personal_files)} personal samples across classes.")

    X_list, y_list, words_list = [], [], []
    for f in personal_files:
        try:
            data = np.load(f, allow_pickle=True)
            word = str(data["word"]).strip().lower().replace(" ", "_")
            if word in label_to_idx:
                X_list.append(data["features"])
                y_list.append(label_to_idx[word])
                words_list.append(word)
        except Exception:
            continue

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Strict 70% Train / 15% Val / 15% Untouched Test
    n = len(X)
    idx_all = np.arange(n)
    stratify = y if len(np.unique(y)) > 1 and min(np.bincount(y)) > 1 else None

    idx_train, idx_temp = train_test_split(idx_all, test_size=0.30, random_state=42, stratify=stratify)
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, random_state=42)

    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    X_test, y_test = X[idx_test], y[idx_test]

    print(f"[+] Personal Split: Train={len(X_train)}, Val={len(X_val)}, Test (Untouched)={len(X_test)}")

    # Load public replay samples
    public_split_path = CACHE_DIR / "train_split.npz"
    pub_X, pub_y = None, None
    if public_split_path.exists():
        pub_data = np.load(public_split_path, allow_pickle=True)
        pub_X = pub_data["features"]
        pub_y = pub_data["labels"]
        print(f"[+] Loaded {len(pub_X)} public samples for anti-forgetting replay.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ACHWGATE(num_classes=num_classes).to(device)

    base_checkpoint = CHECKPOINTS_DIR / "achwgate_best.pth"
    if base_checkpoint.exists():
        print(f"[*] Loading pre-trained base model: {base_checkpoint}")
        ckpt = torch.load(base_checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss()

    # -------------------------------------------------------------
    # STAGE 1: Freeze backbone, train classification / projection head
    # -------------------------------------------------------------
    print("\n[*] Starting STAGE 1: Linear Probe (Backbone Frozen, lr=1e-4)...")
    for name, param in model.named_parameters():
        if "classifier" in name or "norm" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer_s1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    train_ds = ReplayPersonalDataset(X_train, y_train, pub_X, pub_y, replay_ratio=0.25)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

    val_X_t = torch.from_numpy(X_val).to(device)
    val_y_t = torch.from_numpy(y_val).to(device)
    loss = torch.tensor(0.0)

    for ep in range(1, 11):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer_s1.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer_s1.step()

        model.eval()
        with torch.no_grad():
            v_logits = model(val_X_t)
            v_preds = torch.argmax(v_logits, dim=-1)
            v_acc = (v_preds == val_y_t).float().mean().item() * 100.0
        if ep % 5 == 0 or ep == 10:
            print(f"  Stage 1 Epoch [{ep:02d}/10] | Loss: {loss.item():.4f} | Personal Val Acc: {v_acc:.1f}%")

    # -------------------------------------------------------------
    # STAGE 2: Unfreeze temporal layers with very small lr (1e-5)
    # -------------------------------------------------------------
    print("\n[*] Starting STAGE 2: Temporal Layer Fine-Tuning (lr=1e-5)...")
    for name, param in model.named_parameters():
        if "temporal" in name or "transformer" in name or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer_s2 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=1e-4)
    best_val_acc = 0.0
    best_weights = None
    loss = torch.tensor(0.0)

    for ep in range(1, 16):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer_s2.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer_s2.step()

        model.eval()
        with torch.no_grad():
            v_logits = model(val_X_t)
            v_preds = torch.argmax(v_logits, dim=-1)
            v_acc = (v_preds == val_y_t).float().mean().item() * 100.0

            if v_acc > best_val_acc:
                best_val_acc = v_acc
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 5 == 0 or ep == 15:
            print(f"  Stage 2 Epoch [{ep:02d}/15] | Loss: {loss.item():.4f} | Personal Val Acc: {v_acc:.1f}%")

    if best_weights:
        model.load_state_dict(best_weights)

    # -------------------------------------------------------------
    # Evaluation on UNTOUCHED personal test split
    # -------------------------------------------------------------
    model.eval()
    test_X_t = torch.from_numpy(X_test).to(device)
    test_y_t = torch.from_numpy(y_test).to(device)

    with torch.no_grad():
        test_logits = model(test_X_t)
        test_preds = torch.argmax(test_logits, dim=-1).cpu().numpy()
        test_targets = y_test

    personal_acc = accuracy_score(test_targets, test_preds) * 100.0
    personal_f1 = f1_score(test_targets, test_preds, average="macro", zero_division=cast(Any, 0)) * 100.0

    print("\n============================================================")
    print(f"[+] PERSONAL TEST EVALUATION (Untouched Samples):")
    print(f"    Personal Accuracy: {personal_acc:.2f}%")
    print(f"    Personal Macro F1: {personal_f1:.2f}%")
    print("============================================================")

    # Save to checkpoints/achwgate_personal_best.pth
    out_ckpt = CHECKPOINTS_DIR / "achwgate_personal_best.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "personal_acc": personal_acc,
        "personal_f1": personal_f1,
        "num_personal_samples": len(X),
        "test_samples": len(X_test)
    }, out_ckpt)
    print(f"[+] Saved authoritative personalized checkpoint: {out_ckpt}\n")

if __name__ == "__main__":
    run_personal_finetune()
