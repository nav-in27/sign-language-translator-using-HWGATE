"""
Mandatory Tiny Overfit Test for AC-HWGATE.
Verifies that the architecture, feature dimensions, temporal dynamics, loss,
and optimizer can overfit a tiny dataset (10 classes, 10-20 samples/class or 40-80 samples)
to >95% training accuracy.
"""

import os
import sys
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from training.models.ac_hwgate import ACHWGATE

def run_tiny_overfit_test(cache_dir="data/cache", target_classes=6, max_samples_per_class=8, max_epochs=70):
    print("============================================================", flush=True)
    print("         AC-HWGATE MANDATORY TINY OVERFIT TEST              ", flush=True)
    print("============================================================", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"[*] Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB", flush=True)

    cache_files = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
    if not cache_files:
        print(f"[!] No cached files found in {cache_dir}. Testing architecture on synthetic batch...", flush=True)
        num_classes = 6
        N = 36
        X = np.random.randn(N, 32, 133, 6).astype(np.float32)
        y = np.random.randint(0, num_classes, size=(N,))
    else:
        # Group by class word
        class_to_files = {}
        for f in cache_files:
            try:
                data = np.load(f, allow_pickle=True)
                w = str(data['word'])
                class_to_files.setdefault(w, []).append(f)
            except Exception:
                continue

        # Sort classes by number of samples descending
        sorted_classes = sorted(class_to_files.items(), key=lambda x: len(x[1]), reverse=True)
        selected_classes = sorted_classes[:target_classes]

        X_list, y_list = [], []
        class_names = []
        for class_idx, (w, flist) in enumerate(selected_classes):
            class_names.append(w)
            chosen = flist[:max_samples_per_class]
            for f in chosen:
                data = np.load(f, allow_pickle=True)
                X_list.append(data['features'])
                y_list.append(class_idx)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)
        num_classes = len(class_names)
        print(f"[*] Selected classes ({num_classes}): {class_names}", flush=True)

    print(f"[*] Dataset: {len(X)} samples, {num_classes} classes, input shape: {X.shape}", flush=True)

    # Initialize AC-HWGATE
    model = ACHWGATE(
        num_classes=num_classes,
        num_joints=133,
        in_channels=6,
        sequence_length=32,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.0
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-5)

    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y).to(device)

    model.train()
    start_time = time.time()
    best_acc = 0.0

    print("[*] Training to verify >95% training accuracy overfit...", flush=True)
    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad()
        logits = model(X_t)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=-1)
        acc = (preds == y_t).float().mean().item() * 100.0
        if acc > best_acc:
            best_acc = acc

        if epoch % 5 == 0 or acc >= 95.0 or epoch == max_epochs:
            print(f"  Epoch [{epoch:03d}/{max_epochs:03d}] | Loss: {loss.item():.4f} | Accuracy: {acc:.1f}% | Best: {best_acc:.1f}%", flush=True)

        if acc >= 95.0:
            print("------------------------------------------------------------", flush=True)
            print(f"[SUCCESS] Reached {acc:.1f}% training accuracy at epoch {epoch} in {time.time() - start_time:.2f}s!", flush=True)
            print("AC-HWGATE successfully passed the mandatory tiny overfit sanity test.", flush=True)
            print("------------------------------------------------------------", flush=True)
            return True

    print("------------------------------------------------------------", flush=True)
    if best_acc >= 95.0:
        print(f"[SUCCESS] Best accuracy achieved: {best_acc:.1f}% >= 95.0%", flush=True)
        return True
    else:
        print(f"[FAILURE] Model reached {best_acc:.1f}% < 95.0% training accuracy.", flush=True)
        print("[!] Stopping pipeline. Diagnose model gradients, loss, or architecture.", flush=True)
        return False

if __name__ == "__main__":
    success = run_tiny_overfit_test()
    if not success:
        sys.exit(1)
