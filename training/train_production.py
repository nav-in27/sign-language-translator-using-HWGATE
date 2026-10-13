"""
Production AC-HWGATE Training Pipeline.
Architecture: Adaptive Context-Aware Hierarchical Windowed Graph Attention Transformer Encoder (AC-HWGATE).
Dataset: ISL500 / ISL-DATA (500 ISL Word Classes, Temporal MediaPipe Sequences).

Guarantees:
- Strictly Temporal Signing Motion (NO single-frame T=1 conversion).
- 100% Pre-normalized [32, 133, 6] tensors with Position, Velocity, Acceleration dynamics.
- Shared FeatureExtractor parity with Real-Time Webcam Inference.
- Automated ONNX Model Export & Verification.
"""
import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    getattr(sys.stdout, "reconfigure")(line_buffering=True)
import json
import time
import copy
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)
from models.ac_hwgate import ACHWGATE
from export_onnx import export_model_to_onnx

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=-1)
        num_classes = logits.size(-1)
        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for kps, confs, targets in dataloader:
        kps, targets = kps.to(device), targets.to(device)
        optimizer.zero_grad()
        
        # Subtle temporal sequence jitter augmentation
        if model.training:
            jitter = torch.randn_like(kps) * 0.005
            scale = 1.0 + (torch.rand(1, device=device) - 0.5) * 0.02
            kps = (kps + jitter) * scale
            
        logits = model(kps) # [B, num_classes]
        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        
        total_loss += loss.item() * targets.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        
    return total_loss / total, (correct / total) * 100.0

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for kps, confs, targets in dataloader:
            kps, targets = kps.to(device), targets.to(device)
            logits = model(kps)
            loss = criterion(logits, targets)
            total_loss += loss.item() * targets.size(0)
            
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    total = len(all_targets)
    acc = accuracy_score(all_targets, all_preds) * 100.0
    f1 = float(f1_score(all_targets, all_preds, average='macro', zero_division=0))  # type: ignore
    return total_loss / total, acc, f1

def run_production_training(epochs=35, batch_size=32, lr=3e-3):
    set_seed(42)
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70, flush=True)
    print("STARTING AC-HWGATE PRODUCTION TRAINING (ISL500 TEMPORAL DATASET)", flush=True)
    print(f"Device: {device} | Threads: {torch.get_num_threads()} | Epochs: {epochs} | Batch Size: {batch_size}", flush=True)
    print("=" * 70, flush=True)
    
    packed_file = os.path.join(BASE_DIR, "dataset_packed.pt")
    if not os.path.exists(packed_file):
        raise FileNotFoundError(f"Packed dataset not found at {packed_file}. Run pack_isl500.py first!")
        
    data = torch.load(packed_file, map_location='cpu', weights_only=False)
    kps = data['keypoints']      # [N, 32, 133, 6]
    confs = data['confidences']  # [N, 32, 133, 1]
    labels = data['labels']      # [N]
    classes = data['classes']    # 500 class strings
    num_classes = len(classes)
    total_samples = len(labels)
    
    print(f"[+] Loaded ISL500 dataset: {total_samples} temporal motion sequences across {num_classes} classes", flush=True)
    print(f"[+] Tensor shape: {kps.shape} (T=32, J=133, C=6 [P, V, A])", flush=True)
    
    # Train/Val split: 85% train, 15% validation
    num_samples = len(labels)
    test_size = max(int(num_samples * 0.15), 1)
    
    perm = np.random.RandomState(42).permutation(num_samples)
    train_idx = perm[test_size:]
    val_idx = perm[:test_size]
        
    train_loader = DataLoader(
        TensorDataset(kps[train_idx], confs[train_idx], labels[train_idx]),
        batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        TensorDataset(kps[val_idx], confs[val_idx], labels[val_idx]),
        batch_size=batch_size, shuffle=False
    )
    print(f"[+] Splits: Train={len(train_idx)}, Val={len(val_idx)}", flush=True)
    
    # Initialize AC-HWGATE
    model = ACHWGATE(
        num_classes=num_classes,
        num_joints=133,
        in_channels=6,
        sequence_length=32,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[+] Model Architecture: AC-HWGATE ({total_params:,} parameters, {total_params*4/(1024*1024):.2f} MB)", flush=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    
    best_val_acc = -1.0
    best_val_loss = float('inf')
    best_model_weights = None
    chk_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(chk_dir, exist_ok=True)
    best_chk_path = os.path.join(chk_dir, "ac_hwgate_best.pth")
    
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        ep_time = time.time() - ep_t0
        
        is_best = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss) or (epoch == epochs and best_model_weights is None)
        if is_best or best_model_weights is None:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save({
                "model_state_dict": best_model_weights,
                "num_classes": num_classes,
                "classes": classes,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "epoch": epoch,
                "model_name": "AC-HWGATE",
                "config": {
                    "embed_dim": 128,
                    "num_heads": 4,
                    "num_layers": 2,
                    "sequence_length": 32,
                    "in_channels": 6
                }
            }, best_chk_path)
            
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({ep_time:.1f}s) Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | F1: {val_f1:.4f} {'[BEST]' if is_best else ''}", flush=True)
            
    total_time = time.time() - t0
    print("=" * 70, flush=True)
    print(f"[+] Training complete in {total_time:.2f}s | Best Validation Accuracy: {best_val_acc:.2f}%", flush=True)
    print(f"[+] Best checkpoint saved to {best_chk_path}", flush=True)
    print("=" * 70, flush=True)
    
    # Export to ONNX
    print("[+] Exporting trained AC-HWGATE model to ONNX...", flush=True)
    export_model_to_onnx(model_path=best_chk_path)
    
    # Compute class prototypes
    print("[+] Re-computing authoritative class prototypes...", flush=True)
    try:
        from compute_prototypes import compute_prototypes
        compute_prototypes()
    except Exception as pe:
        print(f"[!] Warning computing prototypes: {pe}", flush=True)
        
    print("[+] Production pipeline ready: AC-HWGATE active.", flush=True)

if __name__ == "__main__":
    run_production_training()
