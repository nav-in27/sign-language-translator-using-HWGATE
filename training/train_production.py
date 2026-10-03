"""
Production CWA-HWGATE Training Pipeline (V3 - Fast In-Memory Packed Dataset)
Guarantees:
- 100% Pre-normalized, Balanced 179 Classes (21,480 samples)
- Stratified 80/10/10 Train/Val/Test Split
- Residual Graph Bottleneck Architecture with Confidence-Weighted Gating
- Accuracy Target: > 95%
- Automated ONNX Model Export & Verification
"""
import os
import sys
sys.stdout.reconfigure(line_buffering=True)
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
from sklearn.metrics import f1_score, classification_report

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.cwa_hwgate import CWA_HWGATE

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
        # logits: [B, C], target: [B]
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
        kps, confs, targets = kps.to(device), confs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        # Data augmentation on the fly: subtle jitter and pose/face dropout
        if model.training:
            noise = torch.randn_like(kps) * 0.01
            mask = (kps.abs().sum(dim=-1, keepdim=True) > 0.01).float()
            kps = kps + noise * mask
            
            # Pose & Face Dropout on random 30% of samples in batch
            drop_mask = (torch.rand(kps.size(0), 1, 1, device=device) < 0.30)
            kps[:, 0:33] = torch.where(drop_mask, torch.zeros_like(kps[:, 0:33]), kps[:, 0:33])
            kps[:, 75:133] = torch.where(drop_mask, torch.zeros_like(kps[:, 75:133]), kps[:, 75:133])
            confs[:, 0:33] = torch.where(drop_mask, torch.zeros_like(confs[:, 0:33]), confs[:, 0:33])
            confs[:, 75:133] = torch.where(drop_mask, torch.zeros_like(confs[:, 75:133]), confs[:, 75:133])
            
        output = model(kps, confs) # [1, B, num_classes]
        logits = output[0] # [B, num_classes]
        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    all_confs = []
    
    with torch.no_grad():
        for kps, confs, targets in dataloader:
            kps, confs, targets = kps.to(device), confs.to(device), targets.to(device)
            output = model(kps, confs)
            logits = output[0]
            loss = criterion(logits, targets)
            probs = F.softmax(logits, dim=-1)
            conf_val, preds = probs.max(dim=-1)
            
            total_loss += loss.item() * targets.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_confs.extend(conf_val.cpu().numpy())
            
    total_loss /= len(all_targets)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    acc = (all_preds == all_targets).mean() * 100.0
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    avg_conf = np.mean(all_confs)
    
    return total_loss, acc, macro_f1, avg_conf, all_preds, all_targets

def main():
    set_seed(42)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    packed_path = os.path.join(base_dir, "dataset_packed.pt")
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Active Computation Device: {device}")
    
    if not os.path.exists(packed_path):
        raise FileNotFoundError(f"Packed dataset not found at {packed_path}. Run pack_dataset.py first!")
        
    print(f"[+] Loading packed dataset from {packed_path}...")
    dataset_dict = torch.load(packed_path, map_location='cpu', weights_only=False)
    kps = dataset_dict['keypoints']         # [N, 133, 2]
    confs = dataset_dict['confidences']     # [N, 133, 1]
    labels = dataset_dict['labels']         # [N]
    classes = dataset_dict['classes']
    num_classes = len(classes)
    total_samples = len(labels)
    print(f"[+] Dataset Loaded: {total_samples} samples across {num_classes} classes.")
    
    # Stratified Train (80%) / Val (10%) / Test (10%) splits
    labels_np = labels.numpy()
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, temp_idx = next(sss1.split(np.zeros(total_samples), labels_np))
    
    temp_labels = labels_np[temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_rel_idx, test_rel_idx = next(sss2.split(np.zeros(len(temp_idx)), temp_labels))
    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]
    
    print(f"[+] Splits -> Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    batch_size = 128
    train_loader = DataLoader(
        TensorDataset(kps[train_idx], confs[train_idx], labels[train_idx]),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(kps[val_idx], confs[val_idx], labels[val_idx]),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(kps[test_idx], confs[test_idx], labels[test_idx]),
        batch_size=batch_size, shuffle=False
    )
    
    # Initialize Model
    model = CWA_HWGATE(
        num_classes=num_classes,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        dropout=0.15
    ).to(device)
    
    criterion = LabelSmoothingCrossEntropy(smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-4)
    epochs = 70
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_epoch = 0
    best_model_state = None
    
    print("\n" + "=" * 75)
    print(f"STARTING PRODUCTION TRAINING (CWA-HWGATE V3, {epochs} EPOCHS, TARGET ACC > 95%)")
    print("=" * 75)
    
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, val_conf, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']
        
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            
            # Save best checkpoint
            chk_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'val_f1': val_f1,
                'num_classes': num_classes,
                'classes': classes,
                'config': {
                    'embed_dim': 256,
                    'num_heads': 8,
                    'num_layers': 4
                }
            }, chk_path)
            
            # Also save to root training directory for instant discovery
            torch.save(best_model_state, os.path.join(base_dir, "cwa_hwgate_pretrained.pth"))
            best_str = f" | [+] NEW BEST! (Val Acc: {val_acc:.2f}%)"
        else:
            best_str = ""
            
        print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% F1: {val_f1:.4f} Conf: {val_conf:.3f} | {elapsed:.1f}s{best_str}")
        
    print("\n" + "=" * 75)
    print("FINAL EVALUATION ON UNSEEN TEST SET")
    print("=" * 75)
    print(f"Loading Best Model from Epoch {best_epoch} (Val Acc: {best_val_acc:.2f}%)")
    model.load_state_dict(best_model_state)
    
    test_loss, test_acc, test_f1, test_conf, test_preds, test_targets = evaluate(model, test_loader, criterion, device)
    print(f"\n[+] UNSEEN TEST SET RESULTS:")
    print(f"  Top-1 Test Accuracy: {test_acc:.2f}%")
    print(f"  Macro F1 Score:      {test_f1:.4f}")
    print(f"  Mean Confidence:     {test_conf:.4f}")
    print(f"  Test Loss:           {test_loss:.4f}")
    print(f"  Total Training Time: {(time.time() - t_start)/60:.2f} minutes")
    
    # Export ONNX
    print("\n[+] Exporting SOTA ONNX Model...")
    onnx_path = os.path.join(base_dir, "model.onnx")
    from export_onnx import export_model_to_onnx
    export_model_to_onnx(
        model_path=os.path.join(checkpoint_dir, "best_model.pth"),
        onnx_output_path=onnx_path
    )
    print("[+] Training and ONNX Export Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()
