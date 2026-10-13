"""
AC-HWGATE Production Training Pipeline v2.

Changes from v1:
- Uses corrected unified_v2 data (all 40 classes in all splits)
- Proper mixed-precision training (AMP)
- Gradient accumulation for effective batch size
- Better augmentation (temporal jitter, spatial noise, hand dropout)
- Class-balanced sampling via WeightedRandomSampler
- Cosine annealing with warm restarts
- Per-class metrics saved after each epoch
- ONNX export with dual outputs (logits + embeddings)
- Prototype computation from training data

Target: RTX 3050 4GB VRAM
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from collections import Counter

import functools
print = functools.partial(print, flush=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TRAINING_DIR)
sys.path.insert(0, TRAINING_DIR)

from training.models.ac_hwgate import ACHWGATE


class ISLDataset(Dataset):
    """Dataset with configurable augmentation for temporal skeleton sequences."""
    
    def __init__(self, X, y, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.X[idx].clone()  # [32, 133, 6]
        
        if self.augment:
            # 1. Spatial jitter (small noise to positions)
            pos_noise = torch.randn(x.shape[0], x.shape[1], 2) * 0.005
            x[:, :, :2] += pos_noise
            
            # 2. Scale augmentation (slight body size variation)
            scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.06
            x[:, :, :2] *= scale
            
            # 3. Temporal jitter (slight speed variation via resampling)
            if torch.rand(1).item() < 0.3:
                T = x.shape[0]
                speed = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
                new_indices = torch.linspace(0, T - 1, int(T * speed)).clamp(0, T - 1)
                if len(new_indices) > T:
                    # Subsample back to T
                    step = len(new_indices) / T
                    sel = [int(i * step) for i in range(T)]
                    new_indices = new_indices[sel]
                elif len(new_indices) < T:
                    # Pad to T by repeating last frame indices
                    pad = torch.full((T - len(new_indices),), float(T - 1))
                    new_indices = torch.cat([new_indices, pad])
                new_indices = new_indices[:T].long().clamp(0, T - 1)
                x = x[new_indices]
            
            # 4. Random hand dropout (simulate one hand not visible)
            if torch.rand(1).item() < 0.08:
                if torch.rand(1).item() < 0.5:
                    x[:, 33:54, :] = 0  # Drop left hand
                else:
                    x[:, 54:75, :] = 0  # Drop right hand
            
            # 5. Mirror augmentation (swap left/right hands)
            if torch.rand(1).item() < 0.15:
                lh = x[:, 33:54, :].clone()
                rh = x[:, 54:75, :].clone()
                x[:, 33:54, :] = rh
                x[:, 54:75, :] = lh
                # Flip x-coordinates for mirroring
                x[:, :, 0] = -x[:, :, 0]
                x[:, :, 2] = -x[:, :, 2]  # velocity x
                x[:, :, 4] = -x[:, :, 4]  # acceleration x
        
        return x, self.y[idx]


def evaluate(model, loader, device, criterion=None, class_names=None):
    """Evaluate model and return comprehensive metrics."""
    from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                                  recall_score, confusion_matrix, classification_report)
    
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            with autocast(enabled=(device.type == 'cuda')):
                logits = model(X_batch)
            if criterion:
                loss = criterion(logits.float(), y_batch)
                total_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    metrics = {
        "loss": total_loss / max(len(all_targets), 1) if criterion else 0.0,
        "accuracy": float(accuracy_score(all_targets, all_preds)),
        "macro_f1": float(f1_score(all_targets, all_preds, average='macro', zero_division=0)),  # type: ignore
        "weighted_f1": float(f1_score(all_targets, all_preds, average='weighted', zero_division=0)),  # type: ignore
        "macro_precision": float(precision_score(all_targets, all_preds, average='macro', zero_division=0)),  # type: ignore
        "macro_recall": float(recall_score(all_targets, all_preds, average='macro', zero_division=0)),  # type: ignore
        "preds": all_preds,
        "targets": all_targets,
    }
    
    if class_names:
        raw_report = classification_report(all_targets, all_preds, target_names=class_names, 
                                           output_dict=True, zero_division=0)  # type: ignore
        report: dict = raw_report if isinstance(raw_report, dict) else {}
        per_class = {}
        for cn in class_names:
            if cn in report and isinstance(report[cn], dict):
                per_class[cn] = {
                    "precision": report[cn]["precision"],
                    "recall": report[cn]["recall"],
                    "f1": report[cn]["f1-score"],
                    "support": report[cn]["support"]
                }
        metrics["per_class"] = per_class
        metrics["confusion_matrix"] = confusion_matrix(all_targets, all_preds).tolist()
    
    return metrics


def compute_prototypes(model, loader, device, num_classes):
    """Compute mean embedding prototypes per class for unknown-sign rejection."""
    model.eval()
    class_embeddings = {i: [] for i in range(num_classes)}
    
    embed_dim = 128
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            with autocast(enabled=(device.type == 'cuda')):
                _, embs = model(X_batch, return_embeddings=True)
            embs_np = embs.float().cpu().numpy()
            embed_dim = embs_np.shape[-1]
            for emb, label in zip(embs_np, y_batch.numpy()):
                class_embeddings[int(label)].append(emb)
    
    prototypes = np.zeros((num_classes, embed_dim), dtype=np.float32)
    for cls_idx in range(num_classes):
        if class_embeddings[cls_idx]:
            cls_embs = np.stack(class_embeddings[cls_idx])
            mean_emb = cls_embs.mean(axis=0)
            prototypes[cls_idx] = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
    
    return prototypes


def train(args):
    print("=" * 70)
    print("AC-HWGATE PRODUCTION TRAINING v2")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print(f"[+] Device: {device} ({gpu_name})")
    
    # Load data
    data_dir = os.path.join(ROOT_DIR, args.data_dir)
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    # Load label map
    label_map_path = os.path.join(data_dir, "label_map.json")
    if not os.path.exists(label_map_path):
        label_map_path = os.path.join(ROOT_DIR, "configs", "label_map.json")
    with open(label_map_path) as f:
        label_map = json.load(f)
    
    # Build class names in index order
    if "idx_to_label" in label_map:
        idx_to_label = label_map["idx_to_label"]
        num_classes = len(idx_to_label)
        class_names = [idx_to_label[str(i)] for i in range(num_classes)]
    else:
        # label_map is {name: idx}
        num_classes = len(label_map)
        idx_to_name = {v: k for k, v in label_map.items()}
        class_names = [idx_to_name[i] for i in range(num_classes)]
    
    print(f"[+] Data: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    print(f"[+] Classes: {num_classes}")
    
    # Class-balanced sampling with square-root smoothing (optimal for long-tailed datasets)
    train_counts = Counter(y_train.tolist())
    sample_weights = [(1.0 / train_counts[y]) ** 0.5 for y in y_train]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(y_train), replacement=True)
    
    # Optional mild loss weighting
    if args.use_class_weights:
        total_train = len(y_train)
        loss_weights = torch.zeros(num_classes, dtype=torch.float32)
        for c in range(num_classes):
            count = train_counts.get(c, 1)
            loss_weights[c] = (total_train / (num_classes * max(count, 1))) ** 0.5
        loss_weights = loss_weights.to(device)
    else:
        loss_weights = None
    
    # DataLoaders
    train_ds = ISLDataset(X_train, y_train, augment=True)
    val_ds = ISLDataset(X_val, y_val, augment=False)
    test_ds = ISLDataset(X_test, y_test, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model
    model = ACHWGATE(
        num_classes=num_classes,
        num_joints=133,
        in_channels=6,
        sequence_length=32,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        window_size=8,
        dropout=args.dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[+] Parameters: {total_params:,}")
    
    # Load pretrained weights if available
    if args.resume:
        ckpt_path = os.path.join(ROOT_DIR, args.resume)
        if os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                state = ckpt.get('model_state_dict', ckpt)
                model.load_state_dict(state, strict=False)
                print(f"[+] Resumed from {ckpt_path}")
            except Exception as e:
                print(f"[!] Could not resume: {e}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=args.min_lr
    )
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    
    # Directories
    ckpt_dir = os.path.join(ROOT_DIR, "checkpoints")
    results_dir = os.path.join(ROOT_DIR, "reports")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    best_val_metric = 0.0  # Track by macro_f1 (better for imbalanced classes)
    patience_counter = 0
    history = []
    
    print(f"\n[+] Training config:")
    print(f"    Epochs: {args.epochs}, Batch: {args.batch_size}, Accum: {args.grad_accum}")
    print(f"    Effective batch: {args.batch_size * args.grad_accum}")
    print(f"    LR: {args.lr}, Dropout: {args.dropout}, Label Smooth: {args.label_smoothing}")
    print(f"    Mixed Precision: {device.type == 'cuda'}")
    print()
    
    t_start = time.time()
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad()
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            with autocast(enabled=(device.type == 'cuda')):
                logits = model(X_batch)
                loss = criterion(logits, y_batch) / args.grad_accum
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * args.grad_accum * len(y_batch)
            preds = logits.argmax(dim=-1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
        
        scheduler.step()
        train_loss = epoch_loss / total
        train_acc = correct / total
        
        # Validate
        val_metrics = evaluate(model, val_loader, device, criterion, class_names)
        
        # Track best by macro F1 (rewards balanced class performance)
        val_metric = val_metrics["macro_f1"]
        improved = False
        
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            patience_counter = 0
            improved = True
            
            # Save best checkpoint
            torch.save({
                "model_name": "AC-HWGATE",
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "class_names": class_names,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metric,
                "epoch": epoch,
                "config": {
                    "num_joints": 133, "in_channels": 6, "sequence_length": 32,
                    "embed_dim": args.embed_dim, "num_heads": args.num_heads,
                    "num_layers": args.num_layers, "dropout": args.dropout
                }
            }, os.path.join(ckpt_dir, "achwgate_best.pth"))
        else:
            patience_counter += 1
        
        # Save last checkpoint
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "best_val_metric": best_val_metric,
        }, os.path.join(ckpt_dir, "achwgate_last.pth"))
        
        # Log
        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc * 100, 2),
            "val_loss": round(val_metrics["loss"], 4),
            "val_acc": round(val_metrics["accuracy"] * 100, 2),
            "val_macro_f1": round(val_metrics["macro_f1"] * 100, 2),
            "val_weighted_f1": round(val_metrics["weighted_f1"] * 100, 2),
            "lr": round(optimizer.param_groups[0]["lr"], 6),
        }
        history.append(entry)
        
        mark = " [* BEST]" if improved else ""
        if epoch % 2 == 0 or epoch == 1 or improved:
            print(f"Epoch {epoch:02d}/{args.epochs} | TrLoss: {train_loss:.4f} | TrAcc: {train_acc*100:5.1f}% | "
                  f"ValAcc: {val_metrics['accuracy']*100:5.1f}% | ValF1m: {val_metric*100:5.1f}%{mark}")
        
        # Early stopping
        if patience_counter >= args.patience and epoch >= 20:
            print(f"\n[!] Early stopping at epoch {epoch} (patience={args.patience})")
            break
    
    elapsed = time.time() - t_start
    print(f"\n[+] Training completed in {elapsed:.1f}s")
    print(f"[+] Best val macro F1: {best_val_metric*100:.2f}%")
    
    # Save training history
    with open(os.path.join(results_dir, "training_history_v2.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    # ================================================================
    # FINAL EVALUATION ON TEST SET
    # ================================================================
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON HELD-OUT TEST SET")
    print("=" * 70)
    
    # Load best checkpoint
    best_ckpt = torch.load(os.path.join(ckpt_dir, "achwgate_best.pth"), 
                           map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    
    test_metrics = evaluate(model, test_loader, device, criterion, class_names)
    
    print(f"Test Accuracy:      {test_metrics['accuracy']*100:.2f}%")
    print(f"Macro F1:           {test_metrics['macro_f1']*100:.2f}%")
    print(f"Weighted F1:        {test_metrics['weighted_f1']*100:.2f}%")
    print(f"Macro Precision:    {test_metrics['macro_precision']*100:.2f}%")
    print(f"Macro Recall:       {test_metrics['macro_recall']*100:.2f}%")
    
    # Per-class metrics
    print(f"\n{'Class':<15} {'Prec':>6} {'Recall':>6} {'F1':>6} {'N':>5}")
    print("-" * 42)
    worst_classes = []
    for cn in class_names:
        if cn in test_metrics.get("per_class", {}):
            pc = test_metrics["per_class"][cn]
            print(f"{cn:<15} {pc['precision']*100:5.1f}% {pc['recall']*100:5.1f}% "
                  f"{pc['f1']*100:5.1f}% {pc['support']:>5}")
            if pc['f1'] < 0.5:
                worst_classes.append((cn, pc['f1']))
    
    if worst_classes:
        print(f"\n[!] Classes below 50% F1: {[c[0] for c in worst_classes]}")
    
    # Save per-class metrics CSV
    with open(os.path.join(results_dir, "per_class_accuracy.csv"), "w") as f:
        f.write("class,precision,recall,f1,support\n")
        for cn in class_names:
            if cn in test_metrics.get("per_class", {}):
                pc = test_metrics["per_class"][cn]
                f.write(f"{cn},{pc['precision']:.4f},{pc['recall']:.4f},{pc['f1']:.4f},{pc['support']}\n")
    
    # Confusion matrix
    if "confusion_matrix" in test_metrics:
        cm = np.array(test_metrics["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(f"AC-HWGATE Confusion Matrix (Test Acc: {test_metrics['accuracy']*100:.1f}%)")
        plt.colorbar(im)
        tick_marks = np.arange(num_classes)
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names, fontsize=7)
        ax.set_ylabel('True ISL Sign')
        ax.set_xlabel('Predicted ISL Sign')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=200, bbox_inches='tight')
        plt.close()
    
    # Save final metrics
    final = {
        "model_name": "AC-HWGATE",
        "num_classes": num_classes,
        "train_samples": len(y_train),
        "val_samples": len(y_val),
        "test_samples": len(y_test),
        "test_accuracy": round(test_metrics["accuracy"] * 100, 2),
        "test_macro_f1": round(test_metrics["macro_f1"] * 100, 2),
        "test_weighted_f1": round(test_metrics["weighted_f1"] * 100, 2),
        "test_macro_precision": round(test_metrics["macro_precision"] * 100, 2),
        "test_macro_recall": round(test_metrics["macro_recall"] * 100, 2),
        "best_val_macro_f1": round(best_val_metric * 100, 2),
        "parameters": total_params,
        "training_time_sec": round(elapsed, 1),
        "worst_classes": [{"class": c, "f1": round(f, 3)} for c, f in sorted(worst_classes, key=lambda x: x[1])],
        "config": {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "dropout": args.dropout,
            "embed_dim": args.embed_dim, "num_heads": args.num_heads,
            "num_layers": args.num_layers, "label_smoothing": args.label_smoothing,
        }
    }
    with open(os.path.join(results_dir, "final_evaluation.json"), "w") as f:
        json.dump(final, f, indent=2)
    
    # ================================================================
    # COMPUTE PROTOTYPES
    # ================================================================
    print("\n[+] Computing class prototypes...")
    proto_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    prototypes = compute_prototypes(model, proto_loader, device, num_classes)
    np.save(os.path.join(ROOT_DIR, "models", "class_prototypes.npy"), prototypes)
    print(f"[+] Saved prototypes: {prototypes.shape}")
    
    # ================================================================
    # ONNX EXPORT WITH DUAL OUTPUTS
    # ================================================================
    print("\n[+] Exporting ONNX model with dual outputs (logits + embeddings)...")
    
    # Create wrapper that always returns both outputs
    class DualOutputWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            logits, embeddings = self.model(x, return_embeddings=True)
            return logits, embeddings
    
    wrapper = DualOutputWrapper(model)
    wrapper.eval()
    
    onnx_dir = os.path.join(ROOT_DIR, "backend", "models")
    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = os.path.join(onnx_dir, "ac_hwgate.onnx")
    
    dummy = torch.randn(1, 32, 133, 6, device=device)
    
    try:
        torch.onnx.export(
            wrapper, (dummy,), onnx_path,
            export_params=True, opset_version=14,
            do_constant_folding=True,
            input_names=['input_sequence'],
            output_names=['logits', 'embeddings'],
            dynamic_axes={
                'input_sequence': {0: 'batch'},
                'logits': {0: 'batch'},
                'embeddings': {0: 'batch'}
            }
        )
        print(f"[+] ONNX saved: {onnx_path} ({os.path.getsize(onnx_path)/1e6:.2f} MB)")
        
        # Verify
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path)
            out_names = [o.name for o in sess.get_outputs()]
            print(f"[+] ONNX outputs: {out_names}")
            ort_out = sess.run(None, {'input_sequence': dummy.cpu().numpy()})
            pt_logits, pt_emb = wrapper(dummy)
            diff_logits = float(np.max(np.abs(pt_logits.detach().cpu().numpy() - ort_out[0])))
            diff_emb = float(np.max(np.abs(pt_emb.detach().cpu().numpy() - ort_out[1])))
            print(f"[+] ONNX verification: logits diff={diff_logits:.2e}, emb diff={diff_emb:.2e}")
        except Exception as e:
            print(f"[!] ONNX verification skipped: {e}")
    except Exception as e:
        print(f"[!] ONNX export failed: {e}")
    
    # Also copy to models/ dir
    onnx_copy = os.path.join(ROOT_DIR, "models", "achwgate.onnx")
    import shutil
    if os.path.exists(onnx_path):
        shutil.copy2(onnx_path, onnx_copy)
    
    # Save authoritative label map to all required locations
    auth_label_map = {
        "label_to_idx": {cn: i for i, cn in enumerate(class_names)},
        "idx_to_label": {str(i): cn for i, cn in enumerate(class_names)},
        "classes": class_names,
        "num_classes": num_classes
    }
    # Add "thank you" alias if "thank_you" exists
    if "thank_you" in auth_label_map["label_to_idx"]:
        auth_label_map["label_to_idx"]["thank you"] = auth_label_map["label_to_idx"]["thank_you"]
    
    for path in [
        os.path.join(ROOT_DIR, "models", "label_map.json"),
        os.path.join(ROOT_DIR, "configs", "label_map.json"),
        os.path.join(ROOT_DIR, "backend", "label_map.json"),
    ]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(auth_label_map, f, indent=2)
    
    print("\n[+] All artifacts saved. Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AC-HWGATE Production Training v2")
    parser.add_argument("--data_dir", type=str, default="data/unified_v2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--use_class_weights", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    train(args)
