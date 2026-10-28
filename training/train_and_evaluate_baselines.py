import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DIR = os.path.join(ROOT_DIR, "training")
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "checkpoints")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

sys.path.insert(0, TRAINING_DIR)
from models.ac_hwgate import ACHWGATE

class FastDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=136, in_channels=798):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, return_embeddings=False):
        B, T = x.size(0), x.size(1)
        x_flat = x.view(B, T, -1).transpose(1, 2)
        feat = self.conv(x_flat).squeeze(-1)
        logits = self.fc(feat)
        if return_embeddings:
            return logits, feat
        return logits

class BaselineLSTM(nn.Module):
    def __init__(self, num_classes=136, in_channels=798, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_channels, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, return_embeddings=False):
        B, T = x.size(0), x.size(1)
        x_flat = x.view(B, T, -1)
        out, _ = self.lstm(x_flat)
        attn_w = torch.softmax(self.attention(out), dim=1)
        feat = torch.sum(out * attn_w, dim=1)
        logits = self.fc(feat)
        if return_embeddings:
            return logits, feat
        return logits

class BaselineGRU(nn.Module):
    def __init__(self, num_classes=136, in_channels=798, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(in_channels, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, return_embeddings=False):
        B, T = x.size(0), x.size(1)
        x_flat = x.view(B, T, -1)
        out, hn = self.gru(x_flat)
        feat = torch.cat([hn[-2], hn[-1]], dim=-1)
        logits = self.fc(feat)
        if return_embeddings:
            return logits, feat
        return logits

class BaselineSTGCN(nn.Module):
    def __init__(self, num_classes=136, num_joints=133, in_channels=6, embed_dim=256, dropout=0.2):
        super().__init__()
        self.joint_embed = nn.Linear(in_channels, 64)
        self.spatial_gcn = nn.Linear(64, embed_dim)
        self.temporal_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x, return_embeddings=False):
        B, T, J, C = x.shape
        h = F.relu(self.joint_embed(x))
        h = F.relu(self.spatial_gcn(h.mean(dim=2))) # [B, T, embed_dim]
        h_t = h.transpose(1, 2)
        feat = self.pool(F.relu(self.temporal_conv(h_t))).squeeze(-1)
        logits = self.fc(feat)
        if return_embeddings:
            return logits, feat
        return logits

def train_baseline(model, name, train_loader, val_loader, device, epochs=15, lr=1e-3):
    print(f"\n--- Training Baseline: {name} ({epochs} epochs, lr={lr}) ---")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_weights = None

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
            pred = out.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += len(y)
        scheduler.step()
        ep_time = time.time() - t0
        train_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_correct += (out.argmax(dim=-1) == y).sum().item()
                val_total += len(y)
        val_acc = val_correct / val_total
        print(f"Epoch {ep:02d}/{epochs:02d} [{ep_time:.1f}s] - Loss: {train_loss/total:.4f} - Train Acc: {train_acc*100:.2f}% - Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    ckpt_path = os.path.join(CHECKPOINTS_DIR, f"baseline_{name.lower().replace(' ', '_').replace('-', '_')}_136class_best.pth")
    torch.save({"model_state_dict": best_weights, "val_acc": best_val_acc, "num_classes": 136}, ckpt_path)
    print(f"Saved best {name} checkpoint (Val Acc: {best_val_acc*100:.2f}%) to {ckpt_path}")
    model.load_state_dict(best_weights)
    return model, ckpt_path

def evaluate_model_comprehensive(model, test_loader, class_names, device, name, ckpt_path):
    print(f"\n[+] Evaluating {name} on clean held-out test split (722 samples)...")
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_top5 = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            probs = torch.softmax(out, dim=-1)
            top1 = probs.argmax(dim=-1).cpu().numpy()
            top5 = torch.topk(probs, k=min(5, probs.shape[-1]), dim=-1).indices.cpu().numpy()

            all_preds.extend(top1.tolist())
            all_targets.extend(y.cpu().numpy().tolist())
            for i, target_cls in enumerate(y.cpu().numpy()):
                all_top5.append(bool(target_cls in top5[i]))

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    top1_acc = float(accuracy_score(y_true, y_pred))
    top5_acc = float(np.mean(all_top5))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    # Row normalized cm (percentage)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = (cm / row_sums).round(4)

    # Per class metrics
    per_class = []
    for c_idx, c_name in enumerate(class_names):
        mask_true = (y_true == c_idx)
        mask_pred = (y_pred == c_idx)
        support = int(mask_true.sum())
        tp = int((mask_true & mask_pred).sum())
        fp = int((~mask_true & mask_pred).sum())
        fn = int((mask_true & ~mask_pred).sum())

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class.append({
            "class_idx": c_idx,
            "class_name": c_name,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4)
        })

    # Most frequent confusion pairs
    confusion_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append({
                    "actual_idx": i,
                    "actual_class": class_names[i],
                    "predicted_idx": j,
                    "predicted_class": class_names[j],
                    "count": int(cm[i, j]),
                    "rate": round(float(cm[i, j] / row_sums[i, 0]), 4)
                })
    confusion_pairs.sort(key=lambda item: item["count"], reverse=True)

    # Latency benchmarking: 100 single item runs on device
    dummy_single = torch.randn(1, 32, 133, 6, device=device)
    # Warmup
    for _ in range(15):
        _ = model(dummy_single)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = model(dummy_single)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    p50_lat = float(np.percentile(latencies, 50))
    p95_lat = float(np.percentile(latencies, 95))
    fps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0

    total_params = sum(p.numel() for p in model.parameters())
    artifact_size_bytes = os.path.getsize(ckpt_path) if os.path.exists(ckpt_path) else (total_params * 4)
    artifact_size_mb = artifact_size_bytes / (1024 * 1024)

    print(f"    Top-1 Accuracy:  {top1_acc*100:.2f}% | Top-5: {top5_acc*100:.2f}%")
    print(f"    Macro F1:        {macro_f1*100:.2f}% | Weighted F1: {weighted_f1*100:.2f}%")
    print(f"    Macro Precision: {macro_prec*100:.2f}% | Macro Recall: {macro_rec*100:.2f}%")
    print(f"    Latency:         Mean {mean_lat:.2f}ms (p50: {p50_lat:.2f}ms, p95: {p95_lat:.2f}ms) | FPS: {fps:.1f}")
    print(f"    Params:          {total_params:,} | Size: {artifact_size_mb:.2f} MB")

    return {
        "model_name": name,
        "checkpoint": os.path.relpath(ckpt_path, ROOT_DIR) if os.path.exists(ckpt_path) else ckpt_path,
        "num_classes": len(class_names),
        "split": "data/unified_136class/test_split.npz",
        "sample_count": len(y_true),
        "top1_accuracy": round(top1_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "mean_latency_ms": round(mean_lat, 2),
        "p50_latency_ms": round(p50_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "throughput_fps": round(fps, 1),
        "parameters": total_params,
        "artifact_size_mb": round(artifact_size_mb, 2),
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "per_class": per_class,
        "confusion_matrix_raw": cm.tolist(),
        "confusion_matrix_norm": cm_norm.tolist(),
        "top_confusion_pairs": confusion_pairs[:30]
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Running on device: {device}")

    # Load splits
    train_data = np.load(os.path.join(ROOT_DIR, "data", "unified_136class", "train_split.npz"), allow_pickle=True)
    val_data = np.load(os.path.join(ROOT_DIR, "data", "unified_136class", "val_split.npz"), allow_pickle=True)
    test_data = np.load(os.path.join(ROOT_DIR, "data", "unified_136class", "test_split.npz"), allow_pickle=True)

    with open(os.path.join(ROOT_DIR, "configs", "authoritative_136_classes_manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    class_names = manifest["classes"] if isinstance(manifest, dict) and "classes" in manifest else manifest
    print(f"[+] Loaded {len(class_names)} classes from authoritative manifest")

    train_loader = DataLoader(FastDataset(train_data["features"], train_data["labels"]), batch_size=64, shuffle=True)
    val_loader = DataLoader(FastDataset(val_data["features"], val_data["labels"]), batch_size=64, shuffle=False)
    test_loader = DataLoader(FastDataset(test_data["features"], test_data["labels"]), batch_size=64, shuffle=False)

    trained_models = {}

    # 1. AC-HWGATE Production SOTA
    prod_ckpt = os.path.join(CHECKPOINTS_DIR, "achwgate_136class_production_best.pth")
    prod_model = ACHWGATE(num_classes=136, num_joints=133, in_channels=6, sequence_length=32)
    sd = torch.load(prod_ckpt, map_location=device, weights_only=False)
    prod_model.load_state_dict(sd.get("model_state_dict", sd))
    trained_models["AC-HWGATE (Production SOTA)"] = (prod_model, prod_ckpt)

    # 2. AC-HWGATE Previous Production
    prev_ckpt = os.path.join(CHECKPOINTS_DIR, "achwgate_136class_previous_production_best.pth")
    if os.path.exists(prev_ckpt):
        prev_model = ACHWGATE(num_classes=136, num_joints=133, in_channels=6, sequence_length=32)
        sd_prev = torch.load(prev_ckpt, map_location=device, weights_only=False)
        prev_model.load_state_dict(sd_prev.get("model_state_dict", sd_prev))
        trained_models["AC-HWGATE (Previous Production)"] = (prev_model, prev_ckpt)

    # Train baselines if not already trained
    baselines_to_run = [
        ("BiLSTM with Attention", BaselineLSTM(num_classes=136), 14, 1e-3),
        ("BiGRU (Lightweight)", BaselineGRU(num_classes=136), 14, 1e-3),
        ("1D-CNN (Temporal)", BaselineCNN(num_classes=136), 12, 1e-3),
        ("ST-GCN (Spatial-Temporal)", BaselineSTGCN(num_classes=136), 14, 1e-3),
    ]

    for b_name, b_inst, b_epochs, b_lr in baselines_to_run:
        b_ckpt = os.path.join(CHECKPOINTS_DIR, f"baseline_{b_name.lower().replace(' ', '_').replace('-', '_')}_136class_best.pth")
        if os.path.exists(b_ckpt):
            print(f"[+] Found cached checkpoint for {b_name} at {b_ckpt}")
            sd_b = torch.load(b_ckpt, map_location=device, weights_only=False)
            b_inst.load_state_dict(sd_b.get("model_state_dict", sd_b))
            trained_models[b_name] = (b_inst, b_ckpt)
        else:
            model_trained, saved_ckpt = train_baseline(b_inst, b_name, train_loader, val_loader, device, epochs=b_epochs, lr=b_lr)
            trained_models[b_name] = (model_trained, saved_ckpt)

    # Evaluate all models on clean held-out test split
    comparison_results = []
    confusion_matrix_bundle = {
        "dataset_split": "data/unified_136class/test_split.npz",
        "sample_count": len(test_data["labels"]),
        "num_classes": len(class_names),
        "class_labels": class_names,
        "models": {}
    }

    for model_name, (m_inst, m_ckpt) in trained_models.items():
        eval_dict = evaluate_model_comprehensive(m_inst, test_loader, class_names, device, model_name, m_ckpt)
        
        # Summary row for comparison table
        comparison_results.append({
            "model_name": eval_dict["model_name"],
            "checkpoint": eval_dict["checkpoint"],
            "num_classes": eval_dict["num_classes"],
            "top1_accuracy": eval_dict["top1_accuracy"],
            "top5_accuracy": eval_dict["top5_accuracy"],
            "macro_f1": eval_dict["macro_f1"],
            "weighted_f1": eval_dict["weighted_f1"],
            "macro_precision": eval_dict["macro_precision"],
            "macro_recall": eval_dict["macro_recall"],
            "mean_latency_ms": eval_dict["mean_latency_ms"],
            "p50_latency_ms": eval_dict["p50_latency_ms"],
            "p95_latency_ms": eval_dict["p95_latency_ms"],
            "throughput_fps": eval_dict["throughput_fps"],
            "parameters": eval_dict["parameters"],
            "artifact_size_mb": eval_dict["artifact_size_mb"],
            "evaluation_timestamp": eval_dict["evaluation_timestamp"]
        })

        # Matrix and per-class details
        confusion_matrix_bundle["models"][model_name] = {
            "top1_accuracy": eval_dict["top1_accuracy"],
            "top5_accuracy": eval_dict["top5_accuracy"],
            "macro_f1": eval_dict["macro_f1"],
            "per_class": eval_dict["per_class"],
            "confusion_matrix_raw": eval_dict["confusion_matrix_raw"],
            "confusion_matrix_norm": eval_dict["confusion_matrix_norm"],
            "top_confusion_pairs": eval_dict["top_confusion_pairs"]
        }

    # Save reports
    comp_path = os.path.join(REPORTS_DIR, "model_comparison_benchmark.json")
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "evaluation_split": "data/unified_136class/test_split.npz (722 samples)",
            "num_classes": len(class_names),
            "models": comparison_results
        }, f, indent=2)
    print(f"\n[+] Saved model comparison benchmark to {comp_path}")

    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix_136class.json")
    with open(cm_path, "w", encoding="utf-8") as f:
        json.dump(confusion_matrix_bundle, f, indent=2)
    print(f"[+] Saved 136x136 confusion matrix bundle to {cm_path}")

    print("\n" + "=" * 95)
    print("MODEL COMPARISON BENCHMARK TABLE (722 HELD-OUT TEST SAMPLES)")
    print("=" * 95)
    print(f"{'Model Name':<32} {'Top-1':<9} {'Top-5':<9} {'Macro F1':<10} {'Latency':<10} {'FPS':<8} {'Params':<10}")
    print("-" * 95)
    for r in comparison_results:
        print(f"{r['model_name']:<32} {r['top1_accuracy']*100:>6.2f}%  {r['top5_accuracy']*100:>6.2f}%  {r['macro_f1']*100:>7.2f}%   {r['mean_latency_ms']:>6.2f}ms  {r['throughput_fps']:>6.1f}  {r['parameters']:>9,}")
    print("=" * 95)

if __name__ == "__main__":
    main()
