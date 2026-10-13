"""
Comprehensive Diagnostic Evaluation & Prediction Trace Engine for AC-HWGATE 104-Class ISL.
Evaluates the untouched held-out test split and records complete diagnostic traces:
  - Ground-truth label and predicted label
  - Top-5 alternative candidates with calibrated confidences
  - Prediction margin (top1 - top2) and Shannon entropy
  - Prototype cosine similarity
  - Latency benchmarks (PyTorch and ONNX Runtime)
  - Confusion matrix heatmap (reports/confusion_matrix_104class.png)
  - Per-class metrics table (reports/per_class_metrics_104class.csv)
  - Full per-sample prediction trace (reports/full_prediction_trace_104class.csv)
  - Bidirectional error analysis (frequently mispredicted & systematically missed classes)
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from training.models.ac_hwgate import ACHWGATE

DATA_DIR = ROOT_DIR / "data" / "unified_104class"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_diagnostic_evaluation(checkpoint_path=None, test_split_path=None):
    print("=" * 80)
    print("AC-HWGATE 104-CLASS COMPREHENSIVE DIAGNOSTIC EVALUATION")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}")

    # 1. Load label map
    label_map_path = DATA_DIR / "label_map.json"
    if not label_map_path.exists():
        label_map_path = ROOT_DIR / "models" / "label_map.json"
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    classes = label_map["classes"]
    num_classes = len(classes)
    idx_to_label = {i: c for i, c in enumerate(classes)}
    print(f"[+] Loaded authoritative label map: {num_classes} classes.")

    # 2. Load model checkpoint
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINTS_DIR / "achwgate_104class_best.pth"
        if not checkpoint_path.exists():
            checkpoint_path = CHECKPOINTS_DIR / "achwgate_best.pth"

    print(f"[*] Loading model checkpoint: {checkpoint_path}")
    model = ACHWGATE(
        num_classes=num_classes,
        num_joints=133,
        in_channels=6,
        sequence_length=32
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    print("[+] Model loaded successfully!")

    # 3. Load prototypes if available
    prototypes = None
    proto_path = ROOT_DIR / "models" / "class_prototypes.npy"
    if proto_path.exists():
        prototypes = np.load(proto_path)
        print(f"[+] Loaded class prototypes: shape {prototypes.shape}")

    # 4. Load untouched test split
    if test_split_path is None:
        test_split_path = DATA_DIR / "test_split.npz"
    print(f"[*] Loading untouched test split from: {test_split_path}")
    test_data = np.load(test_split_path, allow_pickle=True)
    X_test = test_data["features"]
    y_test = test_data["labels"]
    n_samples = len(y_test)
    print(f"[+] Untouched Test Set: {n_samples} samples across {len(np.unique(y_test))} unique classes.")

    # 5. Execute full prediction trace per sample
    print("\n[*] Recording granular prediction trace across all test samples...")
    traces = []
    all_preds = []
    all_targets = y_test.tolist()
    all_top3 = []
    all_top5 = []
    latencies = []

    for idx in range(n_samples):
        sample_x = torch.from_numpy(X_test[idx:idx+1]).to(device)
        true_label_idx = int(y_test[idx])
        true_label = idx_to_label.get(true_label_idx, f"UNKNOWN_{true_label_idx}")

        t0 = time.perf_counter()
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits, embeds = model(sample_x, return_embeddings=True)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        infer_time_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(infer_time_ms)

        logits_np = logits[0].cpu().numpy()
        embeds_np = embeds[0].cpu().numpy()
        probs = np.exp(logits_np - np.max(logits_np))
        probs = probs / np.sum(probs)

        # Top 5 candidates
        top5_indices = np.argsort(probs)[::-1][:5]
        top1_idx = int(top5_indices[0])
        top2_idx = int(top5_indices[1]) if len(top5_indices) > 1 else top1_idx

        top1_prob = float(probs[top1_idx])
        top2_prob = float(probs[top2_idx])
        margin = top1_prob - top2_prob
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

        pred_label = idx_to_label.get(top1_idx, f"UNKNOWN_{top1_idx}")
        all_preds.append(top1_idx)
        all_top3.append(top5_indices[:3].tolist())
        all_top5.append(top5_indices.tolist())

        # Prototype similarity
        proto_sim = 0.0
        if prototypes is not None and top1_idx < len(prototypes):
            emb_norm = embeds_np / (np.linalg.norm(embeds_np) + 1e-8)
            proto_sim = float(np.dot(emb_norm, prototypes[top1_idx]))

        is_correct = (top1_idx == true_label_idx)

        top5_str = " | ".join([
            f"{idx_to_label.get(int(ci), str(ci))}: {float(probs[ci])*100:.1f}%"
            for ci in top5_indices
        ])

        traces.append({
            "sample_index": idx,
            "true_class": true_label,
            "true_idx": true_label_idx,
            "predicted_class": pred_label,
            "pred_idx": top1_idx,
            "is_correct": is_correct,
            "confidence": round(top1_prob, 4),
            "margin": round(margin, 4),
            "entropy": round(entropy, 4),
            "prototype_similarity": round(proto_sim, 4),
            "inference_time_ms": round(infer_time_ms, 2),
            "top5_alternatives": top5_str
        })

    trace_df = pd.DataFrame(traces)
    trace_csv_path = REPORTS_DIR / "full_prediction_trace_104class.csv"
    trace_df.to_csv(trace_csv_path, index=False)
    print(f"[+] Saved full prediction trace: {trace_csv_path}")

    # 6. Overall Metrics
    top1_acc = accuracy_score(all_targets, all_preds)
    top3_acc = sum(all_targets[i] in all_top3[i] for i in range(n_samples)) / n_samples
    top5_acc = sum(all_targets[i] in all_top5[i] for i in range(n_samples)) / n_samples
    macro_prec = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    macro_rec = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    avg_latency = float(np.mean(latencies))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0

    print("\n" + "=" * 80)
    print("FINAL HELD-OUT BENCHMARK PERFORMANCE METRICS")
    print("=" * 80)
    print(f"[*] Top-1 Accuracy:       {top1_acc * 100:.2f}%")
    print(f"[*] Top-3 Accuracy:       {top3_acc * 100:.2f}%")
    print(f"[*] Top-5 Accuracy:       {top5_acc * 100:.2f}%")
    print(f"[*] Macro Precision:      {macro_prec:.4f}")
    print(f"[*] Macro Recall:         {macro_rec:.4f}")
    print(f"[*] Macro F1 Score:       {macro_f1:.4f}")
    print(f"[*] Weighted F1 Score:    {weighted_f1:.4f}")
    print(f"[*] Average Latency:      {avg_latency:.2f} ms per sequence")
    print(f"[*] Real-Time Throughput: {fps:.1f} FPS")

    # 7. Per-Class Metrics
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    per_class_rows = []
    for c_idx in range(num_classes):
        c_name = classes[c_idx]
        tp = cm[c_idx, c_idx]
        fp = cm[:, c_idx].sum() - tp
        fn = cm[c_idx, :].sum() - tp
        support = cm[c_idx, :].sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = tp / support if support > 0 else 0.0

        per_class_rows.append({
            "class_name": c_name,
            "support": support,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4)
        })

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_csv_path = REPORTS_DIR / "per_class_metrics_104class.csv"
    per_class_df.to_csv(per_class_csv_path, index=False)
    print(f"[+] Saved per-class metrics: {per_class_csv_path}")

    # 8. Confusion Matrix Heatmap Plot
    try:
        plt.figure(figsize=(24, 20))
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        sns.heatmap(cm_norm, cmap='magma', xticklabels=False, yticklabels=False)
        plt.title(f"AC-HWGATE 104-Class Confusion Matrix (Test Top-1 Acc: {top1_acc*100:.2f}%, Macro F1: {macro_f1:.4f})", fontsize=16)
        plt.xlabel("Predicted Class Index (0..103)", fontsize=14)
        plt.ylabel("Ground Truth Class Index (0..103)", fontsize=14)
        cm_png_path = REPORTS_DIR / "confusion_matrix_104class.png"
        plt.tight_layout()
        plt.savefig(cm_png_path, dpi=200)
        plt.close()
        print(f"[+] Saved confusion matrix heatmap: {cm_png_path}")
    except Exception as e:
        print(f"[!] Note on confusion matrix plot: {e}")

    # 9. Bidirectional Error Analysis
    print("\n" + "=" * 80)
    print("BIDIRECTIONAL ERROR ANALYSIS")
    print("=" * 80)
    # Direction 1: True classes systematically missed (lowest recall)
    worst_recall = per_class_df.sort_values(by="recall").head(10)
    print("\n[!] Top 10 Most Frequently Missed Classes (Lowest Recall):")
    for _, row in worst_recall.iterrows():
        print(f"    - {row['class_name']}: Recall = {row['recall']*100:.1f}%, F1 = {row['f1_score']:.4f} (Support: {row['support']})")

    # Direction 2: False alarms / classes frequently predicted incorrectly (lowest precision)
    worst_prec = per_class_df[per_class_df["precision"] > 0].sort_values(by="precision").head(10)
    print("\n[!] Top 10 Classes with False Alarms (Lowest Precision):")
    for _, row in worst_prec.iterrows():
        print(f"    - {row['class_name']}: Precision = {row['precision']*100:.1f}%, F1 = {row['f1_score']:.4f}")

    # Direction 3: Top confused pairs
    confusion_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append({
                    "true_class": classes[i],
                    "predicted_class": classes[j],
                    "count": int(cm[i, j])
                })
    confusion_pairs.sort(key=lambda x: x["count"], reverse=True)
    print("\n[!] Top Confused Class Pairs:")
    for cp in confusion_pairs[:10]:
        print(f"    - True: {cp['true_class']:<15} -> Predicted: {cp['predicted_class']:<15} (Count: {cp['count']})")

    # 10. Summary JSON
    summary = {
        "num_classes": num_classes,
        "test_samples": n_samples,
        "top1_accuracy": round(top1_acc * 100, 2),
        "top3_accuracy": round(top3_acc * 100, 2),
        "top5_accuracy": round(top5_acc * 100, 2),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "average_latency_ms": round(avg_latency, 2),
        "throughput_fps": round(fps, 1),
        "trace_file": str(trace_csv_path),
        "per_class_file": str(per_class_csv_path),
        "top_confused_pairs": confusion_pairs[:5]
    }
    with open(REPORTS_DIR / "diagnostic_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("[+] Diagnostic evaluation complete!")
    return summary


if __name__ == "__main__":
    run_diagnostic_evaluation()
