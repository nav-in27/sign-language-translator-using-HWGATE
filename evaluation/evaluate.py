"""
Authoritative Evaluation Script for AC-HWGATE 40-Class ISL Recognition.
Computes:
  - Accuracy (Top-1, Top-3)
  - Macro Precision, Macro Recall, Macro F1
  - Weighted F1
  - Latency & FPS Benchmark on GPU and CPU
  - Confusion Matrix plot (reports/confusion_matrix.png)
  - Per-class metrics CSV (reports/per_class_metrics.csv)
  - Full metrics JSON (reports/final_metrics.json)
"""

import os
import sys

import sys
import os
from typing import Any, cast

# Ensure Windows cp1252 terminal never crashes on unicode
try:
    if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        getattr(sys.stdout, "reconfigure")(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, "reconfigure") and sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
        getattr(sys.stderr, "reconfigure")(encoding='utf-8', errors='replace')
except Exception:
    pass

import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from training.models.ac_hwgate import ACHWGATE

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

def evaluate_model(
    model_path=None,
    test_data=None,
    test_labels=None,
    test_signers=None,
    label_map_path=None,
    batch_size=16
):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("============================================================")
    print("           AC-HWGATE COMPREHENSIVE EVALUATION               ")
    print(f"[*] Compute Device: {device}")
    print("============================================================")

    # 1. Load Label Map
    if label_map_path is None:
        label_map_path = MODELS_DIR / "label_map.json"
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    classes = label_map["classes"]
    num_classes = len(classes)

    # 2. Load Model
    if model_path is None:
        model_path = CHECKPOINTS_DIR / "achwgate_best.pth"
    print(f"[*] Loading model checkpoint: {model_path}")
    model = ACHWGATE(num_classes=num_classes).to(device)

    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)
        print("[+] Checkpoint loaded successfully!")
    else:
        print(f"[!] Checkpoint not found at {model_path}. Using initialized model.")

    model.eval()

    # 3. Load Test Data
    if test_data is None or test_labels is None:
        test_split_path = BASE_DIR / "data" / "cache" / "test_split.npz"
        if test_split_path.exists():
            td = np.load(test_split_path, allow_pickle=True)
            test_data = td["features"]
            test_labels = td["labels"]
            test_signers = td.get("signers", ["unknown"] * len(test_labels))
        else:
            print("[ERROR] No test split provided or found on disk!")
            return {}

    N_test = len(test_data)
    unique_signers = len(set(test_signers)) if test_signers is not None else 0
    print(f"[*] Evaluating on {N_test} test videos across {unique_signers} signers...")

    # 4. Inference loop & latency measurement
    all_preds = []
    all_confs = []
    all_top3_matches = []
    latencies = []

    with torch.no_grad():
        for i in range(0, N_test, batch_size):
            batch_x = torch.from_numpy(test_data[i:i + batch_size]).to(device)
            batch_y = test_labels[i:i + batch_size]

            t0 = time.perf_counter()
            logits = model(batch_x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) / len(batch_x))

            probs = F.softmax(logits, dim=-1)
            confs, preds = torch.max(probs, dim=-1)

            # Top-3 accuracy
            top3 = torch.topk(probs, k=min(3, num_classes), dim=-1).indices.cpu().numpy()
            for true_y, top3_pred in zip(batch_y, top3):
                all_top3_matches.append(int(true_y in top3_pred))

            all_preds.extend(preds.cpu().numpy().tolist())
            all_confs.extend(confs.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_confs = np.array(all_confs)
    test_labels = np.array(test_labels)

    # 5. Core Metrics Computation
    top1_acc = float(accuracy_score(test_labels, all_preds) * 100.0)
    top3_acc = float(np.mean(all_top3_matches) * 100.0)
    macro_prec = float(precision_score(test_labels, all_preds, average='macro', zero_division=cast(Any, 0)) * 100.0)
    macro_rec = float(recall_score(test_labels, all_preds, average='macro', zero_division=cast(Any, 0)) * 100.0)
    macro_f1 = float(f1_score(test_labels, all_preds, average='macro', zero_division=cast(Any, 0)) * 100.0)
    weighted_f1 = float(f1_score(test_labels, all_preds, average='weighted', zero_division=cast(Any, 0)) * 100.0)

    avg_latency_ms = float(np.mean(latencies) * 1000.0)
    fps = float(1000.0 / avg_latency_ms) if avg_latency_ms > 0 else 0.0

    param_count = sum(p.numel() for p in model.parameters())
    model_size_mb = float(param_count * 4 / (1024 * 1024))

    print("------------------------------------------------------------")
    print(f"Top-1 Accuracy:     {top1_acc:.2f}%")
    print(f"Top-3 Accuracy:     {top3_acc:.2f}%")
    print(f"Macro Precision:    {macro_prec:.2f}%")
    print(f"Macro Recall:       {macro_rec:.2f}%")
    print(f"Macro F1 Score:     {macro_f1:.2f}%")
    print(f"Weighted F1 Score:  {weighted_f1:.2f}%")
    print(f"Average Latency:    {avg_latency_ms:.2f} ms")
    print(f"Throughput:         {fps:.1f} FPS")
    print(f"Model Parameters:   {param_count:,} ({model_size_mb:.2f} MB)")
    print("------------------------------------------------------------")

    # 6. Confusion Matrix & Per-Class Metrics
    cm = confusion_matrix(test_labels, all_preds, labels=list(range(num_classes)))

    # Per-class metrics table
    per_class_records = []
    for cls_idx, cls_name in enumerate(classes):
        true_mask = (test_labels == cls_idx)
        n_samples = int(np.sum(true_mask))
        if n_samples > 0:
            cls_acc = float(np.mean(all_preds[true_mask] == cls_idx) * 100.0)
        else:
            cls_acc = 0.0

        # Confused with
        confused_with = []
        if n_samples > 0:
            pred_classes = all_preds[true_mask]
            wrong_preds = pred_classes[pred_classes != cls_idx]
            if len(wrong_preds) > 0:
                counts = pd.Series(wrong_preds).value_counts()
                top_confused = [f"{classes[k]} ({v})" for k, v in counts.head(2).items()]
                confused_with = ", ".join(top_confused)

        per_class_records.append({
            "class_index": cls_idx,
            "class_name": cls_name,
            "test_samples": n_samples,
            "accuracy_percent": round(cls_acc, 2),
            "common_confusion": confused_with or "--"
        })

    per_class_df = pd.DataFrame(per_class_records)
    per_class_csv_path = REPORTS_DIR / "per_class_metrics.csv"
    per_class_df.to_csv(per_class_csv_path, index=False)
    print(f"[+] Saved per-class metrics to {per_class_csv_path}")

    # Plot Confusion Matrix
    plt.figure(figsize=(16, 14))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('AC-HWGATE Confusion Matrix (40 Classes)', fontsize=14, pad=15)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, classes, rotation=90, fontsize=8)
    plt.yticks(tick_marks, classes, fontsize=8)
    plt.tight_layout()
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)

    cm_path = REPORTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved confusion matrix plot to {cm_path}")

    # Top best and worst classes
    valid_records = [r for r in per_class_records if r["test_samples"] > 0]
    sorted_records = sorted(valid_records, key=lambda x: x["accuracy_percent"], reverse=True)
    best_classes = [{"class_name": r["class_name"], "accuracy_percent": r["accuracy_percent"]} for r in sorted_records[:10]]
    worst_classes = [{"class_name": r["class_name"], "accuracy_percent": r["accuracy_percent"]} for r in sorted_records[-10:]]

    final_metrics = {
        "model_name": "AC-HWGATE",
        "dataset_name": "vidit031/isl-isolated-40words",
        "num_classes": num_classes,
        "test_samples": N_test,
        "test_signers": unique_signers,
        "top1_accuracy": round(top1_acc, 2),
        "top3_accuracy": round(top3_acc, 2),
        "macro_precision": round(macro_prec, 2),
        "macro_recall": round(macro_rec, 2),
        "macro_f1": round(macro_f1, 2),
        "weighted_f1": round(weighted_f1, 2),
        "inference_latency_ms": round(avg_latency_ms, 2),
        "fps": round(fps, 1),
        "parameters": param_count,
        "model_size_mb": round(model_size_mb, 2),
        "best_classes": best_classes,
        "worst_classes": worst_classes
    }

    metrics_json_path = REPORTS_DIR / "final_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"[+] Saved final metrics JSON to {metrics_json_path}")

    return final_metrics

if __name__ == "__main__":
    evaluate_model()
