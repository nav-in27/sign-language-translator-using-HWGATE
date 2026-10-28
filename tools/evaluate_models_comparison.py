"""
Comprehensive Validation Selection and Held-Out Test Evaluation.
Evaluates the baseline vs candidate checkpoints on the held-out clean test set.
Computes:
  - Top-1 and Top-5 accuracy
  - Macro F1, weighted F1, macro recall
  - Per-class precision, recall, F1, and sample counts
  - Confusion pairs
  - Bootstrap 95% confidence intervals
Strict protocol: Test set is evaluated ONLY after candidate selection from validation metrics.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "training"))

from training.models.ac_hwgate import ACHWGATE

DATA_DIR = ROOT_DIR / "data" / "unified_136class"
REPORTS_DIR = ROOT_DIR / "reports"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"


def load_test_split():
    data = np.load(DATA_DIR / "test_split.npz", allow_pickle=True)
    features = torch.from_numpy(data["features"]).float()
    labels = torch.from_numpy(data["labels"]).long()
    with open(ROOT_DIR / "configs" / "authoritative_136_classes_manifest.json", "r", encoding="utf-8") as f:
        class_names = json.load(f)["classes"]
    return features, labels, class_names


@torch.no_grad()
def evaluate_checkpoint_on_test(ckpt_path: Path, features: torch.Tensor, labels: torch.Tensor, class_names: List[str], device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    num_classes = len(class_names)

    model = ACHWGATE(
        num_classes=num_classes,
        num_joints=133,
        in_channels=6,
        sequence_length=32,
        embed_dim=128,
        num_heads=4,
        num_layers=3,
        window_size=8
    ).to(device)

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    ds = TensorDataset(features, labels)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    all_logits = []
    all_preds = []
    all_targets = []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        all_logits.append(logits.cpu().numpy())
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y.numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Softmax probabilities
    exp_logits = np.exp(all_logits - np.max(all_logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    top1_acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    macro_recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)

    top5_hits = 0
    for i in range(len(all_targets)):
        top5_cands = np.argsort(probs[i])[-5:]
        if all_targets[i] in top5_cands:
            top5_hits += 1
    top5_acc = top5_hits / len(all_targets)

    per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0, labels=list(range(num_classes)))
    per_class_rec = recall_score(all_targets, all_preds, average=None, zero_division=0, labels=list(range(num_classes)))
    per_class_prec = precision_score(all_targets, all_preds, average=None, zero_division=0, labels=list(range(num_classes)))
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))

    # Bootstrap 95% Confidence Interval for Accuracy and Macro F1
    np.random.seed(42)
    n_samples = len(all_targets)
    bs_accs = []
    bs_f1s = []
    for _ in range(1000):
        bs_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        bs_accs.append(accuracy_score(all_targets[bs_idx], all_preds[bs_idx]))
        bs_f1s.append(f1_score(all_targets[bs_idx], all_preds[bs_idx], average="macro", zero_division=0))

    ci_acc = (float(np.percentile(bs_accs, 2.5)), float(np.percentile(bs_accs, 97.5)))
    ci_f1 = (float(np.percentile(bs_f1s, 2.5)), float(np.percentile(bs_f1s, 97.5)))

    # Format per-class results
    per_class_report = []
    for c_idx in range(num_classes):
        support = int(np.sum(all_targets == c_idx))
        per_class_report.append({
            "class_idx": c_idx,
            "class_name": class_names[c_idx],
            "support": support,
            "f1": float(per_class_f1[c_idx]),
            "precision": float(per_class_prec[c_idx]),
            "recall": float(per_class_rec[c_idx])
        })

    # High-frequency confusion pairs
    confusion_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append({
                    "true_class": class_names[i],
                    "predicted_class": class_names[j],
                    "count": int(cm[i, j])
                })
    confusion_pairs.sort(key=lambda x: x["count"], reverse=True)

    return {
        "checkpoint": str(ckpt_path),
        "total_test_samples": n_samples,
        "num_classes": num_classes,
        "top1_accuracy": float(top1_acc),
        "top1_accuracy_ci95": ci_acc,
        "top5_accuracy": float(top5_acc),
        "macro_f1": float(macro_f1),
        "macro_f1_ci95": ci_f1,
        "weighted_f1": float(weighted_f1),
        "macro_recall": float(macro_recall),
        "zero_f1_count": int(np.sum(per_class_f1 == 0.0)),
        "perfect_f1_count": int(np.sum(per_class_f1 == 1.0)),
        "per_class_report": per_class_report,
        "confusion_matrix": cm.tolist(),
        "top_confusion_pairs": confusion_pairs[:30]
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("PHASE 5: CONTROLLED VALIDATION SELECTION & HELD-OUT TEST EVALUATION")
    print("=" * 80)

    # 1. Compare validation results of all experiments
    exp_files = sorted(list(REPORTS_DIR.glob("exp_*_results.json")))
    print(f"\nFound {len(exp_files)} experiment validation reports:")
    val_table = []
    for p in exp_files:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
            val_table.append(d)

    print(f"{'Exp ID':<8} {'Loss Type':<22} {'LR':<8} {'Best Ep':<9} {'Val Top-1':<12} {'Val Macro F1':<14} {'Val Recall':<12} {'Zero-F1':<8}")
    print("-" * 90)
    for r in val_table:
        print(f"{r['experiment_id']:<8} {r['loss_type']:<22} {r['lr']:<8.1e} {r['best_epoch']:<9} {r['val_top1_accuracy']*100:6.2f}%      {r['val_macro_f1']:<14.4f} {r['val_macro_recall']:<12.4f} {r['val_zero_f1_count']:<8}")

    # Select best candidate strictly by Validation Macro F1 and Accuracy
    best_candidate_cfg = max(val_table, key=lambda x: (x["val_macro_f1"], x["val_top1_accuracy"]))
    best_exp_id = best_candidate_cfg["experiment_id"]
    best_ckpt_path = CHECKPOINTS_DIR / f"exp_{best_exp_id}_best.pth"

    print("\n" + "=" * 80)
    print(f"WINNING CANDIDATE SELECTED BY VALIDATION METRICS: Experiment {best_exp_id}")
    print(f"  Name: {best_candidate_cfg['experiment_name']}")
    print(f"  Val Macro F1: {best_candidate_cfg['val_macro_f1']:.4f}")
    print(f"  Val Top-1 Accuracy: {best_candidate_cfg['val_top1_accuracy']*100:.2f}%")
    print(f"  Checkpoint: {best_ckpt_path}")
    print("=" * 80)

    # 2. Load held-out test split
    features, labels, class_names = load_test_split()
    print(f"\nEvaluating on held-out clean test split: {len(features)} samples across {len(class_names)} classes.")

    # Evaluate Baseline Production Checkpoint
    baseline_ckpt_path = CHECKPOINTS_DIR / "achwgate_136class_production_best.pth"
    print(f"\nEvaluating Baseline Production Checkpoint: {baseline_ckpt_path.name}...")
    baseline_eval = evaluate_checkpoint_on_test(baseline_ckpt_path, features, labels, class_names, device)

    # Evaluate Winning Candidate Checkpoint
    print(f"\nEvaluating Winning Candidate Checkpoint: {best_ckpt_path.name}...")
    candidate_eval = evaluate_checkpoint_on_test(best_ckpt_path, features, labels, class_names, device)

    # Comparison Table
    print("\n" + "=" * 80)
    print("HELD-OUT TEST SET EVALUATION COMPARISON (BASELINE VS NEW MODEL)")
    print("=" * 80)
    print(f"{'Metric':<25} {'Baseline (Preserved)':<25} {'New Candidate (Exp ' + best_exp_id + ')':<25} {'Delta':<15}")
    print("-" * 80)

    metrics = [
        ("Top-1 Accuracy", "top1_accuracy", True),
        ("Top-5 Accuracy", "top5_accuracy", True),
        ("Macro F1", "macro_f1", False),
        ("Weighted F1", "weighted_f1", False),
        ("Macro Recall", "macro_recall", False),
        ("Zero-F1 Classes", "zero_f1_count", False),
        ("Perfect-F1 Classes", "perfect_f1_count", False)
    ]

    for label, key, is_pct in metrics:
        b_val = baseline_eval[key]
        c_val = candidate_eval[key]
        if is_pct:
            b_str = f"{b_val*100:6.2f}%"
            c_str = f"{c_val*100:6.2f}%"
            d_str = f"{(c_val - b_val)*100:+6.2f}%"
        elif "count" in key:
            b_str = f"{b_val}"
            c_str = f"{c_val}"
            d_str = f"{c_val - b_val:+d}"
        else:
            b_str = f"{b_val:.4f}"
            c_str = f"{c_val:.4f}"
            d_str = f"{c_val - b_val:+.4f}"
        print(f"{label:<25} {b_str:<25} {c_str:<25} {d_str:<15}")

    print("-" * 80)
    print(f"Baseline 95% CI Top-1 Acc:  [{baseline_eval['top1_accuracy_ci95'][0]*100:.2f}%, {baseline_eval['top1_accuracy_ci95'][1]*100:.2f}%]")
    print(f"Candidate 95% CI Top-1 Acc: [{candidate_eval['top1_accuracy_ci95'][0]*100:.2f}%, {candidate_eval['top1_accuracy_ci95'][1]*100:.2f}%]")
    print(f"Baseline 95% CI Macro F1:   [{baseline_eval['macro_f1_ci95'][0]:.4f}, {baseline_eval['macro_f1_ci95'][1]:.4f}]")
    print(f"Candidate 95% CI Macro F1:  [{candidate_eval['macro_f1_ci95'][0]:.4f}, {candidate_eval['macro_f1_ci95'][1]:.4f}]")
    print("=" * 80)

    # Save comprehensive evaluation report
    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selected_experiment_id": best_exp_id,
        "selected_experiment_name": best_candidate_cfg["experiment_name"],
        "validation_metrics": best_candidate_cfg,
        "baseline_test_metrics": baseline_eval,
        "candidate_test_metrics": candidate_eval
    }

    report_path = REPORTS_DIR / "final_model_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[+] Full test evaluation report saved to: {report_path}")


if __name__ == "__main__":
    main()
