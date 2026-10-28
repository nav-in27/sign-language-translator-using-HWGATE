"""
Task 1: Complete Verification of Dataset, Reconciling 4796 Raw Videos vs 5466 Split Samples,
and Reproducing the 83.31% Test Evaluation from Artifacts and Checkpoint.
"""
import os
import sys
import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "training"))

from training.models.ac_hwgate import ACHWGATE

def step1_reconcile_samples():
    print("=" * 80)
    print("STEP 1: RECONCILING 4,796 RAW VIDEOS VS 5,466 SPLIT SAMPLES")
    print("=" * 80)
    
    train = np.load(ROOT_DIR / "data/unified_136class/train_split.npz", allow_pickle=True)
    val = np.load(ROOT_DIR / "data/unified_136class/val_split.npz", allow_pickle=True)
    test = np.load(ROOT_DIR / "data/unified_136class/test_split.npz", allow_pickle=True)

    n_tr = len(train["features"])
    n_va = len(val["features"])
    n_te = len(test["features"])
    total_samples = n_tr + n_va + n_te

    print(f"Train samples: {n_tr} ({n_tr/total_samples*100:.2f}%)")
    print(f"Val samples:   {n_va} ({n_va/total_samples*100:.2f}%)")
    print(f"Test samples:  {n_te} ({n_te/total_samples*100:.2f}%)")
    print(f"Total samples: {total_samples}")

    all_sources = list(train["sources"]) + list(val["sources"]) + list(test["sources"])
    src_counter = Counter(all_sources)
    print(f"\nSample Source Breakdown in NPZ splits:")
    for src, cnt in src_counter.items():
        print(f"  {src:<25}: {cnt} samples ({cnt/total_samples*100:.2f}%)")

    # Inspect data/additional_landmarks directly
    add_dir = ROOT_DIR / "data/additional_landmarks"
    all_landmark_files = list(add_dir.glob("*/*.npz"))
    print(f"\nTotal landmark NPZ files in data/additional_landmarks: {len(all_landmark_files)}")

    # Check why there are 5466 files vs 4796 raw videos:
    # Let's inspect file names to see where each came from
    patterns = Counter()
    for p in all_landmark_files:
        stem = p.stem
        if "_left_tilt" in stem or "_right_tilt" in stem or "_frontal" in stem or "WIN_" in stem:
            patterns["Dataset 2 (Video_Dataset with tilts)"] += 1
        elif stem.endswith("_0") or stem.endswith("_1") or stem.endswith("_2") or stem.endswith("_3") or stem.endswith("_4"):
            patterns["Augmented / Segmented window takes"] += 1
        elif "vivit" in stem or stem.startswith("MVI_") or stem.startswith("IMG_") or stem.startswith("take"):
            patterns["Dataset 1 (ProcessedData_vivit)"] += 1
        else:
            patterns["Other / Base take"] += 1

    print("\nFile Pattern Breakdown in data/additional_landmarks:")
    for pat, cnt in patterns.items():
        print(f"  {pat:<40}: {cnt}")

    return train, val, test


def step2_verify_split_independence(train, val, test):
    print("\n" + "=" * 80)
    print("STEP 2: VERIFYING SESSION GROUPING & ZERO DATA LEAKAGE")
    print("=" * 80)

    tr_sess = set(train["session_ids"])
    va_sess = set(val["session_ids"])
    te_sess = set(test["session_ids"])

    print(f"Unique Sessions - Train: {len(tr_sess)}, Val: {len(va_sess)}, Test: {len(te_sess)}")

    overlap_tr_va = tr_sess.intersection(va_sess)
    overlap_tr_te = tr_sess.intersection(te_sess)
    overlap_va_te = va_sess.intersection(te_sess)

    print(f"Session Overlap Train <-> Val : {len(overlap_tr_va)}")
    print(f"Session Overlap Train <-> Test: {len(overlap_tr_te)}")
    print(f"Session Overlap Val   <-> Test: {len(overlap_va_te)}")

    if len(overlap_tr_va) > 0:
        print(f"  [!] Note: Overlapping Train-Val sessions: {list(overlap_tr_va)[:5]}")
    if len(overlap_tr_te) > 0:
        print(f"  [!] Note: Overlapping Train-Test sessions: {list(overlap_tr_te)[:5]}")

    # Check exact feature duplicate overlap (SHA-256)
    import hashlib
    def get_hashes(feat_arr):
        return [hashlib.sha256(arr.round(decimals=3).tobytes()).hexdigest() for arr in feat_arr]

    print("\nComputing SHA-256 hashes of landmark feature tensors...")
    tr_h = set(get_hashes(train["features"]))
    va_h = set(get_hashes(val["features"]))
    te_h = set(get_hashes(test["features"]))

    feat_overlap_tr_va = tr_h.intersection(va_h)
    feat_overlap_tr_te = tr_h.intersection(te_h)
    feat_overlap_va_te = va_h.intersection(te_h)

    print(f"Feature Tensor Exact Overlap Train <-> Val : {len(feat_overlap_tr_va)}")
    print(f"Feature Tensor Exact Overlap Train <-> Test: {len(feat_overlap_tr_te)}")
    print(f"Feature Tensor Exact Overlap Val   <-> Test: {len(feat_overlap_va_te)}")


def step3_reproduce_test_evaluation(test):
    print("\n" + "=" * 80)
    print("STEP 3: REPRODUCING EXACT HELD-OUT TEST EVALUATION (RECALCULATING ALL METRICS)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on: {device}")

    # Load authoritative 136-class manifest
    manifest_p = ROOT_DIR / "configs/authoritative_136_classes_manifest.json"
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    classes = manifest["classes"]
    num_classes = len(classes)
    assert num_classes == 136, f"Expected 136 classes, got {num_classes}"

    # Load checkpoint
    ckpt_path = ROOT_DIR / "checkpoints/achwgate_136class_production_best.pth"
    assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = ACHWGATE(
        num_classes=num_classes,
        num_joints=133,
        in_channels=6,
        sequence_length=32,
        embed_dim=128,
        num_heads=4,
        num_layers=3
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    X_test = test["features"]
    y_test = test["labels"]
    
    batch_size = 64
    all_logits = []
    
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            xb = torch.tensor(X_test[i:i+batch_size], dtype=torch.float32, device=device)
            out = model(xb)
            all_logits.append(out.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = probs / np.sum(probs, axis=-1, keepdims=True)

    y_pred = np.argmax(probs, axis=-1)

    # Recalculate metrics
    top1_acc = accuracy_score(y_test, y_pred)
    
    # Top-5 Accuracy
    top5_hits = 0
    for i in range(len(y_test)):
        top5_cands = np.argsort(probs[i])[-5:]
        if y_test[i] in top5_cands:
            top5_hits += 1
    top5_acc = top5_hits / len(y_test)

    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0, labels=list(range(num_classes)))
    per_class_prec = precision_score(y_test, y_pred, average=None, zero_division=0, labels=list(range(num_classes)))
    per_class_rec = recall_score(y_test, y_pred, average=None, zero_division=0, labels=list(range(num_classes)))

    cm = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))

    print(f"\nRECALCULATED METRICS ON UNTOUCHED HELD-OUT TEST SPLIT (N={len(y_test)}):")
    print(f"  Top-1 Accuracy: {top1_acc * 100:.2f}%  (Claimed: 83.31%)")
    print(f"  Top-5 Accuracy: {top5_acc * 100:.2f}%  (Claimed: 93.67%)")
    print(f"  Weighted F1:    {weighted_f1:.4f}   (Claimed: 0.8234)")
    print(f"  Macro F1:       {macro_f1:.4f}   (Claimed: 0.6116)")

    # Compare with saved results JSON
    saved_res_p = ROOT_DIR / "reports/achwgate_136class_production_results.json"
    if saved_res_p.exists():
        with open(saved_res_p, "r", encoding="utf-8") as f:
            saved_d = json.load(f)
        print("\nDiscrepancy Check vs Saved JSON:")
        print(f"  Top-1 Acc Diff: {abs(top1_acc - saved_d['test_top1_accuracy']):.6f}")
        print(f"  Top-5 Acc Diff: {abs(top5_acc - saved_d['test_top5_accuracy']):.6f}")
        print(f"  Macro F1 Diff:  {abs(macro_f1 - saved_d['test_macro_f1']):.6f}")
        print(f"  Weighted F1 Diff:{abs(weighted_f1 - saved_d['test_weighted_f1']):.6f}")

    return {
        "classes": classes,
        "y_test": y_test,
        "y_pred": y_pred,
        "probs": probs,
        "per_class_f1": per_class_f1,
        "per_class_prec": per_class_prec,
        "per_class_rec": per_class_rec,
        "cm": cm
    }


def step4_inspect_weak_classes(eval_data):
    print("\n" + "=" * 80)
    print("STEP 4: DETAILED WEAK-CLASS REPORT & CONFUSION ANALYSIS")
    print("=" * 80)

    classes = eval_data["classes"]
    f1s = eval_data["per_class_f1"]
    precs = eval_data["per_class_prec"]
    recs = eval_data["per_class_rec"]
    y_test = eval_data["y_test"]
    cm = eval_data["cm"]

    test_counts = Counter(y_test)

    class_perf = []
    for i, c in enumerate(classes):
        support = test_counts.get(i, 0)
        class_perf.append({
            "idx": i,
            "class": c,
            "support": support,
            "f1": float(f1s[i]),
            "precision": float(precs[i]),
            "recall": float(recs[i])
        })

    class_perf.sort(key=lambda x: x["f1"])

    print("\n--- ZERO-F1 CLASSES (F1 = 0.0000) ---")
    zero_f1 = [x for x in class_perf if x["f1"] == 0.0]
    print(f"Total Zero-F1 Classes: {len(zero_f1)}")
    for x in zero_f1:
        # Find what they were confused with
        confused_with = []
        for j in range(len(classes)):
            if cm[x["idx"], j] > 0:
                confused_with.append(f"{classes[j]} ({cm[x['idx'], j]})")
        print(f"  {x['class']:<20} | Support: {x['support']:2d} | Prec: {x['precision']:.2f} | Rec: {x['recall']:.2f} | Misclassified as: {', '.join(confused_with)}")

    print("\n--- WEAK CLASSES (0 < F1 < 0.60) ---")
    weak = [x for x in class_perf if 0.0 < x["f1"] < 0.60]
    print(f"Total Weak Classes: {len(weak)}")
    for x in weak:
        confused_with = []
        for j in range(len(classes)):
            if cm[x["idx"], j] > 0 and j != x["idx"]:
                confused_with.append(f"{classes[j]} ({cm[x['idx'], j]})")
        print(f"  {x['class']:<20} | Support: {x['support']:2d} | F1: {x['f1']:.3f} | Prec: {x['precision']:.2f} | Rec: {x['recall']:.2f} | Confusions: {', '.join(confused_with)}")

    print("\n--- PERFECT CLASSES (F1 = 1.0000) ---")
    perfect = [x for x in class_perf if x["f1"] >= 0.999]
    print(f"Total Perfect Classes: {len(perfect)}")
    print(f"  {', '.join(x['class'] for x in perfect[:25])}...")

    # Root Cause Summary
    print("\nROOT CAUSE CATEGORIZATION:")
    insufficient_samples = [x for x in zero_f1 if x["support"] <= 2]
    print(f"  1. Severe Support Deficit in Test (Support <= 2): {len(insufficient_samples)} classes ({[x['class'] for x in insufficient_samples]})")
    
    # Save complete verification and weak-class report
    out_report_p = ROOT_DIR / "reports/weak_classes_detailed_audit.json"
    report_obj = {
        "total_classes": len(classes),
        "perfect_classes_count": len(perfect),
        "zero_f1_classes_count": len(zero_f1),
        "weak_classes_count": len(weak),
        "zero_f1_classes": zero_f1,
        "weak_classes": weak,
        "perfect_classes": perfect,
        "all_class_performance": class_perf
    }
    with open(out_report_p, "w", encoding="utf-8") as f:
        json.dump(report_obj, f, indent=2)
    print(f"\n[+] Detailed weak class audit saved to: {out_report_p}")


if __name__ == "__main__":
    train, val, test = step1_reconcile_samples()
    step2_verify_split_independence(train, val, test)
    eval_data = step3_reproduce_test_evaluation(test)
    step4_inspect_weak_classes(eval_data)
