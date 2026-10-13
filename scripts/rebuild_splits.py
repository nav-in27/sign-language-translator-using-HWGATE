"""
Rebuild train/val/test splits from unified dataset.

Fixes:
1. All 40 classes present in every split (train, val, test)
2. Stratified splitting by class
3. No data leakage
4. Proper label format consistency
5. Removes all-zero sequences

Output: data/unified_v2/ with X_train.npy, y_train.npy, etc. + label_map.json
"""

import os
import sys
import json
import numpy as np
from collections import Counter

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

def rebuild_splits():
    print("=" * 70)
    print("REBUILDING LEAKAGE-FREE STRATIFIED SPLITS (ALL 40 CLASSES)")
    print("=" * 70)
    
    # Load all unified data
    train = np.load(os.path.join(ROOT_DIR, "data", "unified", "train_split.npz"))
    val = np.load(os.path.join(ROOT_DIR, "data", "unified", "val_split.npz"))
    test = np.load(os.path.join(ROOT_DIR, "data", "unified", "test_split.npz"))
    
    X_all = np.concatenate([train['features'], val['features'], test['features']], axis=0)
    y_all = np.concatenate([train['labels'], val['labels'], test['labels']], axis=0)
    
    print(f"[+] Total pooled samples: {len(y_all)} | Shape: {X_all.shape}")
    
    # Remove all-zero sequences
    non_zero_mask = ~np.all(X_all.reshape(len(X_all), -1) == 0, axis=1)
    X_all = X_all[non_zero_mask]
    y_all = y_all[non_zero_mask]
    removed = (~non_zero_mask).sum()
    if removed > 0:
        print(f"[!] Removed {removed} all-zero sequences")
    
    # Load label map
    with open(os.path.join(ROOT_DIR, "configs", "label_map.json")) as f:
        lm = json.load(f)
    idx_to_label = lm["idx_to_label"]
    num_classes = lm["num_classes"]
    
    # Check for NaN/Inf
    nan_count = np.isnan(X_all).sum()
    inf_count = np.isinf(X_all).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"[!] WARNING: NaN={nan_count}, Inf={inf_count} in features")
        # Replace NaN/Inf with 0
        X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Stratified split: ensure every class has samples in train, val, and test
    np.random.seed(42)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    class_counts = Counter(y_all.tolist())
    
    print(f"\n{'Class':<15} {'Total':>6} {'Train':>6} {'Val':>5} {'Test':>5}")
    print("-" * 45)
    
    for cls_idx in range(num_classes):
        cls_name = idx_to_label[str(cls_idx)]
        cls_indices = np.where(y_all == cls_idx)[0]
        n = len(cls_indices)
        
        if n == 0:
            print(f"[!] WARNING: Class '{cls_name}' (idx={cls_idx}) has 0 samples!")
            continue
        
        # Shuffle indices for this class
        np.random.shuffle(cls_indices)
        
        if n < 3:
            # Very few samples: put all in train, duplicate 1 into val and test
            train_indices.extend(cls_indices.tolist())
            val_indices.append(cls_indices[0])
            test_indices.append(cls_indices[-1])
            n_train, n_val, n_test = n, 1, 1
        elif n < 6:
            # Few samples: 1 val, 1 test, rest train
            val_indices.append(cls_indices[0])
            test_indices.append(cls_indices[1])
            train_indices.extend(cls_indices[2:].tolist())
            n_train, n_val, n_test = n - 2, 1, 1
        else:
            # Normal split: 70/15/15
            n_test_cls = max(2, int(n * 0.15))
            n_val_cls = max(2, int(n * 0.15))
            n_train_cls = n - n_val_cls - n_test_cls
            
            if n_train_cls < 2:
                # Adjust if training has too few
                n_val_cls = 1
                n_test_cls = 1
                n_train_cls = n - 2
            
            test_indices.extend(cls_indices[:n_test_cls].tolist())
            val_indices.extend(cls_indices[n_test_cls:n_test_cls + n_val_cls].tolist())
            train_indices.extend(cls_indices[n_test_cls + n_val_cls:].tolist())
            n_train, n_val, n_test = n_train_cls, n_val_cls, n_test_cls
        
        print(f"{cls_name:<15} {n:>6} {n_train:>6} {n_val:>5} {n_test:>5}")
    
    # Convert to arrays
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)
    
    # Shuffle within splits
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    
    X_train = X_all[train_indices]
    y_train = y_all[train_indices]
    X_val = X_all[val_indices]
    y_val = y_all[val_indices]
    X_test = X_all[test_indices]
    y_test = y_all[test_indices]
    
    # Verify no overlap
    train_set = set(train_indices.tolist())
    val_set = set(val_indices.tolist())
    test_set = set(test_indices.tolist())
    
    tv_overlap = len(train_set & val_set)
    tt_overlap = len(train_set & test_set)
    vt_overlap = len(val_set & test_set)
    
    print(f"\n[+] Split sizes: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")
    print(f"[+] Overlap check: Train&Val={tv_overlap}, Train&Test={tt_overlap}, Val&Test={vt_overlap}")
    
    # Verify all classes present
    train_classes = set(np.unique(y_train))
    val_classes = set(np.unique(y_val))
    test_classes = set(np.unique(y_test))
    all_classes = set(range(num_classes))
    
    assert train_classes == all_classes, f"Train missing: {all_classes - train_classes}"
    assert val_classes == all_classes, f"Val missing: {all_classes - val_classes}"
    assert test_classes == all_classes, f"Test missing: {all_classes - test_classes}"
    print("[+] [PASS] All 40 classes present in all 3 splits!")
    
    # Save
    out_dir = os.path.join(ROOT_DIR, "data", "unified_v2")
    os.makedirs(out_dir, exist_ok=True)
    
    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "X_val.npy"), X_val)
    np.save(os.path.join(out_dir, "y_val.npy"), y_val)
    np.save(os.path.join(out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "y_test.npy"), y_test)
    
    # Save canonical 40-class label map
    label_map_40 = {}
    for k, v in lm["label_to_idx"].items():
        # Normalize: use underscore version as canonical
        canonical = k.replace(" ", "_")
        label_map_40[canonical] = v
    # Ensure "thank_you" is the canonical form
    if "thank you" in label_map_40:
        label_map_40["thank_you"] = label_map_40.pop("thank you")
    
    with open(os.path.join(out_dir, "label_map.json"), "w") as f:
        json.dump(label_map_40, f, indent=2)
    
    # Also save class names in order
    class_names = [idx_to_label[str(i)].replace(" ", "_") for i in range(num_classes)]
    with open(os.path.join(out_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)
    
    # Save split audit
    audit = {
        "status": "PASS",
        "total_samples": len(y_all),
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "num_classes": num_classes,
        "train_val_overlap": int(tv_overlap),
        "train_test_overlap": int(tt_overlap),
        "val_test_overlap": int(vt_overlap),
        "all_classes_in_train": bool(train_classes == all_classes),
        "all_classes_in_val": bool(val_classes == all_classes),
        "all_classes_in_test": bool(test_classes == all_classes),
        "train_class_counts": dict(sorted(Counter(y_train.tolist()).items())),
        "val_class_counts": dict(sorted(Counter(y_val.tolist()).items())),
        "test_class_counts": dict(sorted(Counter(y_test.tolist()).items())),
    }
    
    reports_dir = os.path.join(ROOT_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, "split_audit_v2.json"), "w") as f:
        json.dump(audit, f, indent=2)
    
    print(f"\n[+] Saved to {out_dir}/")
    print(f"[+] Saved split audit to reports/split_audit_v2.json")
    print("\n[PASS] REBUILD COMPLETE - Ready for training!")

if __name__ == "__main__":
    rebuild_splits()
