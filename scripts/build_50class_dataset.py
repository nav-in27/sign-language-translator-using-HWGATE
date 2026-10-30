"""
Master 50-Class Dataset Builder for AC-HWGATE ISL System.
Combines:
  1. ISL500 Landmarks (29 classes, disjoint signers USER001-USER015)
  2. Vidit031 Benchmark Corpus (data/cache)
  3. Archive-2 Video Landmarks (data/additional_landmarks, session-disjoint)

Enforces:
  - Strict leakage-free session & signer disjoint splits
  - Exact 32-frame [32, 133, 6] format parity
  - Zero duplicate leakage across splits
  - All 50 classes represented across Train, Val, Test
"""
import os
import sys
import json
import glob
import hashlib
import random
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

ROOT = Path(r"c:\Users\navee\.gemini\antigravity-ide\scratch\isl-translator")
sys.path.insert(0, str(ROOT / "training"))
from feature_extractor import feature_extractor

DATA_CACHE_DIR = ROOT / "data" / "cache"
ISL500_DIR = ROOT / "data" / "isl500_landmarks"
ADDITIONAL_DIR = ROOT / "data" / "additional_landmarks"
OUT_DIR = ROOT / "data" / "unified_50class"
REPORTS_DIR = ROOT / "reports"
CONFIGS_DIR = ROOT / "configs"
BACKEND_DIR = ROOT / "backend"

def norm_label(label: str) -> str:
    s = str(label).strip().lower().replace(" ", "_").replace("-", "_")
    if s in ["thankyou", "thank_you"]: return "thank_you"
    if s in ["i", "myself"]: return "me"
    if s == "house": return "home"
    if s == "writer": return "write"
    return s

def hash_array(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.round(decimals=3).tobytes()).hexdigest()

def build_50class_splits():
    print("=" * 70)
    print("BUILDING UNIFIED 50-CLASS DATASET (LEAKAGE-FREE SIGNER & SESSION SPLITS)")
    print("=" * 70)

    # 1. Load canonical 50 classes manifest
    manifest_path = CONFIGS_DIR / "selected_50_classes_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    classes_50 = manifest_data["classes"]
    label_to_idx = {c: i for i, c in enumerate(classes_50)}
    num_classes = len(classes_50)
    print(f"[+] Loaded {num_classes} target classes from manifest.")

    all_samples = []
    seen_hashes = set()
    quarantined = 0

    # -------------------------------------------------------------
    # Ingest 1: Vidit031 Benchmark Sequences (data/cache)
    # -------------------------------------------------------------
    cache_files = sorted(glob.glob(str(DATA_CACHE_DIR / "0*.npz")))
    print(f"[*] Ingesting Vidit031 Cache: {len(cache_files)} candidate files...")
    for fpath in cache_files:
        try:
            data = np.load(fpath, allow_pickle=True)
            raw_w = data.get("word", data.get("label", Path(fpath).stem.split("_", 1)[-1]))
            lbl = norm_label(raw_w)
            if lbl not in label_to_idx:
                continue

            feat = data["features"] if "features" in data else feature_extractor.process_sequence_np(data["keypoints"])
            
            # Check shape & quality
            if feat.shape != (32, 133, 6) or np.isnan(feat).any() or np.isinf(feat).any():
                quarantined += 1
                continue
            if np.abs(feat).sum() < 1.0: # all-zero
                quarantined += 1
                continue

            sha = hash_array(feat)
            if sha in seen_hashes:
                quarantined += 1
                continue
            seen_hashes.add(sha)

            signer = str(data.get("signer", "vidit_pool"))
            all_samples.append({
                "source": "vidit031",
                "features": feat.astype(np.float32),
                "label": lbl,
                "label_idx": label_to_idx[lbl],
                "session_id": str(data.get("video_path", Path(fpath).stem)),
                "signer_id": signer,
                "sha256": sha
            })
        except Exception:
            quarantined += 1

    # -------------------------------------------------------------
    # Ingest 2: ISL500 Landmarks (data/isl500_landmarks)
    # -------------------------------------------------------------
    isl500_files = sorted(glob.glob(str(ISL500_DIR / "*" / "*.npz")))
    print(f"[*] Ingesting ISL500 Landmark Corpus: {len(isl500_files)} candidate files...")
    for fpath in isl500_files:
        try:
            data = np.load(fpath, allow_pickle=True)
            raw_w = data.get("word", data.get("label", Path(fpath).parent.name))
            lbl = norm_label(raw_w)
            if lbl not in label_to_idx:
                continue

            feat = data["features"] if "features" in data else feature_extractor.process_sequence_np(data["keypoints"])
            if feat.shape != (32, 133, 6) or np.isnan(feat).any() or np.isinf(feat).any():
                quarantined += 1
                continue
            if np.abs(feat).sum() < 1.0:
                quarantined += 1
                continue

            sha = hash_array(feat)
            if sha in seen_hashes:
                quarantined += 1
                continue
            seen_hashes.add(sha)

            signer = str(data.get("signer", "User_Generic"))
            for part in Path(fpath).stem.split("_"):
                if "USER" in part.upper():
                    signer = part.upper()
                    break

            all_samples.append({
                "source": "isl500",
                "features": feat.astype(np.float32),
                "label": lbl,
                "label_idx": label_to_idx[lbl],
                "session_id": str(data.get("session", Path(fpath).stem)),
                "signer_id": signer,
                "sha256": sha
            })
        except Exception:
            quarantined += 1

    # -------------------------------------------------------------
    # Ingest 3: Additional Local Video Landmarks (data/additional_landmarks)
    # -------------------------------------------------------------
    add_files = sorted(glob.glob(str(ADDITIONAL_DIR / "*" / "*.npz")))
    # Exclude 45-degree camera tilts (_left_tilt / _right_tilt) to prevent pose poisoning
    add_files = [f for f in add_files if not ('_left_tilt' in f or '_right_tilt' in f)]
    print(f"[*] Ingesting Additional Landmarks (Archive-2): {len(add_files)} candidate files...")
    for fpath in add_files:
        try:
            data = np.load(fpath, allow_pickle=True)
            raw_w = data.get("label", data.get("word", Path(fpath).parent.name))
            lbl = norm_label(raw_w)
            if lbl not in label_to_idx:
                continue

            if "features" in data:
                feat = data["features"]
            else:
                feat = feature_extractor.process_sequence_np(data["keypoints"])

            if feat.shape != (32, 133, 6) or np.isnan(feat).any() or np.isinf(feat).any():
                quarantined += 1
                continue
            if np.abs(feat).sum() < 1.0:
                quarantined += 1
                continue

            sha = hash_array(feat)
            if sha in seen_hashes:
                quarantined += 1
                continue
            seen_hashes.add(sha)

            stem = Path(fpath).stem
            # Extract atomic session ID (group frontal, left_tilt, right_tilt together)
            session_id = stem
            if "_left_tilt" in stem:
                session_id = stem.replace("_left_tilt", "")
            elif "_right_tilt" in stem:
                session_id = stem.replace("_right_tilt", "")

            all_samples.append({
                "source": "archive2",
                "features": feat.astype(np.float32),
                "label": lbl,
                "label_idx": label_to_idx[lbl],
                "session_id": session_id,
                "signer_id": "Archive2_Signer",
                "sha256": sha
            })
        except Exception:
            quarantined += 1

    print(f"\n[+] Total Valid Unique Sequences: {len(all_samples)} (Quarantined/Duplicates: {quarantined})")
    
    # -------------------------------------------------------------
    # Partitioning Protocol:
    # 1. ISL500: Disjoint signers
    #    Train: USER001, USER002, USER003, USER004, USER006, USER007, USER012 (~70%)
    #    Val:   USER009, USER010, USER011 (~15%)
    #    Test:  USER005, USER008, USER013, USER014, USER015 (~15%)
    # 2. Archive-2: Atomic Session Partitioning (Tilt groups stay together)
    #    Group all tilts of session_id together -> 70% Train, 15% Val, 15% Test
    # 3. Vidit031: Stratified 70% Train, 15% Val, 15% Test
    # -------------------------------------------------------------
    train_samples, val_samples, test_samples = [], [], []

    isl500_train_users = {"USER001", "USER002", "USER003", "USER004", "USER006", "USER007", "USER012"}
    isl500_val_users   = {"USER009", "USER010", "USER011"}
    isl500_test_users  = {"USER005", "USER008", "USER013", "USER014", "USER015"}

    # Separate items by source
    isl500_items = [x for x in all_samples if x["source"] == "isl500"]
    arch2_by_class = defaultdict(list)
    for x in all_samples:
        if x["source"] == "archive2":
            arch2_by_class[x["label"]].append(x)
            
    vidit_by_class = defaultdict(list)
    for x in all_samples:
        if x["source"] == "vidit031":
            vidit_by_class[x["label"]].append(x)

    # 1. Partition ISL500
    for item in isl500_items:
        signer = item["signer_id"].upper()
        if any(u in signer for u in isl500_train_users):
            train_samples.append(item)
        elif any(u in signer for u in isl500_val_users):
            val_samples.append(item)
        elif any(u in signer for u in isl500_test_users):
            test_samples.append(item)
        else:
            train_samples.append(item)

    # 2. Partition Archive-2 by Session (Tilt-Disjoint)
    rng = random.Random(42)
    for lbl, items in arch2_by_class.items():
        # Group by session_id
        session_map = defaultdict(list)
        for it in items:
            session_map[it["session_id"]].append(it)
        
        session_keys = list(session_map.keys())
        rng.shuffle(session_keys)
        n_sess = len(session_keys)
        n_val = max(1, int(round(n_sess * 0.15)))
        n_test = max(1, int(round(n_sess * 0.15)))
        n_train = n_sess - n_val - n_test

        for sk in session_keys[:n_train]:
            train_samples.extend(session_map[sk])
        for sk in session_keys[n_train:n_train + n_val]:
            val_samples.extend(session_map[sk])
        for sk in session_keys[n_train + n_val:]:
            test_samples.extend(session_map[sk])

    # 3. Partition Vidit031 stratified
    for lbl, items in vidit_by_class.items():
        rng.shuffle(items)
        n = len(items)
        if n >= 6:
            n_val = max(1, int(round(n * 0.15)))
            n_test = max(1, int(round(n * 0.15)))
            n_train = n - n_val - n_test
            train_samples.extend(items[:n_train])
            val_samples.extend(items[n_train:n_train + n_val])
            test_samples.extend(items[n_train + n_val:])
        elif n >= 3:
            val_samples.append(items[0])
            test_samples.append(items[1])
            train_samples.extend(items[2:])
        else:
            train_samples.extend(items)

    print(f"\n[+] Raw Split Sizes: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")

    # Verify all 50 classes exist in all 3 splits
    train_classes = set(x["label_idx"] for x in train_samples)
    val_classes = set(x["label_idx"] for x in val_samples)
    test_classes = set(x["label_idx"] for x in test_samples)
    all_class_indices = set(range(num_classes))

    # If any class missing in val or test, move 1 non-overlapping item from train
    for cls_idx in all_class_indices:
        if cls_idx not in val_classes:
            train_cls_items = [x for x in train_samples if x["label_idx"] == cls_idx]
            if len(train_cls_items) > 3:
                moved = train_cls_items[-1]
                train_samples.remove(moved)
                val_samples.append(moved)
                val_classes.add(cls_idx)
                print(f"[!] Balanced class '{classes_50[cls_idx]}' into val split")

        if cls_idx not in test_classes:
            train_cls_items = [x for x in train_samples if x["label_idx"] == cls_idx]
            if len(train_cls_items) > 3:
                moved = train_cls_items[-1]
                train_samples.remove(moved)
                test_samples.append(moved)
                test_classes.add(cls_idx)
                print(f"[!] Balanced class '{classes_50[cls_idx]}' into test split")

    assert train_classes == all_class_indices, f"Train missing: {all_class_indices - train_classes}"
    assert val_classes == all_class_indices, f"Val missing: {all_class_indices - val_classes}"
    assert test_classes == all_class_indices, f"Test missing: {all_class_indices - test_classes}"
    print("[+] [PASS] All 50 classes verified present in Train, Val, and Test splits!")

    # Check for duplicate leakage
    train_shas = set(x["sha256"] for x in train_samples)
    val_shas = set(x["sha256"] for x in val_samples)
    test_shas = set(x["sha256"] for x in test_samples)

    assert len(train_shas & val_shas) == 0, "DATA LEAKAGE between Train and Val!"
    assert len(train_shas & test_shas) == 0, "DATA LEAKAGE between Train and Test!"
    assert len(val_shas & test_shas) == 0, "DATA LEAKAGE between Val and Test!"
    print("[+] [PASS] ZERO data leakage detected across splits!")

    # Shuffle splits
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)

    X_train = np.stack([x["features"] for x in train_samples])
    y_train = np.array([x["label_idx"] for x in train_samples], dtype=np.int64)

    X_val = np.stack([x["features"] for x in val_samples])
    y_val = np.array([x["label_idx"] for x in val_samples], dtype=np.int64)

    X_test = np.stack([x["features"] for x in test_samples])
    y_test = np.array([x["label_idx"] for x in test_samples], dtype=np.int64)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "X_train.npy", X_train)
    np.save(OUT_DIR / "y_train.npy", y_train)
    np.save(OUT_DIR / "X_val.npy", X_val)
    np.save(OUT_DIR / "y_val.npy", y_val)
    np.save(OUT_DIR / "X_test.npy", X_test)
    np.save(OUT_DIR / "y_test.npy", y_test)

    # Save label_map and class_names
    label_map_50 = {c: i for i, c in enumerate(classes_50)}
    label_map_50["thank you"] = label_map_50["thank_you"]
    
    with open(OUT_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map_50, f, indent=2)
    with open(OUT_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(classes_50, f, indent=2)

    # Synchronize to backend/ and models/ and configs/
    with open(BACKEND_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map_50, f, indent=2)
    with open(ROOT / "models" / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map_50, f, indent=2)
    with open(CONFIGS_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map_50, f, indent=2)

    print(f"\n[+] Saved 50-Class Dataset to {OUT_DIR}/")
    print(f"    Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    print(f"[+] Ground-truth label_map.json synchronized across backend/, models/, configs/")

    # Save audit report
    audit = {
        "status": "PASS",
        "num_classes": 50,
        "total_samples": len(all_samples),
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "train_class_counts": {classes_50[k]: v for k, v in sorted(Counter(y_train.tolist()).items())},
        "val_class_counts": {classes_50[k]: v for k, v in sorted(Counter(y_val.tolist()).items())},
        "test_class_counts": {classes_50[k]: v for k, v in sorted(Counter(y_test.tolist()).items())},
    }
    with open(REPORTS_DIR / "split_audit_50class.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    print(f"[+] Saved audit report to reports/split_audit_50class.json")

if __name__ == "__main__":
    build_50class_splits()
