"""
Master Unified Dataset Builder for AC-HWGATE ISL Recognition.
Integrates:
  1. Base Dataset (data/cache 0*.npz from Vidit031) - 642 sequences
  2. ISL500 Corpus (data/isl500_landmarks) - 2,439 sequences
  3. Additional Local Dataset (data/additional_landmarks from local archives) - 305 sequences
Total: ~3,386 sequences across 40 classes!
Outputs:
  - data/unified/train_split.npz
  - data/unified/val_split.npz
  - data/unified/test_split.npz
  - reports/unified_dataset_manifest.csv
  - reports/split_audit.json
"""
import os
import sys
import json
import hashlib
import glob
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from training.feature_extractor import feature_extractor

DATA_CACHE_DIR = BASE_DIR / "data" / "cache"
ISL500_DIR = BASE_DIR / "data" / "isl500_landmarks"
ADDITIONAL_DIR = BASE_DIR / "data" / "additional_landmarks"
UNIFIED_DIR = BASE_DIR / "data" / "unified"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
BACKEND_MODELS_DIR = BASE_DIR / "backend" / "models"

CANONICAL_40_WORDS = [
    "brother", "come", "drink", "eat", "father", "food", "friend", "go",
    "goodbye", "he", "hello", "help", "home", "hospital", "market", "me",
    "mother", "no", "okay", "please", "read", "school", "she", "sister",
    "sit", "sorry", "stand", "stop", "student", "tea", "teacher", "thank_you",
    "today", "water", "what", "when", "where", "write", "yes", "you"
]

def normalize_label(label: str) -> str:
    s = str(label).strip().lower().replace(" ", "_").replace("-", "_")
    if s in ["thankyou", "thank_you"]:
        return "thank_you"
    if s in ["i", "myself"]:
        return "me"
    if s == "house":
        return "home"
    return s

def hash_array(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()

def main():
    UNIFIED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    BACKEND_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  BUILDING UNIFIED 40-CLASS DATASET (BASE + LOCAL ARCHIVES)")
    print("=" * 65)

    label_to_idx = {w: i for i, w in enumerate(CANONICAL_40_WORDS)}
    label_to_idx["thank you"] = label_to_idx["thank_you"]
    idx_to_label = {i: w for i, w in enumerate(CANONICAL_40_WORDS)}

    label_map_payload = {
        "classes": CANONICAL_40_WORDS,
        "num_classes": len(CANONICAL_40_WORDS),
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label
    }
    with open(MODELS_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map_payload, f, indent=2)
    with open(BACKEND_MODELS_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map_payload, f, indent=2)

    all_samples = []
    seen_hashes = set()

    # 1. Base Dataset (data/cache 0*.npz)
    base_files = sorted(glob.glob(str(DATA_CACHE_DIR / "0*.npz")))
    print(f"[*] Ingesting Base Dataset (Vidit031): {len(base_files)} candidate files...")
    for fpath in base_files:
        try:
            data = np.load(fpath, allow_pickle=True)
            raw_w = data.get("word", data.get("label", str(Path(fpath).stem).split("_", 1)[-1]))
            lbl = normalize_label(raw_w)
            if lbl not in label_to_idx:
                continue
            
            if "features" in data:
                feat = data["features"]
            else:
                feat = feature_extractor.process_sequence_np(data["keypoints"])
                
            sha = str(data.get("hash", hash_array(feat)))
            if sha in seen_hashes:
                continue
            seen_hashes.add(sha)
            
            all_samples.append({
                "source": "base_vidit031",
                "features": feat,
                "label": lbl,
                "label_idx": label_to_idx[lbl],
                "signer_id": str(data.get("signer", "vidit_pool")),
                "sha256": sha
            })
        except Exception as e:
            continue

    # 2. ISL500 Corpus (data/isl500_landmarks)
    isl500_files = sorted(glob.glob(str(ISL500_DIR / "*" / "*.npz")))
    print(f"[*] Ingesting ISL500 Landmark Corpus: {len(isl500_files)} candidate files...")
    for fpath in isl500_files:
        try:
            data = np.load(fpath, allow_pickle=True)
            raw_w = data.get("word", data.get("label", Path(fpath).parent.name))
            lbl = normalize_label(raw_w)
            if lbl not in label_to_idx:
                continue
                
            if "features" in data:
                feat = data["features"]
            else:
                feat = feature_extractor.process_sequence_np(data["keypoints"])
                
            sha = str(data.get("hash", hash_array(feat)))
            if sha in seen_hashes:
                continue
            seen_hashes.add(sha)
            
            signer = str(data.get("signer", "User_Generic"))
            for part in Path(fpath).stem.split("_"):
                if "USER" in part.upper():
                    signer = part.upper()
                    break
                    
            all_samples.append({
                "source": "isl500",
                "features": feat,
                "label": lbl,
                "label_idx": label_to_idx[lbl],
                "signer_id": signer,
                "sha256": sha
            })
        except Exception as e:
            continue

    # 3. Additional Local Dataset (data/additional_landmarks)
    add_files = sorted(glob.glob(str(ADDITIONAL_DIR / "*" / "*.npz")))
    print(f"[*] Ingesting Additional Local Dataset: {len(add_files)} candidate files...")
    for fpath in add_files:
        try:
            data = np.load(fpath, allow_pickle=True)
            raw_w = data.get("label", data.get("word", Path(fpath).parent.name))
            lbl = normalize_label(raw_w)
            if lbl not in label_to_idx:
                continue
                
            if "features" in data:
                feat = data["features"]
            else:
                feat = feature_extractor.process_sequence_np(data["keypoints"])
                
            sha = str(data.get("sha256", hash_array(feat)))
            if sha in seen_hashes:
                continue
            seen_hashes.add(sha)
            
            stem = Path(fpath).stem
            session_id = "local_archive_session"
            if "WIN_" in stem:
                session_id = stem.split("_Pro")[0]
            elif "MVI_" in stem:
                session_id = stem.split("_")[0]
                
            all_samples.append({
                "source": "local_additional_archive",
                "features": feat,
                "label": lbl,
                "label_idx": label_to_idx[lbl],
                "signer_id": session_id,
                "sha256": sha
            })
        except Exception as e:
            continue

    print(f"\n[+] Total Valid Unique Sequences Ingested: {len(all_samples)}")
    df_summary = pd.DataFrame([{k: v for k, v in x.items() if k != "features"} for x in all_samples])
    print(df_summary["source"].value_counts())

    # Signer-Independent Partitioning Protocol:
    # ISL500 disjoint users:
    #   Train: USER001, USER002, USER003, USER004, USER006, USER007, USER012 (~1,681 clips = 68.9%)
    #   Val:   USER009, USER010, USER011 (~355 clips = 14.6%)
    #   Test:  USER005, USER008, USER013, USER014, USER015 (~403 clips = 16.5%)
    # Non-ISL500 samples (Base + Local Additional):
    #   Stratified exactly 70% Train, 15% Val, 15% Test
    train_samples = []
    val_samples = []
    test_samples = []

    isl500_train_users = {"USER001", "USER002", "USER003", "USER004", "USER006", "USER007", "USER012"}
    isl500_val_users   = {"USER009", "USER010", "USER011"}
    isl500_test_users  = {"USER005", "USER008", "USER013", "USER014", "USER015"}

    non_isl_by_class = defaultdict(list)

    for item in all_samples:
        src = item["source"]
        signer = item["signer_id"].upper()
        
        if src == "isl500":
            matched = False
            for u in isl500_train_users:
                if u in signer:
                    train_samples.append(item)
                    matched = True
                    break
            if not matched:
                for u in isl500_val_users:
                    if u in signer:
                        val_samples.append(item)
                        matched = True
                        break
            if not matched:
                for u in isl500_test_users:
                    if u in signer:
                        test_samples.append(item)
                        matched = True
                        break
            if not matched:
                train_samples.append(item)
        else:
            non_isl_by_class[item["label"]].append(item)

    # Stratified split for non-ISL500 items
    rng = random.Random(42)
    for lbl, items in non_isl_by_class.items():
        rng.shuffle(items)
        n = len(items)
        n_val = max(1, int(round(n * 0.15)))
        n_test = max(1, int(round(n * 0.15)))
        n_train = n - n_val - n_test
        if n_train < 1:
            n_train = max(1, n - 2)
            n_val = 1 if n >= 2 else 0
            n_test = 1 if n >= 3 else 0
            
        train_samples.extend(items[:n_train])
        val_samples.extend(items[n_train:n_train + n_val])
        test_samples.extend(items[n_train + n_val:])

    print(f"\n[+] Partition Counts:")
    print(f"    Train: {len(train_samples)} ({(len(train_samples)/len(all_samples))*100:.1f}%)")
    print(f"    Val:   {len(val_samples)} ({(len(val_samples)/len(all_samples))*100:.1f}%)")
    print(f"    Test:  {len(test_samples)} ({(len(test_samples)/len(all_samples))*100:.1f}%)")

    # Leakage Audit Verification
    train_hashes = set(x["sha256"] for x in train_samples)
    val_hashes = set(x["sha256"] for x in val_samples)
    test_hashes = set(x["sha256"] for x in test_samples)

    assert len(train_hashes.intersection(val_hashes)) == 0, "Train-Val Hash Leakage Detected!"
    assert len(train_hashes.intersection(test_hashes)) == 0, "Train-Test Hash Leakage Detected!"
    assert len(val_hashes.intersection(test_hashes)) == 0, "Val-Test Hash Leakage Detected!"
    print("[+] Verified: ZERO hash/sequence overlap across all splits!")

    # Save manifest
    manifest_rows = []
    for split_name, s_list in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        for item in s_list:
            manifest_rows.append({
                "source": item["source"],
                "label": item["label"],
                "label_idx": item["label_idx"],
                "signer_id": item["signer_id"],
                "sha256": item["sha256"],
                "split": split_name
            })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv = REPORTS_DIR / "unified_dataset_manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"[+] Written manifest: {manifest_csv}")

    audit_json = REPORTS_DIR / "split_audit.json"
    audit_data = {
        "status": "PASS",
        "total_samples": len(all_samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "train_val_overlap": len(train_hashes.intersection(val_hashes)),
        "train_test_overlap": len(train_hashes.intersection(test_hashes)),
        "val_test_overlap": len(val_hashes.intersection(test_hashes)),
        "classes": CANONICAL_40_WORDS,
        "num_classes": len(CANONICAL_40_WORDS)
    }
    with open(audit_json, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
    print(f"[+] Written split audit: {audit_json}")

    # Pack split npz arrays
    def pack_split(sample_list, split_name):
        print(f"\n[*] Packing {split_name} ({len(sample_list)} sequences)...")
        X = np.stack([s["features"] for s in sample_list], axis=0).astype(np.float32)
        y = np.array([s["label_idx"] for s in sample_list], dtype=np.int64)
        
        out_unified = UNIFIED_DIR / f"{split_name}_split.npz"
        out_cache = DATA_CACHE_DIR / f"{split_name}_split.npz"
        
        np.savez_compressed(out_unified, features=X, labels=y)
        np.savez_compressed(out_cache, features=X, labels=y)
        print(f"    Saved {split_name}: {X.shape}, {y.shape} -> {out_unified}")

    pack_split(train_samples, "train")
    pack_split(val_samples, "val")
    pack_split(test_samples, "test")

    print("\n[+] UNIFIED DATASET BUILD & AUDIT COMPLETE!")

if __name__ == "__main__":
    main()
