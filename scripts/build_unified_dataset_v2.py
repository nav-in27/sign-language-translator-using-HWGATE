"""
=================================================================
UNIFIED DATASET BUILDER V2 — MERGE ISL500 + VIDIT031
=================================================================
Merges:
  1. ISL500 H5 landmarks (29 classes, ~4K samples, 15 signers)
  2. vidit031 original videos (40 classes, 642 clips, ~170 signers)
  
Into a single unified dataset with:
  - Signer-independent train/val/test splits
  - SHA256 deduplication
  - Quality filtering
  - Canonical 40-class label map
  
Output: data/unified_40class_v2/ with NPZ files
        reports/unified_dataset_v2_manifest.csv
        reports/split_audit_v2.json
=================================================================
"""
import os
import sys
import csv
import json
import hashlib
import numpy as np
from pathlib import Path
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    getattr(sys.stdout, 'reconfigure')(line_buffering=True)

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Sources
ISL500_DIR = BASE_DIR / "data" / "isl500_landmarks"
VIDIT_DATA_DIR = BASE_DIR / "data" / "processed"  # vidit031 processed NPZ files
UNIFIED_DIR = BASE_DIR / "data" / "unified_40class_v2"
UNIFIED_DIR.mkdir(parents=True, exist_ok=True)

TARGET_40 = [
    "brother", "come", "drink", "eat", "father", "food", "friend", "go",
    "goodbye", "he", "hello", "help", "home", "hospital", "market", "me",
    "mother", "no", "okay", "please", "read", "school", "she", "sister",
    "sit", "sorry", "stand", "stop", "student", "tea", "teacher", "thank_you",
    "today", "water", "what", "when", "where", "write", "yes", "you"
]


def load_isl500_samples():
    """Load all downloaded ISL500 NPZ files."""
    samples = []
    if not ISL500_DIR.exists():
        print("[!] ISL500 directory not found, skipping")
        return samples
    
    for cls_dir in sorted(ISL500_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        canonical = cls_dir.name
        if canonical not in TARGET_40:
            continue
        for npz_path in sorted(cls_dir.glob("*.npz")):
            try:
                data = np.load(npz_path)
                features = data["features"]
                if features.shape != (32, 133, 6):
                    continue
                signer = str(data.get("signer", "ISL500_UNKNOWN"))
                sha = hashlib.sha256(features.tobytes()).hexdigest()[:16]
                samples.append({
                    "path": str(npz_path),
                    "class": canonical,
                    "source": "ISL500",
                    "signer_id": signer,
                    "hash": sha,
                    "features": features,
                })
            except Exception:
                continue
    return samples


def load_vidit031_samples():
    """Load vidit031 processed samples."""
    samples = []
    
    # Check for existing processed data
    if VIDIT_DATA_DIR.exists():
        for cls_dir in sorted(VIDIT_DATA_DIR.iterdir()):
            if not cls_dir.is_dir():
                continue
            canonical = cls_dir.name
            if canonical not in TARGET_40:
                continue
            for npz_path in sorted(cls_dir.glob("*.npz")):
                try:
                    data = np.load(npz_path)
                    features = data["features"]
                    if features.shape != (32, 133, 6):
                        continue
                    signer = str(data.get("signer", data.get("signer_id", "vidit_unknown")))
                    sha = hashlib.sha256(features.tobytes()).hexdigest()[:16]
                    samples.append({
                        "path": str(npz_path),
                        "class": canonical,
                        "source": "vidit031",
                        "signer_id": signer,
                        "hash": sha,
                        "features": features,
                    })
                except Exception:
                    continue
    
    # Also check the original data/ directory structure
    orig_dir = BASE_DIR / "data"
    for cls_name in TARGET_40:
        canonical = cls_name
        cls_path = orig_dir / cls_name
        if not cls_path.exists():
            continue
        for npz_path in sorted(cls_path.glob("*.npz")):
            try:
                data = np.load(npz_path)
                features = data["features"]
                if features.shape != (32, 133, 6):
                    continue
                signer = str(data.get("signer", data.get("signer_id", "vidit_unknown")))
                sha = hashlib.sha256(features.tobytes()).hexdigest()[:16]
                # Dedup by hash
                samples.append({
                    "path": str(npz_path),
                    "class": canonical,
                    "source": "vidit031",
                    "signer_id": signer,
                    "hash": sha,
                    "features": features,
                })
            except Exception:
                continue
    
    return samples


def deduplicate(samples):
    """Remove exact duplicates by SHA256 hash."""
    seen = set()
    unique = []
    for s in samples:
        if s["hash"] not in seen:
            seen.add(s["hash"])
            unique.append(s)
    removed = len(samples) - len(unique)
    return unique, removed


def quality_filter(samples):
    """Remove low-quality samples (all zeros, NaN, too little motion)."""
    passed = []
    rejected = 0
    for s in samples:
        f = s["features"]
        # Reject NaN
        if np.isnan(f).any():
            rejected += 1
            continue
        # Reject all-zero
        if np.abs(f).max() < 1e-8:
            rejected += 1
            continue
        # Check hand presence: at least some non-zero hand landmarks
        hand_data = f[:, 33:75, :3]  # Both hands position
        hand_nonzero = np.count_nonzero(np.abs(hand_data) > 0.001)
        total_hand = hand_data.size
        if hand_nonzero / total_hand < 0.1:
            rejected += 1
            continue
        passed.append(s)
    return passed, rejected


def create_signer_splits(samples, train_ratio=0.70, val_ratio=0.15):
    """Create signer-independent train/val/test splits."""
    # Group by source+signer
    signer_samples = defaultdict(list)
    for s in samples:
        key = f"{s['source']}_{s['signer_id']}"
        signer_samples[key].append(s)
    
    signers = sorted(signer_samples.keys())
    np.random.seed(42)
    np.random.shuffle(signers)
    
    n = len(signers)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    
    train_signers = set(signers[:n_train])
    val_signers = set(signers[n_train:n_train + n_val])
    test_signers = set(signers[n_train + n_val:])
    
    # Ensure test set is not empty
    if not test_signers:
        test_signers = val_signers
        val_signers = set()
    
    train, val, test = [], [], []
    for s in samples:
        key = f"{s['source']}_{s['signer_id']}"
        if key in train_signers:
            train.append(s)
        elif key in val_signers:
            val.append(s)
        else:
            test.append(s)
    
    return train, val, test, train_signers, val_signers, test_signers


def main():
    print("=" * 65, flush=True)
    print("  UNIFIED DATASET BUILDER V2 — ISL500 + VIDIT031", flush=True)
    print("=" * 65, flush=True)
    
    # Load sources
    print("\n[1/5] Loading ISL500 samples...", flush=True)
    isl500 = load_isl500_samples()
    print(f"  ISL500: {len(isl500)} samples", flush=True)
    
    print("\n[2/5] Loading vidit031 samples...", flush=True)
    vidit = load_vidit031_samples()
    print(f"  vidit031: {len(vidit)} samples", flush=True)
    
    all_samples = isl500 + vidit
    print(f"\n  Combined: {len(all_samples)} total samples", flush=True)
    
    # Class distribution
    class_counts = defaultdict(lambda: {"ISL500": 0, "vidit031": 0})
    for s in all_samples:
        class_counts[s["class"]][s["source"]] += 1
    
    print(f"\n  Per-class distribution:", flush=True)
    for cls in sorted(TARGET_40):
        c = class_counts.get(cls, {"ISL500": 0, "vidit031": 0})
        total = c["ISL500"] + c["vidit031"]
        print(f"    {cls:15s}: ISL500={c['ISL500']:4d}  vidit031={c['vidit031']:3d}  total={total}", flush=True)
    
    # Deduplicate
    print(f"\n[3/5] Deduplicating by SHA256...", flush=True)
    all_samples, n_dupes = deduplicate(all_samples)
    print(f"  Removed {n_dupes} duplicates, {len(all_samples)} unique remaining", flush=True)
    
    # Quality filter
    print(f"\n[4/5] Quality filtering...", flush=True)
    all_samples, n_rejected = quality_filter(all_samples)
    print(f"  Rejected {n_rejected}, {len(all_samples)} passed quality check", flush=True)
    
    # Create splits
    print(f"\n[5/5] Creating signer-independent splits...", flush=True)
    train, val, test, tr_s, va_s, te_s = create_signer_splits(all_samples)
    print(f"  Train: {len(train)} ({len(tr_s)} signers)", flush=True)
    print(f"  Val:   {len(val)} ({len(va_s)} signers)", flush=True)
    print(f"  Test:  {len(test)} ({len(te_s)} signers)", flush=True)
    
    # Verify no leakage
    overlap_tv = tr_s & va_s
    overlap_tt = tr_s & te_s
    overlap_vt = va_s & te_s
    print(f"  Signer overlaps: train-val={len(overlap_tv)}, train-test={len(overlap_tt)}, val-test={len(overlap_vt)}", flush=True)
    assert len(overlap_tv) == 0, "Train-Val signer leakage!"
    assert len(overlap_tt) == 0, "Train-Test signer leakage!"
    
    # Save split assignments and manifest
    manifest = []
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        for s in split_data:
            manifest.append({
                "sample_id": Path(s["path"]).name,
                "class": s["class"],
                "source": s["source"],
                "signer_id": s["signer_id"],
                "hash": s["hash"],
                "split": split_name,
                "path": s["path"],
            })
    
    manifest_path = REPORTS_DIR / "unified_dataset_v2_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)
    print(f"\n  Manifest: {manifest_path}", flush=True)
    
    # Save split audit
    split_audit = {
        "total_samples": len(all_samples),
        "split_counts": {"train": len(train), "val": len(val), "test": len(test)},
        "signer_counts": {"train": len(tr_s), "val": len(va_s), "test": len(te_s)},
        "signer_overlaps": {
            "train_val": len(overlap_tv),
            "train_test": len(overlap_tt),
            "val_test": len(overlap_vt),
        },
        "sources": {
            "ISL500": sum(1 for s in all_samples if s["source"] == "ISL500"),
            "vidit031": sum(1 for s in all_samples if s["source"] == "vidit031"),
        },
        "class_distribution": {
            cls: sum(1 for s in all_samples if s["class"] == cls)
            for cls in TARGET_40
        },
    }
    audit_path = REPORTS_DIR / "split_audit_v2.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(split_audit, f, indent=2)
    print(f"  Split audit: {audit_path}", flush=True)
    
    print(f"\n{'=' * 65}", flush=True)
    print(f"  UNIFIED DATASET V2 BUILT SUCCESSFULLY", flush=True)
    print(f"  Total samples: {len(all_samples)}", flush=True)
    print(f"  Classes: {len(set(s['class'] for s in all_samples))}", flush=True)
    print(f"{'=' * 65}", flush=True)


if __name__ == "__main__":
    main()
