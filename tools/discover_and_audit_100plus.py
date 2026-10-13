import os
import sys
import json
import glob
from pathlib import Path
from collections import defaultdict
import numpy as np

# Ensure stdout handles encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent.parent

def norm(c):
    c = c.strip().lower().replace(" ", "_").replace("-", "_")
    if c == "writer":
        c = "write"
    if c == "thankyou":
        c = "thank_you"
    return c

print("=" * 70)
print("PHASE 2: COMPREHENSIVE 100+ CLASS DISCOVERY & SAMPLE AUDIT")
print("=" * 70)

# 1. Existing verified landmark sequences in data/
landmarks_by_class = defaultdict(list)

# Check data/cache
cache_dir = ROOT_DIR / "data" / "cache"
if cache_dir.exists():
    for f in cache_dir.glob("*.npz"):
        if f.name.endswith("_split.npz"):
            continue
        try:
            d = np.load(f, allow_pickle=True)
            feat = d.get("features")
            word = str(d.get("word", "")).strip()
            if not word and "_" in f.stem:
                word = f.stem.split("_", 1)[1]
            word = norm(word)
            if feat is not None and feat.shape == (32, 133, 6):
                landmarks_by_class[word].append({
                    "path": str(f),
                    "source": "cache",
                    "signer": str(d.get("signer", "unknown")),
                    "session_id": f.stem
                })
        except Exception:
            pass

print(f"[+] Loaded cache landmarks: {len(landmarks_by_class)} classes")

# Check data/isl500_landmarks
isl_dir = ROOT_DIR / "data" / "isl500_landmarks"
if isl_dir.exists():
    for sub in isl_dir.iterdir():
        if sub.is_dir() and not sub.name.startswith("."):
            c_name = norm(sub.name)
            for f in sub.glob("*.npz"):
                try:
                    d = np.load(f, allow_pickle=True)
                    feat = d.get("features")
                    if feat is not None and feat.shape == (32, 133, 6):
                        # Extract signer ID from filename (e.g. USER003_...)
                        signer = f.stem.split("_")[0] if "_" in f.stem else "isl500"
                        landmarks_by_class[c_name].append({
                            "path": str(f),
                            "source": "isl500",
                            "signer": signer,
                            "session_id": f.stem
                        })
                except Exception:
                    pass

print(f"[+] Combined with ISL500 landmarks: {len(landmarks_by_class)} classes")

# Check data/additional_landmarks
add_dir = ROOT_DIR / "data" / "additional_landmarks"
if add_dir.exists():
    for sub in add_dir.iterdir():
        if sub.is_dir() and not sub.name.startswith("."):
            c_name = norm(sub.name)
            for f in sub.glob("*.npz"):
                try:
                    d = np.load(f, allow_pickle=True)
                    feat = d.get("features")
                    if feat is not None and feat.shape == (32, 133, 6):
                        sess = str(d.get("session_id", f.stem))
                        landmarks_by_class[c_name].append({
                            "path": str(f),
                            "source": str(d.get("source", "additional")),
                            "signer": sess.split("_")[0] if "_" in sess else "add",
                            "session_id": sess
                        })
                except Exception:
                    pass

print(f"[+] Combined with additional_landmarks: {len(landmarks_by_class)} classes")

# Report counts
summary = []
for c, samples in sorted(landmarks_by_class.items()):
    # Remove exact duplicate filepaths or hashes
    unique_paths = set()
    valid_samples = []
    for s in samples:
        if s["path"] not in unique_paths:
            unique_paths.add(s["path"])
            valid_samples.append(s)
    landmarks_by_class[c] = valid_samples
    summary.append({
        "class": c,
        "count": len(valid_samples),
        "sources": list(set(s["source"] for s in valid_samples))
    })

summary.sort(key=lambda x: x["count"], reverse=True)
print("\n" + "=" * 70)
print(f"TOTAL CLASSES WITH PRE-EXTRACTED VALID SEQUENCES: {len(summary)}")
print("=" * 70)

ge_15 = [s for s in summary if s["count"] >= 15]
ge_20 = [s for s in summary if s["count"] >= 20]
ge_50 = [s for s in summary if s["count"] >= 50]
print(f"Classes with >= 15 valid samples: {len(ge_15)}")
print(f"Classes with >= 20 valid samples: {len(ge_20)}")
print(f"Classes with >= 50 valid samples: {len(ge_50)}")

print("\nSample top 30 classes:")
for s in summary[:30]:
    print(f"  {s['class']:<20}: {s['count']:>4} samples | Sources: {s['sources']}")

print("\nSample classes ranked 80 to 110:")
for s in summary[80:110]:
    print(f"  {s['class']:<20}: {s['count']:>4} samples | Sources: {s['sources']}")

# Save discovery json
with open(ROOT_DIR / "reports" / "preextracted_landmarks_summary.json", "w") as f:
    json.dump({
        "total_classes": len(summary),
        "classes_ge_15": len(ge_15),
        "classes_ge_20": len(ge_20),
        "classes_ge_50": len(ge_50),
        "classes": summary
    }, f, indent=2)

print("\nSaved summary to reports/preextracted_landmarks_summary.json")
