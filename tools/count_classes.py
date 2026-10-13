"""Count extracted npz files per class in both data sources."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def count_dir(base_dir, label=""):
    if not base_dir.exists():
        print(f"  [MISSING] {base_dir}")
        return {}
    counts = {}
    for cls_dir in sorted(base_dir.iterdir()):
        if cls_dir.is_dir():
            npz = list(cls_dir.rglob("*.npz"))
            npy = list(cls_dir.rglob("*.npy"))
            total = len(npz) + len(npy)
            counts[cls_dir.name] = total
    print(f"\n{'='*60}")
    print(f"  {label}: {base_dir}")
    print(f"  Classes: {len(counts)}, Total files: {sum(counts.values())}")
    print(f"{'='*60}")
    for cls, n in sorted(counts.items()):
        print(f"  {cls:30s} {n:5d}")
    return counts

# additional_landmarks (from extraction)
c1 = count_dir(ROOT / "data" / "additional_landmarks", "additional_landmarks")

# isl500_landmarks (pre-existing)
c2 = count_dir(ROOT / "data" / "isl500_landmarks", "isl500_landmarks")

# Merge
all_classes = set(list(c1.keys()) + list(c2.keys()))
print(f"\n{'='*60}")
print(f"  MERGED TOTALS")
print(f"  Unique classes: {len(all_classes)}")
print(f"{'='*60}")
merged = {}
for cls in sorted(all_classes):
    n = c1.get(cls, 0) + c2.get(cls, 0)
    merged[cls] = n
    src = []
    if cls in c1: src.append(f"add:{c1[cls]}")
    if cls in c2: src.append(f"isl500:{c2[cls]}")
    print(f"  {cls:30s} {n:5d}  ({', '.join(src)})")

eligible = {k: v for k, v in merged.items() if v >= 15}
print(f"\n  Classes with >= 15 samples: {len(eligible)}")
print(f"  Classes with >= 20 samples: {len({k:v for k,v in merged.items() if v >= 20})}")
