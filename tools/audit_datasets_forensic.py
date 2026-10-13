import os
import sys
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

# Ensure stdout handles encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def norm(c):
    return c.strip().lower().replace(" ", "_").replace("-", "_")

datasets = {
    "Archive2_Videos": Path(r"C:\Users\navee\Downloads\archive (2)\Video_Dataset\Video_Dataset"),
    "Vivit_Videos": Path(r"C:\Users\navee\Downloads\archive\ProcessedData_vivit"),
    "Raw40_Videos": Path("data/raw_40words"),
    "ISL500_Landmarks": Path("data/isl500_landmarks"),
    "Additional_Landmarks": Path("data/additional_landmarks"),
    "Cache_npz": Path("data/cache"),
}

summary = {}
all_classes = set()

for name, p in datasets.items():
    if not p.exists():
        summary[name] = {"exists": False}
        continue
    
    if name == "Cache_npz":
        # Cache has files directly in folder: e.g. 0000_thank_you.npz
        files = list(p.glob("*.npz"))
        class_stats = defaultdict(int)
        for f in files:
            if f.name.endswith("_split.npz"):
                continue
            parts = f.stem.split("_", 1)
            if len(parts) == 2:
                w = norm(parts[1])
                class_stats[w] += 1
                all_classes.add(w)
        summary[name] = {
            "exists": True,
            "path": str(p),
            "num_classes": len(class_stats),
            "total_files": sum(class_stats.values()),
            "class_stats": dict(class_stats)
        }
        continue

    classes = [d for d in p.iterdir() if d.is_dir() and not d.name.startswith(".")]
    class_stats = {}
    file_count = 0
    for d in classes:
        c_norm = norm(d.name)
        all_classes.add(c_norm)
        files = [f for f in d.iterdir() if f.is_file()]
        class_stats[c_norm] = len(files)
        file_count += len(files)
    
    summary[name] = {
        "exists": True,
        "path": str(p),
        "num_classes": len(classes),
        "total_files": file_count,
        "class_stats": class_stats
    }

print("=" * 60)
print("DATASET INVENTORY AUDIT")
print("=" * 60)
for name, s in summary.items():
    if s["exists"]:
        print(f"[{name}]")
        print(f"  Path: {s['path']}")
        print(f"  Classes: {s['num_classes']}, Files: {s['total_files']}")
    else:
        print(f"[{name}] NOT FOUND")

print("\n" + "=" * 60)
print(f"TOTAL DISTINCT CLASSES DISCOVERED: {len(all_classes)}")
print("=" * 60)

# Build a comprehensive table across all classes
table = []
for c in sorted(all_classes):
    a2 = summary.get("Archive2_Videos", {}).get("class_stats", {}).get(c, 0)
    viv = summary.get("Vivit_Videos", {}).get("class_stats", {}).get(c, 0)
    r40 = summary.get("Raw40_Videos", {}).get("class_stats", {}).get(c, 0)
    isl = summary.get("ISL500_Landmarks", {}).get("class_stats", {}).get(c, 0)
    add = summary.get("Additional_Landmarks", {}).get("class_stats", {}).get(c, 0)
    cache = summary.get("Cache_npz", {}).get("class_stats", {}).get(c, 0)
    
    total_raw_vids = a2 + viv + r40
    total_preextracted = isl + add
    
    table.append({
        "class": c,
        "archive2": a2,
        "vivit": viv,
        "raw40": r40,
        "isl500_lm": isl,
        "additional_lm": add,
        "cache_npz": cache,
        "total_vids": total_raw_vids,
        "total_preextracted_lm": total_preextracted,
        "grand_total": total_raw_vids + isl # avoid double counting additional_lm if extracted from vids
    })

# Check classes with sufficient samples
ge_10_samples = [r for r in table if r["total_vids"] >= 10 or r["isl500_lm"] >= 10 or r["additional_lm"] >= 10]
ge_15_samples = [r for r in table if r["total_vids"] >= 15 or r["isl500_lm"] >= 15 or r["additional_lm"] >= 15]
ge_20_samples = [r for r in table if r["total_vids"] >= 20 or r["isl500_lm"] >= 20 or r["additional_lm"] >= 20]

print(f"Classes with >= 10 total samples: {len(ge_10_samples)}")
print(f"Classes with >= 15 total samples: {len(ge_15_samples)}")
print(f"Classes with >= 20 total samples: {len(ge_20_samples)}")

# Write json report
out_path = Path("reports/forensic_dataset_audit.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "summary": summary,
        "total_distinct_classes": len(all_classes),
        "classes_ge_10": len(ge_10_samples),
        "classes_ge_15": len(ge_15_samples),
        "classes_ge_20": len(ge_20_samples),
        "table": table
    }, f, indent=2)

print(f"\n[+] Saved detailed audit to {out_path}")
