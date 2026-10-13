import os
import json
from pathlib import Path

CURRENT_DATASET_PATH = Path(r"c:\Users\navee\.gemini\antigravity-ide\scratch\isl-translator\data\raw_40words")
ADDITIONAL_PATH_1 = Path(r"C:\Users\navee\Downloads\archive\ProcessedData_vivit")
ADDITIONAL_PATH_2 = Path(r"C:\Users\navee\Downloads\archive (2)\Video_Dataset\Video_Dataset")
if not ADDITIONAL_PATH_2.exists():
    ADDITIONAL_PATH_2 = Path(r"C:\Users\navee\Downloads\archive (2)\Video_Dataset")

curr = sorted([d.name for d in CURRENT_DATASET_PATH.iterdir() if d.is_dir() and not d.name.startswith(".")])
add1 = sorted([d.name for d in ADDITIONAL_PATH_1.iterdir() if d.is_dir()])
add2 = sorted([d.name for d in ADDITIONAL_PATH_2.iterdir() if d.is_dir()])

print("CURRENT CLASSES (40):")
print(curr)

print("\nADDITIONAL DATASET 1 CLASSES (76):")
print(add1)

print("\nADDITIONAL DATASET 2 CLASSES (61):")
print(add2)

# Check overlaps
curr_lower = {c.lower(): c for c in curr}
add1_lower = {c.lower(): c for c in add1}
add2_lower = {c.lower(): c for c in add2}

overlap_1 = set(curr_lower.keys()).intersection(set(add1_lower.keys()))
overlap_2 = set(curr_lower.keys()).intersection(set(add2_lower.keys()))
overlap_1_2 = set(add1_lower.keys()).intersection(set(add2_lower.keys()))

print("\n" + "="*50)
print(f"Overlap between Current and Add1: {len(overlap_1)} classes -> {sorted(list(overlap_1))}")
print(f"Overlap between Current and Add2: {len(overlap_2)} classes -> {sorted(list(overlap_2))}")
print(f"Overlap between Add1 and Add2: {len(overlap_1_2)} classes -> {sorted(list(overlap_1_2))}")
print(f"Total Unique Classes across ALL: {len(set(curr_lower.keys()).union(set(add1_lower.keys())).union(set(add2_lower.keys())))}")
