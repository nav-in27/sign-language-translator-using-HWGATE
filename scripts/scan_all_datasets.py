import os
import json
import csv
from pathlib import Path
import cv2
import hashlib

CURRENT_DATASET_PATH = Path(r"c:\Users\navee\.gemini\antigravity-ide\scratch\isl-translator\data\raw_40words")
ADDITIONAL_PATH_1 = Path(r"C:\Users\navee\Downloads\archive\ProcessedData_vivit")
ADDITIONAL_PATH_2 = Path(r"C:\Users\navee\Downloads\archive (2)\Video_Dataset\Video_Dataset")
if not ADDITIONAL_PATH_2.exists():
    ADDITIONAL_PATH_2 = Path(r"C:\Users\navee\Downloads\archive (2)\Video_Dataset")

print(f"Current Dataset exists: {CURRENT_DATASET_PATH.exists()}")
print(f"Additional Dataset 1 exists: {ADDITIONAL_PATH_1.exists()}")
print(f"Additional Dataset 2 exists: {ADDITIONAL_PATH_2.exists()}")

def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = frames / fps if fps > 0 else 0
    cap.release()
    return {
        "fps": round(fps, 2) if fps > 0 else 30.0,
        "frames": frames,
        "resolution": f"{w}x{h}",
        "duration": round(dur, 2)
    }

def scan_dataset(base_path, name):
    classes = {}
    video_exts = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
    
    for root, dirs, files in os.walk(base_path):
        vids = [f for f in files if Path(f).suffix.lower() in video_exts]
        if vids:
            rel = os.path.relpath(root, base_path)
            # normalize class name from directory
            class_name = Path(rel).name
            if class_name not in classes:
                classes[class_name] = []
            for v in vids:
                classes[class_name].append(Path(root) / v)
                
    return classes

print("\nScanning Current Dataset...")
curr_classes = scan_dataset(CURRENT_DATASET_PATH, "current")
print(f"Current classes: {len(curr_classes)}, total videos: {sum(len(v) for v in curr_classes.values())}")

print("\nScanning Additional Dataset 1 (ProcessedData_vivit)...")
add1_classes = scan_dataset(ADDITIONAL_PATH_1, "archive_vivit")
print(f"Add1 classes: {len(add1_classes)}, total videos: {sum(len(v) for v in add1_classes.values())}")

print("\nScanning Additional Dataset 2 (Video_Dataset)...")
add2_classes = scan_dataset(ADDITIONAL_PATH_2, "archive_2")
print(f"Add2 classes: {len(add2_classes)}, total videos: {sum(len(v) for v in add2_classes.values())}")

# Print sample classes from each
print("\nCurrent Classes (sample 10):", sorted(list(curr_classes.keys()))[:10])
print("Add1 Classes (sample 10):", sorted(list(add1_classes.keys()))[:10])
print("Add2 Classes (sample 10):", sorted(list(add2_classes.keys()))[:10])
