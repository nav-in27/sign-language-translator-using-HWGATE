import os
import json
import csv
from pathlib import Path
import cv2
import re

CURRENT_DATASET_PATH = Path(r"c:\Users\navee\.gemini\antigravity-ide\scratch\isl-translator\data\raw_40words")
ADD1_PATH = Path(r"C:\Users\navee\Downloads\archive\ProcessedData_vivit")
ADD2_PATH = Path(r"C:\Users\navee\Downloads\archive (2)\Video_Dataset\Video_Dataset")
if not ADD2_PATH.exists():
    ADD2_PATH = Path(r"C:\Users\navee\Downloads\archive (2)\Video_Dataset")

reports_dir = Path(r"c:\Users\navee\.gemini\antigravity-ide\scratch\isl-translator\reports")
reports_dir.mkdir(parents=True, exist_ok=True)

video_exts = {".mp4", ".mov", ".avi", ".webm", ".mkv"}

def probe_video(video_path):
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
        "fps": round(fps, 1) if fps > 0 else 30.0,
        "frames": frames,
        "res": f"{w}x{h}",
        "duration": round(dur, 2)
    }

def analyze_dataset(base_path, dataset_name):
    class_stats = {}
    for root, dirs, files in os.walk(base_path):
        vids = [Path(root) / f for f in files if Path(f).suffix.lower() in video_exts]
        if vids:
            class_name = Path(root).name
            if class_name.lower() == "video_dataset" or class_name.lower() == "sample videos":
                continue
            
            # Sample up to 3 videos for metadata probing
            samples = vids[:3]
            sample_infos = [probe_video(s) for s in samples]
            sample_infos = [si for si in sample_infos if si is not None]
            
            if sample_infos:
                avg_dur = round(sum(si["duration"] for si in sample_infos) / len(sample_infos), 2)
                mode_fps = sample_infos[0]["fps"]
                mode_res = sample_infos[0]["res"]
            else:
                avg_dur = 0.0
                mode_fps = 0.0
                mode_res = "Unknown"
                
            file_types = list(set(v.suffix.lower() for v in vids))
            
            # Signer information heuristic
            signers = set()
            for v in vids:
                stem = v.stem
                # Check for patterns like User001, signer_1, WIN_..., MVI_...
                if "user" in stem.lower():
                    m = re.search(r"user\d+", stem.lower())
                    if m: signers.add(m.group(0))
                elif "signer" in stem.lower():
                    m = re.search(r"signer\d+", stem.lower())
                    if m: signers.add(m.group(0))
                elif stem.startswith("MVI_"):
                    # Canon sequence
                    signers.add("Canon_Session")
                elif stem.startswith("WIN_"):
                    # Windows webcam session
                    signers.add("Webcam_Session")
                else:
                    signers.add("Generic_Signer")
                    
            signer_info = f"{len(signers)} session(s)/signer(s)" if signers else "Unspecified"
            
            class_stats[class_name] = {
                "dataset": dataset_name,
                "class_name": class_name,
                "sample_count": len(vids),
                "video_count": len(vids),
                "file_type": "/".join(file_types),
                "average_duration": avg_dur,
                "fps": mode_fps,
                "resolution": mode_res,
                "signer_information": signer_info,
                "usable_samples": len(vids)
            }
    return class_stats

print("[*] Analyzing Current Dataset (raw_40words)...")
curr_stats = analyze_dataset(CURRENT_DATASET_PATH, "Current_Dataset (Vidit031)")

print("[*] Analyzing Additional Dataset 1 (ProcessedData_vivit)...")
add1_stats = analyze_dataset(ADD1_PATH, "Additional_1 (ProcessedData_vivit)")

print("[*] Analyzing Additional Dataset 2 (Video_Dataset)...")
add2_stats = analyze_dataset(ADD2_PATH, "Additional_2 (Video_Dataset)")

# 1. Generate reports/dataset_comparison.csv
comparison_csv = reports_dir / "dataset_comparison.csv"
fieldnames = [
    "dataset", "class_name", "sample_count", "video_count", "file_type",
    "average_duration", "fps", "resolution", "signer_information", "usable_samples"
]

all_rows = list(curr_stats.values()) + list(add1_stats.values()) + list(add2_stats.values())

with open(comparison_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)

print(f"[+] Written: {comparison_csv} with {len(all_rows)} rows")

# 2. Generate reports/dataset_comparison_summary.json
summary_json = reports_dir / "dataset_comparison_summary.json"
summary_data = {
    "current_dataset": {
        "name": "Current_Dataset (Vidit031 + ISL500 unified)",
        "number_of_classes": len(curr_stats),
        "number_of_videos": sum(x["video_count"] for x in curr_stats.values()),
        "classes_available": sorted(list(curr_stats.keys())),
        "samples_per_class": {k: curr_stats[k]["sample_count"] for k in sorted(curr_stats.keys())}
    },
    "additional_dataset_1": {
        "name": "Additional_1 (ProcessedData_vivit)",
        "number_of_classes": len(add1_stats),
        "number_of_videos": sum(x["video_count"] for x in add1_stats.values()),
        "classes_available": sorted(list(add1_stats.keys())),
        "samples_per_class": {k: add1_stats[k]["sample_count"] for k in sorted(add1_stats.keys())}
    },
    "additional_dataset_2": {
        "name": "Additional_2 (Video_Dataset)",
        "number_of_classes": len(add2_stats),
        "number_of_videos": sum(x["video_count"] for x in add2_stats.values()),
        "classes_available": sorted(list(add2_stats.keys())),
        "samples_per_class": {k: add2_stats[k]["sample_count"] for k in sorted(add2_stats.keys())}
    },
    "total_combined": {
        "total_datasets": 3,
        "total_classes_across_all": len(set(list(curr_stats.keys()) + list(add1_stats.keys()) + list(add2_stats.keys()))),
        "total_videos_across_all": sum(x["video_count"] for x in all_rows)
    }
}

with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(summary_data, f, indent=2)

print(f"[+] Written: {summary_json}")
