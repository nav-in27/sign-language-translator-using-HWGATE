"""
Download and Validation Script for ISL Isolated 40-Words Dataset
Dataset: vidit031/isl-isolated-40words on Hugging Face
Downloads directly to data/raw_40words/ and generates reports.
"""

import os
import sys
import json
import hashlib
import cv2
import pandas as pd
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download

REPO_ID = "vidit031/isl-isolated-40words"
DATA_DIR = Path("data/raw_40words")
REPORTS_DIR = Path("reports")

def check_internet():
    import urllib.request
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=5)
        return True
    except Exception as e:
        print(f"[ERROR] Cannot connect to Hugging Face: {e}")
        return False

def verify_file_sha256(filepath, expected_sha256):
    if not os.path.exists(filepath):
        return False
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower() == expected_sha256.lower()

def download_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    metadata_path = DATA_DIR / "metadata.csv"

    print("============================================================")
    print("      ISL 40-WORD DATASET AUTOMATIC DOWNLOAD & AUDIT        ")
    print(f"      Repository: {REPO_ID}                                ")
    print(f"      Destination: {DATA_DIR.resolve()}                    ")
    print("============================================================")

    # Check if internet is available
    has_internet = check_internet()

    # Step 1: Download repository snapshot
    if has_internet:
        print(f"[*] Downloading repository {REPO_ID} via huggingface_hub...")
        try:
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                local_dir=str(DATA_DIR),
                max_workers=8
            )
            print("[+] Download / Sync complete!")
        except Exception as e:
            print(f"[WARNING] snapshot_download encountered: {e}. Checking local files...")
    else:
        print("[!] Offline mode: verifying existing local files...")

    if not metadata_path.exists():
        print(f"[FATAL] {metadata_path} not found! Cannot proceed without metadata.")
        sys.exit(1)

    # Step 2: Parse and Audit metadata.csv
    df = pd.read_csv(metadata_path)
    print(f"[+] Loaded metadata.csv with {len(df)} entries.")

    expected_words = [
        "hello", "goodbye", "thank you", "sorry", "please", "yes", "no", "help",
        "stop", "okay", "me", "you", "he", "she", "mother", "father", "brother",
        "sister", "friend", "teacher", "student", "home", "school", "hospital",
        "market", "eat", "drink", "water", "food", "tea", "come", "go", "sit",
        "stand", "read", "write", "what", "where", "when", "today"
    ]

    # Verify columns
    word_col = 'word' if 'word' in df.columns else 'normalized_word'
    classes_found = sorted(df[word_col].astype(str).unique().tolist())
    print(f"[+] Found {len(classes_found)} unique classes in metadata.")

    # Validate video files
    valid_videos = []
    missing_videos = []
    corrupt_videos = []
    video_durations = []
    video_fps_list = []
    video_frame_counts = []

    print("[*] Auditing all video files on disk...")
    for idx, row in df.iterrows():
        rel_path = str(row.get('video_path') or '')
        local_path = DATA_DIR / rel_path
        if not local_path.exists():
            # Try searching by original_filename or basename
            fname = os.path.basename(rel_path)
            # search
            matches = list(DATA_DIR.glob(f"**/{fname}"))
            if matches:
                local_path = matches[0]
            else:
                missing_videos.append(rel_path)
                continue

        # Check if video opens and has frames
        cap = cv2.VideoCapture(str(local_path))
        if not cap.isOpened():
            corrupt_videos.append(str(local_path))
            continue

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if frame_count <= 0:
            corrupt_videos.append(str(local_path))
            continue

        duration = frame_count / fps if fps > 0 else 0
        video_durations.append(duration)
        video_fps_list.append(fps)
        video_frame_counts.append(frame_count)
        valid_videos.append({
            "index": idx,
            "word": str(row[word_col]),
            "signer": str(row.get('signer', 'unknown')),
            "local_path": str(local_path.resolve()),
            "frame_count": frame_count,
            "fps": fps,
            "duration": duration
        })

    print(f"[+] Valid videos verified: {len(valid_videos)} / {len(df)}")
    if missing_videos:
        print(f"[!] Missing videos: {len(missing_videos)}")
    if corrupt_videos:
        print(f"[!] Corrupt videos: {len(corrupt_videos)}")

    # Class distribution statistics
    class_counts = df[word_col].value_counts().to_dict()
    counts_list = list(class_counts.values())

    report_data = {
        "dataset_name": REPO_ID,
        "total_metadata_entries": len(df),
        "total_valid_videos": len(valid_videos),
        "total_missing_videos": len(missing_videos),
        "total_corrupt_videos": len(corrupt_videos),
        "num_classes": len(classes_found),
        "classes": classes_found,
        "class_counts": class_counts,
        "min_samples_per_class": min(counts_list) if counts_list else 0,
        "max_samples_per_class": max(counts_list) if counts_list else 0,
        "median_samples_per_class": float(pd.Series(counts_list).median()) if counts_list else 0,
        "mean_samples_per_class": float(pd.Series(counts_list).mean()) if counts_list else 0,
        "num_signers": len(df['signer'].unique()) if 'signer' in df.columns else 0,
        "signers": sorted(df['signer'].astype(str).unique().tolist()) if 'signer' in df.columns else [],
        "avg_duration_sec": float(pd.Series(video_durations).mean()) if video_durations else 0,
        "avg_fps": float(pd.Series(video_fps_list).mean()) if video_fps_list else 0,
        "avg_frames": float(pd.Series(video_frame_counts).mean()) if video_frame_counts else 0,
    }

    # Save JSON report
    json_report_path = REPORTS_DIR / "dataset_report.json"
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"[+] Saved JSON report to {json_report_path}")

    # Save TXT report
    txt_report_path = REPORTS_DIR / "dataset_report.txt"
    with open(txt_report_path, "w", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write("      DATASET AUDIT REPORT: ISL ISOLATED 40-WORDS           \n")
        f.write("============================================================\n\n")
        f.write(f"Dataset: {REPO_ID}\n")
        f.write(f"Total entries in metadata: {report_data['total_metadata_entries']}\n")
        f.write(f"Valid verified video files: {report_data['total_valid_videos']}\n")
        f.write(f"Missing videos: {report_data['total_missing_videos']}\n")
        f.write(f"Corrupt videos: {report_data['total_corrupt_videos']}\n")
        f.write(f"Number of classes: {report_data['num_classes']}\n")
        f.write(f"Unique signers: {report_data['num_signers']}\n")
        f.write(f"Samples per class: Min={report_data['min_samples_per_class']}, Max={report_data['max_samples_per_class']}, Median={report_data['median_samples_per_class']:.1f}\n")
        f.write(f"Average video duration: {report_data['avg_duration_sec']:.2f} seconds\n")
        f.write(f"Average FPS: {report_data['avg_fps']:.1f} frames/sec\n")
        f.write(f"Average frame count: {report_data['avg_frames']:.1f} frames\n\n")
        f.write("Class Distribution:\n")
        for cls, count in sorted(class_counts.items(), key=lambda x: x[0]):
            f.write(f"  - {cls:<15}: {count} samples\n")
        f.write("\nSigners List:\n")
        for s in report_data['signers']:
            f.write(f"  - {s}\n")
        f.write("============================================================\n")
    print(f"[+] Saved Text report to {txt_report_path}")

    if len(valid_videos) < 100:
        print("[ERROR] Insufficient valid videos! Check network or local directory.")
        sys.exit(1)

    print("[SUCCESS] Dataset download and audit completed successfully!")
    return report_data

if __name__ == "__main__":
    download_dataset()
