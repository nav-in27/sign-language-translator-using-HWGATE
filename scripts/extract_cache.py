"""
Extracts and caches MediaPipe Holistic landmark sequences (T=32, 133 joints, 6 channels)
from all verified videos in data/raw_40words/ to data/cache/.
Multi-process worker pool for high-throughput processing.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import mediapipe as mp
import cv2
from typing import Any, Optional

# Add parent directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from training.feature_extractor import FeatureExtractor

DATA_DIR = BASE_DIR / "data" / "raw_40words"
CACHE_DIR = BASE_DIR / "data" / "cache"
MODELS_DIR = BASE_DIR / "models"
CONFIGS_DIR = BASE_DIR / "configs"

# Global worker holistic instance
worker_extractor: Optional[FeatureExtractor] = None
worker_holistic: Any = None

def init_worker():
    global worker_extractor, worker_holistic
    worker_extractor = FeatureExtractor(num_joints=133, sequence_length=32, include_motion=True)
    mp_solutions: Any = mp.solutions
    mp_holistic = mp_solutions.holistic
    worker_holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

def process_single_video(item):
    global worker_extractor, worker_holistic
    idx, word, signer, rel_path, label_idx = item
    out_filename = f"{idx:04d}_{word.replace(' ', '_')}.npz"
    out_path = CACHE_DIR / out_filename

    # If already cached and valid, skip
    if out_path.exists():
        try:
            with np.load(out_path, allow_pickle=True) as data:
                if 'features' in data and data['features'].shape == (32, 133, 6):
                    return {
                        "index": idx,
                        "word": word,
                        "label_idx": label_idx,
                        "signer": signer,
                        "cache_file": out_filename,
                        "status": "cached"
                    }
        except Exception:
            pass

    full_video_path = DATA_DIR / rel_path
    if not full_video_path.exists():
        fname = Path(rel_path).name
        matches = list(DATA_DIR.glob(f"**/{fname}"))
        if matches:
            full_video_path = matches[0]
        else:
            return {"index": idx, "word": word, "status": "missing_video"}

    assert worker_holistic is not None and worker_extractor is not None
    cap = cv2.VideoCapture(str(full_video_path))
    raw_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = worker_holistic.process(frame_rgb)
        kps = worker_extractor.extract_landmarks_from_holistic(results)
        raw_frames.append(kps)
    cap.release()

    if len(raw_frames) == 0:
        return {"index": idx, "word": word, "status": "empty_frames"}

    # Process to shape [32, 133, 6]
    features = worker_extractor.process_sequence_np(raw_frames)

    # Save to disk
    np.savez_compressed(
        out_path,
        features=features.astype(np.float32),
        word=word,
        label_idx=label_idx,
        signer=signer,
        video_path=str(rel_path)
    )

    return {
        "index": idx,
        "word": word,
        "label_idx": label_idx,
        "signer": signer,
        "cache_file": out_filename,
        "status": "success",
        "raw_frame_count": len(raw_frames)
    }

def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    metadata_path = DATA_DIR / "metadata.csv"
    if not metadata_path.exists():
        print(f"[ERROR] metadata.csv not found in {DATA_DIR}!")
        sys.exit(1)

    df = pd.read_csv(metadata_path)
    word_col = 'word' if 'word' in df.columns else 'normalized_word'
    unique_words = sorted(df[word_col].astype(str).unique().tolist())
    print(f"[+] Found {len(unique_words)} unique classes across {len(df)} video entries.")

    # Create authoritative label map
    label_to_idx = {w: i for i, w in enumerate(unique_words)}
    idx_to_label = {i: w for i, w in enumerate(unique_words)}
    label_map = {
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "classes": unique_words,
        "num_classes": len(unique_words)
    }

    # Save label maps
    for dest in [MODELS_DIR / "label_map.json", CONFIGS_DIR / "label_map.json"]:
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(label_map, f, indent=2)
    print(f"[+] Saved label_map.json ({len(unique_words)} classes)")

    # Save preprocessing config
    prep_config = {
        "num_joints": 133,
        "channels_per_joint": 6,
        "sequence_length": 32,
        "feature_dim": 798,
        "joints_breakdown": {
            "pose": 33,
            "left_hand": 21,
            "right_hand": 21,
            "face": 58
        },
        "motion_channels": ["pos_x", "pos_y", "vel_x", "vel_y", "acc_x", "acc_y"],
        "normalization": "hierarchical_torso_anchored",
        "num_classes": len(unique_words)
    }
    for dest in [MODELS_DIR / "preprocessing_config.json", CONFIGS_DIR / "preprocessing_config.json"]:
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(prep_config, f, indent=2)
    print("[+] Saved preprocessing_config.json")

    # Build tasks list
    tasks = []
    for idx, row in df.iterrows():
        word = str(row[word_col])
        signer = str(row.get('signer', 'unknown'))
        rel_path = str(row.get('video_path', ''))
        label_idx = label_to_idx[word]
        tasks.append((idx, word, signer, rel_path, label_idx))

    print(f"[*] Starting parallel extraction of {len(tasks)} videos into {CACHE_DIR}...")
    num_workers = min(6, os.cpu_count() or 4)
    print(f"[*] Using {num_workers} parallel worker processes.")

    t0 = time.time()
    results = []
    completed = 0
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor:
        futures = {executor.submit(process_single_video, item): item for item in tasks}
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            completed += 1
            if completed % 25 == 0 or completed == len(tasks):
                elapsed = time.time() - t0
                fps = completed / elapsed if elapsed > 0 else 0
                print(f"  [{completed}/{len(tasks)}] processed ({completed*100/len(tasks):.1f}%) | {fps:.2f} videos/sec | Elapsed: {elapsed:.1f}s")

    # Save cache manifest
    manifest_path = CACHE_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    successes = [r for r in results if r.get('status') in ['success', 'cached']]
    print("============================================================")
    print(f"[+] Extraction complete in {time.time() - t0:.2f}s!")
    print(f"[+] Total cached: {len(successes)} / {len(tasks)} video sequences")
    print(f"[+] Cache manifest saved to {manifest_path}")
    print("============================================================")

    if len(successes) < 200:
        print("[ERROR] Insufficient cached sequences!")
        sys.exit(1)

if __name__ == "__main__":
    main()
