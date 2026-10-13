"""
Automated Multi-Threaded Landmark Extractor for Full 104-Class ISL Vocabulary.
Automatically discovers classes from:
  1. Archive-2 Video Dataset (C:\\Users\\navee\\Downloads\\archive (2)\\Video_Dataset\\Video_Dataset)
  2. Vivit Processed Data (C:\\Users\\navee\\Downloads\\archive\\ProcessedData_vivit)

Saves all extracted 32-frame [32, 133, 6] sequences to:
  data/additional_landmarks/<class_name>/<stem>.npz

Features:
  - 8-worker MediaPipe Holistic pool for maximum CPU utilization
  - Bit-exact parity with FeatureExtractor
  - Skip existing cached files
  - Automated session tracking for leakage-free splitting
"""

import os
import sys
import time
import queue
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
import mediapipe as mp
from typing import Any, List, Dict, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "training"))

from training.feature_extractor import feature_extractor

VID_DATASET_DIR = Path(r"C:\Users\navee\Downloads\archive (2)\Video_Dataset\Video_Dataset")
VIVIT_DIR = Path(r"C:\Users\navee\Downloads\archive\ProcessedData_vivit")
OUT_DIR = ROOT_DIR / "data" / "additional_landmarks"
CONFIGS_DIR = ROOT_DIR / "configs"
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


class HolisticWorkerPool:
    def __init__(self, num_workers=8):
        self.num_workers = num_workers
        self.instances = []
        mp_solutions: Any = mp.solutions
        mp_holistic = mp_solutions.holistic
        print(f"[*] Initializing {num_workers} MediaPipe Holistic workers in pool...")
        for _ in range(num_workers):
            h = mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4
            )
            self.instances.append(h)
        self.pool = queue.Queue()
        for h in self.instances:
            self.pool.put(h)
        print("[+] Workers initialized successfully.")

    def get_worker(self):
        return self.pool.get()

    def return_worker(self, h):
        self.pool.put(h)

    def close(self):
        while not self.pool.empty():
            h = self.pool.get()
            try:
                h.close()
            except Exception:
                pass


def process_single_video(video_path: Path, out_path: Path, label: str, source: str, pool: HolisticWorkerPool) -> Tuple[bool, str]:
    if out_path.exists():
        try:
            d = np.load(out_path, allow_pickle=True)
            if "features" in d and d["features"].shape == (32, 133, 6):
                return True, "cached"
        except Exception:
            pass

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, "cannot_open"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 6:
        cap.release()
        return False, "too_short"

    indices = np.linspace(0, total_frames - 1, 32, dtype=int)
    target_set = set(indices)

    frames_dict = {}
    current_idx = 0
    while cap.isOpened() and len(frames_dict) < 32:
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in target_set:
            frames_dict[current_idx] = frame
        current_idx += 1
    cap.release()

    if len(frames_dict) < 12:
        return False, "not_enough_frames"

    ordered_frames = [frames_dict[i] for i in indices if i in frames_dict]

    worker = pool.get_worker()
    kps_list = []
    try:
        for f in ordered_frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            res = worker.process(rgb)
            kps = feature_extractor.extract_landmarks_from_holistic(res)
            kps_list.append(kps)
    finally:
        pool.return_worker(worker)

    # Validate hand presence (at least 3 frames with hand detection)
    raw_arr = np.array(kps_list, dtype=np.float32)
    lh_detected = np.sum(np.abs(raw_arr[:, 33:54, :]) > 1e-4, axis=(1, 2))
    rh_detected = np.sum(np.abs(raw_arr[:, 54:75, :]) > 1e-4, axis=(1, 2))
    active_hands = np.sum((lh_detected > 0) | (rh_detected > 0))
    if active_hands < 3:
        # User hands not sufficiently detected
        return False, "no_hands_detected"

    feat = feature_extractor.process_sequence_np(kps_list)
    if feat.shape != (32, 133, 6) or np.isnan(feat).any() or np.isinf(feat).any():
        return False, "corrupted_features"

    stem = video_path.stem
    session_id = stem
    if "_left_tilt" in stem:
        session_id = stem.replace("_left_tilt", "")
    elif "_right_tilt" in stem:
        session_id = stem.replace("_right_tilt", "")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        features=feat,
        keypoints=raw_arr,
        label=label,
        source=source,
        session_id=session_id,
        source_video=str(video_path)
    )
    return True, "extracted"


def discover_and_extract_all():
    print("=" * 75)
    print("AUTOMATED 104-CLASS DATASET DISCOVERY & EXTRACTOR")
    print("=" * 75)

    # 1. Discover Video_Dataset classes (61 classes total)
    vid_classes = {}
    for d in sorted(VID_DATASET_DIR.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            norm = d.name.lower().replace(" ", "_").replace("-", "_")
            if norm == "writer":
                norm = "write"
            vids = list(d.glob("*.mp4")) + list(d.glob("*.MOV"))
            if vids:
                vid_classes[norm] = (d.name, vids)

    print(f"[+] Discovered {len(vid_classes)} classes in Video_Dataset.")

    # 2. Discover Vivit classes with >= 12 samples (42 classes)
    vivit_classes = {}
    for d in sorted(VIVIT_DIR.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            norm = d.name.lower().replace(" ", "_").replace("-", "_")
            vids = list(d.glob("*.MOV")) + list(d.glob("*.mp4"))
            if len(vids) >= 12:
                vivit_classes[norm] = (d.name, vids)

    print(f"[+] Discovered {len(vivit_classes)} eligible classes in Vivit (>= 12 samples).")

    all_class_names = sorted(list(set(vid_classes.keys()).union(set(vivit_classes.keys()))))
    print(f"\n[+] TOTAL UNIFIED VOCABULARY: {len(all_class_names)} CLASSES")
    assert len(all_class_names) >= 100, f"Expected >= 100 classes, found {len(all_class_names)}"

    # Save manifest
    manifest = {
        "num_classes": len(all_class_names),
        "classes": all_class_names,
        "class_to_idx": {c: i for i, c in enumerate(all_class_names)},
        "idx_to_class": {i: c for i, c in enumerate(all_class_names)},
        "sources": {
            "video_dataset_classes": sorted(list(vid_classes.keys())),
            "vivit_classes": sorted(list(vivit_classes.keys()))
        }
    }
    manifest_path = CONFIGS_DIR / "selected_104_classes_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[+] Saved authoritative manifest to: {manifest_path}")

    # Build queue of tasks
    tasks = []
    for norm_cls in all_class_names:
        dst_dir = OUT_DIR / norm_cls
        if norm_cls in vid_classes:
            _, vids = vid_classes[norm_cls]
            for v in vids:
                out_p = dst_dir / f"{v.stem}.npz"
                tasks.append((v, out_p, norm_cls, "video_dataset"))
        elif norm_cls in vivit_classes:
            _, vids = vivit_classes[norm_cls]
            for v in vids:
                out_p = dst_dir / f"{v.stem}.npz"
                tasks.append((v, out_p, norm_cls, "vivit"))

    print(f"[*] Total video tasks to process: {len(tasks)}")

    # Initialize worker pool
    pool = HolisticWorkerPool(num_workers=8)
    t_start = time.time()
    extracted_count = 0
    cached_count = 0
    failed_count = 0

    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_task = {
                executor.submit(process_single_video, v, out_p, lbl, src, pool): (v, lbl)
                for v, out_p, lbl, src in tasks
            }

            for i, future in enumerate(as_completed(future_to_task)):
                v, lbl = future_to_task[future]
                try:
                    ok, status = future.result()
                    if ok:
                        if status == "cached":
                            cached_count += 1
                        else:
                            extracted_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    failed_count += 1

                done = i + 1
                if done % 100 == 0 or done == len(tasks):
                    elapsed = time.time() - t_start
                    rate = done / elapsed if elapsed > 0 else 0
                    print(f"  [{done}/{len(tasks)}] Extracted: {extracted_count} | Cached: {cached_count} | Failed: {failed_count} | Speed: {rate:.1f} vids/s")

    finally:
        pool.close()

    total_time = time.time() - t_start
    print("=" * 75)
    print(f"[+] Extraction complete in {total_time:.1f}s ({total_time/60:.1f} minutes)!")
    print(f"    Newly extracted: {extracted_count}")
    print(f"    Cached from prior runs: {cached_count}")
    print(f"    Failed/Skipped: {failed_count}")
    print(f"    Total valid sequences: {extracted_count + cached_count}")
    print("=" * 75)


if __name__ == "__main__":
    discover_and_extract_all()
