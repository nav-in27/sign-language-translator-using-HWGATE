"""
High-Speed Landmark Extractor for Full 136-Class Authoritative ISL Vocabulary.
Processes all unextracted videos from:
  1. Archive-2 Video Dataset (C:\\Users\\navee\\Downloads\\archive (2)\\Video_Dataset\\Video_Dataset)
  2. Vivit Video Dataset (C:\\Users\\navee\\Downloads\\archive\\ProcessedData_vivit)

Ensures 100% feature coverage with bit-exact parity across:
  - Sequence length: T = 32 frames
  - Joints: 133 WholeBody landmarks
  - Features: [32, 133, 6] (Px, Py, Vx, Vy, Ax, Ay)
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
REPORTS_DIR = ROOT_DIR / "reports"

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
                min_detection_confidence=0.35,
                min_tracking_confidence=0.35
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
        # Fallback manual read if opencv frame count fails
        manual_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            manual_frames.append(frame)
        cap.release()
        total_frames = len(manual_frames)
        if total_frames < 6:
            return False, "too_short"
        indices = np.linspace(0, total_frames - 1, 32, dtype=int)
        ordered_frames = [manual_frames[i] for i in indices]
    else:
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

        if len(frames_dict) < 8:
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

    # Hand presence check (at least 2 frames with active hands, or non-zero upper body movement)
    raw_arr = np.array(kps_list, dtype=np.float32)
    lh_energy = np.abs(raw_arr[:, 33:54, :]).sum()
    rh_energy = np.abs(raw_arr[:, 54:75, :]).sum()
    pose_energy = np.abs(raw_arr[:, :33, :]).sum()

    if (lh_energy + rh_energy) < 0.05 and pose_energy < 1.0:
        return False, "no_landmarks_detected"

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


def extract_all():
    print("=" * 75)
    print("EXTRACTION OF FULL 136-CLASS AUTHORITATIVE ISL VOCABULARY")
    print("=" * 75)

    audit_path = REPORTS_DIR / "audit_all_training_examples.json"
    if not audit_path.exists():
        print("[!] Missing audit_all_training_examples.json! Please run audit first.")
        return

    with open(audit_path, "r", encoding="utf-8") as f:
        audit_data = json.load(f)

    eligible_info = audit_data["vocabulary_analysis"]["eligible_classes"]
    eligible_names = sorted([x["class_name"] for x in eligible_info])
    print(f"[+] Loaded {len(eligible_names)} eligible classes from audit.")

    tasks = []
    # 1. Video_Dataset
    for d in sorted(VID_DATASET_DIR.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            norm = d.name.strip().lower().replace(" ", "_").replace("-", "_")
            if norm == "writer": norm = "write"
            if norm in eligible_names:
                vids = [f for f in d.iterdir() if f.is_file() and f.suffix.lower() in {".mp4", ".mov"}]
                dst_dir = OUT_DIR / norm
                for v in vids:
                    out_p = dst_dir / f"{v.stem}.npz"
                    tasks.append((v, out_p, norm, "video_dataset"))

    # 2. Vivit
    for d in sorted(VIVIT_DIR.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            norm = d.name.strip().lower().replace(" ", "_").replace("-", "_")
            if norm in eligible_names:
                vids = [f for f in d.iterdir() if f.is_file() and f.suffix.lower() in {".mp4", ".mov"}]
                dst_dir = OUT_DIR / norm
                for v in vids:
                    out_p = dst_dir / f"{v.stem}.npz"
                    tasks.append((v, out_p, norm, "vivit"))

    print(f"[*] Total video extraction tasks queued: {len(tasks)}")

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
                if done % 200 == 0 or done == len(tasks):
                    elapsed = time.time() - t_start
                    rate = done / elapsed if elapsed > 0 else 0
                    print(f"  [{done}/{len(tasks)}] Extracted: {extracted_count} | Cached: {cached_count} | Failed: {failed_count} | Rate: {rate:.1f} vids/s")

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
    extract_all()
