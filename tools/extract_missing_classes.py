import os
import sys
import time
import queue
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List, Dict

# Ensure stdout handles encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "training"))

from training.feature_extractor import feature_extractor

VID_DATASET_DIR = Path(r"C:\Users\navee\Downloads\archive (2)\Video_Dataset\Video_Dataset")
RAW40_DIR = ROOT_DIR / "data" / "raw_40words"
VIVIT_DIR = Path(r"C:\Users\navee\Downloads\archive\ProcessedData_vivit")
OUT_DIR = ROOT_DIR / "data" / "additional_landmarks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

class HolisticWorkerPool:
    def __init__(self, num_workers=6):
        self.num_workers = num_workers
        self.instances = []
        mp_holistic = mp.solutions.holistic
        print(f"[*] Initializing {num_workers} MediaPipe Holistic workers...")
        for _ in range(num_workers):
            h = mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.25,
                min_tracking_confidence=0.25
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

    # Read all frames or sample up to 48 frames
    step = max(1, total_frames // 48)
    frames = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            # If resolution is very small (Vivit 224x224), upscale to 384x384 for hand detection
            if frame.shape[1] < 300:
                frame = cv2.resize(frame, (384, 384), interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
        idx += 1
    cap.release()

    if len(frames) < 6:
        return False, "not_enough_frames"

    worker = pool.get_worker()
    kps_list = []
    try:
        for f in frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            res = worker.process(rgb)
            kps = feature_extractor.extract_landmarks_from_holistic(res)
            kps_list.append(kps)
    finally:
        pool.return_worker(worker)

    raw_arr = np.array(kps_list, dtype=np.float32) # [T, 133, 2]
    
    # Check that at least pose or hands are active
    lh_det = np.sum(np.abs(raw_arr[:, 33:54, :]) > 1e-4, axis=(1, 2))
    rh_det = np.sum(np.abs(raw_arr[:, 54:75, :]) > 1e-4, axis=(1, 2))
    pose_det = np.sum(np.abs(raw_arr[:, 0:33, :]) > 1e-4, axis=(1, 2))
    
    has_activity = (np.sum(lh_det > 0) + np.sum(rh_det > 0) >= 1) or (np.sum(pose_det > 0) >= 6)
    if not has_activity:
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

def main():
    print("=" * 70)
    print("EXTRACTION OF REMAINING 108-CLASS VOCABULARY VIDEOS")
    print("=" * 70)

    # 1. Archive-2 classes to extract
    archive2_targets = {
        "sparrow": "Sparrow",
        "switch": "Switch",
        "turtle": "Turtle",
        "vegetables": "Vegetables",
        "volcano": "Volcano",
        "write": "Writer"
    }

    # 2. Raw40 classes to extract
    raw40_targets = ["brother", "goodbye", "hospital", "read", "sorry", "stand", "stop", "when"]

    # 3. Vivit classes to extract
    vivit_targets = [
        "animal", "bad", "big", "bird", "cat", "cold", "cow", "dog", 
        "fast", "fish", "good", "happy", "hat", "healthy", "horse", "hot"
    ]

    tasks = []

    # Queue Archive-2
    for norm_cls, folder_name in archive2_targets.items():
        cls_dir = VID_DATASET_DIR / folder_name
        if cls_dir.exists():
            vids = list(cls_dir.glob("*.mp4")) + list(cls_dir.glob("*.MOV"))
            dst_dir = OUT_DIR / norm_cls
            for v in vids:
                out_p = dst_dir / f"a2_{v.stem}.npz"
                tasks.append((v, out_p, norm_cls, "archive2"))

    # Queue Raw40
    for norm_cls in raw40_targets:
        cls_dir = RAW40_DIR / norm_cls
        if cls_dir.exists():
            vids = list(cls_dir.glob("*.mp4")) + list(cls_dir.glob("*.MOV"))
            dst_dir = OUT_DIR / norm_cls
            for v in vids:
                out_p = dst_dir / f"r40_{v.stem}.npz"
                tasks.append((v, out_p, norm_cls, "raw40"))

    # Queue Vivit
    for norm_cls in vivit_targets:
        cls_dir = VIVIT_DIR / norm_cls
        if cls_dir.exists():
            vids = list(cls_dir.glob("*.MOV")) + list(cls_dir.glob("*.mp4"))
            dst_dir = OUT_DIR / norm_cls
            for v in vids:
                out_p = dst_dir / f"viv_{v.stem}.npz"
                tasks.append((v, out_p, norm_cls, "vivit"))

    print(f"[*] Total video extraction tasks: {len(tasks)}")

    pool = HolisticWorkerPool(num_workers=6)
    t0 = time.time()
    extracted = 0
    cached = 0
    failed = 0

    try:
        with ThreadPoolExecutor(max_workers=6) as executor:
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
                            cached += 1
                        else:
                            extracted += 1
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1

                if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
                    elapsed = time.time() - t0
                    print(f"Progress: [{i+1}/{len(tasks)}] | Extracted: {extracted} | Cached: {cached} | Failed: {failed} | Time: {elapsed:.1f}s", flush=True)
    finally:
        pool.close()

    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print(f"EXTRACTION COMPLETED in {total_time:.1f}s")
    print(f"Total Processed: {len(tasks)} (Extracted: {extracted}, Cached: {cached}, Failed: {failed})")
    print("=" * 70)

if __name__ == "__main__":
    main()
