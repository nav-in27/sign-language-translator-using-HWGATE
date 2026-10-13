import os
import sys
import time
import queue
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
import mediapipe as mp
from typing import Any

ROOT = Path(r"c:\Users\navee\.gemini\antigravity-ide\scratch\isl-translator")
sys.path.insert(0, str(ROOT / "training"))
from feature_extractor import feature_extractor

ARCHIVE2_DIR = Path(r"C:\Users\navee\Downloads\archive (2)\Video_Dataset\Video_Dataset")
OUT_BASE = ROOT / "data" / "additional_landmarks"

CLASSES_TO_EXTRACT = [
    ("Bear", "bear"),
    ("Break", "break"),
    ("Clean", "clean"),
    ("Close", "close"),
    ("Cook", "cook"),
    ("Cry", "cry"),
    ("Elephant", "elephant"),
    ("Give", "give"),
    ("Jump", "jump"),
    ("Key", "key"),
    ("Knife", "knife"),
    ("Lion", "lion"),
    ("Monkey", "monkey"),
    ("Temple", "temple"),
    ("Tiger", "tiger"),
    ("Umbrella", "umbrella"),
    ("Uncle", "uncle"),
    ("Wife", "wife"),
    ("Wrong", "wrong")
]

class HolisticWorkerPool:
    def __init__(self, num_workers=6):
        self.num_workers = num_workers
        self.instances = []
        mp_solutions: Any = mp.solutions
        mp_holistic = mp_solutions.holistic
        print(f"[*] Initializing {num_workers} MediaPipe Holistic workers...")
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
            try: h.close()
            except: pass

def process_single_video(vpath, out_file, label, pool):
    if out_file.exists():
        return True, "cached"
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        return False, "cannot_open"
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 8:
        cap.release()
        return False, "too_short"
    
    indices = np.linspace(0, total_frames - 1, 32, dtype=int)
    target_set = set(indices)
    
    frames_dict = {}
    current_idx = 0
    while cap.isOpened() and len(frames_dict) < 32:
        ret, frame = cap.read()
        if not ret: break
        if current_idx in target_set:
            frames_dict[current_idx] = frame
        current_idx += 1
    cap.release()
    
    if len(frames_dict) < 16:
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
        
    feat = feature_extractor.process_sequence_np(kps_list)
    np.savez_compressed(out_file, features=feat, keypoints=np.array(kps_list, dtype=np.float32), label=label, source_video=str(vpath))
    return True, "extracted"

def main():
    print("=" * 70)
    print("EXTRACTING 19 SUPPLEMENTARY CLASSES (60 SAMPLES EACH) FOR 50-CLASS DATASET")
    print("=" * 70)
    
    pool = HolisticWorkerPool(num_workers=6)
    total_extracted = 0
    t_start = time.time()
    
    for raw_folder, norm_label in CLASSES_TO_EXTRACT:
        src_dir = ARCHIVE2_DIR / raw_folder
        dst_dir = OUT_BASE / norm_label
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        videos = list(src_dir.glob("*.mp4"))
        t0 = time.time()
        
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [
                executor.submit(process_single_video, v, dst_dir / f"{v.stem}.npz", norm_label, pool)
                for v in videos
            ]
            results = [f.result() for f in futures]
            
        success_count = sum(1 for ok, _ in results if ok)
        cached_count = sum(1 for _, st in results if st == "cached")
        total_extracted += success_count
        elapsed = time.time() - t0
        print(f"[+] Class '{norm_label}': {success_count}/{len(videos)} valid ({cached_count} cached) in {elapsed:.1f}s")
        
    pool.close()
    t_total = time.time() - t_start
    print("=" * 70)
    print(f"[+] All 19 classes extracted: {total_extracted} total sequences in {t_total:.1f}s ({t_total/60:.1f}m)")
    print("=" * 70)

if __name__ == "__main__":
    main()
