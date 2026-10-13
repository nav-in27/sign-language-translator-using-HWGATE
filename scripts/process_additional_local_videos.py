import os
import sys
import hashlib
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
import mediapipe as mp
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from training.feature_extractor import feature_extractor

ADD1_PATH = Path(r"C:\Users\navee\Downloads\archive\ProcessedData_vivit")
ADD2_PATH = Path(r"C:\Users\navee\Downloads\archive (2)\Video_Dataset\Video_Dataset")
if not ADD2_PATH.exists():
    ADD2_PATH = Path(r"C:\Users\navee\Downloads\archive (2)\Video_Dataset")

OUTPUT_DIR = BASE_DIR / "data" / "additional_landmarks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Matching classes mapping
CLASS_MAP = {
    # From Add2
    "Hello": "hello",
    "Come": "come",
    "Drink": "drink",
    "Tea": "tea",
    "Thank you": "thank_you",
    "Writer": "write",
    # From Add1
    "today": "today"
}

def get_file_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def process_single_video(video_path, label):
    try:
        sha = get_file_sha256(video_path)
        out_label_dir = OUTPUT_DIR / label
        out_label_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_label_dir / f"{sha[:10]}_{video_path.stem}.npz"
        if out_file.exists():
            return str(out_file), "success"
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, "cannot_open"
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            # Count manually if needed
            frames_list = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                frames_list.append(frame)
            cap.release()
            total_frames = len(frames_list)
            if total_frames == 0:
                return None, "empty_video"
        else:
            frames_list = None
            
        # Target: exactly 32 temporal frames
        indices = np.linspace(0, total_frames - 1, 32, dtype=int)
        
        extracted_frames = []
        mp_solutions: Any = mp.solutions
        mp_holistic = mp_solutions.holistic
        
        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:
            if frames_list is not None:
                for idx in indices:
                    frame = frames_list[idx]
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = holistic.process(rgb)
                    kps = feature_extractor.extract_landmarks_from_holistic(res)
                    extracted_frames.append(kps)
            else:
                current_idx = 0
                next_target_pos = 0
                while cap.isOpened() and next_target_pos < 32:
                    ret, frame = cap.read()
                    if not ret: break
                    if current_idx == indices[next_target_pos]:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        res = holistic.process(rgb)
                        kps = feature_extractor.extract_landmarks_from_holistic(res)
                        extracted_frames.append(kps)
                        next_target_pos += 1
                        # Handle duplicate indices if total_frames < 32
                        while next_target_pos < 32 and indices[next_target_pos] == current_idx:
                            extracted_frames.append(kps.copy())
                            next_target_pos += 1
                    current_idx += 1
                cap.release()
                
        # Pad if missing
        while len(extracted_frames) < 32:
            if extracted_frames:
                extracted_frames.append(extracted_frames[-1].copy())
            else:
                extracted_frames.append(np.zeros((133, 2), dtype=np.float32))
                
        raw_seq = np.array(extracted_frames[:32], dtype=np.float32) # [32, 133, 2]
        
        # Verify hand presence in at least 4 frames
        lh_conf = np.sum(np.abs(raw_seq[:, 33:54, :]) > 1e-4, axis=(1,2))
        rh_conf = np.sum(np.abs(raw_seq[:, 54:75, :]) > 1e-4, axis=(1,2))
        active_hand_frames = np.sum((lh_conf > 0) | (rh_conf > 0))
        if active_hand_frames < 4:
            return None, "insufficient_hands"
            
        sha = get_file_sha256(video_path)
        out_label_dir = OUTPUT_DIR / label
        out_label_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_label_dir / f"{sha[:10]}_{video_path.stem}.npz"
        
        np.savez_compressed(
            out_file,
            keypoints=raw_seq,
            label=label,
            sha256=sha,
            source_video=str(video_path)
        )
        return str(out_file), "success"
    except Exception as e:
        return None, str(e)

def main():
    print("==================================================")
    print("[*] Extracting Landmarks from Additional Local Videos")
    print("==================================================")
    
    videos_to_process = []
    
    # 1. Add2 matches
    for add_cls, target_label in CLASS_MAP.items():
        if add_cls == "today": continue
        folder = ADD2_PATH / add_cls
        if folder.exists():
            vids = [f for f in folder.iterdir() if f.suffix.lower() in [".mp4", ".mov", ".avi"]]
            print(f"  - Add2 '{add_cls}' -> target '{target_label}': {len(vids)} videos found")
            for v in vids:
                videos_to_process.append((v, target_label))
                
    # 2. Add1 matches
    today_folder = ADD1_PATH / "today"
    if today_folder.exists():
        vids = [f for f in today_folder.iterdir() if f.suffix.lower() in [".mp4", ".mov", ".avi"]]
        print(f"  - Add1 'today' -> target 'today': {len(vids)} videos found")
        for v in vids:
            videos_to_process.append((v, "today"))
            
    print(f"\nTotal videos scheduled for landmark extraction: {len(videos_to_process)}")
    
    t0 = time.time()
    success_count = 0
    skipped_count = 0
    
    # Use 4 parallel workers for stability and speed
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_single_video, v, lbl): (v, lbl) for v, lbl in videos_to_process}
        for future in as_completed(futures):
            res_path, status = future.result()
            if status == "success":
                success_count += 1
            else:
                skipped_count += 1
                
            done = success_count + skipped_count
            if done % 25 == 0 or done == len(videos_to_process):
                elapsed = time.time() - t0
                fps_rate = done / elapsed if elapsed > 0 else 0
                print(f"  Progress: {done}/{len(videos_to_process)} ({success_count} success, {skipped_count} skipped) - {fps_rate:.1f} vids/sec")
                
    print(f"\n[+] Extraction Complete in {time.time()-t0:.1f}s!")
    print(f"    Total Success: {success_count} landmark sequences")
    print(f"    Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
