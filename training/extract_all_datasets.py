"""
Comprehensive keypoint extraction from ALL available ISL datasets.
Merges multiple datasets into a single unified training directory.
"""
import os
import cv2
import mediapipe as mp
import torch
import sys
sys.stdout.reconfigure(line_buffering=True)
import json
import time
from collections import defaultdict

def get_133_keypoints(results):
    """
    Extracts 133 keypoints from MediaPipe Holistic results.
    Layout: Pose(33) + LeftHand(21) + RightHand(21) + Face(58) = 133
    Returns: keypoints (133, 2), confidences (133, 1)
    """
    keypoints = torch.zeros(133, 2)
    confidences = torch.zeros(133, 1)
    
    idx = 0
    # 1. Pose (33)
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            keypoints[idx] = torch.tensor([lm.x, lm.y])
            confidences[idx] = torch.tensor([lm.visibility])
            idx += 1
    else:
        idx += 33
        
    # 2. Left Hand (21)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            keypoints[idx] = torch.tensor([lm.x, lm.y])
            confidences[idx] = torch.tensor([1.0])
            idx += 1
    else:
        idx += 21
        
    # 3. Right Hand (21)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            keypoints[idx] = torch.tensor([lm.x, lm.y])
            confidences[idx] = torch.tensor([1.0])
            idx += 1
    else:
        idx += 21
        
    # 4. Face (First 58 of 468)
    if results.face_landmarks:
        for lm in list(results.face_landmarks.landmark)[:58]:
            keypoints[idx] = torch.tensor([lm.x, lm.y])
            confidences[idx] = torch.tensor([1.0])
            idx += 1
    else:
        idx += 58
        
    return keypoints, confidences

def has_valid_landmarks(keypoints, confidences):
    """Check if extracted keypoints are non-trivial (at least hands or pose detected)."""
    # Check if at least pose landmarks exist (indices 0-32)
    pose_active = (keypoints[0:33].abs().sum() > 0.01)
    # Check if at least one hand detected (indices 33-74)
    hand_active = (keypoints[33:75].abs().sum() > 0.01)
    return pose_active or hand_active

def normalize_class_name(name):
    """Normalize class names to lowercase with underscores."""
    name = name.strip().lower()
    name = name.replace(' ', '_')
    name = name.replace('-', '_')
    name = name.replace('__', '_')
    return name

def extract_video_frames(video_path, max_frames=5):
    """Extract evenly spaced frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []
    
    # Select evenly spaced frame indices
    if total_frames <= max_frames:
        indices = list(range(total_frames))
    else:
        indices = [int(i * total_frames / max_frames) for i in range(max_frames)]
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    return frames

def process_image_file(holistic, img_path):
    """Process a single image and return keypoints, confidences or None."""
    image = cv2.imread(img_path)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    kp, conf = get_133_keypoints(results)
    
    if not has_valid_landmarks(kp, conf):
        return None
    
    return kp, conf

def process_frame(holistic, frame):
    """Process a single video frame and return keypoints, confidences or None."""
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    kp, conf = get_133_keypoints(results)
    
    if not has_valid_landmarks(kp, conf):
        return None
    
    return kp, conf


def process_dataset_images(holistic, input_dir, output_dir, max_per_class=0, source_name=""):
    """Process an image-based dataset (folders of JPG/PNG images)."""
    stats = defaultdict(int)
    
    classes = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    
    for cls_raw in classes:
        cls_name = normalize_class_name(cls_raw)
        cls_in_dir = os.path.join(input_dir, cls_raw)
        cls_out_dir = os.path.join(output_dir, cls_name)
        os.makedirs(cls_out_dir, exist_ok=True)
        
        images = [f for f in os.listdir(cls_in_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if max_per_class > 0:
            images = images[:max_per_class]
        
        processed = 0
        skipped = 0
        for img_name in images:
            img_path = os.path.join(cls_in_dir, img_name)
            # Create unique output name with source prefix
            base_name = os.path.splitext(img_name)[0]
            out_name = f"{source_name}_{base_name}.pt" if source_name else f"{base_name}.pt"
            out_path = os.path.join(cls_out_dir, out_name)
            
            if os.path.exists(out_path):
                processed += 1
                continue
            
            result = process_image_file(holistic, img_path)
            if result is None:
                skipped += 1
                continue
            
            kp, conf = result
            data = {
                'keypoints': kp.unsqueeze(0),
                'confidences': conf.unsqueeze(0),
                'label': cls_name
            }
            torch.save(data, out_path)
            processed += 1
        
        stats[cls_name] += processed
        print(f"  [{source_name}] Class '{cls_name}': {processed} extracted, {skipped} skipped (no landmarks)", flush=True)
    
    return stats


def process_dataset_videos(holistic, input_dir, output_dir, max_per_class=0, frames_per_video=5, source_name=""):
    """Process a video-based dataset (folders of MP4/MOV videos)."""
    stats = defaultdict(int)
    
    classes = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    
    for cls_raw in classes:
        cls_name = normalize_class_name(cls_raw)
        cls_in_dir = os.path.join(input_dir, cls_raw)
        cls_out_dir = os.path.join(output_dir, cls_name)
        os.makedirs(cls_out_dir, exist_ok=True)
        
        videos = [f for f in os.listdir(cls_in_dir) 
                  if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
        
        if max_per_class > 0:
            videos = videos[:max_per_class]
        
        processed = 0
        skipped = 0
        for vid_name in videos:
            vid_path = os.path.join(cls_in_dir, vid_name)
            frames = extract_video_frames(vid_path, max_frames=frames_per_video)
            
            for fi, frame in enumerate(frames):
                base_name = os.path.splitext(vid_name)[0]
                out_name = f"{source_name}_{base_name}_f{fi}.pt"
                out_path = os.path.join(cls_out_dir, out_name)
                
                if os.path.exists(out_path):
                    processed += 1
                    continue
                
                result = process_frame(holistic, frame)
                if result is None:
                    skipped += 1
                    continue
                
                kp, conf = result
                data = {
                    'keypoints': kp.unsqueeze(0),
                    'confidences': conf.unsqueeze(0),
                    'label': cls_name
                }
                torch.save(data, out_path)
                processed += 1
        
        stats[cls_name] += processed
        print(f"  [{source_name}] Class '{cls_name}': {processed} extracted, {skipped} skipped", flush=True)
    
    return stats


def main():
    start_time = time.time()
    
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_unified")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 70)
    print("ISL UNIFIED DATASET EXTRACTOR")
    print("=" * 70)
    
    # Initialize MediaPipe Holistic once
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=True,
        min_detection_confidence=0.3,  # Lower threshold to capture more samples
        min_tracking_confidence=0.3
    )
    
    all_stats = defaultdict(int)
    
    # ================================================================
    # DATASET 1: archive/ProcessedData_vivit (Video .MOV files)
    # ================================================================
    ds1_path = r"C:\Users\navee\Downloads\archive\ProcessedData_vivit"
    if os.path.exists(ds1_path):
        print(f"\n--- Processing Dataset 1: ProcessedData_vivit (Videos) ---")
        stats = process_dataset_videos(
            holistic, ds1_path, OUTPUT_DIR, 
            max_per_class=0, frames_per_video=8, source_name="ds1"
        )
        for k, v in stats.items():
            all_stats[k] += v
    else:
        print(f"Dataset 1 not found at {ds1_path}, skipping.")
    
    # ================================================================
    # DATASET 2: archive (1)/ISL_Dataset (Image .JPG files, A-Z)
    # ================================================================
    ds2_path = r"C:\Users\navee\Downloads\archive (1)\ISL_Dataset"
    if os.path.exists(ds2_path):
        print(f"\n--- Processing Dataset 2: ISL_Dataset (Images) ---")
        stats = process_dataset_images(
            holistic, ds2_path, OUTPUT_DIR,
            max_per_class=0, source_name="ds2"
        )
        for k, v in stats.items():
            all_stats[k] += v
    else:
        print(f"Dataset 2 not found at {ds2_path}, skipping.")
    
    # ================================================================
    # DATASET 3: archive (2)/Video_Dataset/Video_Dataset (Video .MP4 files)
    # ================================================================
    ds3_path = r"C:\Users\navee\Downloads\archive (2)\Video_Dataset\Video_Dataset"
    if os.path.exists(ds3_path):
        print(f"\n--- Processing Dataset 3: Video_Dataset (Videos) ---")
        stats = process_dataset_videos(
            holistic, ds3_path, OUTPUT_DIR,
            max_per_class=0, frames_per_video=8, source_name="ds3"
        )
        for k, v in stats.items():
            all_stats[k] += v
    else:
        print(f"Dataset 3 not found at {ds3_path}, skipping.")
    
    # ================================================================
    # DATASET 4: archive (3)/isl_dataset (Image .JPG, A-Z + 0-9, 1000/class)
    # ================================================================
    ds4_path = r"C:\Users\navee\Downloads\archive (3)\isl_dataset"
    if os.path.exists(ds4_path):
        print(f"\n--- Processing Dataset 4: isl_dataset (Images, Large) - capped at 200/class ---")
        stats = process_dataset_images(
            holistic, ds4_path, OUTPUT_DIR,
            max_per_class=200, source_name="ds4"
        )
        for k, v in stats.items():
            all_stats[k] += v
    else:
        print(f"Dataset 4 not found at {ds4_path}, skipping.")
    
    holistic.close()
    
    # ================================================================
    # COPY EXISTING EXTRACTED DATA
    # ================================================================
    existing_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_extracted_full")
    if os.path.exists(existing_dir):
        print(f"\n--- Merging existing extracted data from data_extracted_full ---")
        import shutil
        for cls_name in os.listdir(existing_dir):
            cls_src = os.path.join(existing_dir, cls_name)
            if not os.path.isdir(cls_src) or cls_name.startswith('.'):
                continue
            norm_name = normalize_class_name(cls_name)
            cls_dst = os.path.join(OUTPUT_DIR, norm_name)
            os.makedirs(cls_dst, exist_ok=True)
            
            count = 0
            for f in os.listdir(cls_src):
                if f.endswith('.pt'):
                    src_file = os.path.join(cls_src, f)
                    # Prefix to avoid name collision
                    dst_name = f"existing_{f}" if not f.startswith("existing_") else f
                    dst_file = os.path.join(cls_dst, dst_name)
                    if not os.path.exists(dst_file):
                        shutil.copy2(src_file, dst_file)
                    count += 1
            all_stats[norm_name] += count
            print(f"  [existing] Class '{norm_name}': {count} files merged")
    
    # ================================================================
    # SUMMARY
    # ================================================================
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"EXTRACTION COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 70)
    
    total_samples = sum(all_stats.values())
    print(f"\nTotal classes: {len(all_stats)}")
    print(f"Total samples: {total_samples}")
    print("\nPer-class breakdown:")
    for cls_name in sorted(all_stats.keys()):
        print(f"  {cls_name}: {all_stats[cls_name]}")
    
    # Save stats
    stats_path = os.path.join(OUTPUT_DIR, "extraction_stats.json")
    with open(stats_path, "w") as f:
        json.dump({
            "total_classes": len(all_stats),
            "total_samples": total_samples,
            "per_class": dict(all_stats),
            "extraction_time_seconds": elapsed
        }, f, indent=2)
    
    print(f"\nStats saved to {stats_path}")
    print(f"Unified dataset saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
