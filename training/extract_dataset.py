import os
import sys
import json
import glob
import cv2
import argparse
import numpy as np
import mediapipe as mp
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any

# Add training dir to path
TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TRAINING_DIR)
from feature_extractor import feature_extractor

def parse_args():
    parser = argparse.ArgumentParser(description="Extract temporal landmark sequences from ISL videos")
    parser.add_argument("--video_dir", type=str, 
                        default=r"C:\Users\navee\Downloads\archive (2)\Video_Dataset\Video_Dataset",
                        help="Path to raw ISL video dataset")
    parser.add_argument("--output_dir", type=str, default="data_temporal_sequences",
                        help="Output directory for sequence arrays and statistics")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Limit to first N classes for quick prototyping / tiny sanity test")
    parser.add_argument("--selected_classes", type=str, nargs="+", default=None,
                        help="Specific class names to extract")
    parser.add_argument("--include_tilts", action="store_true", default=False,
                        help="Include left/right tilt variations in addition to frontal videos")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation fraction")
    parser.add_argument("--test_split", type=float, default=0.15, help="Test fraction")
    return parser.parse_args()


def extract_landmarks_from_video(
    vpath: str, 
    holistic: Any,
    target_frames: int = 32
) -> Optional[np.ndarray]:
    """
    Extracts raw 133 keypoints [target_frames, 133, 2] from video.
    Returns None if video is corrupt or unreadable.
    """
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 10:
        cap.release()
        return None

    # Uniform temporal sampling
    indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
    raw_sequence = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            # Fallback black frame if read fails
            raw_sequence.append(np.zeros((133, 2), dtype=np.float32))
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res: Any = holistic.process(rgb)

        frame_kps = []
        # 1. Pose: 33 keypoints
        if res.pose_landmarks:
            for lm in res.pose_landmarks.landmark[:33]:
                frame_kps.append([lm.x, lm.y])
        else:
            for _ in range(33):
                frame_kps.append([0.0, 0.0])

        # 2. Left Hand: 21 keypoints
        if res.left_hand_landmarks:
            for lm in res.left_hand_landmarks.landmark[:21]:
                frame_kps.append([lm.x, lm.y])
        else:
            for _ in range(21):
                frame_kps.append([0.0, 0.0])

        # 3. Right Hand: 21 keypoints
        if res.right_hand_landmarks:
            for lm in res.right_hand_landmarks.landmark[:21]:
                frame_kps.append([lm.x, lm.y])
        else:
            for _ in range(21):
                frame_kps.append([0.0, 0.0])

        # 4. Face: 58 selected keypoints
        if res.face_landmarks:
            for lm in list(res.face_landmarks.landmark)[:58]:
                frame_kps.append([lm.x, lm.y])
        else:
            for _ in range(58):
                frame_kps.append([0.0, 0.0])

        raw_sequence.append(np.array(frame_kps, dtype=np.float32))

    cap.release()

    if len(raw_sequence) != target_frames:
        return None

    raw_arr = np.stack(raw_sequence, axis=0) # [32, 133, 2]
    # Check if video had any non-trivial human presence
    if np.sum(np.abs(raw_arr)) < 1.0:
        return None

    return raw_arr


def main():
    args = parse_args()
    print("=" * 70)
    print("AC-HWGATE AUTHENTIC TEMPORAL SEQUENCE EXTRACTOR")
    print(f"Video Base Directory: {args.video_dir}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 70)

    if not os.path.exists(args.video_dir):
        print(f"[!] Critical Error: Video directory not found at {args.video_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("configs", exist_ok=True)

    # 1. Discover classes
    all_classes = sorted([d for d in os.listdir(args.video_dir) if os.path.isdir(os.path.join(args.video_dir, d))])
    
    if args.selected_classes:
        target_classes = [c for c in all_classes if c in args.selected_classes]
    elif args.num_classes:
        target_classes = all_classes[:args.num_classes]
    else:
        target_classes = all_classes

    print(f"[+] Total available classes: {len(all_classes)}")
    print(f"[+] Processing {len(target_classes)} target classes:")
    print("    " + ", ".join(target_classes[:15]) + ("..." if len(target_classes) > 15 else ""))

    # Create authoritative label map
    label_map = {name: idx for idx, name in enumerate(target_classes)}
    label_map_path = os.path.join(args.output_dir, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    # Also write to configs/label_map.json
    with open(os.path.join("configs", "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"[+] Saved authoritative label map to {label_map_path}")

    # 2. Initialize MediaPipe Holistic
    print("[+] Initializing MediaPipe Holistic...")
    mp_solutions: Any = mp.solutions
    holistic = mp_solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Data containers
    train_samples = []
    train_labels = []
    val_samples = []
    val_labels = []
    test_samples = []
    test_labels = []

    stats_per_class = {}
    corrupted_count = 0
    total_videos_scanned = 0

    # 3. Process videos grouped by recording session
    for c_idx, c_name in enumerate(target_classes):
        c_dir = os.path.join(args.video_dir, c_name)
        vfiles = sorted([f for f in os.listdir(c_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))])
        
        # Filter tilts if requested
        if not args.include_tilts:
            vfiles = [f for f in vfiles if not ('_left_tilt' in f or '_right_tilt' in f)]

        total_videos_scanned += len(vfiles)

        # Group by recording session prefix
        sessions = defaultdict(list)
        for f in vfiles:
            sess_key = f.replace('_left_tilt', '').replace('_right_tilt', '').replace('.mp4', '')
            sessions[sess_key].append(os.path.join(c_dir, f))

        sess_list = sorted(list(sessions.keys()))
        n_sess = len(sess_list)
        n_val = max(1, int(round(n_sess * args.val_split)))
        n_test = max(1, int(round(n_sess * args.test_split)))
        n_train = max(1, n_sess - n_val - n_test)

        train_sess = set(sess_list[:n_train])
        val_sess = set(sess_list[n_train:n_train + n_val])
        test_sess = set(sess_list[n_train + n_val:])

        c_train_cnt = 0
        c_val_cnt = 0
        c_test_cnt = 0

        for s_idx, (sess_key, vpaths) in enumerate(sessions.items()):
            for vp in vpaths:
                raw_kps = extract_landmarks_from_video(vp, holistic, target_frames=32)
                if raw_kps is None:
                    corrupted_count += 1
                    continue

                # Run feature extractor to compute part normalization + motion dynamics (P, V, A)
                norm_feat = feature_extractor.process_sequence_np(raw_kps) # [32, 133, 6]

                if sess_key in train_sess:
                    train_samples.append(norm_feat)
                    train_labels.append(c_idx)
                    c_train_cnt += 1
                elif sess_key in val_sess:
                    val_samples.append(norm_feat)
                    val_labels.append(c_idx)
                    c_val_cnt += 1
                else:
                    test_samples.append(norm_feat)
                    test_labels.append(c_idx)
                    c_test_cnt += 1

        stats_per_class[c_name] = {
            "class_id": c_idx,
            "sessions": n_sess,
            "train_samples": c_train_cnt,
            "val_samples": c_val_cnt,
            "test_samples": c_test_cnt,
            "total_samples": c_train_cnt + c_val_cnt + c_test_cnt
        }

        progress = (c_idx + 1) / len(target_classes) * 100
        print(f"[{c_idx+1:02d}/{len(target_classes):02d}] {c_name:18s} | "
              f"Train: {c_train_cnt:02d} | Val: {c_val_cnt:02d} | Test: {c_test_cnt:02d} | Progress: {progress:.1f}%")

    holistic.close()

    # 4. Save numpy datasets
    print("[+] Packaging and saving tensor arrays...")
    X_train = np.array(train_samples, dtype=np.float32)
    y_train = np.array(train_labels, dtype=np.int64)
    X_val = np.array(val_samples, dtype=np.float32)
    y_val = np.array(val_labels, dtype=np.int64)
    X_test = np.array(test_samples, dtype=np.float32)
    y_test = np.array(test_labels, dtype=np.int64)

    np.save(os.path.join(args.output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(args.output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(args.output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(args.output_dir, "y_val.npy"), y_val)
    np.save(os.path.join(args.output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(args.output_dir, "y_test.npy"), y_test)

    # 5. Generate dataset_statistics.json
    total_valid = len(y_train) + len(y_val) + len(y_test)
    counts = [s["total_samples"] for s in stats_per_class.values()]

    dataset_stats = {
        "dataset_name": "ISL-61 Authentic Video Dataset",
        "num_classes": len(target_classes),
        "sequence_length": 32,
        "num_joints": 133,
        "feature_dim_per_joint": 6,
        "total_feature_dim": 798,
        "total_videos_scanned": total_videos_scanned,
        "total_valid_sequences": total_valid,
        "corrupted_or_empty_rejected": corrupted_count,
        "train_samples": len(y_train),
        "val_samples": len(y_val),
        "test_samples": len(y_test),
        "min_samples_per_class": min(counts) if counts else 0,
        "max_samples_per_class": max(counts) if counts else 0,
        "median_samples_per_class": float(np.median(counts)) if counts else 0.0,
        "classes": target_classes,
        "per_class_statistics": stats_per_class
    }

    with open("dataset_statistics.json", "w") as f:
        json.dump(dataset_stats, f, indent=2)
    with open(os.path.join(args.output_dir, "dataset_statistics.json"), "w") as f:
        json.dump(dataset_stats, f, indent=2)

    # 6. Generate DATASET_REPORT.md
    with open("DATASET_REPORT.md", "w") as f:
        f.write("# Authoritative Dataset Report: ISL Authentic Temporal Sequences\n\n")
        f.write(f"**Date:** 2026-10-03\n")
        f.write(f"**Model:** AC-HWGATE\n\n")
        f.write(f"## 1. Summary Statistics\n\n")
        f.write(f"- **Classes Evaluated:** {len(target_classes)}\n")
        f.write(f"- **Temporal Sequence Length ($T$):** 32 frames\n")
        f.write(f"- **Joints per Frame:** 133 holistic landmarks\n")
        f.write(f"- **Channels per Joint:** 6 (Position $P_x, P_y$, Velocity $V_x, V_y$, Acceleration $A_x, A_y$)\n")
        f.write(f"- **Input Tensor Shape:** $[B, 32, 133, 6]$\n")
        f.write(f"- **Total Valid Sequences:** {total_valid}\n")
        f.write(f"- **Train Sequences:** {len(y_train)}\n")
        f.write(f"- **Validation Sequences:** {len(y_val)}\n")
        f.write(f"- **Test Sequences (Zero-Leakage Unseen Sessions):** {len(y_test)}\n")
        f.write(f"- **Corrupt / Invalid Rejected:** {corrupted_count}\n")
        f.write(f"- **Min / Max / Median Samples per Class:** {min(counts)} / {max(counts)} / {np.median(counts):.1f}\n\n")
        f.write(f"## 2. Session Split Integrity\n\n")
        f.write("All video sequences were partitioned strictly by recording session prefix (`WIN_YYYYMMDD_HH_MM_SS`). ")
        f.write("No session's video was split between train and test, ensuring **strict zero-leakage test integrity**.\n\n")
        f.write(f"## 3. Class Breakdown\n\n")
        f.write("| Index | Class Name | Train | Val | Test | Total |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: | :---: |\n")
        for c_name, st in stats_per_class.items():
            f.write(f"| {st['class_id']} | {c_name} | {st['train_samples']} | {st['val_samples']} | {st['test_samples']} | {st['total_samples']} |\n")

    print("\n" + "=" * 70)
    print(f"[+] Dataset extraction complete!")
    print(f"    Train: {X_train.shape}, Labels: {y_train.shape}")
    print(f"    Val:   {X_val.shape}, Labels: {y_val.shape}")
    print(f"    Test:  {X_test.shape}, Labels: {y_test.shape}")
    print(f"    Saved: dataset_statistics.json and DATASET_REPORT.md")
    print("=" * 70)

if __name__ == "__main__":
    main()
