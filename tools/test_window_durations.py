import os
import sys
import json
import cv2
import numpy as np
import mediapipe as mp

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from backend.inference import isl_model
from training.feature_extractor import feature_extractor

def test_video_windows(video_path, gt_name):
    print(f"\nEvaluating {gt_name} from: {os.path.basename(video_path)}")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total video frames: {total_frames}")
    
    mp_h = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True
    )
    
    all_kps = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_h.process(rgb)
        kp = feature_extractor.extract_landmarks_from_holistic(res)
        all_kps.append(kp)
    cap.release()
    mp_h.close()
    
    with open('configs/authoritative_136_classes_manifest.json') as mf:
        classes = json.load(mf)['classes']
        
    print(f"Extracted {len(all_kps)} frames of keypoints.")
    
    L = len(all_kps)
    windows = [
        ("Full Video (0..L)", 0, L),
        ("Gesture Lifecycle (25..100)", 25, min(100, L)),
        ("Gesture Lifecycle (30..105)", 30, min(105, L)),
        ("Gesture Lifecycle (20..95)", 20, min(95, L)),
        ("Premature fragment (25..65)", 25, min(65, L)),
    ]
    
    for name, start, end in windows:
        window_kps = all_kps[start:end]
        feat = feature_extractor.process_sequence_np(window_kps)
        out = isl_model.ort_session.run(None, {'keypoints': feat[None, ...].astype(np.float32)})[0]
        logits = out[0]
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / np.sum(exp_l)
        top1 = np.argmax(probs)
        top1_class = classes[top1]
        top1_prob = probs[top1]
        print(f"  {name:25s} -> Predicted: '{top1_class:15s}' (Prob: {top1_prob:.4f}, Logit: {logits[top1]:.2f})")

if __name__ == "__main__":
    v1 = r"C:\Users\navee\Downloads\archive (2)\Video_Dataset\Video_Dataset\Thank you\WIN_20231103_13_12_51_Pro.mp4"
    v2 = r"C:\Users\navee\Downloads\archive (2)\Video_Dataset\Video_Dataset\Good Morning\WIN_20231103_13_17_04_Pro.mp4"
    test_video_windows(v1, "thank_you")
    test_video_windows(v2, "good_morning")
