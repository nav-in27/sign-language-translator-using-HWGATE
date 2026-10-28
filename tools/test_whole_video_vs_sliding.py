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

def test_video(video_path, ground_truth):
    print(f"\nEvaluating: {os.path.basename(video_path)} (GT: {ground_truth})")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total video frames: {total_frames}")
    
    indices = np.linspace(0, total_frames - 1, 32, dtype=int)
    frames_dict = {}
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if idx in set(indices):
            frames_dict[idx] = frame
        idx += 1
    cap.release()
    
    ordered_frames = [frames_dict[i] for i in indices if i in frames_dict]
    print(f"Subsampled frames: {len(ordered_frames)}")
    
    mp_h = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    kps_list = []
    for f in ordered_frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        res = mp_h.process(rgb)
        kp = feature_extractor.extract_landmarks_from_holistic(res)
        kps_list.append(kp)
    mp_h.close()
    
    feat = feature_extractor.process_sequence_np(kps_list)
    out = isl_model.ort_session.run(None, {'keypoints': feat[None, ...].astype(np.float32)})[0]
    logits = out[0]
    exp_l = np.exp(logits - np.max(logits))
    probs = exp_l / np.sum(exp_l)
    
    with open('configs/authoritative_136_classes_manifest.json') as mf:
        classes = json.load(mf)['classes']
        
    top5_idx = np.argsort(probs)[-5:][::-1]
    print("Top 5 Predictions (Whole Video Subsampled):")
    for r, c_idx in enumerate(top5_idx):
        print(f"  #{r+1}: {classes[c_idx]} (idx {c_idx}, prob {probs[c_idx]:.4f}, logit {logits[c_idx]:.2f})")

if __name__ == "__main__":
    v1 = r"C:\Users\navee\Downloads\archive (2)\Video_Dataset\Video_Dataset\Thank you\WIN_20231103_13_12_51_Pro.mp4"
    v2 = r"C:\Users\navee\Downloads\archive (2)\Video_Dataset\Video_Dataset\Good Morning\WIN_20231103_13_17_04_Pro.mp4"
    v3 = r"C:\Users\navee\Downloads\archive (2)\Video_Dataset\Video_Dataset\Hello\WIN_20231103_13_18_55_Pro.mp4"
    test_video(v1, "thank_you")
    test_video(v2, "good_morning")
    test_video(v3, "hello")
