import os
import sys
import json
import time
import cv2
import numpy as np
import mediapipe as mp

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from backend.inference import isl_model
from backend.translator import nmt_engine
from training.feature_extractor import feature_extractor

def run_trace():
    print("="*75)
    print("STEP 2 TRACE: RUNNING RAW 'THANK YOU' VIDEO FRAME-BY-FRAME THROUGH LIVE PIPELINE")
    print("="*75)
    
    video_path = r"C:\Users\navee\Downloads\archive (2)\Video_Dataset\Video_Dataset\Thank you\WIN_20231103_13_12_51_Pro.mp4"
    if not os.path.exists(video_path):
        print(f"Error: video not found at {video_path}")
        return
        
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[+] Loaded video: {video_path}")
    print(f"    Total frames: {total_frames}, FPS: {fps}")
    
    mp_holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    isl_model.raw_frame_buffer.clear()
    isl_model.candidate_history.clear()
    isl_model.last_confirmed_sign = None
    isl_model.cooldown_counter = 0
    isl_model.missing_frames_counter = 0
    
    frame_idx = 0
    predictions = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_holistic.process(rgb)
        
        # Simulate CameraFeed.jsx get133Keypoints
        keypoints = []
        confidences = []
        
        # 1. Pose (33)
        if res.pose_landmarks:
            for i in range(33):
                lm = res.pose_landmarks.landmark[i]
                keypoints.append([lm.x, lm.y, lm.z])
                confidences.append([getattr(lm, "visibility", 1.0)])
        else:
            for _ in range(33):
                keypoints.append([0.0, 0.0, 0.0])
                confidences.append([0.0])
                
        # 2. Left Hand (21)
        if res.left_hand_landmarks:
            for i in range(21):
                lm = res.left_hand_landmarks.landmark[i]
                keypoints.append([lm.x, lm.y, lm.z])
                confidences.append([1.0])
        else:
            for _ in range(21):
                keypoints.append([0.0, 0.0, 0.0])
                confidences.append([0.0])
                
        # 3. Right Hand (21)
        if res.right_hand_landmarks:
            for i in range(21):
                lm = res.right_hand_landmarks.landmark[i]
                keypoints.append([lm.x, lm.y, lm.z])
                confidences.append([1.0])
        else:
            for _ in range(21):
                keypoints.append([0.0, 0.0, 0.0])
                confidences.append([0.0])
                
        # 4. Face (58)
        if res.face_landmarks:
            for i in range(58):
                lm = res.face_landmarks.landmark[i]
                keypoints.append([lm.x, lm.y, lm.z])
                confidences.append([1.0])
        else:
            for _ in range(58):
                keypoints.append([0.0, 0.0, 0.0])
                confidences.append([0.0])
                
        label, conf, is_unk = isl_model.predict(keypoints, confidences)
        status = isl_model.last_status
        buf_len = len(isl_model.raw_frame_buffer)
        
        if status in ("CONFIRMED", "DETECTING") or not is_unk:
            predictions.append((frame_idx, label, conf, status, is_unk, buf_len))
            print(f"Frame {frame_idx:02d}: Status={status:9s} | Label={label:15s} | Conf={conf:.4f} | is_unk={is_unk} | BufLen={buf_len}")
            
    cap.release()
    mp_holistic.close()
    
    print("\n--- Summary of confirmed signs ---")
    confirmed = [p for p in predictions if p[3] == "CONFIRMED"]
    if confirmed:
        for c in confirmed:
            trans = nmt_engine.translate_gloss_to_english(c[1])
            print(f"Frame {c[0]}: CONFIRMED '{c[1]}' (Conf: {c[2]:.4f}) -> English: '{trans}'")
    else:
        print("No sign was confirmed! All remained DETECTING or UNKNOWN.")

if __name__ == "__main__":
    run_trace()
