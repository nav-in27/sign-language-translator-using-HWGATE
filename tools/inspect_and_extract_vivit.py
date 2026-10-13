import os
import sys
import glob
from pathlib import Path
from collections import defaultdict
import cv2
import mediapipe as mp
import numpy as np

v_dir = Path(r"C:\Users\navee\Downloads\archive\ProcessedData_vivit")
if not v_dir.exists():
    print(f"Vivit directory {v_dir} not found!")
    sys.exit(1)

classes = sorted([d.name for d in v_dir.iterdir() if d.is_dir()])
print(f"Scanning {len(classes)} Vivit classes...")

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.15,
    min_tracking_confidence=0.15
)

results = {}
for c in classes:
    vids = list((v_dir / c).glob("*.MOV")) + list((v_dir / c).glob("*.mp4"))
    valid_vids = 0
    total_frames_with_hands = 0
    total_frames_with_pose = 0
    
    for v in vids:
        cap = cv2.VideoCapture(str(v))
        f_hands = 0
        f_pose = 0
        total_f = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            total_f += 1
            # Resize slightly if too small (Vivit is 224x224, upscaling to 336x336 improves MediaPipe palm detector)
            if frame.shape[0] < 300:
                frame = cv2.resize(frame, (336, 336))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)
            if res.pose_landmarks:
                f_pose += 1
            if res.left_hand_landmarks or res.right_hand_landmarks:
                f_hands += 1
        cap.release()
        
        # If at least 1 frame with hand detected or active pose/arm landmarks
        if f_hands >= 1 or f_pose >= 10:
            valid_vids += 1
        total_frames_with_hands += f_hands
        total_frames_with_pose += f_pose

    results[c] = {
        "total_vids": len(vids),
        "valid_vids": valid_vids,
        "total_frames_with_hands": total_frames_with_hands,
        "total_frames_with_pose": total_frames_with_pose
    }
    print(f"  {c}: total={len(vids)}, valid={valid_vids}, hand_frames={total_frames_with_hands}, pose_frames={total_frames_with_pose}", flush=True)

holistic.close()

import json
with open("reports/vivit_extraction_analysis.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved vivit analysis to reports/vivit_extraction_analysis.json")
