import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

v_dir = Path(r"C:\Users\navee\Downloads\archive\ProcessedData_vivit")
test_classes = ["animal", "bad", "big", "bird", "cat", "cold", "cow", "dog", "fast", "fish", "good", "happy", "horse", "loud", "old", "sick", "warm", "young"]

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.15,
    min_tracking_confidence=0.15
)

extracted_counts = {}

for c in test_classes:
    vids = list((v_dir / c).glob("*.MOV")) + list((v_dir / c).glob("*.mp4"))
    success = 0
    for v in vids:
        cap = cv2.VideoCapture(str(v))
        kps_list = []
        hand_detected_frames = 0
        wrist_motion = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            h, w = frame.shape[:2]
            # Upscale 224x224 to 384x384 to provide adequate pixel density for hand detector
            if w < 300:
                frame = cv2.resize(frame, (384, 384), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)
            
            # Check pose wrists (joints 15 and 16)
            if res.pose_landmarks:
                lw = res.pose_landmarks.landmark[15]
                rw = res.pose_landmarks.landmark[16]
                wrist_motion.append([lw.x, lw.y, rw.x, rw.y])
            
            if res.left_hand_landmarks or res.right_hand_landmarks:
                hand_detected_frames += 1
        cap.release()
        
        # Check motion energy
        has_wrist_motion = False
        if len(wrist_motion) > 5:
            wm = np.array(wrist_motion)
            if np.std(wm, axis=0).mean() > 0.01:
                has_wrist_motion = True
        
        if hand_detected_frames >= 1 or has_wrist_motion:
            success += 1
            
    extracted_counts[c] = (success, len(vids))
    print(f"Class '{c}': {success} / {len(vids)} videos detected with gesture motion/hands")

holistic.close()
