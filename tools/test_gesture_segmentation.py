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

def simulate_stream_with_segmentation(video_path, gt_label):
    print("="*75)
    print(f"TESTING GESTURE SEGMENTATION ON: {os.path.basename(video_path)} (GT: {gt_label})")
    print("="*75)
    
    cap = cv2.VideoCapture(video_path)
    mp_h = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    with open('configs/authoritative_136_classes_manifest.json') as mf:
        classes = json.load(mf)['classes']
        
    gt_idx = classes.index(gt_label)
    
    # State tracking
    gesture_buffer = []
    is_signing = False
    confirmed_events = []
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_h.process(rgb)
        
        # 1. Pose landmarks for reference shoulder and nose
        sh_y = 0.45
        if res.pose_landmarks:
            l_sh = res.pose_landmarks.landmark[11].y
            r_sh = res.pose_landmarks.landmark[12].y
            sh_y = (l_sh + r_sh) / 2.0
            
        signing_space_boundary = sh_y + 0.28 # active signing space
        
        # Check hand elevation
        lh_active = False
        rh_active = False
        if res.left_hand_landmarks:
            wrist_y = res.left_hand_landmarks.landmark[0].y
            if wrist_y < signing_space_boundary:
                lh_active = True
        if res.right_hand_landmarks:
            wrist_y = res.right_hand_landmarks.landmark[0].y
            if wrist_y < signing_space_boundary:
                rh_active = True
                
        hands_in_signing_space = (lh_active or rh_active)
        
        # Extract 133 WholeBody keypoints
        kp = feature_extractor.extract_landmarks_from_holistic(res)
        
        if hands_in_signing_space:
            gesture_buffer.append(kp)
            if not is_signing and len(gesture_buffer) >= 6:
                is_signing = True
                print(f"  Frame {frame_idx:03d}: Sign stroke INITIATED (Hands in signing space).")
        else:
            if is_signing:
                # Gesture stroke has ENDED (hands returned to rest)
                duration = len(gesture_buffer)
                print(f"  Frame {frame_idx:03d}: Sign stroke COMPLETED. Duration: {duration} frames ({duration/30.0:.2f}s).")
                if duration >= 16:
                    # Classify the completed gesture trajectory
                    feat = feature_extractor.process_sequence_np(gesture_buffer)
                    out = isl_model.ort_session.run(None, {'keypoints': feat[None, ...].astype(np.float32)})[0]
                    logits = out[0]
                    exp_l = np.exp(logits - np.max(logits))
                    probs = exp_l / np.sum(exp_l)
                    top1 = np.argmax(probs)
                    pred_class = classes[top1]
                    conf = float(probs[top1])
                    is_correct = (pred_class == gt_label)
                    confirmed_events.append((frame_idx, pred_class, conf, is_correct, duration))
                    print(f"    --> PREDICTION: '{pred_class}' (Conf: {conf:.4f}, Logit: {logits[top1]:.2f}) -> {'CORRECT' if is_correct else 'WRONG'}")
                else:
                    print(f"    --> Transient stroke (<16 frames) ignored as noise/rejection.")
                gesture_buffer = []
                is_signing = False
            else:
                gesture_buffer = []
                
    cap.release()
    mp_h.close()
    
    print("\n--- Summary of Confirmed Sign Events ---")
    if confirmed_events:
        for ev in confirmed_events:
            print(f"  End Frame {ev[0]}: Sign='{ev[1]}' | Conf={ev[2]:.4f} | Correct={ev[3]} | Duration={ev[4]} frames")
    else:
        print("  No completed signs confirmed.")

if __name__ == "__main__":
    v1 = r"C:\Users\navee\Downloads\archive (2)\Video_Dataset\Video_Dataset\Thank you\WIN_20231103_13_12_51_Pro.mp4"
    v2 = r"C:\Users\navee\Downloads\archive (2)\Video_Dataset\Video_Dataset\Good Morning\WIN_20231103_13_17_04_Pro.mp4"
    simulate_stream_with_segmentation(v1, "thank_you")
    simulate_stream_with_segmentation(v2, "good_morning")
