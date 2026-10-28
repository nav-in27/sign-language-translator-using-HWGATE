import os
import sys
import cv2
import numpy as np
import mediapipe as mp

def measure():
    video_path = r"C:\Users\navee\Downloads\archive (2)\Video_Dataset\Video_Dataset\Thank you\WIN_20231103_13_12_51_Pro.mp4"
    cap = cv2.VideoCapture(video_path)
    
    mp_h = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True
    )
    
    records = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_h.process(rgb)
        
        rh_kps = np.zeros((21, 2))
        rh_vis = False
        if res.right_hand_landmarks:
            rh_vis = True
            for i, lm in enumerate(res.right_hand_landmarks.landmark):
                rh_kps[i] = [lm.x, lm.y]
                
        # Also get wrist and shoulders from pose
        pose_kps = np.zeros((33, 2))
        if res.pose_landmarks:
            for i, lm in enumerate(res.pose_landmarks.landmark):
                pose_kps[i] = [lm.x, lm.y]
                
        records.append({
            'frame': frame_idx,
            'rh_kps': rh_kps,
            'rh_vis': rh_vis,
            'pose': pose_kps
        })
    cap.release()
    mp_h.close()
    
    print(f"Total frames: {len(records)}")
    
    # Analyze resting segment: frames 5 to 25
    rest_pts = np.stack([r['rh_kps'] for r in records[4:25]], axis=0) # [21, 21, 2]
    rest_std = float(np.mean(np.std(rest_pts, axis=0)))
    rest_y = float(np.mean(rest_pts[:, 0, 1])) # wrist y
    
    # Analyze active stroke: frames 35 to 75
    active_pts = np.stack([r['rh_kps'] for r in records[34:75]], axis=0)
    active_std = float(np.mean(np.std(active_pts, axis=0)))
    active_y = float(np.mean(active_pts[:, 0, 1]))
    
    # Nose / Shoulder y
    nose_y = float(np.mean([r['pose'][0, 1] for r in records]))
    left_sh_y = float(np.mean([r['pose'][11, 1] for r in records]))
    right_sh_y = float(np.mean([r['pose'][12, 1] for r in records]))
    mid_sh_y = (left_sh_y + right_sh_y) / 2.0
    
    print(f"\nPose Reference Y:")
    print(f"  Nose Y: {nose_y:.3f}")
    print(f"  Shoulder Midpoint Y: {mid_sh_y:.3f}")
    print(f"\nResting Phase (Frames 5..25):")
    print(f"  Hand Motion Std: {rest_std:.5f}")
    print(f"  Wrist Mean Y:    {rest_y:.3f} (Lower than shoulders by {rest_y - mid_sh_y:+.3f})")
    print(f"\nActive Signing Phase (Frames 35..75):")
    print(f"  Hand Motion Std: {active_std:.5f} ({active_std / max(1e-6, rest_std):.1f}x higher than rest!)")
    print(f"  Wrist Mean Y:    {active_y:.3f} (Near chin/face: nose={nose_y:.3f}, sh={mid_sh_y:.3f})")

if __name__ == "__main__":
    measure()
