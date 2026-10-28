import numpy as np
import json
import torch
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from backend.inference import isl_model

def test_webcam_simulations():
    print("="*70)
    print("TESTING WEBCAM SIMULATION INPUTS THROUGH ISL_MODEL")
    print("="*70)
    
    with open('configs/authoritative_136_classes_manifest.json') as f:
        classes = json.load(f)['classes']
        
    # Test 1: Typical webcam standing pose with hands at chest (neutral rest)
    # Pose: 33 joints around (0.5, 0.5)
    # Left hand: 21 joints near left hip (0.4, 0.8)
    # Right hand: 21 joints near right hip (0.6, 0.8)
    # Face: 58 joints around (0.5, 0.2)
    
    # 1. Idle sitting with hands resting on table / lap
    print("\n[Scenario 1] Hands resting in lap (motion energy low, vis high)")
    for frame in range(25):
        kps = np.zeros((133, 2), dtype=np.float32)
        confs = np.zeros(133, dtype=np.float32)
        # Pose
        kps[:33, 0] = 0.5 + 0.05 * np.random.randn(33)
        kps[:33, 1] = 0.5 + 0.05 * np.random.randn(33)
        confs[:33] = 0.9
        # Hands resting low (lap)
        kps[33:54, 0] = 0.4 + 0.001 * np.random.randn(21)
        kps[33:54, 1] = 0.85 + 0.001 * np.random.randn(21)
        confs[33:54] = 0.8
        kps[54:75, 0] = 0.6 + 0.001 * np.random.randn(21)
        kps[54:75, 1] = 0.85 + 0.001 * np.random.randn(21)
        confs[54:75] = 0.8
        
        label, conf, is_unk = isl_model.predict(kps.tolist(), confs.tolist())
        status = isl_model.last_status
    print(f"Final output: label='{label}', conf={conf:.4f}, is_unk={is_unk}, status='{status}'")
    
    # 2. Random hand waving / talking with hands (unrelated gestures)
    print("\n[Scenario 2] Random hand waving (gesturing while talking)")
    isl_model.raw_frame_buffer.clear()
    isl_model.candidate_history.clear()
    detected_labels = []
    for frame in range(60):
        t = frame / 30.0
        kps = np.zeros((133, 2), dtype=np.float32)
        confs = np.zeros(133, dtype=np.float32)
        # Pose
        kps[:33, 0] = 0.5
        kps[:33, 1] = 0.4
        confs[:33] = 0.9
        # Right hand waving in circular motion
        kps[54:75, 0] = 0.5 + 0.15 * np.sin(2 * np.pi * t) + 0.02 * np.random.randn(21)
        kps[54:75, 1] = 0.4 + 0.10 * np.cos(2 * np.pi * t) + 0.02 * np.random.randn(21)
        confs[54:75] = 0.95
        
        label, conf, is_unk = isl_model.predict(kps.tolist(), confs.tolist())
        status = isl_model.last_status
        if status == "CONFIRMED" and not is_unk:
            detected_labels.append((frame, label, conf))
            
    print(f"Spurious detections during random waving: {detected_labels}")

if __name__ == "__main__":
    test_webcam_simulations()
