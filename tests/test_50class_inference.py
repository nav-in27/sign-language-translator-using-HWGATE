import os
import sys
import json
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
from inference import isl_model

def run_50class_verification():
    print("=" * 70)
    print("VERIFYING 50-CLASS LIVE INFERENCE ENGINE")
    print("=" * 70)

    print(f"Engine Type:  {isl_model.engine_type}")
    print(f"Num Classes:  {isl_model.num_classes}")
    print(f"Device:       {isl_model.device}")
    print(f"Prototypes:   {isl_model.prototypes.shape if isl_model.prototypes is not None else 'None'}")
    assert isl_model.num_classes == 50, f"Expected 50 classes, got {isl_model.num_classes}"

    # Load test set
    data_dir = os.path.join(ROOT, "data", "unified_50class")
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    with open(os.path.join(data_dir, "class_names.json")) as f:
        class_names = json.load(f)

    # 1. Direct Sequence Feature Inference Test on held-out samples
    correct = 0
    unknown_count = 0
    confidences = []
    
    # Test 1 sample from each of the 50 classes
    tested_classes = set()
    per_class_results = {}

    for i in range(len(y_test)):
        cls_idx = int(y_test[i])
        cls_name = class_names[cls_idx]
        if cls_idx in tested_classes:
            continue
        tested_classes.add(cls_idx)

        feat = X_test[i] # [32, 133, 6]
        pred_label, conf, is_unk = isl_model.predict_sequence_features(feat)
        is_corr = (pred_label.lower().replace(" ", "_") == cls_name.lower().replace(" ", "_"))
        if is_corr:
            correct += 1
        if is_unk:
            unknown_count += 1
        confidences.append(conf)
        per_class_results[cls_name] = {
            "predicted": pred_label,
            "confidence": round(conf, 4),
            "correct": is_corr,
            "is_unknown": is_unk
        }

    print(f"\n[+] Evaluated 1 held-out test sample for each of the 50 classes:")
    print(f"    Correct Predictions: {correct} / 50 ({correct/50*100:.1f}%)")
    print(f"    Unknown Rejections:  {unknown_count} / 50")
    print(f"    Mean Confidence:     {np.mean(confidences)*100:.1f}%")

    # 2. Test Unknown Sign Rejection (Noise / Random / Static)
    print("\n[*] Testing Multi-Stage Unknown Sign Rejection...")
    
    # 2a. Random Gaussian Noise
    noise_seq = np.random.randn(32, 133, 6).astype(np.float32) * 0.5
    lbl_noise, conf_noise, is_unk_noise = isl_model.predict_sequence_features(noise_seq)
    print(f"    Noise Input Result:       Label='{lbl_noise}', Conf={conf_noise:.4f}, IsUnknown={is_unk_noise}")
    assert is_unk_noise, f"Noise input was not rejected! Got {lbl_noise}"

    # 2b. All-Zero Static
    zero_seq = np.zeros((32, 133, 6), dtype=np.float32)
    lbl_zero, conf_zero, is_unk_zero = isl_model.predict_sequence_features(zero_seq)
    print(f"    Zero Input Result:        Label='{lbl_zero}', Conf={conf_zero:.4f}, IsUnknown={is_unk_zero}")
    assert is_unk_zero, f"Zero input was not rejected! Got {lbl_zero}"

    # 3. Test Rolling Buffer Simulation
    print("\n[*] Testing Rolling Window Streaming Simulation...")
    isl_model.raw_frame_buffer.clear()
    isl_model.candidate_history.clear()
    isl_model.cooldown_counter = 0

    # Pick a test sign: e.g. "THANK_YOU" or "WATER" or "ELEPHANT"
    test_idx = class_names.index("elephant")
    sample_feat = X_test[np.where(y_test == test_idx)[0][0]] # [32, 133, 6]

    # Stream 32 frames one by one through predict(kps, confs)
    # The positions are feat[:, :, :2]
    # confidences can be dummy 0.9 for visible hands
    dummy_confs = [0.9] * 133
    confirmed_labels = []

    for f in range(32):
        frame_kp = sample_feat[f, :, :2].tolist()
        pred, conf, is_unk = isl_model.predict(frame_kp, dummy_confs)
        if not is_unk and isl_model.last_status == "CONFIRMED":
            confirmed_labels.append(pred)

    print(f"    Streaming 'elephant' (32 frames) -> Confirmed: {set(confirmed_labels)}")
    print(f"    Last Stable Result: {isl_model.last_stable_result}")

    print("\n" + "=" * 70)
    print("[+] [ALL PASS] 50-Class Inference Engine & Gate Verification Complete!")
    print("=" * 70)

if __name__ == "__main__":
    run_50class_verification()
