import os
import sys
import json
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
from inference import isl_model
from translator import nmt_engine

def run_live_stream_simulation():
    print("=" * 75)
    print("LIVE WEBCAM STREAM SIMULATION ACROSS 50 CANONICAL ISL CLASSES")
    print("=" * 75)

    data_dir = os.path.join(ROOT, "data", "unified_50class")
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    with open(os.path.join(data_dir, "class_names.json")) as f:
        class_names = json.load(f)

    with open(os.path.join(ROOT, "configs", "translation_map.json")) as f:
        trans_map = json.load(f)

    # For each class, pick a held-out test sequence and stream it frame-by-frame
    total_classes = len(class_names)
    confirmed_count = 0
    correct_count = 0
    latencies = []

    results = []

    print(f"{'CLASS':15} | {'PREDICTED':15} | {'CONF':6} | {'STATUS':10} | {'TRANSLATION':30} | {'LATENCY'}")
    print("-" * 95)

    for cls_idx in range(total_classes):
        cls_name = class_names[cls_idx]
        matches = np.where(y_test == cls_idx)[0]
        if len(matches) == 0:
            continue
        test_seq = X_test[matches[0]] # [32, 133, 6]

        # Reset engine state between simulated gestures
        isl_model.raw_frame_buffer.clear()
        isl_model.candidate_history.clear()
        isl_model.cooldown_counter = 0

        # Simulate user raising hand and performing gesture (32 frames)
        stream_confirmed = None
        stream_conf = 0.0
        frame_times = []

        dummy_confs = [0.9] * 133

        # Stream up to 45 frames (sign + brief hold)
        T_len = len(test_seq)
        for f in range(45):
            idx = min(f, T_len - 1)
            kp_frame = test_seq[idx, :, :2].tolist()
            t0 = time.perf_counter()
            pred, conf, is_unk = isl_model.predict(kp_frame, dummy_confs)
            t_ms = (time.perf_counter() - t0) * 1000.0
            frame_times.append(t_ms)

            if not is_unk and isl_model.last_status == "CONFIRMED":
                stream_confirmed = pred
                stream_conf = conf

        mean_lat = np.mean(frame_times)
        latencies.append(mean_lat)

        norm_pred = (stream_confirmed or "NONE").lower().replace(" ", "_")
        is_corr = (norm_pred == cls_name.lower().replace(" ", "_"))

        if stream_confirmed is not None:
            confirmed_count += 1
            if is_corr:
                correct_count += 1

        translation = trans_map.get(norm_pred, nmt_engine.translate_gloss_to_english(norm_pred))
        status_str = "MATCH [PASS]" if is_corr else ("CONFIRMED" if stream_confirmed else "WAITING")

        print(f"{cls_name:15} | {norm_pred:15} | {stream_conf:5.2f} | {status_str:10} | {translation[:30]:30} | {mean_lat:.2f}ms")

        results.append({
            "class": cls_name,
            "predicted": norm_pred,
            "confidence": round(float(stream_conf), 4),
            "correct": is_corr,
            "latency_ms": round(float(mean_lat), 2)
        })

    print("-" * 95)
    print(f"\n[+] SUMMARY OF 50-CLASS LIVE WEBCAM SIMULATION:")
    print(f"    Total Signs Tested:      {total_classes}")
    print(f"    Confirmed Gestures:      {confirmed_count} / {total_classes} ({confirmed_count/total_classes*100:.1f}%)")
    print(f"    Correct Live Predictions: {correct_count} / {total_classes} ({correct_count/total_classes*100:.1f}%)")
    print(f"    Average Frame Latency:   {np.mean(latencies):.2f} ms (~{1000.0/np.mean(latencies):.0f} FPS capability)")
    print("=" * 75)

    # Save verification report
    report_path = os.path.join(ROOT, "reports", "live_stream_verification_50class.json")
    with open(report_path, "w") as f:
        json.dump({
            "total_tested": total_classes,
            "confirmed": confirmed_count,
            "correct": correct_count,
            "live_accuracy": round(correct_count / total_classes, 4),
            "mean_latency_ms": round(float(np.mean(latencies)), 2),
            "details": results
        }, f, indent=2)
    print(f"[+] Saved live stream verification report to {report_path}")

if __name__ == "__main__":
    run_live_stream_simulation()
