"""
Comprehensive Recognition Event Tracer for AC-HWGATE Live Pipeline.
Instruments the complete pipeline:
Webcam frame -> landmarks -> temporal buffer -> model input -> AC-HWGATE output ->
class index -> class label -> confirmation filter -> English translation -> UI payload -> speech status.

Traces every stage and records:
- Sequence ID
- Timestamp
- Model Version
- Raw Top Predictions
- Confidence
- Decoded Class
- Confirmation Status (State)
- Translation
- Speech Status
"""

import os
import sys
import time
import json
import numpy as np

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TOOLS_DIR)
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, BACKEND_DIR)

from backend.inference import isl_model, RecognitionState, PredictionResult
from backend.translator import nmt_engine

def trace_event_for_sample(class_name: str, sample_path: str):
    print(f"\n{'='*90}")
    print(f"TRACING RECOGNITION EVENT FOR GROUND-TRUTH SIGN: '{class_name.upper()}'")
    print(f"Sample source: {os.path.basename(sample_path)}")
    print(f"{'='*90}\n")

    # Load landmark sample
    data = np.load(sample_path, allow_pickle=True)
    if 'keypoints' in data:
        kps_seq = data['keypoints'] # [32, 133, 2]
    else:
        # Fallback to features slice
        feats = data['features']
        kps_seq = feats[:, :, :2]

    # Reset engine
    isl_model.raw_frame_buffer.clear()
    isl_model.candidate_history.clear()
    isl_model.last_confirmed_sign = None
    isl_model.last_confirmed_translation = ""
    isl_model.last_confirmed_event_id = None
    isl_model.state = RecognitionState.WAITING
    isl_model.last_processed_seq_id = 0
    isl_model.cooldown_counter = 0

    trace_records = []
    base_time = 1730000000.000

    print(f"{'Seq':<5} | {'State':<17} | {'Decoded Class':<15} | {'Conf':<6} | {'Top-1 Candidate':<15} | {'Translation':<18} | {'Speech':<7} | {'Event ID'}")
    print("-" * 110)

    # We feed 40 frames:
    # Frames 1-3: Hand entering zone (buffer building)
    # Frames 4-32: Accumulating temporal buffer
    # Frames 33-40: Stable recognition & confirmation
    num_frames = len(kps_seq)
    for frame_idx in range(40):
        seq_id = frame_idx + 1
        t_stamp = base_time + (seq_id * 0.0333) # 30 fps
        
        # Use circular index into sample sequence
        kp_frame = kps_seq[frame_idx % num_frames].tolist()
        # High confidence for active signing
        confs = [0.95 for _ in range(133)]

        # Pipeline Stage 1-2: Frame & Landmarks
        # Pipeline Stage 3: Temporal Buffer accumulation
        # Pipeline Stage 4-6: Model Inference & Confirmation filter
        res = isl_model.predict(kp_frame, confs, seq_id=seq_id, timestamp=t_stamp)

        top1_cand = res.top_predictions[0]["class"] if res.top_predictions else "N/A"
        top1_p = res.top_predictions[0]["prob"] if res.top_predictions else 0.0

        rec = {
            "sequence_id": res.seq_id,
            "timestamp": round(res.timestamp, 4),
            "model_version": res.model_version,
            "raw_top_predictions": res.top_predictions[:3],
            "confidence": round(float(res.confidence), 4),
            "decoded_class": res.label,
            "confirmation_status": res.state,
            "translation": res.translation if res.state == RecognitionState.CONFIRMED else (f"(Prev: {res.previous_translation})" if res.previous_translation else "None"),
            "speech_status": res.speech_action,
            "event_id": res.event_id or "N/A"
        }
        trace_records.append(rec)

        print(f"{res.seq_id:<5} | {res.state:<17} | {res.label:<15} | {res.confidence:<6.3f} | {top1_cand:<15} | {rec['translation']:<18} | {res.speech_action:<7} | {rec['event_id']}")

    print("\n[PIPELINE TRACE SUMMARY]")
    confirmed_events = [r for r in trace_records if r["confirmation_status"] == RecognitionState.CONFIRMED and r["speech_status"] == "speak"]
    print(f"Total frames processed: {len(trace_records)}")
    print(f"Confirmed events generated: {len(confirmed_events)}")
    if confirmed_events:
        first_c = confirmed_events[0]
        print(f"First confirmation at sequence ID: {first_c['sequence_id']}")
        print(f"Confirmed Class: {first_c['decoded_class']}")
        print(f"Model Confidence: {first_c['confidence']}")
        print(f"English Translation: '{first_c['translation']}'")
        print(f"Speech Trigger: '{first_c['speech_status']}' on Event '{first_c['event_id']}'")
    
    return trace_records

def main():
    print("================================================================================")
    print("      AC-HWGATE LIVE RECOGNITION EVENT INSTRUMENTATION & TRACE AUDIT            ")
    print("================================================================================")

    samples = [
        ("thank_you", os.path.join(ROOT_DIR, "data", "additional_landmarks", "thank_you", "WIN_20231103_13_12_51_Pro.npz")),
        ("good_morning", os.path.join(ROOT_DIR, "data", "additional_landmarks", "good_morning", "WIN_20231103_13_17_04_Pro.npz")),
        ("today", os.path.join(ROOT_DIR, "data", "additional_landmarks", "today", "a1bdea67f8_MVI_5032.npz"))
    ]

    for cname, spath in samples:
        if os.path.exists(spath):
            trace_event_for_sample(cname, spath)
        else:
            print(f"[!] Warning: sample path {spath} not found.")

    # Save detailed JSON report
    print("\nTracing complete.")

if __name__ == "__main__":
    main()
