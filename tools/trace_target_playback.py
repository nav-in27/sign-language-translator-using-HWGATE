import os
import sys
import glob
import numpy as np
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.inference import isl_model, RecognitionState

def trace_sign(class_name):
    pattern = f"data/additional_landmarks/{class_name}/*.npz"
    files = glob.glob(pattern)
    if not files:
        print(f"No files found for pattern: {pattern}")
        return
    sample_path = files[0]
    print("=" * 80)
    print(f"TRACING SIGN: {class_name} using {os.path.basename(sample_path)}")
    print("=" * 80)
    
    data = np.load(sample_path, allow_pickle=True)
    if "keypoints" in data:
        kps = data["keypoints"]
    else:
        print(f"Keys in data: {data.files}")
        return
        
    isl_model.reset()
    confs = [[0.95] for _ in range(133)]
    
    for t in range(len(kps)):
        res = isl_model.predict(kps[t].tolist(), confs, seq_id=100 + t, timestamp=t * 0.033)
        top_str = ""
        if res.top_predictions:
            top_str = ", ".join([f"{p['class']}:{p['prob']:.2f}" for p in res.top_predictions[:3]])
        if res.state != RecognitionState.COLLECTING_FRAMES:
            print(f"Frame {t:2d} (seq {100+t}): State={res.state:15s} Prov={str(res.provisional_sign):12s} Confirmed={str(res.confirmed_sign):12s} Top: [{top_str}] Rej: {res.rejection_reason}")

for name in ['thank_you', 'good', 'beautiful', 'give', 'onion', 'hello']:
    trace_sign(name)
