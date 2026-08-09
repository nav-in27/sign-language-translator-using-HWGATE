import asyncio
import sys
import os

# Mock the session state
session_state = {
    "last_gloss": "",
    "gloss_buffer": [],
    "last_translated": "",
    "last_sign_time": 0.0
}

from inference import isl_model

print("Model loaded!")

keypoints = [[0.1, 0.2, 0.3] for _ in range(133)]
confidences = [[1.0] for _ in range(133)]

print("Testing prediction loop...")
for i in range(40):
    try:
        label, conf = isl_model.predict(keypoints, confidences)
        print(f"Frame {i}: label={label}, conf={conf}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("Success!")
