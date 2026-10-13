import time
import requests
import numpy as np

payload = {
    "keypoints": [[0.5, 0.5] for _ in range(133)],
    "confidences": [1.0 for _ in range(133)]
}

times = []
r = None
for _ in range(25):
    t0 = time.perf_counter()
    r = requests.post("http://localhost:8000/predict", json=payload, timeout=2)
    times.append((time.perf_counter() - t0) * 1000.0)

print(f"HTTP /predict round-trip: Avg: {np.mean(times[1:]):.2f} ms | Min: {np.min(times[1:]):.2f} ms | P50: {np.median(times[1:]):.2f} ms")
if r is not None:
    print("Response:", r.json())
