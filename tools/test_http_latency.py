import time
import requests
import numpy as np

kps = np.random.randn(133, 2).tolist()
confs = np.ones(133).tolist()
times = []
for _ in range(15):
    t0 = time.perf_counter()
    r = requests.post("http://localhost:8000/predict", json={"keypoints": kps, "confidences": confs}, timeout=5)
    times.append((time.perf_counter() - t0) * 1000.0)

print(f"HTTP /predict round-trip: Avg: {np.mean(times[1:]):.2f} ms | Min: {np.min(times[1:]):.2f} ms | Max: {np.max(times[1:]):.2f} ms")
