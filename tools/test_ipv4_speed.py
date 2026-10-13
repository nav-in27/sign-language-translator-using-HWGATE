import time
import requests
import numpy as np

payload = {
    "keypoints": [[0.5, 0.5] for _ in range(133)],
    "confidences": [1.0 for _ in range(133)]
}

session = requests.Session()

# Test 1: 127.0.0.1 (IPv4 direct)
times_ip = []
r = None
for _ in range(25):
    t0 = time.perf_counter()
    r = session.post("http://127.0.0.1:8000/predict", json=payload, timeout=2)
    times_ip.append((time.perf_counter() - t0) * 1000.0)

print(f"Direct IPv4 (127.0.0.1) round-trip: Avg: {np.mean(times_ip[1:]):.2f} ms | Min: {np.min(times_ip[1:]):.2f} ms | P50: {np.median(times_ip[1:]):.2f} ms")
if r is not None:
    print("Response:", r.json())
