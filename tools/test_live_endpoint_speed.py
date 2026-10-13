import time
import json
import asyncio
import websockets
import requests
import numpy as np

kps = np.random.randn(133, 2).tolist()
confs = np.ones(133).tolist()

async def test_ws():
    uri = "ws://localhost:8000/ws/predict"
    async with websockets.connect(uri) as ws:
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            await ws.send(json.dumps({"keypoints": kps, "confidences": confs}))
            resp = await ws.recv()
            times.append((time.perf_counter() - t0) * 1000.0)
        print(f"WebSocket /ws/predict Latency : Avg: {np.mean(times[1:]):.2f} ms | Min: {np.min(times[1:]):.2f} ms | P50: {np.median(times[1:]):.2f} ms")

def test_http():
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        r = requests.post("http://localhost:8000/predict", json={"keypoints": kps, "confidences": confs})
        times.append((time.perf_counter() - t0) * 1000.0)
    print(f"HTTP /predict Latency        : Avg: {np.mean(times[1:]):.2f} ms | Min: {np.min(times[1:]):.2f} ms | P50: {np.median(times[1:]):.2f} ms")

if __name__ == "__main__":
    print("Testing Live Endpoint Performance...")
    asyncio.run(test_ws())
    test_http()
