"""
Test FastAPI backend endpoints directly with TestClient.
"""
import sys
import os
import time
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
sys.path.insert(0, BACKEND_DIR)

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    print("\n[+] /health endpoint response:", data)
    assert data["num_classes"] == 136, f"Expected 136 classes, got {data['num_classes']}"
    assert "136 classes" in data["engine"]

def test_predict_http_speed():
    kps = np.random.randn(133, 2).tolist()
    confs = np.ones(133).tolist()
    
    # Warmup
    for _ in range(5):
        client.post("/predict", json={"keypoints": kps, "confidences": confs})
        
    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        res = client.post("/predict", json={"keypoints": kps, "confidences": confs})
        latencies.append((time.perf_counter() - t0) * 1000.0)
        assert res.status_code == 200
        
    avg_l = float(np.mean(latencies))
    p95_l = float(np.percentile(latencies, 95))
    print(f"[+] Direct HTTP /predict: Avg Latency = {avg_l:.2f} ms | P95 = {p95_l:.2f} ms")

def test_websocket_predict():
    kps = np.random.randn(133, 2).tolist()
    confs = np.ones(133).tolist()
    
    with client.websocket_connect("/ws/predict") as ws:
        latencies = []
        for _ in range(30):
            t0 = time.perf_counter()
            ws.send_json({"keypoints": kps, "confidences": confs})
            data = ws.receive_json()
            latencies.append((time.perf_counter() - t0) * 1000.0)
            assert "label" in data
            assert "confidence" in data
        avg_l = float(np.mean(latencies))
        print(f"[+] WebSocket /ws/predict: Avg Latency = {avg_l:.2f} ms | P50 = {float(np.median(latencies)):.2f} ms")

def test_translate_endpoint():
    res = client.post("/translate", json={"text": "what_is_your_name"})
    assert res.status_code == 200
    data = res.json()
    print("[+] /translate ('what_is_your_name') ->", data["translation"])
    assert data["translation"] == "What is your name?"

if __name__ == "__main__":
    test_health()
    test_translate_endpoint()
    test_predict_http_speed()
    test_websocket_predict()
    print("\n[SUCCESS] All live backend endpoints validated!")
