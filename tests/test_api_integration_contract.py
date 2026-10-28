from fastapi.testclient import TestClient
import numpy as np
from pathlib import Path

from backend.app import app


def test_model_status_reports_loaded_production_artifact():
    response = TestClient(app).get("/model/status")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    assert data["artifact_kind"] == "onnx"
    assert data["artifact_path"].endswith("backend\\models\\ac_hwgate.onnx")
    assert len(data["artifact_sha256"]) == 64
    assert data["manifest_compatible"] is True
    assert data["num_classes"] == 136


def test_evaluation_summary_is_versioned_and_current():
    response = TestClient(app).get("/evaluation/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "current_checkpoint"
    assert data["sample_count"] == 722
    assert data["per_class"] and len(data["per_class"]) == 136
    assert data["current_checkpoint_sha256"] == data["report_checkpoint_sha256"]


def test_prediction_response_contains_trace_id_and_contract_fields():
    payload = {
        "keypoints": [[0.0, 0.0, 0.0] for _ in range(133)],
        "confidences": [[0.0] for _ in range(133)],
        "seq_id": 1,
        "timestamp": 1.0,
    }
    response = TestClient(app).post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["trace_id"]
    assert data["window_id"] == "window_1"
    assert data["live_diagnostics"]["required_frames"] == 16
    assert "buffer_length" in data["live_diagnostics"]
    assert data["engine"].startswith("AC-HWGATE")
    assert data["state"] in {"WAITING", "UNKNOWN_SIGN", "COLLECTING_FRAMES", "RECOGNIZING", "UNCERTAIN", "CONFIRMED"}


def test_prediction_preserves_client_trace_id():
    payload = {
        "keypoints": [[0.0, 0.0, 0.0] for _ in range(133)],
        "confidences": [[0.0] for _ in range(133)],
        "seq_id": 1,
        "timestamp": 1.0,
        "trace_id": "trace-regression-001",
    }
    response = TestClient(app).post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["trace_id"] == "trace-regression-001"


def test_authentic_landmark_sequence_reaches_production_api():
    path = next(Path("data/additional_landmarks").rglob("*.npz"))
    sample = np.load(path, allow_pickle=True)
    keypoints = sample["keypoints"].astype(np.float32)
    keypoints = np.concatenate([keypoints, np.zeros((len(keypoints), 133, 1), dtype=np.float32)], axis=2)
    confidences = np.ones((133, 1), dtype=np.float32)
    client = TestClient(app)
    responses = []
    for seq_id, frame in enumerate(keypoints, start=1):
        response = client.post("/predict", json={
            "keypoints": frame.tolist(),
            "confidences": confidences.tolist(),
            "seq_id": seq_id,
            "timestamp": float(seq_id),
        })
        assert response.status_code == 200
        responses.append(response.json())
    final = responses[-1]
    assert final["trace_id"]
    assert final["engine"].startswith("AC-HWGATE")
    assert all(item["class"] for item in final["top_predictions"])


def test_authentic_landmark_sequence_reaches_websocket_client():
    path = next(Path("data/additional_landmarks").rglob("*.npz"))
    sample = np.load(path, allow_pickle=True)
    keypoints = sample["keypoints"].astype(np.float32)
    keypoints = np.concatenate([keypoints, np.zeros((len(keypoints), 133, 1), dtype=np.float32)], axis=2)
    confidences = np.ones((133, 1), dtype=np.float32)
    client = TestClient(app)
    with client.websocket_connect("/ws/predict") as websocket:
        messages = []
        for seq_id, frame in enumerate(keypoints, start=1):
            websocket.send_json({
                "keypoints": frame.tolist(),
                "confidences": confidences.tolist(),
                "seq_id": seq_id,
                "timestamp": float(seq_id),
            })
            messages.append(websocket.receive_json())
    assert len(messages) == 32
    assert all(message["trace_id"] for message in messages)
    assert all(message["engine"].startswith("AC-HWGATE") for message in messages)
    assert all("live_diagnostics" in message for message in messages)
    assert messages[-1]["state"] != "WAITING"
