"""Regression tests for browser stream reconnect and sequence handling."""

from fastapi.testclient import TestClient

from backend.app import app, session_state
from backend.inference import RecognitionState, isl_model


def test_websocket_connect_resets_previous_stream_state():
    isl_model.last_processed_seq_id = 900
    isl_model.state = RecognitionState.RECOGNIZING
    session_state["gloss_buffer"] = ["GOOD"]
    session_state["last_gloss"] = "GOOD"

    with TestClient(app).websocket_connect("/ws/predict"):
        assert isl_model.last_processed_seq_id == 0
        assert isl_model.state == RecognitionState.WAITING
        assert session_state["gloss_buffer"] == []


def test_http_sequence_one_starts_new_stream():
    isl_model.last_processed_seq_id = 900
    isl_model.state = RecognitionState.RECOGNIZING
    payload = {
        "keypoints": [[0.0, 0.0] for _ in range(133)],
        "confidences": [[0.0] for _ in range(133)],
        "seq_id": 1,
        "timestamp": 1.0,
    }
    response = TestClient(app).post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["state"] == RecognitionState.WAITING
    assert response.json()["label"] != "STALE_SEQUENCE"
    assert isl_model.last_processed_seq_id == 1

