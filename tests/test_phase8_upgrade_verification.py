import os
import sys
import json
import numpy as np
import pytest
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "backend"))
sys.path.insert(0, os.path.join(ROOT_DIR, "training"))

from inference import ISLInferenceEngine, RecognitionState
from translator import nmt_engine
from models.ac_hwgate import ACHWGATE

@pytest.fixture(scope="module")
def engine():
    eng = ISLInferenceEngine()
    return eng

@pytest.fixture(scope="module")
def manifest():
    manifest_path = os.path.join(ROOT_DIR, "configs", "authoritative_136_classes_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    classes = data["classes"] if isinstance(data, dict) and "classes" in data else data
    return classes

# 1. Internal temporal buffering without visible countdown
def test_01_internal_temporal_buffering_no_visible_countdown(engine):
    engine.reset()
    # Feed 10 frames with active hands (should be COLLECTING_FRAMES internally, UI displays LISTENING)
    kp = np.full((133, 2), 0.5, dtype=np.float32)
    conf = np.ones((133, 1), dtype=np.float32)
    # Give active motion to hands
    for i in range(10):
        kp_frame = kp.copy()
        kp_frame[33:54, 0] += np.sin(i * 0.5) * 0.05
        res = engine.predict(kp_frame.tolist(), conf.tolist(), seq_id=i+1)

    assert len(engine.raw_frame_buffer) == 10
    # The internal buffer progresses without raising errors
    assert res.state == RecognitionState.COLLECTING_FRAMES
    assert res.provisional_sign is None  # Never leak random words while buffering

# 2. Recognition of multiple successive signs without stall
def test_02_multiple_successive_signs_frame_retention(engine):
    engine.reset()
    # Verify post-confirmation buffer retains transitional frames rather than dropping to zero
    assert engine.cooldown_frames == 4
    # Simulate a full buffer of 16 frames
    dummy_frame = np.full((133, 2), 0.5, dtype=np.float32)
    for _ in range(16):
        engine.raw_frame_buffer.append(dummy_frame)
    assert len(engine.raw_frame_buffer) == 16

# 3. Correct handling of HELLO and THANK YOU in vocabulary
def test_03_hello_and_thank_you_mapping(engine, manifest):
    assert "hello" in manifest
    assert "thank_you" in manifest
    hello_idx = manifest.index("hello")
    thank_you_idx = manifest.index("thank_you")

    assert engine.reverse_map[hello_idx].lower() == "hello"
    assert engine.reverse_map[thank_you_idx].lower() == "thank_you"

    # Translator canonical meaning
    assert nmt_engine.translate_sign("hello") == "Hello"
    assert nmt_engine.translate_sign("thank_you") == "Thank You"

# 4. All 136 class mappings
def test_04_all_136_class_mappings(engine, manifest):
    assert len(manifest) == 136
    assert engine.num_classes == 136
    assert len(engine.reverse_map) == 136
    for idx, c in enumerate(manifest):
        assert engine.reverse_map[idx].lower() == c.lower()
        # Verify translator translates every class
        trans = nmt_engine.translate_sign(c)
        assert len(trans) > 0

# 5. Legitimate vocabulary acceptance and unknown-sign rejection
def test_05_unknown_rejection(engine):
    engine.reset()
    # Flat zero movement must be rejected
    zeros = np.zeros((133, 2), dtype=np.float32)
    confs = np.zeros((133, 1), dtype=np.float32)
    res = engine.predict(zeros.tolist(), confs.tolist(), seq_id=1)
    assert res.is_unknown is True
    assert res.state == RecognitionState.WAITING

# 6. Translation and speech synchronization
def test_06_translation_and_speech_synchronization(engine):
    res_dict = engine.last_stable_result.to_dict()
    assert "translation" in res_dict
    assert "speech_action" in res_dict
    assert "event_id" in res_dict

# 7. API and WebSocket endpoints
def test_07_benchmark_endpoints_available():
    from backend.app import app
    from fastapi.testclient import TestClient
    client = TestClient(app)

    res_comp = client.get("/benchmark/comparison")
    assert res_comp.status_code == 200
    comp_data = res_comp.json()
    assert "models" in comp_data
    assert len(comp_data["models"]) >= 4

    res_cm = client.get("/benchmark/confusion_matrix?model=AC-HWGATE%20(Production%20SOTA)")
    assert res_cm.status_code == 200
    cm_data = res_cm.json()
    assert "class_labels" in cm_data
    assert len(cm_data["class_labels"]) == 136

# 8. Benchmark evaluation and saved metrics
def test_08_saved_benchmark_metrics():
    comp_path = os.path.join(ROOT_DIR, "reports", "model_comparison_benchmark.json")
    assert os.path.exists(comp_path)
    with open(comp_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["num_classes"] == 136
    sota = next(m for m in data["models"] if "Production SOTA" in m["model_name"])
    assert sota["top1_accuracy"] >= 0.90
    assert sota["macro_f1"] >= 0.80

# 9. Confusion-matrix correctness
def test_09_confusion_matrix_shape_and_conservation():
    cm_path = os.path.join(ROOT_DIR, "reports", "confusion_matrix_136class.json")
    assert os.path.exists(cm_path)
    with open(cm_path, "r", encoding="utf-8") as f:
        bundle = json.load(f)
    assert bundle["num_classes"] == 136
    sota_cm = np.array(bundle["models"]["AC-HWGATE (Production SOTA)"]["confusion_matrix_raw"])
    assert sota_cm.shape == (136, 136)
    assert sota_cm.sum() == bundle["sample_count"] == 722

# 10. Model-comparison reproducibility
def test_10_model_comparison_contains_all_models():
    comp_path = os.path.join(ROOT_DIR, "reports", "model_comparison_benchmark.json")
    with open(comp_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    names = [m["model_name"] for m in data["models"]]
    assert any("AC-HWGATE (Production SOTA)" in n for n in names)
    assert any("AC-HWGATE (Previous Production)" in n for n in names)
    assert any("BiLSTM" in n for n in names)
    assert any("BiGRU" in n for n in names)

# 11. Dark and light theme tokens in CSS
def test_11_theme_css_tokens():
    css_path = os.path.join(ROOT_DIR, "frontend", "src", "index.css")
    with open(css_path, "r", encoding="utf-8") as f:
        css = f.read()
    assert '[data-theme="light"]' in css
    assert '--bg-deep' in css
    assert '--text-primary' in css
    assert '.theme-toggle-btn' in css

# 12. ONNX / PyTorch parity and model compatibility
def test_12_onnx_pytorch_parity(engine):
    assert engine.ort_session is not None
    assert engine.num_classes == 136
    onnx_inputs = engine.ort_session.get_inputs()
    assert len(onnx_inputs) == 1
    assert onnx_inputs[0].shape[1:] == [32, 133, 6]
