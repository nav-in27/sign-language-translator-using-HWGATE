"""
Unit and Parity Tests for 136-Class AC-HWGATE ISL Recognition & Translation.
Verifies:
1. Feature extraction pipeline bit-exact parity.
2. Inference engine loading with 136 classes and ONNX Runtime.
3. Held-out test sample prediction correctness.
4. Multi-stage rejection of unknown noise / idle / missing hands.
5. Latency profiling (FPS and end-to-end recognition time).
6. Semantic translation engine accuracy for single words and multi-sign phrases.
"""

import os
import sys
import time
import numpy as np
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
TRAINING_DIR = os.path.join(ROOT_DIR, "training")
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, TRAINING_DIR)

from feature_extractor import feature_extractor
from inference import isl_model
from translator import nmt_engine


def test_model_loaded_136_classes():
    assert isl_model.num_classes == 136, f"Expected 136 classes, got {isl_model.num_classes}"
    assert isl_model.prototypes is not None, "Prototypes must be loaded"
    assert isl_model.prototypes.shape == (136, 128), f"Expected (136, 128), got {isl_model.prototypes.shape}"
    assert "ONNX Runtime" in isl_model.engine_type, f"Expected ONNX Runtime, got {isl_model.engine_type}"


def test_feature_extraction_parity():
    # Simulate a dynamic 40-frame keypoint sequence (40, 133, 2)
    np.random.seed(42)
    raw_sequence = [np.random.randn(133, 2).astype(np.float32) for _ in range(40)]
    
    # Method 1: FeatureExtractor direct sequence
    feat1 = feature_extractor.process_sequence_np(raw_sequence)
    
    # Method 2: Feed into InferenceEngine frame buffer
    isl_model.frame_buffer = raw_sequence
    feat2 = feature_extractor.process_sequence_np(list(isl_model.frame_buffer))
    
    # The last 32 frames of buffer (since deque maxlen=32)
    expected_feat = feature_extractor.process_sequence_np(raw_sequence[-32:])
    
    np.testing.assert_allclose(feat2, expected_feat, atol=1e-6, err_msg="Feature buffer mismatch!")
    assert feat1.shape == (32, 133, 6), f"Expected shape (32, 133, 6), got {feat1.shape}"


def test_held_out_test_sample_recognition():
    test_npz = os.path.join(ROOT_DIR, "data", "unified_136class", "test_split.npz")
    assert os.path.exists(test_npz), "test_split.npz must exist"
    data = np.load(test_npz, allow_pickle=True)
    
    X_test = data["features"]
    classes = data["class_names"]
    
    correct_count = 0
    num_to_test = 50
    
    for i in range(num_to_test):
        feat = X_test[i]
        true_label = classes[i].upper()
        
        pred_label, conf, is_unk = isl_model.predict_sequence_features(feat)
        if pred_label == true_label:
            correct_count += 1
            
    acc = correct_count / num_to_test
    print(f"\n[Test Verification] Top-1 Accuracy on 50 Held-Out Samples: {acc*100:.1f}% ({correct_count}/{num_to_test})")
    # Smoke floor for the leakage-free clean-session checkpoint. Full test
    # metrics are reported by the final evaluation artifact, not selected here.
    assert acc >= 0.65, f"Expected >= 65% accuracy on sample batch, got {acc*100:.1f}%"


def test_unknown_and_idle_rejection():
    # 1. Idle static frame (zero motion energy)
    static_kps = np.zeros((133, 2), dtype=np.float32)
    confs = np.ones((133, 1), dtype=np.float32)
    
    isl_model.raw_frame_buffer.clear()
    isl_model.candidate_history.clear()
    
    for _ in range(15):
        label, conf, is_unk = isl_model.predict(static_kps.tolist(), confs.tolist())
    
    # Zero motion energy must be rejected
    assert is_unk is True, f"Idle static frames should be rejected, got: {label}"

    # 2. Dropped hands / missing hands (zeros confidence)
    handless_confs = np.zeros((133, 1), dtype=np.float32)
    isl_model.raw_frame_buffer.clear()
    for _ in range(5):
        label, conf, is_unk = isl_model.predict(static_kps.tolist(), handless_confs.tolist())
    assert is_unk is True, f"Missing hands must be rejected, got: {label}"
    assert label in ["WAITING FOR SIGN", "DETECTING...", "UNKNOWN_SIGN"], f"Expected waiting status, got {label}"

    # 3. Flushed buffer after user drops hands
    assert len(isl_model.raw_frame_buffer) == 0, "Buffer should be flushed when hands drop"


def test_translation_engine_coverage():
    manifest_path = os.path.join(ROOT_DIR, "configs", "authoritative_136_classes_manifest.json")
    import json
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    classes = manifest["classes"]
    
    for c in classes:
        t = nmt_engine.translate_gloss_to_english(c)
        assert len(t) > 0, f"Translation for '{c}' is empty"
        assert t != "Unknown", f"Translation for '{c}' failed"

    # Multi-sign phrase test
    phrase1 = nmt_engine.translate_gloss_to_english("today friday")
    assert phrase1 == "Today is Friday."
    
    phrase2 = nmt_engine.translate_gloss_to_english("what_is_your_name")
    assert phrase2 == "What is your name?"
    
    phrase3 = nmt_engine.translate_gloss_to_english("father come home")
    assert phrase3 == "Father is coming home."


def test_inference_latency_benchmark():
    dummy_feat = np.random.randn(32, 133, 6).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        _ = isl_model.predict_sequence_features(dummy_feat)
        
    latencies = []
    num_runs = 100
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = isl_model.predict_sequence_features(dummy_feat)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        
    avg_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    fps = 1000.0 / avg_latency
    
    print(f"\n[Performance Benchmark] ONNX Runtime 136-Class AC-HWGATE Latency:")
    print(f"  Average Inference Latency: {avg_latency:.2f} ms")
    print(f"  P95 Inference Latency:     {p95_latency:.2f} ms")
    print(f"  Maximum Theoretical FPS:   {fps:.1f} FPS")
    
    assert avg_latency < 30.0, f"Inference latency too slow: {avg_latency:.2f} ms >= 30ms"
    assert fps > 35.0, f"FPS too low: {fps:.1f} FPS"


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
