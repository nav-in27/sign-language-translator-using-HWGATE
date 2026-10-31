"""
Comprehensive Regression Test Suite for AC-HWGATE Live Translator Repair.
Validates:
1. Correct model output-to-label mapping.
2. Correct English meaning for confirmed classes.
3. Stale sequence IDs discarded.
4. State transitions completing correctly without flickering.
5. Uncertain inputs not becoming fabricated words.
6. Speech matching current confirmed event without repeated playback of held signs.
7. PyTorch and ONNX numerical consistency.
8. Sustained inference and end-to-end latency.
"""

import os
import sys
import time
import json
import pytest
import numpy as np
import torch

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TESTS_DIR)
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
TRAINING_DIR = os.path.join(ROOT_DIR, "training")

for p in [ROOT_DIR, BACKEND_DIR, TRAINING_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.inference import isl_model, RecognitionState, PredictionResult
from backend.translator import nmt_engine
from training.feature_extractor import feature_extractor
from models.ac_hwgate import ACHWGATE


def test_01_model_output_to_label_mapping():
    """1. Verify model output dimension, manifest synchronization, and index mapping."""
    manifest_path = os.path.join(ROOT_DIR, "configs", "selected_50_classes_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    classes = manifest_data["classes"]

    assert isl_model.num_classes == 50, f"Expected 136 classes, got {isl_model.num_classes}"
    assert len(classes) == 50, f"Manifest classes length mismatch: {len(classes)}"
    assert isl_model.reverse_map[0] == classes[0]
    assert isl_model.reverse_map[49] == classes[49]

    # Verify specific critical benchmark classes
    assert isl_model.reverse_map[classes.index("thank_you")] == "thank_you"
    assert isl_model.reverse_map[classes.index("hello")] == "hello"
    assert isl_model.reverse_map[classes.index("today")] == "today"
    assert isl_model.reverse_map[classes.index("student")] == "student"


def test_02_correct_english_meaning_for_confirmed_classes():
    """2. Verify authentic English translation mapping for isolated sign classes."""
    test_cases = [
        ("thank_you", "Thank You"),
        ("good_morning", "Good Morning"),
        ("today", "Today"),
        ("good", "Good"),
        ("hello", "Hello"),
        ("help", "Help"),
        ("please", "Please"),
        ("afternoon", "Afternoon"),
        ("night", "Night"),
    ]
    for gloss, expected in test_cases:
        trans = nmt_engine.translate_sign(gloss)
        assert trans.lower() == expected.lower(), f"Expected '{expected}' for '{gloss}', got '{trans}'"

    # Verify no multi-word hallucination or sentence fabrication for isolated signs
    single_sign_trans = nmt_engine.translate_sign("lion")
    assert "roaring" not in single_sign_trans.lower(), "Fabricated phrase detected!"


def test_03_stale_results_discarded():
    """3. Verify out-of-order sequence IDs are discarded as STALE_SEQUENCE."""
    isl_model.raw_frame_buffer.clear()
    isl_model.candidate_history.clear()
    isl_model.last_processed_seq_id = 100
    isl_model.state = RecognitionState.WAITING

    dummy_kps = [[0.5, 0.5] for _ in range(133)]
    dummy_confs = [[0.9] for _ in range(133)]

    # Packet with older seq_id = 95 arriving after seq_id = 100
    stale_res = isl_model.predict(dummy_kps, dummy_confs, seq_id=95)
    assert stale_res.label == "STALE_SEQUENCE"
    assert stale_res.is_unknown is True
    assert stale_res.speech_action == "none"
    assert isl_model.last_processed_seq_id == 100  # Stale packet did not advance sequence


def test_04_state_machine_transitions():
    """4. Verify consistent state machine progression: WAITING -> COLLECTING_FRAMES -> RECOGNIZING -> CONFIRMED -> WAITING."""
    isl_model.reset()

    # Step A: Idle frames with dropped hands -> WAITING
    zero_confs = [[0.0] for _ in range(133)]
    dummy_kps = [[0.0, 0.0] for _ in range(133)]
    for f in range(5):
        res = isl_model.predict(dummy_kps, zero_confs, seq_id=f + 1)
    assert res.state == RecognitionState.WAITING
    assert len(isl_model.raw_frame_buffer) == 0

    # Step B: Hands raised into active signing zone -> COLLECTING_FRAMES
    # Load authentic good_morning landmarks
    sample_path = os.path.join(ROOT_DIR, "data", "additional_landmarks", "today", "a1bdea67f8_MVI_5032.npz")
    d = np.load(sample_path, allow_pickle=True)
    kps = d["keypoints"]
    valid_confs = [[0.95] for _ in range(133)]

    # Frames 10 to 19 (10 frames): Buffering frames
    for f in range(10, 20):
        res = isl_model.predict(kps[f].tolist(), valid_confs, seq_id=100 + f)
        assert res.state == RecognitionState.COLLECTING_FRAMES

    # Continue to frame 25: Buffer reaches min_frames (16) -> RECOGNIZING or CONFIRMED
    for f in range(20, 26):
        res = isl_model.predict(kps[f].tolist(), valid_confs, seq_id=100 + f)
    assert res.state in (RecognitionState.RECOGNIZING, RecognitionState.CONFIRMED)
    if res.state == RecognitionState.CONFIRMED:
        assert res.confirmed_sign == "TODAY"
    else:
        assert res.provisional_sign in ("TODAY", "YES")

    # Step C: Completed gesture stroke confirmed on hand drop
    res_drop = isl_model.predict(dummy_kps, zero_confs, seq_id=200)
    assert res_drop.state == RecognitionState.CONFIRMED
    assert res_drop.confirmed_sign in ("TODAY", "YES")
    assert res_drop.event_id is not None


def test_05_uncertain_inputs_do_not_fabricate_words():
    """5. Verify that uncertain inputs transition cleanly to UNCERTAIN without fabricating words."""
    isl_model.reset()

    # Raise min_confidence to force Gate 3 failure
    orig_conf = isl_model.min_confidence
    isl_model.min_confidence = 0.999

    valid_confs = [[0.95] for _ in range(133)]
    saw_uncertain = False
    try:
        for f in range(40):
            # Active continuous hand movement inside signing zone
            kps = [[0.5 + 0.1 * np.sin(f * 0.5 + j), 0.4 + 0.1 * np.cos(f * 0.5 + j)] for j in range(133)]
            res = isl_model.predict(kps, valid_confs, seq_id=f + 1)
            if res.state == RecognitionState.UNCERTAIN:
                saw_uncertain = True
                assert res.label == "UNCERTAIN"
                assert res.is_unknown is True
                assert "Uncertain" in res.translation or "not recognized" in res.translation
                assert res.speech_action == "none"
                break
    finally:
        isl_model.min_confidence = orig_conf

    assert saw_uncertain, "Persistent low-confidence input must trigger UNCERTAIN state without fabricating words"


def test_06_speech_matching_current_confirmed_event():
    """6. Verify unique event ID generation and single speech trigger for held signs."""
    isl_model.reset()

    sample_path = os.path.join(ROOT_DIR, "data", "additional_landmarks", "today", "a1bdea67f8_MVI_5032.npz")
    d = np.load(sample_path, allow_pickle=True)
    kps = d["keypoints"]
    valid_confs = [[0.95] for _ in range(133)]

    confirmed_results = []
    for f in range(len(kps)):
        res = isl_model.predict(kps[f].tolist(), valid_confs, seq_id=f + 1)
        if res.state == RecognitionState.CONFIRMED:
            confirmed_results.append(res)

    assert len(confirmed_results) > 0, "Expected confirmation for today sign"

    # First confirmed frame must have speech_action == 'speak' and valid event_id
    first_res = confirmed_results[0]
    assert first_res.speech_action == "speak"
    assert first_res.event_id is not None
    initial_event_id = first_res.event_id

    # Subsequent frames during cooldown / held sign must have speech_action == 'none'
    for subsequent_res in confirmed_results[1:8]:
        assert subsequent_res.speech_action == "none", "Speech must NOT re-trigger for held sign!"
        assert subsequent_res.event_id == initial_event_id, "Event ID must remain identical for held sign!"


def test_07_pytorch_and_onnx_consistency():
    """7. Verify numerical parity between PyTorch production checkpoint and ONNX model."""
    weights_path = os.path.join(ROOT_DIR, "checkpoints", "achwgate_50class_production_best.pth")
    assert os.path.exists(weights_path), f"PyTorch weights not found at {weights_path}"
    assert isl_model.ort_session is not None, "ONNX session must be loaded"

    # Load PyTorch model
    device = torch.device("cpu")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    pt_model = ACHWGATE(num_classes=50, num_joints=133, in_channels=6, sequence_length=32).to(device)
    pt_model.load_state_dict(state_dict, strict=False)
    pt_model.eval()

    # Generate synthetic input
    np.random.seed(42)
    dummy_input = np.random.randn(1, 32, 133, 6).astype(np.float32)

    # ONNX inference
    onnx_input_name = isl_model.ort_session.get_inputs()[0].name
    onnx_logits = isl_model.ort_session.run(None, {onnx_input_name: dummy_input})[0]

    # PyTorch inference
    with torch.no_grad():
        pt_logits = pt_model(torch.tensor(dummy_input, device=device)).cpu().numpy()

    # Numerical consistency check
    max_diff = float(np.max(np.abs(onnx_logits - pt_logits)))
    assert max_diff < 1e-4, f"PyTorch vs ONNX logit difference ({max_diff:.6f}) exceeds 1e-4!"
    assert np.argmax(onnx_logits) == np.argmax(pt_logits), "PyTorch and ONNX predicted different classes!"


def test_08_sustained_inference_and_latency():
    """8. Verify sustained operation for 100 iterations, latency < 10ms, and memory stability."""
    dummy_kps = [[0.5, 0.5] for _ in range(133)]
    dummy_confs = [[0.9] for _ in range(133)]

    isl_model.raw_frame_buffer.clear()
    latencies = []

    for i in range(100):
        t0 = time.perf_counter()
        _ = isl_model.predict(dummy_kps, dummy_confs, seq_id=i + 1)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))

    # Buffer must not exceed window_size
    assert len(isl_model.raw_frame_buffer) <= isl_model.window_size, "Buffer memory leak detected!"
    # Mean latency should be well under 10ms (real-time 100+ FPS)
    assert mean_latency < 10.0, f"Inference latency too slow: {mean_latency:.2f} ms"
    assert p95_latency < 15.0, f"P95 latency too slow: {p95_latency:.2f} ms"
    isl_model.reset()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
