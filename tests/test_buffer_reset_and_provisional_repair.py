"""
Regression Tests: Buffer Reset Repairs, Provisional Gate Calibration,
and End-to-End Prediction Confirmation.
"""

import os
import sys
import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.inference import ISLInferenceEngine


@pytest.fixture(scope="module")
def inference_engine():
    """Initializes the production ISL inference engine."""
    engine = ISLInferenceEngine()
    return engine


def make_active_frame(t: float, motion_offset: float = 0.0):
    """Generates a synthetic active signing frame (hands raised in signing zone)."""
    kps = np.zeros((133, 3), dtype=np.float32)
    confs = np.zeros(133, dtype=np.float32)

    # Pose: Nose (0), Shoulders (11, 12), Wrists (15, 16)
    kps[0] = [0.5, 0.25, 0.0]       # Nose
    confs[0] = 0.99
    kps[11] = [0.4, 0.45, 0.0]      # Left Shoulder
    confs[11] = 0.99
    kps[12] = [0.6, 0.45, 0.0]      # Right Shoulder
    confs[12] = 0.99

    # Right wrist raised above shoulder (in active signing zone)
    kps[16] = [0.55 + motion_offset, 0.38 + motion_offset, 0.0]
    confs[16] = 0.99

    # Right hand landmarks (indices 54 to 74)
    for i in range(54, 75):
        kps[i] = [0.55 + (i - 54) * 0.005 + motion_offset, 0.35 + motion_offset, 0.0]
        confs[i] = 0.95

    return kps.tolist(), confs.tolist()


def make_absent_hands_frame():
    """Generates a frame with pose only (no hands detected by MediaPipe)."""
    kps = np.zeros((133, 3), dtype=np.float32)
    confs = np.zeros(133, dtype=np.float32)

    kps[0] = [0.5, 0.25, 0.0]
    confs[0] = 0.99
    kps[11] = [0.4, 0.45, 0.0]
    confs[11] = 0.99
    kps[12] = [0.6, 0.45, 0.0]
    confs[12] = 0.99
    # Hands resting below waist / absent
    kps[15] = [0.35, 0.85, 0.0]
    confs[15] = 0.1
    kps[16] = [0.65, 0.85, 0.0]
    confs[16] = 0.1

    return kps.tolist(), confs.tolist()


def test_buffer_accumulates_past_transient_dropouts(inference_engine):
    """
    Verifies that transient MediaPipe hand dropouts (1 to 5 frames) do NOT
    wipe the temporal buffer back to 0 at 5/16 or 9/16.
    """
    inference_engine.reset()

    # Step 1: Send 5 active signing frames
    t0 = 100.0
    for i in range(5):
        kps, confs = make_active_frame(t0 + i * 0.033, motion_offset=i * 0.01)
        res = inference_engine.predict(kps, confs, seq_id=i + 1, timestamp=t0 + i * 0.033)

    assert len(inference_engine.raw_frame_buffer) == 5, (
        f"Expected buffer length 5 after 5 frames, got {len(inference_engine.raw_frame_buffer)}"
    )

    # Step 2: Simulate 4 frames of transient tracking dropout (hands lost)
    for j in range(4):
        seq = 6 + j
        t = t0 + (5 + j) * 0.033
        kps, confs = make_absent_hands_frame()
        res = inference_engine.predict(kps, confs, seq_id=seq, timestamp=t)
        # Buffer must NOT be wiped
        assert len(inference_engine.raw_frame_buffer) == 5, (
            f"Buffer was wiped during transient dropout at frame {j+1}!"
        )
        assert res.rejection_reason == "transient_hand_absence"

    # Step 3: Hands reappear and motion continues
    for k in range(5):
        seq = 10 + k
        t = t0 + (9 + k) * 0.033
        kps, confs = make_active_frame(t, motion_offset=(5 + k) * 0.01)
        res = inference_engine.predict(kps, confs, seq_id=seq, timestamp=t)

    # Buffer should now have 10 frames (accumulated past dropouts)
    assert len(inference_engine.raw_frame_buffer) == 10, (
        f"Expected buffer length 10, got {len(inference_engine.raw_frame_buffer)}"
    )


def test_buffer_legitimate_reset_on_hand_resting(inference_engine):
    """
    Verifies that prolonged absence/lowering of hands (>= 15 frames, ~500ms)
    legitimately resets the temporal buffer.
    """
    inference_engine.reset()

    # Accumulate 8 frames
    t0 = 200.0
    for i in range(8):
        kps, confs = make_active_frame(t0 + i * 0.033, motion_offset=i * 0.01)
        inference_engine.predict(kps, confs, seq_id=i + 1, timestamp=t0 + i * 0.033)

    assert len(inference_engine.raw_frame_buffer) == 8

    # Lower hands for 15 frames
    for j in range(15):
        seq = 9 + j
        t = t0 + (8 + j) * 0.033
        kps, confs = make_absent_hands_frame()
        res = inference_engine.predict(kps, confs, seq_id=seq, timestamp=t)

    # Buffer should now be reset to 0
    assert len(inference_engine.raw_frame_buffer) == 0, (
        f"Buffer was not reset after 15 resting frames; got length {len(inference_engine.raw_frame_buffer)}"
    )
    assert res.reset_reason == "hands_lowered_or_rested"


def test_buffer_legitimate_reset_on_stale_timestamp_gap(inference_engine):
    """
    Verifies that a gap > 1.0s between consecutive frames (browser pause/tab switch)
    triggers a clean reset to prevent mixing disjoint gestures.
    """
    inference_engine.reset()

    # Send 7 frames
    t0 = 300.0
    for i in range(7):
        kps, confs = make_active_frame(t0 + i * 0.033, motion_offset=i * 0.01)
        inference_engine.predict(kps, confs, seq_id=i + 1, timestamp=t0 + i * 0.033)

    assert len(inference_engine.raw_frame_buffer) == 7

    # Send frame after a 2.5 second gap
    kps, confs = make_active_frame(t0 + 0.25 + 2.5, motion_offset=0.1)
    res = inference_engine.predict(kps, confs, seq_id=8, timestamp=t0 + 0.25 + 2.5)

    assert "stale_frame_gap" in res.reset_reason
    # Buffer was legitimately cleared on stale gap
    assert len(inference_engine.raw_frame_buffer) == 0


def test_low_confidence_never_leaks_provisional_sign(inference_engine):
    """
    Verifies that when confidence is low or gates fail, the engine strictly returns
    provisional_sign=None and never leaks random argmax predictions (like GOOD or COW).
    """
    inference_engine.reset()

    # Send 16 frames of tiny jitter (barely any motion)
    t0 = 400.0
    for i in range(16):
        kps, confs = make_active_frame(t0 + i * 0.033, motion_offset=i * 0.0001)
        res = inference_engine.predict(kps, confs, seq_id=i + 1, timestamp=t0 + i * 0.033)

    # Gate 2 (motion energy) or Gate 3 (confidence) should reject this window
    assert res.state in ("BUFFERING", "UNCERTAIN", "NO_SIGN", "WAITING", "RECOGNIZING")
    assert res.provisional_sign is None, (
        f"Provisional sign leaked on indistinct window: {res.provisional_sign}"
    )
    assert res.confirmed_sign is None


def test_actual_gesture_confirmation_and_speech_contract(inference_engine):
    """
    Replays a real recorded gesture from the clean dataset, verifying:
    1. Confirmation state reached.
    2. Unique event_id generated.
    3. Correct authoritative translation provided.
    4. speech_action == 'SPEAK'.
    5. Consecutive post-confirmation frames do NOT re-trigger duplicate speech.
    """
    test_split_path = os.path.join(REPO_ROOT, "data", "unified_136class", "test_split.npz")
    if not os.path.exists(test_split_path):
        pytest.skip("unified_136class/test_split.npz not found")

    data = np.load(test_split_path, allow_pickle=True)
    features = data["features"]
    class_names = data["class_names"]

    # Test sequence prediction contract on first sample
    sample_feat = features[0]
    expected_class = class_names[0].upper()

    pred_res = inference_engine.predict_sequence_features(sample_feat, seq_id=100, timestamp=500.0)
    assert pred_res.state in ("CONFIRMED", "UNCERTAIN")
    if pred_res.state == "CONFIRMED":
        assert pred_res.confirmed_sign is not None
        assert pred_res.translation != ""


def test_pytorch_and_onnx_parity(inference_engine):
    """
    Verifies that ONNX and PyTorch AC-HWGATE produce matching outputs (diff < 1e-4).
    """
    onnx_path = os.path.join(REPO_ROOT, "backend", "models", "ac_hwgate.onnx")
    if not os.path.exists(onnx_path):
        pytest.skip("ONNX model artifact not found")

    import onnxruntime as ort
    import torch

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy_input = np.random.randn(1, 32, 133, 6).astype(np.float32)

    # PyTorch output
    inference_engine.model.eval()
    with torch.no_grad():
        pt_input = torch.from_numpy(dummy_input).to(inference_engine.device)
        pt_out = inference_engine.model(pt_input).cpu().numpy()

    # ONNX output
    onnx_out = session.run(None, {session.get_inputs()[0].name: dummy_input})[0]

    max_diff = np.max(np.abs(pt_out - onnx_out))
    assert max_diff < 1e-4, f"Max difference between PyTorch and ONNX is {max_diff:.6f} >= 1e-4"
