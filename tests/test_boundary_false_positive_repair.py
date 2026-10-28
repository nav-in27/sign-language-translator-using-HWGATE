import os
import sys
import numpy as np
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "backend"))

from backend.inference import isl_model, RecognitionState

def test_01_thank_you_replay_has_zero_boundary_false_positives():
    """Verify authentic thank_you replay confirms THANK_YOU with zero boundary false positives (e.g. STILL or GIVE)."""
    isl_model.reset()
    sample_path = os.path.join(ROOT_DIR, "data", "additional_landmarks", "thank_you", "WIN_20231103_13_13_18_Pro.npz")
    if not os.path.exists(sample_path):
        pytest.skip("thank_you sample not found")

    data = np.load(sample_path, allow_pickle=True)
    kps = data["keypoints"]
    T = len(kps)
    confs = [[0.95] for _ in range(133)]

    confirmed_signs = []
    event_ids = set()

    # Active gesture playback
    for t in range(T):
        res = isl_model.predict(kps[t].tolist(), confs, seq_id=100 + t, timestamp=t * 0.033)
        if res.state == RecognitionState.CONFIRMED and res.confirmed_sign:
            confirmed_signs.append(res.confirmed_sign)
            if res.event_id:
                event_ids.add(res.event_id)

    # Post-gesture hand lowering / resting
    zero_confs = [[0.0] for _ in range(133)]
    rest_kps = [[0.0, 0.0] for _ in range(133)]
    for t in range(20):
        res = isl_model.predict(rest_kps, zero_confs, seq_id=100 + T + t, timestamp=(T + t) * 0.033)
        if res.state == RecognitionState.CONFIRMED and res.confirmed_sign:
            confirmed_signs.append(res.confirmed_sign)
            if res.event_id:
                event_ids.add(res.event_id)

    unique_signs = set(confirmed_signs)
    assert "THANK_YOU" in unique_signs, f"Expected THANK_YOU, got: {unique_signs}"
    assert "STILL" not in unique_signs, "Boundary false positive STILL occurred!"
    assert "GIVE" not in unique_signs, "Boundary false positive GIVE occurred!"
    assert len(unique_signs) == 1, f"Expected only THANK_YOU, got additional signs: {unique_signs}"
    assert len(event_ids) == 1, f"Expected exactly 1 unique confirmed event_id, got: {len(event_ids)}"


def test_02_stationary_hands_rejected_by_gate2_without_model_inference():
    """Verify stationary hands in signing space are cleanly rejected by Gate 2 and do not predict STILL."""
    isl_model.reset()
    confs = [[0.95] for _ in range(133)]
    # Stationary hands with minimal sensor jitter (~0.001) over 40 frames (~1.3s)
    for f in range(40):
        jitter = np.random.normal(0, 0.001, (133, 2)).astype(np.float32)
        kps = np.zeros((133, 2), dtype=np.float32)
        kps[11] = [0.45, 0.40]
        kps[12] = [0.55, 0.40]
        for j in range(21):
            kps[54 + j] = [0.55 + 0.01 * (j % 5), 0.45 + 0.01 * (j // 5)]
        kps += jitter
        res = isl_model.predict(kps.tolist(), confs, seq_id=500 + f, timestamp=f * 0.033)

    assert res.state in (RecognitionState.WAITING, RecognitionState.COLLECTING_FRAMES)
    assert res.confirmed_sign is None
    assert res.provisional_sign != "STILL", "Stationary hands collapsed into STILL prediction!"


def test_03_hand_drop_does_not_confirm_stray_retraction():
    """Verify hand drop after already confirming a sign does not emit an extra sign."""
    isl_model.reset()
    isl_model.bout_confirmed_sign = "HELLO"
    isl_model.provisional_sign = "STILL"
    isl_model.provisional_confidence = 0.90
    isl_model.state = RecognitionState.RECOGNIZING
    # Append 16 frames to buffer
    for _ in range(16):
        isl_model.raw_frame_buffer.append(np.zeros((133, 2), dtype=np.float32))

    # Hand drop occurs (missing_frames_counter = 1)
    zero_confs = [[0.0] for _ in range(133)]
    rest_kps = [[0.0, 0.0] for _ in range(133)]
    res = isl_model.predict(rest_kps, zero_confs, seq_id=1, timestamp=0.033)

    # Must NOT confirm STILL
    assert res.state != RecognitionState.CONFIRMED or res.confirmed_sign != "STILL"
    assert isl_model.bout_confirmed_sign == "HELLO"
