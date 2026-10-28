import pytest
import os
import sys
import json
import time
import numpy as np
import cv2

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "backend"))
sys.path.insert(0, os.path.join(ROOT_DIR, "training"))

from backend.inference import isl_model
from backend.translator import nmt_engine
from training.feature_extractor import feature_extractor


def test_01_hand_motion_dynamics_handles_single_hand_gesture():
    """Verify Gate 2 properly calculates motion energy on active hand without averaging zero-hand."""
    isl_model.reset()
    # 32 frames of single-hand movement (Right hand active, Left hand all zeros)
    frames = []
    for t in range(32):
        kps = np.zeros((133, 2), dtype=np.float32)
        # Shoulders
        kps[11] = [0.45, 0.40]
        kps[12] = [0.55, 0.40]
        # Right hand moving naturally (amplitude 0.04)
        offset_y = 0.45 - 0.04 * np.sin(np.pi * t / 31.0)
        for j in range(21):
            kps[54 + j] = [0.55 + 0.01 * (j % 5), offset_y + 0.01 * (j // 5)]
        frames.append(kps)
        
    confs = np.zeros((133, 1), dtype=np.float32)
    confs[0:33] = 1.0     # pose visible
    confs[33:54] = 0.0    # left hand invisible
    confs[54:75] = 1.0    # right hand visible
    confs[75:133] = 1.0   # face visible

    res = None
    for t, frame in enumerate(frames):
        res = isl_model.predict(frame.tolist(), confs.tolist(), seq_id=100 + t, timestamp=t * 0.033)

    # Must NOT be trapped in WAITING due to zero left hand
    assert res is not None
    assert res.state in ("RECOGNIZING", "CONFIRMED", "UNCERTAIN"), f"Expected active state, got {res.state}"
    assert res.provisional_sign != "WAITING FOR SIGN", "Provisional sign was trapped in WAITING FOR SIGN"


def test_02_signing_space_boundary_allows_chest_level_hands():
    """Verify Gate 1 allows hands in upper 95% of frame when shoulders are at normal position."""
    isl_model.reset()
    kps = np.zeros((133, 2), dtype=np.float32)
    kps[11] = [0.45, 0.45]
    kps[12] = [0.55, 0.45]
    # Hand at chest/mid-torso level (y = 0.65)
    for j in range(21):
        kps[54 + j] = [0.55, 0.65]

    confs = np.zeros((133, 1), dtype=np.float32)
    confs[11] = 1.0
    confs[12] = 1.0
    confs[54:75] = 1.0

    res = isl_model.predict(kps.tolist(), confs.tolist(), seq_id=1, timestamp=0.033)
    # 1st frame with hands present transitions to COLLECTING_FRAMES or preserves buffer
    assert isl_model.hand_present_counter >= 1
    assert isl_model.missing_frames_counter == 0


def test_03_telemetry_trace_records_all_pipeline_stages():
    """Verify _record_trace captures landmarks validity, buffer length, shapes, rejection reasons."""
    isl_model.reset()
    # Feed 20 frames
    confs = np.ones((133, 1), dtype=np.float32)
    for t in range(20):
        kps = np.random.uniform(0.3, 0.7, (133, 2)).astype(np.float32)
        _ = isl_model.predict(kps.tolist(), confs.tolist(), seq_id=500 + t, timestamp=t * 0.033)

    trace = isl_model.get_latest_event_trace()
    assert len(trace) > 0, "No trace events recorded"
    latest = trace[-1]
    required_keys = [
        "seq_id", "timestamp", "landmarks_valid", "landmarks_shape",
        "buffer_length", "model_input_shape", "raw_top_predictions",
        "model_input_dtype", "model_input_finite", "confidence", "decoded_class", "confirmation_status", "rejection_reason"
    ]
    for key in required_keys:
        assert key in latest, f"Missing telemetry key: {key}"


def test_04_real_video_sample_reaches_confirmed_and_translation():
    """Verify that feeding a real video sample produces CONFIRMED sign, translation, and speech action."""
    video_path = os.path.join(ROOT_DIR, "data", "raw_40words", "hello", "hello__INCLUDE__00566__MVI_0029.mp4")
    if not os.path.exists(video_path):
        pytest.skip(f"Video {video_path} not found")

    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    isl_model.reset()
    confirmed_events = []

    frame_idx = 0
    while cap.isOpened() and frame_idx < 90:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_idx += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        keypoints = []
        confidences = []
        # Pose
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                keypoints.append([lm.x, lm.y, lm.z])
                confidences.append([getattr(lm, 'visibility', 1.0)])
        else:
            for i in range(33):
                keypoints.append([0.0, 0.0, 0.0])
                confidences.append([0.0])

        # Left Hand
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                keypoints.append([lm.x, lm.y, lm.z])
                confidences.append([1.0])
        else:
            for i in range(21):
                keypoints.append([0.0, 0.0, 0.0])
                confidences.append([0.0])

        # Right Hand
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                keypoints.append([lm.x, lm.y, lm.z])
                confidences.append([1.0])
        else:
            for i in range(21):
                keypoints.append([0.0, 0.0, 0.0])
                confidences.append([0.0])

        # Face
        if results.face_landmarks:
            for lm in list(results.face_landmarks.landmark)[:58]:
                keypoints.append([lm.x, lm.y, lm.z])
                confidences.append([1.0])
        else:
            for i in range(58):
                keypoints.append([0.0, 0.0, 0.0])
                confidences.append([0.0])

        res = isl_model.predict(keypoints, confidences, seq_id=frame_idx, timestamp=frame_idx / 30.0)
        if res.state == "CONFIRMED" and res.confirmed_sign:
            confirmed_events.append({
                "frame": frame_idx,
                "sign": res.confirmed_sign,
                "translation": res.translation,
                "event_id": res.event_id,
                "speech": res.speech_action
            })

    cap.release()
    holistic.close()

    assert len(confirmed_events) > 0, "No signs were confirmed during video playback"
    first_event = confirmed_events[0]
    assert first_event["sign"] != "UNKNOWN_SIGN"
    assert first_event["translation"] != ""
    assert first_event["event_id"] is not None and first_event["event_id"].startswith("evt_")
    assert first_event["speech"] == "speak"
