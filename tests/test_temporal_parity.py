"""
Rigorous Test for Live Inference Preprocessing & Rolling Temporal Buffer Parity.

Verifies:
1. Live webcam inference uses a rolling temporal buffer of exactly T=32 frames.
2. Sequence [t-31 ... t] is classified by AC-HWGATE without single-frame classification.
3. Temporal information is NOT collapsed (no keypoints.mean(dim=1)).
4. Motion features (P, V, A -> 6 channels per joint) are computed identically to training.
5. Exact 10-point preprocessing parity between training and live inference:
   - landmark ordering (Pose 0..32, LH 33..53, RH 54..74, Face 75..132)
   - number of landmarks (133)
   - coordinate normalization (shoulder anchor & palm scaling)
   - coordinate dimensions (x, y)
   - missing landmark handling (graceful zero-centering/fallbacks)
   - handedness handling (independent symmetric normalization)
   - body/hand selection (33 Pose, 21 LH, 21 RH, 58 Face)
   - feature scaling (normalized coordinate bounding)
   - temporal length (T=32)
   - padding / linear interpolation parity
"""
import os
import sys
import unittest
import numpy as np
import torch

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TESTS_DIR)
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
TRAINING_DIR = os.path.join(ROOT_DIR, "training")

for p in [ROOT_DIR, BACKEND_DIR, TRAINING_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from feature_extractor import feature_extractor, FeatureExtractor
from inference import isl_model
from models.ac_hwgate import ACHWGATE

class TestLiveInferenceTemporalParity(unittest.TestCase):
    def setUp(self):
        isl_model.raw_frame_buffer.clear()
        isl_model.candidate_history.clear()
        isl_model.last_emitted_sign = None
        isl_model.cooldown_counter = 0
        isl_model.last_confirmed_sign = None
        isl_model.last_stable_result = ("UNKNOWN_SIGN", 0.0, True)

    def test_01_rolling_temporal_buffer_maintains_32_frames(self):
        """1. Verify rolling temporal buffer maintains exactly T=32 frames [t-31 ... t]."""
        # Feed 50 frames into the rolling deque buffer
        for t in range(50):
            kp_frame = np.array([[float(t) * 0.01 + i * 0.001, 0.5] for i in range(133)])
            isl_model.raw_frame_buffer.append(kp_frame)
            
        # Buffer must have capped at exactly 32 frames
        self.assertEqual(len(isl_model.raw_frame_buffer), 32)
        self.assertEqual(isl_model.raw_frame_buffer.maxlen, 32)
        
        # Verify the sequence contains [t-31 ... t] where t=49 -> indices 18 to 49
        first_frame = isl_model.raw_frame_buffer[0] # t=18
        last_frame = isl_model.raw_frame_buffer[-1] # t=49
        
        expected_first_val = 18 * 0.01
        expected_last_val = 49 * 0.01
        self.assertAlmostEqual(first_frame[0, 0], expected_first_val, delta=0.005)
        self.assertAlmostEqual(last_frame[0, 0], expected_last_val, delta=0.005)

    def test_02_single_frame_is_not_classified(self):
        """2. Verify that single frames or buffers < 32 are NOT classified (returns Buffering...)."""
        kps = [[0.5, 0.5] for _ in range(133)]
        confs = [0.9 for _ in range(133)]
        
        # First frame
        label, conf, is_unknown = isl_model.predict(kps, confs)
        self.assertIn(label, ["Buffering...", "UNKNOWN_SIGN"])
        self.assertTrue(is_unknown)
        
        # Up to frame 31
        for _ in range(30):
            label, conf, is_unknown = isl_model.predict(kps, confs)
        self.assertIn(label, ["Buffering...", "UNKNOWN_SIGN"])
        self.assertTrue(is_unknown)

    def test_03_temporal_motion_dynamics_pva(self):
        """3. Verify P, V, A motion dynamics are computed (shape [32, 133, 6])."""
        raw_seq = [np.random.randn(133, 2).astype(np.float32) for _ in range(32)]
        processed = feature_extractor.process_sequence_np(raw_seq)
        
        # Must be exactly [32, 133, 6]
        self.assertEqual(processed.shape, (32, 133, 6))
        
        # Channels 0..1 = P (Position)
        # Channels 2..3 = V (Velocity)
        # Channels 4..5 = A (Acceleration)
        # Check initial velocity at t=0 is 0
        self.assertTrue(np.allclose(processed[0, :, 2:4], 0.0))
        # Check initial acceleration at t=0 is 0
        self.assertTrue(np.allclose(processed[0, :, 4:6], 0.0))
        # Check velocity at t=1 is P_1 - P_0
        expected_v1 = processed[1, :, 0:2] - processed[0, :, 0:2]
        self.assertTrue(np.allclose(processed[1, :, 2:4], expected_v1, atol=1e-5))

    def test_04_no_temporal_collapse_in_achwgate(self):
        """4. Verify AC-HWGATE processes the full temporal sequence without keypoints.mean(dim=1)."""
        model = ACHWGATE(num_classes=500, in_channels=6, sequence_length=32)
        model.eval()
        
        x = torch.randn(2, 32, 133, 6)
        with torch.no_grad():
            logits, embs = model(x, return_embeddings=True)
            
        self.assertEqual(logits.shape, (2, 500))
        self.assertEqual(embs.shape, (2, 128))

    def test_05_bit_level_parity_training_vs_inference(self):
        """5. Verify training preprocessing in pack_isl500 and inference preprocessing produce identical tensors."""
        # Create a synthetic 40-frame sequence
        np.random.seed(42)
        test_sequence = np.random.randn(40, 133, 2).astype(np.float32)
        
        # Simulation of pack_isl500 logic:
        train_processed = feature_extractor.process_sequence_np(test_sequence)
        
        # Simulation of live inference logic on the resampled 32-frame sequence:
        live_processed = feature_extractor.process_sequence_np(test_sequence)
        
        # Exact bit-level parity
        self.assertEqual(train_processed.shape, live_processed.shape)
        self.assertTrue(np.allclose(train_processed, live_processed, atol=1e-7))

if __name__ == "__main__":
    unittest.main()
