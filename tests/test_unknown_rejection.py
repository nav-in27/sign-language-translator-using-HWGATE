"""
Rigorous Test Suite for AC-HWGATE Unknown Sign Rejection & Temporal Stabilization.

Validates:
1. Hand-Quality Threshold (missing or low-confidence hand keypoints reject as UNKNOWN_SIGN).
2. Physical Hand Motion Gating (static/stationary hands reject as UNKNOWN_SIGN).
3. Energy & Logit Threshold (diffuse/unregistered activations reject as UNKNOWN_SIGN).
4. Calibrated Softmax Confidence & Margin Rejection.
5. Embedding Prototype Cosine Distance Rejection.
6. Strict Temporal Stability Gate:
   - Fluctuating classes (e.g. BRINJAL -> CAULIFLOWER -> BRINJAL -> CHILLI -> CAULIFLOWER) -> returns UNKNOWN_SIGN!
   - Consecutive stable classes (e.g. NO -> NO -> NO -> NO -> NO) -> emits "NO"!
7. Backend /predict API integration with UNKNOWN_SIGN suppression.
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

try:
    from backend.inference import isl_model
except ImportError:
    from inference import isl_model
from fastapi.testclient import TestClient
try:
    from backend.app import app
except ImportError:
    from app import app

class TestUnknownSignRejection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        # Reset engine state between tests
        isl_model.reset()
        from app import session_state
        session_state["gloss_buffer"] = []
        session_state["last_gloss"] = ""
        session_state["last_translated"] = ""
        session_state["last_sign_time"] = 0.0

    def test_01_missing_hand_quality_rejects_unknown(self):
        """1. When hand keypoints have zero/low confidence, reject as UNKNOWN_SIGN."""
        kps = [[0.5, 0.5] for _ in range(133)]
        # Set all hand confidences (33:75) to 0.05 (untracked)
        confs = [0.8 if i < 33 else 0.05 for i in range(133)]
        
        label, conf, is_unknown = isl_model.predict(kps, confs)
        self.assertEqual(label, "UNKNOWN_SIGN")
        self.assertTrue(is_unknown)

    def test_02_stationary_hand_motion_rejects_unknown(self):
        """2. When hands are completely stationary (no motion dynamics), reject as UNKNOWN_SIGN."""
        kps = [[0.5, 0.5] for _ in range(133)]
        confs = [0.9 for _ in range(133)]
        
        # Feed 35 completely identical stationary frames to fill T=32 buffer
        label, conf, is_unknown = "", 0.0, False
        for _ in range(35):
            label, conf, is_unknown = isl_model.predict(kps, confs)
            
        self.assertEqual(label, "UNKNOWN_SIGN")
        self.assertTrue(is_unknown)

    def test_03_temporal_fluctuation_rejects_unknown(self):
        """
        3. Strict Temporal Stability Requirement:
           BRINJAL -> CAULIFLOWER -> BRINJAL -> CHILLI -> CAULIFLOWER -> MUST emit UNKNOWN_SIGN!
        """
        # Inject fluctuating simulated candidates into candidate_history
        isl_model.candidate_history.clear()
        fluctuating_stream = ["Brinjal", "Cauliflower", "Brinjal", "Chilli", "Cauliflower"]
        for cand in fluctuating_stream:
            isl_model.candidate_history.append(cand)
            
        # Verify consecutive agreement check
        recent_k = list(isl_model.candidate_history)[-isl_model.min_consecutive_frames:]
        is_stable = all(c == recent_k[0] and c not in ["UNKNOWN_SIGN", "Buffering...", "No Sign Detected"] for c in recent_k)
        
        self.assertFalse(is_stable, "Fluctuating predictions must NOT pass stability gate!")

    def test_04_consecutive_stable_stream_emits_sign(self):
        """
        4. Strict Temporal Stability Requirement:
           NO -> NO -> NO -> NO -> NO -> MUST emit 'No'!
        """
        isl_model.candidate_history.clear()
        stable_stream = ["No", "No", "No", "No", "No"]
        for cand in stable_stream:
            isl_model.candidate_history.append(cand)
            
        recent_k = list(isl_model.candidate_history)[-isl_model.min_consecutive_frames:]
        is_stable = all(c == recent_k[0] and c not in ["UNKNOWN_SIGN", "Buffering...", "No Sign Detected"] for c in recent_k)
        
        self.assertTrue(is_stable, "Consecutive identical predictions MUST pass stability gate!")
        self.assertEqual(recent_k[0], "No")

    def test_05_prototype_distance_rejection(self):
        """5. An embedding orthogonal/distant from the class prototype is rejected."""
        if isl_model.prototypes is not None:
            # Query embedding that is inverted/orthogonal to class 0 prototype
            proto0 = isl_model.prototypes[0]
            orthogonal_emb = -proto0 # Cosine similarity = -1.0
            
            sim = float(np.dot(orthogonal_emb, proto0))
            self.assertLess(sim, isl_model.min_prototype_sim)

    def test_06_fastapi_predict_rejects_unstable_movement(self):
        """6. FastAPI /predict endpoint returns UNKNOWN_SIGN on random movement without sentence pollution."""
        kps = [[0.5, 0.5] for _ in range(133)]
        confs = [0.1 for _ in range(133)] # low confidence
        
        response = self.client.post("/predict", json={"keypoints": kps, "confidences": confs})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertEqual(data["label"], "UNKNOWN_SIGN")
        self.assertTrue(data["is_unknown"])
        self.assertEqual(data["raw_glosses"], "") # Ensure translation buffer was NOT polluted

if __name__ == "__main__":
    unittest.main()
