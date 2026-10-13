import unittest
import numpy as np
import torch
import sys
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from inference import ISLInferenceEngine

class TestISLSystemPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ISLInferenceEngine()
        
    def test_landmark_preprocessing(self):
        """Verify 133 keypoints are correctly formatted to (133, 2) normalized features."""
        dummy_kps = [[0.5, 0.5] for _ in range(133)]
        dummy_kps[11] = [0.4, 0.3]
        dummy_kps[12] = [0.6, 0.3]
        
        proc_kps = self.engine.preprocess_frame(dummy_kps)
        self.assertEqual(proc_kps.shape, (133, 2))
        self.assertFalse(np.isnan(proc_kps).any(), "Preprocessed landmarks contain NaNs")

    def test_label_mapping_integrity(self):
        """Verify label mapping dictionary is non-empty and maps correctly."""
        self.assertGreater(len(self.engine.label_map), 0)
        self.assertIn(0, self.engine.reverse_map)

    def test_prediction_output(self):
        """Verify inference returns valid label, float confidence, and boolean rejection flag."""
        dummy_kps = [[0.5, 0.5, 0.0] for _ in range(133)]
        dummy_confs = [[1.0] for _ in range(133)]
        
        label, conf, is_unknown = self.engine.predict(dummy_kps, dummy_confs)
        self.assertIsInstance(label, str)
        self.assertIsInstance(conf, float)
        self.assertIsInstance(is_unknown, bool)
        self.assertTrue(0.0 <= conf <= 1.0)

    def test_rejection_mechanism(self):
        """Verify low-confidence or noise predictions trigger UNKNOWN SIGN or Buffering/No Sign."""
        # Frame buffering phase
        self.engine.frame_buffer = []
        dummy_kps = [[0.5, 0.5, 0.0] for _ in range(133)]
        dummy_confs = [[1.0] for _ in range(133)]
        
        label, conf, is_unknown = self.engine.predict(dummy_kps, dummy_confs)
        self.assertIn(label, ["UNKNOWN_SIGN", "Buffering...", "DETECTING..."])
        self.assertTrue(is_unknown)

if __name__ == "__main__":
    unittest.main()
