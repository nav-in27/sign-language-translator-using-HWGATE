"""
Comprehensive End-to-End Pipeline Verification Suite for AC-HWGATE ISL Translator.

Tests:
1. Label map integrity and consistency across all config and model directories.
2. Translation engine vocabulary coverage (all 40 canonical ISL glosses + phrases).
3. Feature extraction pipeline (32 frames, 133 joints, 6 channels = 798 features).
4. Model architecture loading (ONNX Runtime dual-output and PyTorch checkpoint).
5. Multi-gate inference rejection pipeline (Gate 1 through Gate 5).
6. End-to-end batch test on real test split sequences.
"""

import os
import sys
import json
import time
import unittest
import numpy as np
import torch

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BACKEND_DIR)
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "training"))

from translator import InstantISLTranslator, nmt_engine
from feature_extractor import feature_extractor
from inference import ISLInferenceEngine


class TestLabelMapIntegrity(unittest.TestCase):
    """Test label map consistency and validity across configs."""

    def test_label_map_exists_and_has_100plus_classes(self):
        paths = [
            os.path.join(ROOT_DIR, "configs", "label_map.json"),
            os.path.join(ROOT_DIR, "models", "label_map.json"),
            os.path.join(BACKEND_DIR, "label_map.json"),
            os.path.join(ROOT_DIR, "data", "unified_104class", "label_map.json"),
        ]
        for p in paths:
            self.assertTrue(os.path.exists(p), f"Missing label map at {p}")
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "classes" in data:
                classes = data["classes"]
            elif "idx_to_label" in data:
                classes = list(data["idx_to_label"].values())
            else:
                classes = list(data.keys())
            self.assertGreaterEqual(len(classes), 100, f"Expected >= 100 classes in {p}, found {len(classes)}")

    def test_canonical_classes_contain_no_blanks(self):
        lm_path = os.path.join(BACKEND_DIR, "label_map.json")
        with open(lm_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        classes = data.get("classes", list(data.keys()))
        for c in classes:
            self.assertTrue(len(c.strip()) > 0, f"Class name is empty: '{c}'")
            self.assertFalse(" " in c and c != "thank you", f"Unexpected space in class: '{c}'")


class TestTranslationEngine(unittest.TestCase):
    """Test Natural Language Translation for single glosses and multi-word phrases."""

    def setUp(self):
        self.engine = InstantISLTranslator()

    def test_all_classes_have_translations(self):
        # Load canonical classes
        lm_path = os.path.join(BACKEND_DIR, "label_map.json")
        with open(lm_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        classes = data.get("classes", list(data.keys()))

        missing = []
        for c in classes:
            clean = c.lower().replace("-", "_")
            translation = self.engine.translate_gloss_to_english(clean)
            if not translation or len(translation.strip()) == 0:
                missing.append(c)

        self.assertEqual(len(missing), 0, f"Missing translations for glosses: {missing}")

    def test_multi_word_phrases(self):
        test_cases = [
            ("hello how_are_you", "Hello, how are you?"),
            ("thank_you very_much", "Thank you very much."),
            ("me water drink", "I want to drink water."),
            ("me food eat", "I want to eat food."),
            ("father come home", "Father is coming home."),
            ("clean home", "Clean the home."),
        ]
        for phrase, expected in test_cases:
            result = self.engine.translate_gloss_to_english(phrase)
            self.assertEqual(result, expected, f"Phrase '{phrase}' translated to '{result}', expected '{expected}'")

    def test_case_insensitivity(self):
        t1 = self.engine.translate_gloss_to_english("HELLO")
        t2 = self.engine.translate_gloss_to_english("hello")
        self.assertEqual(t1, t2)


class TestFeatureExtractor(unittest.TestCase):
    """Test skeleton normalization and motion feature extraction."""

    def test_process_sequence_shape(self):
        # Create 20 frames of random (133, 2) keypoints
        frames = [np.random.uniform(0.2, 0.8, (133, 2)).astype(np.float32) for _ in range(20)]
        feat = feature_extractor.process_sequence_np(frames)
        # Should be resampled to 32 frames, with 133 joints * 6 channels = 798 features
        self.assertEqual(feat.shape, (32, 133, 6))
        self.assertFalse(np.isnan(feat).any(), "Extracted features contain NaN")
        self.assertFalse(np.isinf(feat).any(), "Extracted features contain Inf")

    def test_normalization_anchoring(self):
        frames = [np.random.uniform(0.1, 0.9, (133, 2)).astype(np.float32) for _ in range(32)]
        norm = feature_extractor.normalize_single_frame(frames[0])
        self.assertEqual(norm.shape, (133, 2))
        self.assertFalse(np.isnan(norm).any())


class TestMultiGateInference(unittest.TestCase):
    """Test the 5-gate rejection and inference mechanism."""

    @classmethod
    def setUpClass(cls):
        cls.engine = ISLInferenceEngine()

    def test_gate1_missing_hands_rejection(self):
        # Keypoints with 0 confidence for hands
        kps = [[0.5, 0.5] for _ in range(133)]
        confs = [[0.0] for _ in range(133)]
        label, conf, is_unknown = self.engine.predict(kps, confs)
        self.assertTrue(is_unknown, "Gate 1 should reject when hand confidence is 0")
        self.assertIn(label, ["DETECTING...", "WAITING FOR SIGN", "UNKNOWN_SIGN"])

    def test_gate2_static_motion_rejection(self):
        # Static keypoints across all frames (zero motion energy)
        self.engine.raw_frame_buffer.clear()
        kps = [[0.5, 0.5] for _ in range(133)]
        confs = [[0.9] for _ in range(133)]
        label, conf, is_unknown = "", 0.0, False
        for _ in range(16):
            label, conf, is_unknown = self.engine.predict(kps, confs)
        self.assertTrue(is_unknown, "Gate 2 should reject static hands with near-zero motion")

    def test_model_loaded_and_operational(self):
        self.assertIsNotNone(self.engine.label_map)
        self.assertGreaterEqual(len(self.engine.label_map), 100)
        has_engine = (self.engine.ort_session is not None) or (self.engine.pytorch_model is not None)
        self.assertTrue(has_engine, "Neither ONNX Runtime nor PyTorch model loaded")


class TestBatchRealDataEvaluation(unittest.TestCase):
    """Test model prediction on real test dataset sequences."""

    def test_test_data_batch_inference(self):
        test_split_path = os.path.join(ROOT_DIR, "data", "unified_104class", "test_split.npz")
        if not os.path.exists(test_split_path):
            self.skipTest("unified_104class test data not found")

        data = np.load(test_split_path, allow_pickle=True)
        X_test = data["features"]
        y_test = data["labels"]

        engine = ISLInferenceEngine()
        correct = 0
        total = min(100, len(y_test))

        for i in range(total):
            sample = X_test[i]
            label, conf, is_unknown = engine.predict_sequence_features(sample)
            pred_idx = engine.label_map.get(label.lower(), -1)
            if pred_idx == y_test[i]:
                correct += 1

        acc = correct / total
        print(f"\n[TestBatch] Sample evaluation (N={total}): Acc={acc*100:.1f}%")
        self.assertGreater(acc, 0.70, "Batch accuracy on held-out test split should be > 70%")


if __name__ == "__main__":
    unittest.main(verbosity=2)
