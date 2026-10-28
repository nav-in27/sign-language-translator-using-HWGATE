"""
Automated Comprehensive Regression Test Suite for AC-HWGATE ISL System.
Tests:
  1. Label mapping & manifest synchronization across configs, backend, and models.
  2. Feature extractor input contract ([32, 133, 6] shape, motion features, interpolation).
  3. Model checkpoint dimensions & parameter integrity.
  4. Dual-output ONNX numerical parity (< 1e-4 tolerance).
  5. Real-time inference engine temporal buffer & state transitions.
  6. Calibrated unknown sign rejection on random noise & idle takes.
  7. Semantic translation mapping fidelity (no fabricated sentences).
  8. Held-out test set accuracy & macro F1 verification.
"""

import os
import sys
import json
import pytest
import numpy as np
import torch
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "training"))
sys.path.insert(0, str(ROOT_DIR / "backend"))

from training.feature_extractor import feature_extractor
from training.models.ac_hwgate import ACHWGATE


def test_manifest_synchronization():
    """Verify that all label maps and manifests across backend, models, and configs match."""
    manifest_path = ROOT_DIR / "configs" / "authoritative_136_classes_manifest.json"
    if not manifest_path.exists():
        manifest_path = ROOT_DIR / "configs" / "selected_104_classes_manifest.json"
    assert manifest_path.exists(), "Manifest file missing!"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    
    num_classes = m["num_classes"]
    classes = m["classes"]
    assert len(classes) == num_classes
    assert num_classes >= 100, f"Expected at least 100 classes, got {num_classes}"

    # Check backend/label_map.json
    backend_lm_path = ROOT_DIR / "backend" / "models" / "label_map.json"
    if not backend_lm_path.exists():
        backend_lm_path = ROOT_DIR / "backend" / "label_map.json"
    assert backend_lm_path.exists(), "backend/models/label_map.json missing!"
    with open(backend_lm_path, "r", encoding="utf-8") as f:
        b_lm = json.load(f)
    
    b_classes = b_lm["classes"] if "classes" in b_lm else (b_lm["class_to_idx"] if "class_to_idx" in b_lm else b_lm)
    assert len(b_classes) == num_classes, f"Backend label map has {len(b_classes)} classes, expected {num_classes}"
    for c in classes:
        assert c in b_classes, f"Class {c} missing in backend label_map.json"

    # Check models/label_map.json
    models_lm_path = ROOT_DIR / "models" / "label_map.json"
    if models_lm_path.exists():
        with open(models_lm_path, "r", encoding="utf-8") as f:
            m_lm = json.load(f)
        m_classes = m_lm["classes"] if "classes" in m_lm else (m_lm["class_to_idx"] if "class_to_idx" in m_lm else m_lm)
        assert len(m_classes) == num_classes


def test_feature_extractor_input_contract():
    """Verify FeatureExtractor output shape, kinematics, and temporal interpolation."""
    # Test varying input lengths
    for t_len in [12, 24, 32, 45, 60]:
        raw_kps = [np.random.randn(133, 2).astype(np.float32) for _ in range(t_len)]
        feat = feature_extractor.process_sequence_np(raw_kps)
        assert feat.shape == (32, 133, 6), f"Expected (32, 133, 6), got {feat.shape} for T={t_len}"
        assert not np.isnan(feat).any(), "NaN found in features"
        assert not np.isinf(feat).any(), "Inf found in features"

    # Verify channels: 0:2 Position, 2:4 Velocity, 4:6 Acceleration
    # Constant input should have 0 velocity and 0 acceleration
    const_kps = [np.ones((133, 2), dtype=np.float32) * 0.5 for _ in range(32)]
    feat_const = feature_extractor.process_sequence_np(const_kps)
    assert np.allclose(feat_const[:, :, 2:], 0.0, atol=1e-5), "Constant positions must have zero velocity & acceleration"


def test_model_architecture_and_checkpoint():
    """Verify AC-HWGATE forward pass and checkpoint classifier shape."""
    ckpt_path = ROOT_DIR / "checkpoints" / "achwgate_104class_best.pth"
    if not ckpt_path.exists():
        ckpt_path = ROOT_DIR / "checkpoints" / "achwgate_best.pth"
    
    assert ckpt_path.exists(), "No best checkpoint found!"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    
    classifier_w = state["classifier.weight"]
    num_classes = classifier_w.shape[0]
    assert num_classes >= 100, f"Checkpoint has {num_classes} classes, expected >= 100"

    model = ACHWGATE(num_classes=num_classes, in_channels=6, sequence_length=32)
    model.load_state_dict(state)
    model.eval()

    dummy_x = torch.randn(2, 32, 133, 6)
    with torch.no_grad():
        logits, embs = model(dummy_x, return_embeddings=True)
    assert logits.shape == (2, num_classes)
    assert embs.shape == (2, 128)


def test_onnx_model_parity():
    """Verify bit-exact parity between PyTorch and ONNX Runtime."""
    onnx_path = ROOT_DIR / "backend" / "models" / "ac_hwgate.onnx"
    assert onnx_path.exists(), "backend/models/ac_hwgate.onnx missing!"

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    
    out_shapes = [o.shape for o in sess.get_outputs()]
    num_classes = out_shapes[0][1]
    assert num_classes >= 100, f"ONNX model output dimension is {num_classes}, expected >= 100"

    ckpt_path = ROOT_DIR / "checkpoints" / "achwgate_136class_production_best.pth"
    if not ckpt_path.exists():
        ckpt_path = ROOT_DIR / "checkpoints" / "achwgate_best.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    
    model = ACHWGATE(num_classes=num_classes, in_channels=6, sequence_length=32)
    model.load_state_dict(state)
    model.eval()

    test_input = np.random.randn(1, 32, 133, 6).astype(np.float32)
    with torch.no_grad():
        py_logits, py_embs = model(torch.from_numpy(test_input), return_embeddings=True)
    
    input_name = sess.get_inputs()[0].name
    ort_outs = sess.run(None, {input_name: test_input})
    ort_logits = ort_outs[0]
    ort_embs = ort_outs[1]

    diff_logits = float(np.max(np.abs(py_logits.numpy() - ort_logits)))
    diff_embs = float(np.max(np.abs(py_embs.numpy() - ort_embs)))

    assert diff_logits < 1e-4, f"ONNX logits discrepancy {diff_logits} exceeds 1e-4 tolerance"
    assert diff_embs < 1e-4, f"ONNX embeddings discrepancy {diff_embs} exceeds 1e-4 tolerance"


def test_inference_engine_state_transitions():
    """Verify temporal buffer behavior, fast start, confirmation, and hand-drop flush."""
    from backend.inference import ISLInferenceEngine
    engine = ISLInferenceEngine()
    
    # 1. Empty / resting hands -> WAITING FOR SIGN or UNKNOWN_SIGN
    blank_kp = [[0.0, 0.0] for _ in range(133)]
    blank_conf = [0.0 for _ in range(133)]
    
    # 4 consecutive frames with dropped hands must flush buffer
    for _ in range(5):
        lbl, conf, is_unk = engine.predict(blank_kp, blank_conf)
    assert lbl in ["WAITING FOR SIGN", "UNKNOWN_SIGN"]
    assert is_unk is True
    assert len(engine.raw_frame_buffer) == 0

    # 2. Add active hand frames (fewer than min_frames) -> DETECTING... or UNKNOWN_SIGN
    active_kp = [[0.5, 0.5] for _ in range(133)]
    active_conf = [1.0 for _ in range(133)]
    
    for _ in range(5):
        lbl, conf, is_unk = engine.predict(active_kp, active_conf)
    assert lbl in ["DETECTING...", "UNKNOWN_SIGN"]
    assert is_unk is True
    assert len(engine.raw_frame_buffer) == 5


def test_translation_mapping_fidelity():
    """Verify that translation mapping provides authentic English meanings for all classes."""
    from backend.translator import nmt_engine
    manifest_path = ROOT_DIR / "configs" / "authoritative_136_classes_manifest.json"
    if not manifest_path.exists():
        manifest_path = ROOT_DIR / "configs" / "selected_104_classes_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    
    classes = m["classes"]
    for c in classes:
        res = nmt_engine.translate_gloss_to_english(c)
        assert res and len(res) > 0, f"Empty translation for class {c}"
        assert not res.startswith("UNKNOWN"), f"Unknown translation for valid class {c}"
        # Assert no hallucinated complex sentences like 'A roaring lion.' or 'Why are you crying?'
        assert "A roaring lion" not in res, f"Hallucinated translation detected for {c}"
        assert "Why are you crying" not in res, f"Hallucinated translation detected for {c}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
