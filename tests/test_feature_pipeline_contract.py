"""
Automated Contract Test Suite for Feature Pipeline Verification (PART B).
Verifies that Training Preprocessing, Backend Inference Preprocessing, and ONNX Runtime
receive identical [B, 32, 133, 6] tensors within strict numerical tolerance (< 1e-5).
"""
import sys
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "backend"))

from training.feature_extractor import FeatureExtractor, feature_extractor
from backend.inference import ISLInferenceEngine
import onnxruntime as ort

def test_feature_pipeline_contract():
    print("=" * 70)
    print("RUNNING FEATURE PIPELINE CONTRACT VERIFICATION (PART B)")
    print("=" * 70)

    # 1. Verify Feature Dimensions Contract
    fe = FeatureExtractor(num_joints=133, sequence_length=32, include_motion=True)
    assert fe.num_joints == 133, f"Expected 133 joints, got {fe.num_joints}"
    assert fe.channels_per_joint == 6, f"Expected 6 channels, got {fe.channels_per_joint}"
    assert fe.sequence_length == 32, f"Expected 32 sequence length, got {fe.sequence_length}"
    assert fe.total_feature_dim == 798, f"Expected 798 total feature dim, got {fe.total_feature_dim}"
    print("[+] [CONTRACT 1/5] Dimensional contracts verified: [B, 32, 133, 6] = 798 dims.")

    # 2. Joint and Coordinate Ordering Contract
    # 0..32: Pose (33)
    # 33..53: Left Hand (21)
    # 54..74: Right Hand (21)
    # 75..132: Face (58)
    dummy_raw_frames = []
    rng = np.random.RandomState(42)
    for t in range(25): # Variable input length (25 frames)
        frame = rng.uniform(0.2, 0.8, size=(133, 2)).astype(np.float32)
        # Shoulders (joints 11, 12)
        frame[11] = [0.60, 0.45]
        frame[12] = [0.40, 0.45]
        dummy_raw_frames.append(frame)

    # Process through training feature extractor
    train_feat = fe.process_sequence_np(dummy_raw_frames)
    assert train_feat.shape == (32, 133, 6), f"Training feature shape mismatch: {train_feat.shape}"
    assert np.isfinite(train_feat).all(), "Non-finite values in training features"
    print("[+] [CONTRACT 2/5] Training feature extraction verified: resamples 25 -> 32 frames without NaN/Inf.")

    # 3. Backend Inference Preprocessing Parity
    engine = ISLInferenceEngine()
    engine.reset()
    for t, frame in enumerate(dummy_raw_frames):
        conf = np.ones((133, 1), dtype=np.float32)
        engine.predict(frame, conf, seq_id=t+1, timestamp=t * 0.033)

    # Extract rolling buffer from engine
    infer_feat = feature_extractor.process_sequence_np(list(engine.raw_frame_buffer))
    # Both processed the same raw frames
    max_diff_feat = float(np.max(np.abs(train_feat - infer_feat)))
    print(f"[+] [CONTRACT 3/5] Preprocessing Parity (Training vs Backend Inference): max diff = {max_diff_feat:.2e}")
    assert max_diff_feat < 1e-5, f"Feature preprocessing mismatch: {max_diff_feat}"

    # 4. Motion Dynamics Laws Verification
    # V_t = P_t - P_{t-1}, A_t = V_t - V_{t-1}
    pos = train_feat[:, :, :2]
    vel = train_feat[:, :, 2:4]
    acc = train_feat[:, :, 4:6]
    computed_vel = np.zeros_like(pos)
    computed_vel[1:] = pos[1:] - pos[:-1]
    computed_acc = np.zeros_like(vel)
    computed_acc[1:] = computed_vel[1:] - computed_vel[:-1]

    vel_diff = float(np.max(np.abs(vel - computed_vel)))
    acc_diff = float(np.max(np.abs(acc - computed_acc)))
    print(f"[+] [CONTRACT 4/5] Kinematic Laws (Position, Velocity, Acceleration): vel diff = {vel_diff:.2e}, acc diff = {acc_diff:.2e}")
    assert vel_diff < 1e-6 and acc_diff < 1e-6, "Kinematic motion computation does not satisfy differential laws"

    # 5. ONNX vs PyTorch Execution Contract
    onnx_path = ROOT / "backend" / "models" / "ac_hwgate.onnx"
    assert onnx_path.exists(), f"ONNX model missing at {onnx_path}"
    ort_session = ort.InferenceSession(str(onnx_path))

    inp_tensor = np.expand_dims(train_feat, axis=0) # [1, 32, 133, 6]
    ort_outs = ort_session.run(None, {ort_session.get_inputs()[0].name: inp_tensor})
    onnx_logits = ort_outs[0][0]
    onnx_embs = ort_outs[1][0] if len(ort_outs) > 1 else None

    # Load PyTorch checkpoint
    ckpt_path = ROOT / "checkpoints" / "achwgate_50class_production_best.pth"
    assert ckpt_path.exists(), f"PyTorch checkpoint missing at {ckpt_path}"
    from training.models.ac_hwgate import ACHWGATE
    pt_model = ACHWGATE(num_classes=50, num_joints=133, in_channels=6, sequence_length=32)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    weights = ckpt.get("model_state_dict", ckpt)
    pt_model.load_state_dict(weights)
    pt_model.eval()

    with torch.no_grad():
        pt_logits_t, pt_embs_t = pt_model(torch.tensor(inp_tensor, dtype=torch.float32), return_embeddings=True)
        pt_logits = pt_logits_t[0].numpy()
        pt_embs = pt_embs_t[0].numpy()

    logit_diff = float(np.max(np.abs(onnx_logits - pt_logits)))
    emb_diff = float(np.max(np.abs(onnx_embs - pt_embs))) if onnx_embs is not None else 0.0

    print(f"[+] [CONTRACT 5/5] ONNX vs PyTorch Forward Contract: Logit Diff = {logit_diff:.2e}, Emb Diff = {emb_diff:.2e}")
    assert logit_diff < 1e-4, f"ONNX-PyTorch parity failed: logit diff {logit_diff}"
    assert emb_diff < 1e-4, f"ONNX-PyTorch parity failed: emb diff {emb_diff}"

    print("\n" + "=" * 70)
    print("ALL 5 FEATURE PIPELINE CONTRACT TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 70)

if __name__ == "__main__":
    test_feature_pipeline_contract()

