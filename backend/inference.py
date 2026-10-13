"""
AC-HWGATE High-Speed Real-Time Inference Engine with Multi-Stage Unknown Sign Rejection.
Optimized for 104-Class ISL Recognition with Rolling Temporal Buffer and Instant Confirmation.
Bit-Exact Parity with FeatureExtractor and ONNX Runtime Acceleration.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import json
import re
from collections import deque, Counter
from typing import Tuple, List, Optional, Union, Any, Dict

try:
    import onnxruntime as ort
    HAS_ONNX_RUNTIME = True
except ImportError:
    HAS_ONNX_RUNTIME = False
    ort = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
TRAINING_DIR = os.path.join(ROOT_DIR, "training")

sys.path.insert(0, TRAINING_DIR)
from feature_extractor import feature_extractor
from models.ac_hwgate import ACHWGATE


class ISLInferenceEngine:
    """
    AC-HWGATE High-Speed Real-Time Inference Engine with Multi-Stage Unknown Sign Rejection.
    Calibrated for 104-class ISL Recognition with Rolling 32-Frame Buffer and Buffer-Pollution Guards.
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Rolling Temporal Buffer Parameters (T = 32 window, continuous sliding)
        self.window_size = 32
        self.min_frames = 12             # Fast start: require only 12 frames (~0.4s) before predicting
        self.stride = 2                  # Rolling stride: predict every 2 frames (~15 Hz updates)
        self.frame_counter = 0
        self.last_inference_time = 0.0
        self.last_stable_result = ("WAITING FOR SIGN", 0.0, True)
        self.last_status = "WAITING"

        # Calibrated thresholds for 104 classes (uniform random = 0.0096 / ~1.0%)
        self.temperature_cal = 1.0
        self.min_confidence = 0.25       # 26x above uniform random
        self.min_margin = 0.03           # Minimum difference between top-1 and top-2
        self.max_entropy = 4.40          # Maximum Shannon entropy (log(104) ~ 4.64)
        self.min_prototype_sim = 0.35    # Prototype cosine similarity threshold

        # Hand quality thresholds
        self.min_hand_points = 3         # At least 3 hand joints detected
        self.min_hand_conf = 0.10        # Per-joint confidence threshold
        self.min_motion_energy = 0.0002  # Minimum temporal variance to avoid static idle

        # Stability tracking & post-confirmation cooldown
        self.min_consecutive_frames = 2  # 2 consecutive matching predictions to confirm
        self.cooldown_frames = 5         # Cooldown ~0.16s after confirmation to prevent double-triggering
        self.cooldown_counter = 0
        self.missing_frames_counter = 0

        self.raw_frame_buffer = deque(maxlen=self.window_size) # Rolling 32-frame buffer
        self.candidate_history = deque(maxlen=4)               # Rolling prediction history
        self.last_confirmed_sign = None
        self.last_emitted_sign = None

        # Load authoritative ground-truth label map & prototypes
        self.label_map = self._load_label_map()
        self.reverse_map = {}
        for k, v in self.label_map.items():
            idx = int(v)
            if idx not in self.reverse_map or "_" in k:
                self.reverse_map[idx] = str(k)
        self.num_classes = len(set(self.label_map.values()))
        self.prototypes = self._load_prototypes()

        self.ort_session = None
        self.pytorch_model = None
        self.engine_type = "None"

        self.load_model()
        self.warmup()

    def _load_label_map(self) -> Dict[str, int]:
        paths = [
            os.path.join(ROOT_DIR, "data", "unified_104class", "label_map.json"),
            os.path.join(ROOT_DIR, "configs", "selected_104_classes_manifest.json"),
            os.path.join(ROOT_DIR, "data", "unified_104class", "class_names.json"),
            os.path.join(BASE_DIR, "models", "label_map.json"),
            os.path.join(ROOT_DIR, "models", "label_map.json"),
            os.path.join(ROOT_DIR, "data", "unified_50class", "label_map.json"),
        ]
        for p in paths:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        print(f"[AC-HWGATE] Loaded class names list from {p} ({len(data)} classes)")
                        return {c: i for i, c in enumerate(data)}
                    elif "classes" in data and isinstance(data["classes"], list):
                        cls_list = data["classes"]
                        print(f"[AC-HWGATE] Loaded ground-truth classes from manifest {p} ({len(cls_list)} classes)")
                        return {c: i for i, c in enumerate(cls_list)}
                    elif "label_to_idx" in data:
                        print(f"[AC-HWGATE] Loaded ground-truth label map from {p} ({len(data['label_to_idx'])} classes)")
                        return data["label_to_idx"]
                    elif "idx_to_label" in data:
                        idx_map = {int(k): str(v) for k, v in data["idx_to_label"].items()}
                        print(f"[AC-HWGATE] Loaded ground-truth label map from {p} ({len(idx_map)} classes)")
                        return {v: k for k, v in idx_map.items()}
                    elif isinstance(data, dict):
                        print(f"[AC-HWGATE] Loaded ground-truth label map from {p} ({len(data)} classes)")
                        return data
        raise FileNotFoundError("Critical Error: No ground-truth label_map.json found in backend/ or models/!")

    def _load_prototypes(self) -> Optional[np.ndarray]:
        paths = [
            os.path.join(BASE_DIR, "models", "class_prototypes.npy"),
            os.path.join(ROOT_DIR, "models", "class_prototypes.npy"),
            os.path.join(TRAINING_DIR, "class_prototypes.npy"),
        ]
        for p in paths:
            if os.path.exists(p):
                prototypes = np.load(p)
                print(f"[AC-HWGATE] Loaded class prototypes from {p} (shape: {prototypes.shape})")
                return prototypes
        print("[!] Note: class_prototypes.npy not found, prototype distance gate fallback to weights.")
        return None

    def load_model(self):
        """Loads AC-HWGATE preferring optimized ONNX Runtime."""
        self.label_map = self._load_label_map()
        self.reverse_map = {}
        for k, v in self.label_map.items():
            idx = int(v)
            if idx not in self.reverse_map or "_" in k:
                self.reverse_map[idx] = str(k)
        self.num_classes = len(set(self.label_map.values()))

        onnx_paths = [
            os.path.join(BASE_DIR, "models", "ac_hwgate.onnx"),
            os.path.join(ROOT_DIR, "models", "achwgate.onnx")
        ]
        for onnx_path in onnx_paths:
            if HAS_ONNX_RUNTIME and ort is not None and os.path.exists(onnx_path):
                try:
                    opts = ort.SessionOptions()
                    opts.intra_op_num_threads = 4
                    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
                    # Check output shape compatibility
                    out_shape = session.get_outputs()[0].shape
                    if len(out_shape) == 2 and out_shape[1] is not None and out_shape[1] != self.num_classes:
                        print(f"[!] Note: ONNX model output dimension ({out_shape[1]}) differs from label map ({self.num_classes}). Fallback to PyTorch.")
                    else:
                        self.ort_session = session
                        self.engine_type = f"AC-HWGATE ONNX Runtime ({self.num_classes} classes)"
                        print(f"[AC-HWGATE] Loaded ONNX Runtime model from {onnx_path} ({self.num_classes} classes)")
                        return
                except Exception as e:
                    print(f"[AC-HWGATE] Failed to load ONNX model: {e}")
                    self.ort_session = None

        self._load_pytorch_model()

    def _load_pytorch_model(self):
        weights_paths = [
            os.path.join(ROOT_DIR, "checkpoints", "achwgate_104class_best.pth"),
            os.path.join(ROOT_DIR, "checkpoints", "achwgate_best.pth"),
            os.path.join(ROOT_DIR, "checkpoints", "achwgate_50class_best.pth"),
        ]
        for wpath in weights_paths:
            if os.path.exists(wpath):
                try:
                    checkpoint = torch.load(wpath, map_location=self.device, weights_only=False)
                    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
                    model = ACHWGATE(
                        num_classes=self.num_classes,
                        num_joints=133,
                        in_channels=6,
                        sequence_length=32
                    ).to(self.device)
                    model.load_state_dict(state_dict, strict=False)
                    model.eval()
                    self.pytorch_model = model
                    self.engine_type = f"AC-HWGATE PyTorch ({self.num_classes} classes)"
                    print(f"[AC-HWGATE] Loaded PyTorch checkpoint from {wpath} ({self.num_classes} classes)")
                    return
                except Exception as e:
                    print(f"[AC-HWGATE] Failed loading PyTorch weights from {wpath}: {e}")

        print(f"[AC-HWGATE] Initializing untransferred AC-HWGATE fallback ({self.num_classes} classes).")
        self.pytorch_model = ACHWGATE(num_classes=self.num_classes, in_channels=6).to(self.device)
        self.pytorch_model.eval()

    def warmup(self):
        """Warm up ONNX / PyTorch model with dummy inputs to eliminate first-inference delay."""
        try:
            dummy_input = np.zeros((1, self.window_size, 133, 6), dtype=np.float32)
            if self.ort_session is not None:
                inp_name = self.ort_session.get_inputs()[0].name
                for _ in range(3):
                    _ = self.ort_session.run(None, {inp_name: dummy_input})
                print("[AC-HWGATE Engine] ONNX Runtime model warmed up and ready.")
            elif self.pytorch_model is not None:
                with torch.no_grad():
                    inp_t = torch.zeros((1, self.window_size, 133, 6), dtype=torch.float32).to(self.device)
                    for _ in range(3):
                        _ = self.pytorch_model(inp_t)
                print("[AC-HWGATE Engine] PyTorch CUDA model warmed up and ready.")
        except Exception as e:
            print(f"[AC-HWGATE Engine] Warmup note: {e}")

    @property
    def frame_buffer(self) -> deque:
        return self.raw_frame_buffer

    @frame_buffer.setter
    def frame_buffer(self, val: Any) -> None:
        self.raw_frame_buffer.clear()
        if val:
            self.raw_frame_buffer.extend(val)

    def preprocess_frame(self, keypoints: Union[List[List[float]], np.ndarray]) -> np.ndarray:
        arr = np.array(keypoints, dtype=np.float32)
        if arr.shape[-1] > 2:
            arr = arr[..., :2]
        return arr

    def predict(self, keypoints: List[List[float]], confidences: Union[List[List[float]], List[float]]) -> Tuple[str, float, bool]:
        """
        Fast Live Prediction with 32-frame Rolling Buffer, Buffer-Pollution Guards, and Instant Confirmation.
        Returns: Tuple[str, float, bool] -> (label, confidence, is_unknown)
        """
        self.frame_counter += 1

        kp_arr = np.array(keypoints, dtype=np.float32)
        if kp_arr.shape[-1] > 2:
            kp_arr = kp_arr[..., :2]

        conf_arr = np.array(confidences, dtype=np.float32).reshape(-1, 1)

        # Gate 1: Hand-Quality & Keypoint Integrity Check
        lh_conf = conf_arr[33:54]
        rh_conf = conf_arr[54:75]
        lh_vis = int(np.sum(lh_conf >= self.min_hand_conf))
        rh_vis = int(np.sum(rh_conf >= self.min_hand_conf))
        lh_mean = float(np.mean(lh_conf))
        rh_mean = float(np.mean(rh_conf))

        has_hands = (lh_vis >= self.min_hand_points and lh_mean >= 0.10) or \
                    (rh_vis >= self.min_hand_points and rh_mean >= 0.10)

        if not has_hands:
            self.missing_frames_counter += 1
            if self.missing_frames_counter >= 4:
                # User has dropped or rested hands: FLUSH temporal buffer to prevent polluting next sign
                self.raw_frame_buffer.clear()
                self.candidate_history.clear()
                self.last_confirmed_sign = None
                self.last_status = "WAITING"
                self.last_stable_result = ("WAITING FOR SIGN", 0.0, True)
                return "WAITING FOR SIGN", 0.0, True
            self.last_status = "WAITING"
            return "DETECTING...", 0.0, True

        self.missing_frames_counter = 0
        self.raw_frame_buffer.append(kp_arr)

        # Cooldown check: if in post-confirmation cooldown, preserve stable result
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return self.last_stable_result

        # Fast start: require at least self.min_frames (12 frames ~ 0.4s) before predicting
        if len(self.raw_frame_buffer) < self.min_frames:
            self.last_status = "DETECTING"
            return "DETECTING...", 0.0, True

        # Rolling stride check: evaluate every `self.stride` frames (e.g. 2 frames)
        if (self.frame_counter % self.stride != 0):
            return self.last_stable_result

        # Gate 2: Temporal Hand Motion Dynamics Check
        recent_kps = np.stack(list(self.raw_frame_buffer), axis=0)
        hand_pts = recent_kps[:, 33:75, :]
        motion_energy = float(np.mean(np.std(hand_pts, axis=0)))
        if motion_energy < self.min_motion_energy:
            self.candidate_history.append("UNKNOWN_SIGN")
            self.last_status = "DETECTING"
            self.last_stable_result = ("DETECTING...", 0.0, True)
            return "DETECTING...", 0.0, True

        # Feature Extraction & Model Inference (process_sequence_np resamples buffer to 32 frames)
        seq_features = feature_extractor.process_sequence_np(list(self.raw_frame_buffer))
        seq_tensor = np.expand_dims(seq_features, axis=0)

        embeddings: Optional[np.ndarray] = None
        if self.ort_session is not None:
            input_name = self.ort_session.get_inputs()[0].name
            ort_outs = self.ort_session.run(None, {input_name: seq_tensor})
            logits = np.asarray(ort_outs[0], dtype=np.float32)
            if len(logits.shape) == 2:
                logits = logits[0]
            if len(ort_outs) > 1 and ort_outs[1] is not None:
                embeddings = np.asarray(ort_outs[1], dtype=np.float32)
                if len(embeddings.shape) == 2:
                    embeddings = embeddings[0]
        elif self.pytorch_model is not None:
            with torch.no_grad():
                seq_t = torch.tensor(seq_tensor, dtype=torch.float32).to(self.device)
                logits_t, feat_t = self.pytorch_model(seq_t, return_embeddings=True)
                logits = np.asarray(logits_t[0].cpu().numpy(), dtype=np.float32)
                embeddings = np.asarray(feat_t[0].cpu().numpy(), dtype=np.float32)
        else:
            raise RuntimeError("No inference model loaded")

        # Gate 3: Probabilities, Margin & Entropy
        cal_logits = logits / self.temperature_cal
        exp_logits = np.exp(cal_logits - np.max(cal_logits))
        probs = exp_logits / np.sum(exp_logits)

        sorted_probs = np.sort(probs)[::-1]
        top1_prob = float(sorted_probs[0])
        top2_prob = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
        prob_margin = top1_prob - top2_prob
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

        predicted_idx = int(np.argmax(probs))

        if top1_prob < self.min_confidence or prob_margin < self.min_margin or entropy > self.max_entropy:
            self.candidate_history.append("UNKNOWN_SIGN")
            self.last_status = "DETECTING"
            self.last_stable_result = ("DETECTING...", top1_prob, True)
            return "DETECTING...", top1_prob, True

        # Gate 4: Prototype Cosine Distance Gate
        if self.prototypes is not None and embeddings is not None and predicted_idx < len(self.prototypes):
            emb_norm = embeddings / (np.linalg.norm(embeddings) + 1e-8)
            proto = self.prototypes[predicted_idx]
            proto_sim = float(np.dot(emb_norm, proto))
            if proto_sim < self.min_prototype_sim:
                self.candidate_history.append("UNKNOWN_SIGN")
                self.last_status = "DETECTING"
                self.last_stable_result = ("DETECTING...", top1_prob, True)
                return "DETECTING...", top1_prob, True

        raw_class = self.reverse_map.get(predicted_idx, "UNKNOWN_SIGN")
        clean_class = raw_class.upper()

        # Gate 5: Fast Temporal Consistency (2 consecutive matching predictions confirm!)
        self.candidate_history.append(clean_class)
        recent_candidates = list(self.candidate_history)

        if len(recent_candidates) >= 2 and recent_candidates[-1] == recent_candidates[-2] and recent_candidates[-1] not in ["UNKNOWN_SIGN", "IDLE"]:
            stable_class = recent_candidates[-1]
            self.last_confirmed_sign = stable_class
            self.last_status = "CONFIRMED"
            self.cooldown_counter = self.cooldown_frames
            self.candidate_history.clear()

            # BUFFER GUARD: Retain only the trailing 6 frames so completed sign does not contaminate next sign
            while len(self.raw_frame_buffer) > 6:
                self.raw_frame_buffer.popleft()

            res = (stable_class, top1_prob, False)
            self.last_stable_result = res
            return res

        # If not yet 2 consecutive matches, keep state as DETECTING with current top probability
        self.last_status = "DETECTING"
        res = ("DETECTING...", top1_prob, True)
        self.last_stable_result = res
        return res

    def predict_sequence_features(self, seq_features: np.ndarray) -> Tuple[str, float, bool]:
        """
        Runs model inference directly on pre-extracted [32, 133, 6] feature tensor.
        Applies Gate 3 (margin/entropy), Gate 4 (prototype similarity), and returns label, conf, is_unknown.
        """
        seq_tensor = np.expand_dims(seq_features, axis=0) if seq_features.ndim == 3 else seq_features
        embeddings: Optional[np.ndarray] = None
        if self.ort_session is not None:
            input_name = self.ort_session.get_inputs()[0].name
            ort_outs = self.ort_session.run(None, {input_name: seq_tensor.astype(np.float32)})
            logits = np.asarray(ort_outs[0], dtype=np.float32)
            if len(logits.shape) == 2:
                logits = logits[0]
            if len(ort_outs) > 1 and ort_outs[1] is not None:
                embeddings = np.asarray(ort_outs[1], dtype=np.float32)
                if len(embeddings.shape) == 2:
                    embeddings = embeddings[0]
        elif self.pytorch_model is not None:
            with torch.no_grad():
                seq_t = torch.tensor(seq_tensor, dtype=torch.float32).to(self.device)
                logits_t, feat_t = self.pytorch_model(seq_t, return_embeddings=True)
                logits = np.asarray(logits_t[0].cpu().numpy(), dtype=np.float32)
                embeddings = np.asarray(feat_t[0].cpu().numpy(), dtype=np.float32)
        else:
            raise RuntimeError("No inference model loaded")

        cal_logits = logits / self.temperature_cal
        exp_logits = np.exp(cal_logits - np.max(cal_logits))
        probs = exp_logits / np.sum(exp_logits)

        sorted_probs = np.sort(probs)[::-1]
        top1_prob = float(sorted_probs[0])
        top2_prob = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
        prob_margin = top1_prob - top2_prob
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

        predicted_idx = int(np.argmax(probs))

        if top1_prob < self.min_confidence or prob_margin < self.min_margin or entropy > self.max_entropy:
            return "UNKNOWN_SIGN", top1_prob, True

        if self.prototypes is not None and embeddings is not None and predicted_idx < len(self.prototypes):
            emb_norm = embeddings / (np.linalg.norm(embeddings) + 1e-8)
            proto = self.prototypes[predicted_idx]
            proto_sim = float(np.dot(emb_norm, proto))
            if proto_sim < self.min_prototype_sim:
                return "UNKNOWN_SIGN", top1_prob, True

        raw_class = self.reverse_map.get(predicted_idx, "UNKNOWN_SIGN")
        return raw_class.upper(), top1_prob, False


# Singleton engine instance
isl_model = ISLInferenceEngine()
