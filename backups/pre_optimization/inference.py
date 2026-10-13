import sys
import os
import torch
import torch.nn as nn
import numpy as np
import json
import re
from collections import deque, Counter
from typing import Tuple, List, Optional, Union

try:
    import onnxruntime as ort
    HAS_ONNX_RUNTIME = True
except ImportError:
    HAS_ONNX_RUNTIME = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
TRAINING_DIR = os.path.join(ROOT_DIR, "training")

sys.path.insert(0, TRAINING_DIR)
from feature_extractor import feature_extractor
from models.ac_hwgate import ACHWGATE

class ISLInferenceEngine:
    """
    AC-HWGATE Real-Time Inference Engine with Multi-Stage Unknown Sign Rejection.
    Calibrated for 40-class ISL Recognition.
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Buffer and temporal window parameters
        self.window_size = 32
        self.min_frames = 32             # Min frames to start predicting (full 32-frame temporal buffer)
        self.min_consecutive_frames = 3  # Stable consensus threshold
        
        # Calibration and threshold parameters for 40 classes (uniform random = 0.025)
        self.temperature_cal = 1.0
        self.min_confidence = 0.45       # Calibrated threshold for 40 classes
        self.min_margin = 0.05           # Minimum difference between top 1 and top 2 prob
        self.max_entropy = 3.60          # Maximum Shannon entropy (log(40) ~ 3.688)
        self.min_max_logit = 0.20        # Minimum max logit
        self.min_energy = 0.50           # Energy score threshold
        self.min_prototype_sim = 0.10    # Minimum cosine similarity to class prototype
        
        # Hand quality thresholds
        self.min_hand_points = 4         # At least 4 hand joints detected
        self.min_hand_conf = 0.15        # Per-joint confidence threshold
        self.min_motion_energy = 0.0005  # Minimum temporal variance to avoid static idle
        
        # Cooldown and state
        self.cooldown_frames = 12
        self.cooldown_counter = 0
        self.missing_frames_counter = 0

        self.raw_frame_buffer = []       # Stores raw 133 keypoints [T, 133, 2]
        self.candidate_history = deque(maxlen=8) # Queue of raw frame candidate predictions
        self.last_emitted_sign = None

        # Load authoritative ground-truth label map & prototypes
        self.label_map = self._load_label_map()
        self.reverse_map = {int(v): str(k) for k, v in self.label_map.items()}
        self.num_classes = len(self.label_map)
        self.prototypes = self._load_prototypes()

        self.ort_session = None
        self.pytorch_model = None
        self.engine_type = "None"

        self.load_model()

    def _load_label_map(self):
        paths = [
            os.path.join(ROOT_DIR, "models", "label_map.json"),
            os.path.join(BASE_DIR, "label_map.json"),
            os.path.join(ROOT_DIR, "configs", "label_map.json"),
            os.path.join(TRAINING_DIR, "label_map.json"),
        ]
        for p in paths:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "label_to_idx" in data:
                        print(f"[AC-HWGATE] Loaded ground-truth label map from {p} ({len(data['label_to_idx'])} classes)")
                        return data["label_to_idx"]
                    elif isinstance(data, dict):
                        print(f"[AC-HWGATE] Loaded ground-truth label map from {p} ({len(data)} classes)")
                        return data
        raise FileNotFoundError("Critical Error: No ground-truth label_map.json found in backend/ or models/!")

    def _load_prototypes(self):
        paths = [
            os.path.join(ROOT_DIR, "models", "class_prototypes.npy"),
            os.path.join(BASE_DIR, "models", "class_prototypes.npy"),
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
        self.reverse_map = {int(v): str(k) for k, v in self.label_map.items()}
        self.num_classes = len(self.label_map)

        onnx_paths = [
            os.path.join(BASE_DIR, "models", "ac_hwgate.onnx"),
            os.path.join(ROOT_DIR, "models", "achwgate.onnx")
        ]
        for onnx_path in onnx_paths:
            if HAS_ONNX_RUNTIME and os.path.exists(onnx_path):
                try:
                    opts = ort.SessionOptions()
                    opts.intra_op_num_threads = 4
                    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    self.ort_session = ort.InferenceSession(onnx_path, opts)
                    self.engine_type = f"AC-HWGATE ONNX Runtime ({self.num_classes} classes)"
                    print(f"[AC-HWGATE] Loaded ONNX Runtime model from {onnx_path} ({self.num_classes} classes)")
                    return
                except Exception as e:
                    print(f"[AC-HWGATE] Failed to load ONNX model: {e}")
                    self.ort_session = None

        self._load_pytorch_model()

    def _load_pytorch_model(self):
        weights_paths = [
            os.path.join(ROOT_DIR, "checkpoints", "achwgate_best.pth"),
            os.path.join(TRAINING_DIR, "checkpoints", "ac_hwgate_best.pth"),
        ]
        for wpath in weights_paths:
            if os.path.exists(wpath):
                try:
                    checkpoint = torch.load(wpath, map_location=self.device, weights_only=False)
                    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
                    self.pytorch_model = ACHWGATE(
                        num_classes=self.num_classes,
                        num_joints=133,
                        in_channels=6,
                        sequence_length=32
                    ).to(self.device)
                    self.pytorch_model.load_state_dict(state_dict, strict=False)
                    self.pytorch_model.eval()
                    self.engine_type = f"AC-HWGATE PyTorch ({self.num_classes} classes)"
                    print(f"[AC-HWGATE] Loaded PyTorch checkpoint from {wpath} ({self.num_classes} classes)")
                    return
                except Exception as e:
                    print(f"[AC-HWGATE] Failed loading PyTorch weights from {wpath}: {e}")

        print(f"[AC-HWGATE] Initializing untransferred AC-HWGATE fallback ({self.num_classes} classes).")
        self.pytorch_model = ACHWGATE(num_classes=self.num_classes, in_channels=6).to(self.device)
        self.pytorch_model.eval()
        self.engine_type = f"AC-HWGATE PyTorch (Fallback, {self.num_classes} classes)"

    def predict(self, keypoints: List[List[float]], confidences: Union[List[List[float]], List[float]]) -> Tuple[str, float, bool]:
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        kp_arr = np.array(keypoints, dtype=np.float32)
        if kp_arr.shape[-1] > 2:
            kp_arr = kp_arr[..., :2]

        conf_arr = np.array(confidences, dtype=np.float32).reshape(-1, 1)

        # Gate 1: Hand-Quality & Keypoint Integrity Check
        lh_conf = conf_arr[33:54]
        rh_conf = conf_arr[54:75]
        lh_vis = np.sum(lh_conf >= self.min_hand_conf)
        rh_vis = np.sum(rh_conf >= self.min_hand_conf)
        lh_mean = float(np.mean(lh_conf))
        rh_mean = float(np.mean(rh_conf))

        has_hands = (lh_vis >= self.min_hand_points and lh_mean >= 0.10) or \
                    (rh_vis >= self.min_hand_points and rh_mean >= 0.10)

        if not has_hands:
            self.missing_frames_counter += 1
            if self.missing_frames_counter > 15:
                self.raw_frame_buffer.clear()
                self.candidate_history.clear()
                self.last_emitted_sign = None
                return "No Sign Detected", 0.0, True
            self.candidate_history.append("UNKNOWN_SIGN")
            return "UNKNOWN_SIGN", 0.0, True
        else:
            self.missing_frames_counter = 0
            self.raw_frame_buffer.append(kp_arr)
            if len(self.raw_frame_buffer) > self.window_size:
                self.raw_frame_buffer.pop(0)

        if len(self.raw_frame_buffer) < self.min_frames:
            self.candidate_history.append("Buffering...")
            return "Buffering...", 0.0, True

        # Gate 2: Temporal Hand Motion Dynamics
        recent_kps = np.stack(self.raw_frame_buffer, axis=0)
        hand_pts = recent_kps[:, 33:75, :]
        motion_energy = float(np.mean(np.std(hand_pts, axis=0)))
        if motion_energy < self.min_motion_energy:
            self.candidate_history.append("UNKNOWN_SIGN")
            return "UNKNOWN_SIGN", 0.0, True

        # Feature Extraction & Model Inference
        seq_features = feature_extractor.process_sequence_np(self.raw_frame_buffer)
        seq_tensor = np.expand_dims(seq_features, axis=0)

        embeddings = None
        if self.ort_session is not None:
            input_name = self.ort_session.get_inputs()[0].name
            ort_outs = self.ort_session.run(None, {input_name: seq_tensor})
            logits = ort_outs[0]
            if len(logits.shape) == 2:
                logits = logits[0]
            if len(ort_outs) > 1:
                embeddings = ort_outs[1]
                if len(embeddings.shape) == 2:
                    embeddings = embeddings[0]
        else:
            with torch.no_grad():
                seq_t = torch.tensor(seq_tensor, dtype=torch.float32).to(self.device)
                logits_t, feat_t = self.pytorch_model(seq_t, return_embeddings=True)
                logits = logits_t[0].cpu().numpy()
                embeddings = feat_t[0].cpu().numpy()

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
            return "UNKNOWN_SIGN", top1_prob, True

        # Gate 4: Prototype Cosine Distance
        if self.prototypes is not None and embeddings is not None and predicted_idx < len(self.prototypes):
            emb_norm = embeddings / (np.linalg.norm(embeddings) + 1e-8)
            proto = self.prototypes[predicted_idx]
            proto_sim = float(np.dot(emb_norm, proto))
            if proto_sim < self.min_prototype_sim:
                self.candidate_history.append("UNKNOWN_SIGN")
                return "UNKNOWN_SIGN", top1_prob, True

        raw_class = self.reverse_map.get(predicted_idx, "UNKNOWN_SIGN")
        clean_class = raw_class.upper()

        # Gate 5: Temporal Prediction Stability Gate
        self.candidate_history.append(clean_class)
        recent_window = list(self.candidate_history)[-5:]
        valid_candidates = [c for c in recent_window if c not in ["UNKNOWN_SIGN", "Buffering...", "No Sign Detected"]]

        if len(valid_candidates) >= 3:
            counts = Counter(valid_candidates)
            most_common_class, freq = counts.most_common(1)[0]
            if freq >= 3:
                stable_class = most_common_class
            else:
                return "UNKNOWN_SIGN", top1_prob, True
        else:
            return "UNKNOWN_SIGN", top1_prob, True

        # Duplicate suppression
        if stable_class != self.last_emitted_sign and self.cooldown_counter == 0:
            self.last_emitted_sign = stable_class
            self.cooldown_counter = self.cooldown_frames
            return stable_class, top1_prob, False
        else:
            return stable_class, top1_prob, False

# Singleton instance
isl_model = ISLInferenceEngine()
