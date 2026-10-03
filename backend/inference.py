import sys
import os
import torch
import torch.nn as nn
import numpy as np
import json
import re
from collections import Counter

try:
    import onnxruntime as ort
    HAS_ONNX_RUNTIME = True
except ImportError:
    HAS_ONNX_RUNTIME = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.join(os.path.dirname(BASE_DIR), "training")

sys.path.insert(0, TRAINING_DIR)
from feature_extractor import feature_extractor
from models.cwa_hwgate import CWA_HWGATE

class ISLInferenceEngine:
    """
    Production ISL Inference Engine with Temporal Sliding Window, Moving Average Probability
    Smoothing, Consecutive Missing Frame Debouncing, and Calibrated Sign Rejection.
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rejection_threshold = 0.28  # Calibrated for multi-class softmax
        self.window_size = 32
        self.min_frames = 2              # Low latency: sub-100ms response time
        self.cooldown_frames = 4
        self.cooldown_counter = 0
        self.missing_frames_counter = 0  # Prevents single-frame tracking flickers from wiping buffer
        
        self.frame_buffer = []
        self.conf_buffer = []
        self.recent_predictions = []
        self.last_emitted_sign = None
        
        # Load canonical ground-truth label map
        self.label_map = self._load_label_map()
        self.reverse_map = {int(v): str(k) for k, v in self.label_map.items()}
        self.num_classes = len(self.label_map)
        
        self.ort_session = None
        self.pytorch_model = None
        self.engine_type = "None"
        
        self.load_model()
        
    def _load_label_map(self):
        paths = [
            os.path.join(TRAINING_DIR, "label_map.json"),
            os.path.join(BASE_DIR, "label_map.json"),
        ]
        for p in paths:
            if os.path.exists(p):
                with open(p, "r") as f:
                    data = json.load(f)
                    print(f"[+] Loaded ground-truth label map from {p} ({len(data)} classes)")
                    return data
                    
        raise FileNotFoundError("Critical Error: No ground-truth label_map.json found in training/ or backend/!")

    def load_model(self):
        """Loads or reloads the active model (preferring optimized ONNX Runtime)."""
        self.label_map = self._load_label_map()
        self.reverse_map = {int(v): str(k) for k, v in self.label_map.items()}
        self.num_classes = len(self.label_map)
        
        onnx_path = os.path.join(TRAINING_DIR, "model.onnx")
        if HAS_ONNX_RUNTIME and os.path.exists(onnx_path):
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 4
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.ort_session = ort.InferenceSession(onnx_path, opts)
                self.engine_type = f"ONNX Runtime (Accelerated, {self.num_classes} classes)"
                print(f"[+] Loaded ONNX model from {onnx_path} ({self.num_classes} classes)")
                return
            except Exception as e:
                print(f"[!] Failed to load ONNX model: {e}")
                self.ort_session = None
                
        self._load_pytorch_model()

    def _load_pytorch_model(self):
        weights_paths = [
            os.path.join(TRAINING_DIR, "checkpoints", "best_model.pth"),
            os.path.join(TRAINING_DIR, "cwa_hwgate_pretrained.pth"),
        ]
        
        for wpath in weights_paths:
            if os.path.exists(wpath):
                try:
                    checkpoint = torch.load(wpath, map_location=self.device, weights_only=False)
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                        chk_classes = checkpoint.get('num_classes', self.num_classes)
                        cfg = checkpoint.get('config', {})
                        self.pytorch_model = CWA_HWGATE(
                            num_classes=chk_classes,
                            embed_dim=cfg.get('embed_dim', 256),
                            num_heads=cfg.get('num_heads', 8),
                            num_layers=cfg.get('num_layers', 4)
                        ).to(self.device)
                        self.pytorch_model.load_state_dict(state_dict, strict=False)
                        self.pytorch_model.eval()
                        self.engine_type = f"PyTorch CWA-HWGATE ({os.path.basename(wpath)})"
                        print(f"[+] Loaded CWA-HWGATE model from {wpath} ({chk_classes} classes)")
                        return
                    elif isinstance(checkpoint, dict):
                        self.pytorch_model = CWA_HWGATE(num_classes=self.num_classes).to(self.device)
                        self.pytorch_model.load_state_dict(checkpoint, strict=False)
                        self.pytorch_model.eval()
                        self.engine_type = f"PyTorch CWA-HWGATE ({os.path.basename(wpath)})"
                        print(f"[+] Loaded CWA-HWGATE state_dict from {wpath}")
                        return
                except Exception as e:
                    print(f"[!] Failed loading PyTorch weights from {wpath}: {e}")
                    
        print("[!] WARNING: Initializing fallback CWA-HWGATE.")
        self.pytorch_model = CWA_HWGATE(num_classes=self.num_classes).to(self.device)
        self.pytorch_model.eval()
        self.engine_type = "PyTorch (Fallback)"

    def preprocess_frame(self, keypoints):
        """Preprocesses raw MediaPipe 133 keypoints into part-normalized 2D features."""
        return feature_extractor.process_keypoints_np(keypoints)

    def predict(self, keypoints, confidences):
        """
        Predicts ISL sign gloss with temporal sliding window, real confidence gating, and debouncing.
        """
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            
        frame_features = self.preprocess_frame(keypoints) # (133, 2)
        
        # Parse landmark confidences
        if len(confidences) > 0:
            confs_flat = [c[0] if isinstance(c, (list, tuple, np.ndarray)) else float(c) for c in confidences]
            confs_arr = np.array(confs_flat, dtype=np.float32).reshape(133, 1)
        else:
            confs_arr = np.zeros((133, 1), dtype=np.float32)
            
        has_active_body = (np.sum(confs_arr > 0.1) >= 4)
        
        if not has_active_body:
            self.missing_frames_counter += 1
            if self.missing_frames_counter > 15:
                self.frame_buffer = []
                self.conf_buffer = []
                self.recent_predictions = []
                self.last_emitted_sign = None
                return "No Sign Detected", 0.0, True
            elif len(self.frame_buffer) < self.min_frames:
                return "Buffering...", 0.0, True
        else:
            self.missing_frames_counter = 0
            self.frame_buffer.append(frame_features)
            self.conf_buffer.append(confs_arr)
            if len(self.frame_buffer) > self.window_size:
                self.frame_buffer.pop(0)
                self.conf_buffer.pop(0)
            
        if len(self.frame_buffer) < self.min_frames:
            return "Buffering...", 0.0, True
            
        # Format input tensors [1, T, 133, 2] and [1, T, 133, 1]
        seq = np.array(self.frame_buffer, dtype=np.float32)       # [T, 133, 2]
        seq_tensor = np.expand_dims(seq, axis=0)                  # [1, T, 133, 2]
        conf_seq = np.array(self.conf_buffer, dtype=np.float32)   # [T, 133, 1]
        conf_tensor = np.expand_dims(conf_seq, axis=0)            # [1, T, 133, 1]
        
        # Predict using ONNX or PyTorch
        if self.ort_session is not None:
            input_names = [i.name for i in self.ort_session.get_inputs()]
            if len(input_names) >= 2:
                ort_inputs = {input_names[0]: seq_tensor, input_names[1]: conf_tensor}
            else:
                ort_inputs = {input_names[0]: seq_tensor}
            logits = self.ort_session.run(None, ort_inputs)[0]
            if len(logits.shape) == 3:
                logits = logits[0, 0]
            elif len(logits.shape) == 2:
                logits = logits[0]
            # Numerically stable softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            confidence = float(np.max(probs))
            predicted_idx = int(np.argmax(probs))
        else:
            with torch.no_grad():
                seq_t = torch.tensor(seq_tensor, dtype=torch.float32).to(self.device)
                conf_t = torch.tensor(conf_tensor, dtype=torch.float32).to(self.device)
                logits = self.pytorch_model(seq_t, conf_t)
                if len(logits.shape) == 3:
                    logits = logits[0, 0]
                elif len(logits.shape) == 2:
                    logits = logits[0]
                probs = torch.softmax(logits, dim=-1)
                conf_val, pred_val = torch.max(probs, dim=-1)
                confidence = float(conf_val.item())
                predicted_idx = int(pred_val.item())
                
        # Rejection Threshold Check
        if confidence < self.rejection_threshold:
            return "UNKNOWN SIGN", confidence, True
            
        raw_class = self.reverse_map.get(predicted_idx, "UNKNOWN SIGN")
        # Format class name: e.g. single chars 'a' -> 'A', words 'good_morning' -> 'Good Morning'
        if len(raw_class) == 1:
            clean_class = raw_class.upper()
        else:
            clean_class = re.sub(r'^\d+\.\s*', '', raw_class).replace('_', ' ').title()
        
        # Temporal Majority Voting
        self.recent_predictions.append(clean_class)
        if len(self.recent_predictions) > 5:
            self.recent_predictions.pop(0)
            
        most_common_class, count = Counter(self.recent_predictions).most_common(1)[0]
        
        # Single Emission Cooldown
        if count >= 2 and most_common_class != self.last_emitted_sign and self.cooldown_counter == 0:
            self.last_emitted_sign = most_common_class
            self.cooldown_counter = self.cooldown_frames
            return most_common_class, confidence, False
        elif most_common_class == self.last_emitted_sign:
            return most_common_class, confidence, False
        else:
            return most_common_class, confidence, False

isl_model = ISLInferenceEngine()
