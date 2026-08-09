import sys
import os
import torch
import torch.nn as nn
import numpy as np
import json
import re

try:
    import onnxruntime as ort
    HAS_ONNX_RUNTIME = True
except ImportError:
    HAS_ONNX_RUNTIME = False

# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.join(os.path.dirname(BASE_DIR), "training")
SCRATCH_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
ISL_REC_DIR = os.path.join(SCRATCH_DIR, "isl_recognition")

# ============================================================
# PyTorch Model Architecture (Fallback)
# ============================================================
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, lstm_out):
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        return context

class ISLModelLite(nn.Module):
    def __init__(self, num_classes, input_dim=225, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = TemporalAttention(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, confidences=None):
        B, T = x.shape[0], x.shape[1]
        x = x[:, :, :75, :3].reshape(B, T, -1)
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        return self.classifier(context)

# ============================================================
# Production Inference Engine
# ============================================================
class ISLInferenceEngine:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rejection_threshold = 0.65
        self.window_size = 64
        self.min_frames = 15
        self.frame_buffer = []
        self.recent_predictions = []
        
        # 1. Load label mapping
        self.label_map = self._load_label_map()
        self.reverse_map = {v: k for k, v in self.label_map.items()}
        self.num_classes = len(self.label_map)
        
        # 2. Try loading ONNX model first, fall back to PyTorch
        self.ort_session = None
        self.pytorch_model = None
        self.engine_type = "None"
        
        onnx_path = os.path.join(TRAINING_DIR, "model.onnx")
        if HAS_ONNX_RUNTIME and os.path.exists(onnx_path):
            try:
                self.ort_session = ort.InferenceSession(onnx_path)
                self.engine_type = "ONNX Runtime"
                print(f"Loaded ONNX model from {onnx_path}")
            except Exception as e:
                print(f"Failed to load ONNX model: {e}")
                
        if self.ort_session is None:
            self._load_pytorch_model()
            
    def _load_label_map(self):
        paths = [
            os.path.join(ISL_REC_DIR, "label_map.json"),
            os.path.join(TRAINING_DIR, "label_map.json"),
            os.path.join(BASE_DIR, "label_map.json")
        ]
        for p in paths:
            if os.path.exists(p):
                with open(p, "r") as f:
                    print(f"Loaded label map from: {p}")
                    return json.load(f)
                    
        # Fallback default map
        default_map = {f"Class_{i}": i for i in range(264)}
        return default_map

    def _load_pytorch_model(self):
        weights_paths = [
            os.path.join(ISL_REC_DIR, "best_model_lite.pth"),
            os.path.join(TRAINING_DIR, "LSTM_weights.pth"),
            os.path.join(ISL_REC_DIR, "best_model.pth")
        ]
        
        for wpath in weights_paths:
            if os.path.exists(wpath):
                try:
                    state_dict = torch.load(wpath, map_location=self.device)
                    # Check num classes in state dict
                    chk_classes = self.num_classes
                    for key in ['classifier.3.weight', 'fc.weight', 'classifier.weight']:
                        if key in state_dict:
                            chk_classes = state_dict[key].shape[0]
                            break
                            
                    self.pytorch_model = ISLModelLite(num_classes=chk_classes).to(self.device)
                    self.pytorch_model.load_state_dict(state_dict, strict=False)
                    self.pytorch_model.eval()
                    self.engine_type = f"PyTorch ({os.path.basename(wpath)})"
                    print(f"Loaded PyTorch model from {wpath}")
                    return
                except Exception as e:
                    print(f"Failed loading PyTorch weights from {wpath}: {e}")
                    
        print("WARNING: No model weights found! Initializing randomly for structural fallback.")
        self.pytorch_model = ISLModelLite(num_classes=self.num_classes).to(self.device)
        self.pytorch_model.eval()
        self.engine_type = "PyTorch (Random Initialized)"

    def preprocess_frame(self, keypoints):
        """
        Input: 133 keypoints [x, y, z] in [0, 1] screen space.
        Returns: (80, 3) normalized landmark array.
        """
        kp = np.array(keypoints, dtype=np.float32) # (133, 3)
        
        pose = kp[0:33, :3]   # 33 x 3
        lh = kp[33:54, :3]    # 21 x 3
        rh = kp[54:75, :3]    # 21 x 3
        
        combined = np.concatenate([pose, lh, rh], axis=0) # (75, 3)
        
        # 3D Normalization: Center origin at shoulder midpoint (Pose 11 & 12)
        left_shoulder = combined[11, :]
        right_shoulder = combined[12, :]
        center = (left_shoulder + right_shoulder) / 2.0
        
        normalized = combined - center
        
        shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
        if shoulder_dist > 1e-4:
            normalized = normalized / shoulder_dist
            
        # Pad 5 dummy keypoints to reach 80 keypoints
        pad_kps = np.zeros((5, 3), dtype=np.float32)
        normalized_80 = np.concatenate([normalized, pad_kps], axis=0) # (80, 3)
        
        return normalized_80

    def predict(self, keypoints, confidences):
        """
        Predicts ISL sign gloss from single incoming frame keypoints.
        Applies temporal sliding window, confidence rejection, and debouncing.
        """
        frame_features = self.preprocess_frame(keypoints)
        
        # Check hand visibility / activity
        hand_confs = np.array([c[0] for c in confidences[33:75]])
        pose_confs = np.array([c[0] for c in confidences[0:33]])
        
        has_hands = (np.sum(hand_confs > 0.1) >= 3) or (pose_confs[15] > 0.3 or pose_confs[16] > 0.3)
        
        if not has_hands:
            self.frame_buffer = []
            self.recent_predictions = []
            return "No Sign Detected", 0.0, True
            
        self.frame_buffer.append(frame_features)
        if len(self.frame_buffer) > self.window_size:
            self.frame_buffer.pop(0)
            
        if len(self.frame_buffer) < self.min_frames:
            return "Buffering...", 0.0, True
            
        # Build 64-frame sequence: [1, 64, 80, 3]
        seq = np.array(self.frame_buffer, dtype=np.float32)
        if seq.shape[0] < self.window_size:
            pad_len = self.window_size - seq.shape[0]
            padding = np.zeros((pad_len, 80, 3), dtype=np.float32)
            seq = np.vstack((seq, padding))
            
        seq_tensor = np.expand_dims(seq, axis=0) # (1, 64, 80, 3)
        conf_tensor = np.ones((1, 64, 80, 1), dtype=np.float32)
        
        # Predict using ONNX or PyTorch
        if self.ort_session is not None:
            input_names = [i.name for i in self.ort_session.get_inputs()]
            if len(input_names) >= 2:
                ort_inputs = {input_names[0]: seq_tensor, input_names[1]: conf_tensor}
            else:
                ort_inputs = {input_names[0]: seq_tensor}
            logits = self.ort_session.run(None, ort_inputs)[0]
            if len(logits.shape) == 3:
                logits = logits[0]
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
            confidence = float(np.max(probs))
            predicted_idx = int(np.argmax(probs))
        else:
            with torch.no_grad():
                seq_t = torch.tensor(seq_tensor, dtype=torch.float32).to(self.device)
                conf_t = torch.tensor(conf_tensor, dtype=torch.float32).to(self.device)
                logits = self.pytorch_model(seq_t, conf_t)
                if len(logits.shape) == 3:
                    logits = logits[0]
                probs = torch.softmax(logits, dim=-1)
                conf_val, pred_val = torch.max(probs, dim=-1)
                confidence = float(conf_val.item())
                predicted_idx = int(pred_val.item())
                
        # Rejection thresholding
        if confidence < self.rejection_threshold:
            return "UNKNOWN SIGN", confidence, True
            
        raw_class = self.reverse_map.get(predicted_idx, "UNKNOWN SIGN")
        clean_class = re.sub(r'^\d+\.\s*', '', raw_class)
        
        # Temporal debouncing (majority voting over last 5 predictions)
        self.recent_predictions.append(clean_class)
        if len(self.recent_predictions) > 5:
            self.recent_predictions.pop(0)
            
        from collections import Counter
        most_common_class, count = Counter(self.recent_predictions).most_common(1)[0]
        
        return most_common_class, confidence, False

# Singleton instance
isl_model = ISLInferenceEngine()
