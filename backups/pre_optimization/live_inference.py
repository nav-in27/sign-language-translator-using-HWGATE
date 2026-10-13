"""
Production Real-Time Live Inference Engine for AC-HWGATE ISL Recognition.
Implements:
  - 32-frame rolling temporal buffer with authoritative FeatureExtractor.
  - Multi-signal Unknown Rejection (confidence threshold, temporal stability,
    prototype embedding distance, hand presence/movement gating).
  - State Machine: IDLE -> SIGNING -> CONFIRMED -> COOLDOWN.
  - Duplicate suppression & cooldown timer.
  - Deterministic English grammar reconstruction templates.
  - Low-latency TTS trigger.
"""

import os
import sys
import time
import json
import collections
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from training.feature_extractor import FeatureExtractor, feature_extractor
from training.models.ac_hwgate import ACHWGATE

# Translation templates for the 40 ISL vocabulary classes
TRANSLATION_MAP = {
    "hello": "Hello, how are you?",
    "goodbye": "Goodbye, have a great day!",
    "thank you": "Thank you very much.",
    "sorry": "I am truly sorry.",
    "please": "Please help me.",
    "yes": "Yes, I agree.",
    "no": "No, I do not.",
    "help": "I need some help.",
    "stop": "Please stop now.",
    "okay": "Everything is okay.",
    "me": "This is me.",
    "you": "You are welcome.",
    "he": "He is there.",
    "she": "She is there.",
    "mother": "This is my mother.",
    "father": "This is my father.",
    "brother": "This is my brother.",
    "sister": "This is my sister.",
    "friend": "You are my good friend.",
    "teacher": "Our teacher is speaking.",
    "student": "I am an active student.",
    "home": "I am heading home.",
    "school": "I am going to school.",
    "hospital": "We need to visit the hospital.",
    "market": "I am going to the market.",
    "eat": "I want to eat food.",
    "drink": "I would like something to drink.",
    "water": "Please give me some water.",
    "food": "The food is ready.",
    "tea": "I would like a cup of tea.",
    "come": "Please come here.",
    "go": "You may go ahead.",
    "sit": "Please have a seat.",
    "stand": "Please stand up.",
    "read": "I am reading this book.",
    "write": "I am writing notes.",
    "what": "What is this?",
    "where": "Where is it located?",
    "when": "When will it happen?",
    "today": "Today is a wonderful day."
}

class LiveISLInferenceEngine:
    def __init__(
        self,
        config_path: Optional[str] = None,
        model_path: Optional[str] = None,
        label_map_path: Optional[str] = None,
        prototypes_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        self.base_dir = BASE_DIR
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[AC-HWGATE Engine] Initializing on device: {self.device}")

        # 1. Load Label Map
        if label_map_path is None:
            label_map_path = self.base_dir / "models" / "label_map.json"
        if not os.path.exists(label_map_path):
            label_map_path = self.base_dir / "configs" / "label_map.json"

        if os.path.exists(label_map_path):
            with open(label_map_path, "r", encoding="utf-8") as f:
                self.label_map = json.load(f)
            self.classes = self.label_map["classes"]
            self.num_classes = len(self.classes)
        else:
            self.classes = sorted(list(TRANSLATION_MAP.keys()))
            self.num_classes = len(self.classes)

        # 2. Load Model
        if model_path is None:
            model_path = self.base_dir / "checkpoints" / "achwgate_best.pth"

        self.model = ACHWGATE(num_classes=self.num_classes).to(self.device)
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            self.model.load_state_dict(state_dict)
            print(f"[AC-HWGATE Engine] Loaded checkpoint: {model_path}")
        else:
            print(f"[AC-HWGATE Engine] Warning: Checkpoint not found at {model_path}. Using initialized weights.")

        self.model.eval()

        # 3. Class Prototypes for Unknown-Sign Distance Rejection
        if prototypes_path is None:
            prototypes_path = self.base_dir / "models" / "class_prototypes.npy"
        if os.path.exists(prototypes_path):
            self.prototypes = np.load(prototypes_path)
            print(f"[AC-HWGATE Engine] Loaded class prototypes from {prototypes_path}")
        else:
            self.prototypes = None

        # 4. Rolling Temporal Buffers & State Machine
        self.rolling_buffer_size = 32
        self.raw_frame_buffer = collections.deque(maxlen=self.rolling_buffer_size)
        self.prediction_history = collections.deque(maxlen=8)
        self.confidence_history = collections.deque(maxlen=8)

        # Tuning Thresholds
        self.confidence_threshold = 0.60
        self.unknown_threshold = 0.35
        self.min_stability_count = 3
        self.cooldown_frames = 15
        self.cooldown_counter = 0

        # State Tracking
        self.current_state = "IDLE" # IDLE, SIGNING, CONFIRMED, COOLDOWN
        self.last_confirmed_sign = None
        self.confirmed_sentence = []
        self.last_latency_ms = 0.0

    def reset(self):
        self.raw_frame_buffer.clear()
        self.prediction_history.clear()
        self.confidence_history.clear()
        self.current_state = "IDLE"
        self.last_confirmed_sign = None
        self.cooldown_counter = 0

    def process_frame(self, holistic_results) -> Dict:
        """
        Processes a single webcam frame.
        holistic_results: MediaPipe holistic output.
        Returns: Dict containing sign, confidence, status, translation, fps, latency, etc.
        """
        t0 = time.perf_counter()

        # 1. Extract raw landmarks (133, 2)
        raw_kps = feature_extractor.extract_landmarks_from_holistic(holistic_results)
        self.raw_frame_buffer.append(raw_kps)

        # Check hand activity (left hand joints 33:54, right hand joints 54:75)
        lh = raw_kps[33:54]
        rh = raw_kps[54:75]
        hands_present = (np.any(np.abs(lh) > 1e-4) or np.any(np.abs(rh) > 1e-4))

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        # Default result payload
        result = {
            "sign": "NO SIGN",
            "confidence": 0.0,
            "status": "IDLE",
            "translation": "--",
            "speak_trigger": False,
            "latency_ms": 0.0,
            "sentence": " ".join([TRANSLATION_MAP.get(s, s) for s in self.confirmed_sentence])
        }

        # If insufficient frames or hands not present, return IDLE
        if len(self.raw_frame_buffer) < 16 or not hands_present:
            self.current_state = "IDLE"
            self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
            result["latency_ms"] = round(self.last_latency_ms, 2)
            result["status"] = "IDLE"
            return result

        # 2. Extract authoritative temporal sequence [32, 133, 6]
        seq_frames = list(self.raw_frame_buffer)
        features = feature_extractor.process_sequence_np(seq_frames) # [32, 133, 6]

        # 3. Model Forward Pass
        feat_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device) # [1, 32, 133, 6]
        with torch.no_grad():
            logits, embeddings = self.model(feat_tensor, return_embeddings=True)
            probs = F.softmax(logits, dim=-1)[0]
            top_prob, pred_idx = torch.max(probs, dim=-1)
            raw_conf = float(top_prob.item())
            predicted_class = self.classes[int(pred_idx.item())]
            emb = embeddings[0].cpu().numpy()

        # 4. Unknown Rejection Tests
        is_unknown = False

        # Signal 1: Raw confidence floor
        if raw_conf < self.unknown_threshold:
            is_unknown = True

        # Signal 2: Embedding distance to class prototype
        if not is_unknown and self.prototypes is not None:
            target_proto = self.prototypes[int(pred_idx.item())]
            dist = float(np.linalg.norm(emb - target_proto))
            if dist > 2.5: # Out-of-distribution distance
                is_unknown = True

        # Signal 3: Hand movement energy
        # Calculate velocity of wrists across the rolling buffer
        lh_wrists = [f[33] for f in seq_frames if np.any(np.abs(f[33]) > 1e-4)]
        rh_wrists = [f[54] for f in seq_frames if np.any(np.abs(f[54]) > 1e-4)]
        motion_energy = 0.0
        if len(lh_wrists) > 2:
            motion_energy += float(np.mean(np.linalg.norm(np.diff(lh_wrists, axis=0), axis=-1)))
        if len(rh_wrists) > 2:
            motion_energy += float(np.mean(np.linalg.norm(np.diff(rh_wrists, axis=0), axis=-1)))

        if motion_energy < 0.005: # Idle hands floating
            is_unknown = True

        # Update prediction histories
        if is_unknown:
            self.prediction_history.append("UNKNOWN")
            self.confidence_history.append(raw_conf)
        else:
            self.prediction_history.append(predicted_class)
            self.confidence_history.append(raw_conf)

        # 5. Temporal Smoothing & Majority Voting
        recent_preds = list(self.prediction_history)
        if len(recent_preds) >= self.min_stability_count:
            most_common = collections.Counter(recent_preds).most_common(1)[0]
            candidate_sign, count = most_common
            avg_conf = float(np.mean([c for p, c in zip(self.prediction_history, self.confidence_history) if p == candidate_sign]))
        else:
            candidate_sign = "UNKNOWN"
            count = 0
            avg_conf = raw_conf

        t1 = time.perf_counter()
        self.last_latency_ms = (t1 - t0) * 1000.0

        # 6. State Machine & Emission Logic
        if candidate_sign == "UNKNOWN" or is_unknown:
            self.current_state = "UNKNOWN"
            result["sign"] = "UNKNOWN"
            result["confidence"] = round(raw_conf * 100.0, 1)
            result["status"] = "NOT RECOGNIZED"
            result["translation"] = "--"
            result["latency_ms"] = round(self.last_latency_ms, 2)
            return result

        # Known candidate sign
        if count >= self.min_stability_count and avg_conf >= self.confidence_threshold:
            # Check duplicate suppression and cooldown
            if candidate_sign != self.last_confirmed_sign and self.cooldown_counter == 0:
                self.current_state = "CONFIRMED"
                self.last_confirmed_sign = candidate_sign
                self.cooldown_counter = self.cooldown_frames
                self.confirmed_sentence.append(candidate_sign)
                if len(self.confirmed_sentence) > 6:
                    self.confirmed_sentence.pop(0)

                result["sign"] = candidate_sign.upper()
                result["confidence"] = round(avg_conf * 100.0, 1)
                result["status"] = "CONFIRMED"
                result["translation"] = TRANSLATION_MAP.get(candidate_sign, candidate_sign.capitalize() + ".")
                result["speak_trigger"] = True
            else:
                self.current_state = "SIGNING"
                result["sign"] = candidate_sign.upper()
                result["confidence"] = round(avg_conf * 100.0, 1)
                result["status"] = "SIGNING"
                result["translation"] = TRANSLATION_MAP.get(candidate_sign, candidate_sign.capitalize() + ".")
        else:
            self.current_state = "SIGNING"
            result["sign"] = candidate_sign.upper()
            result["confidence"] = round(avg_conf * 100.0, 1)
            result["status"] = "DETECTING"
            result["translation"] = TRANSLATION_MAP.get(candidate_sign, "--")

        result["latency_ms"] = round(self.last_latency_ms, 2)
        result["sentence"] = " ".join([TRANSLATION_MAP.get(s, s) for s in self.confirmed_sentence])
        return result
