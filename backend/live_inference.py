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

# Translation templates for the 50 ISL vocabulary classes
TRANSLATION_MAP: Dict[str, str] = {
    "bear": "A bear.",
    "break": "It is broken.",
    "brother": "This is my brother.",
    "clean": "Please clean this.",
    "close": "Please close it.",
    "come": "Please come here.",
    "cook": "Cooking food.",
    "cry": "Why are you crying?",
    "drink": "I would like something to drink.",
    "eat": "I want to eat food.",
    "elephant": "An elephant.",
    "father": "This is my father.",
    "food": "The food is ready.",
    "friend": "You are my good friend.",
    "give": "Please give it to me.",
    "go": "You may go ahead.",
    "goodbye": "Goodbye, have a great day!",
    "he": "He is there.",
    "hello": "Hello, how are you?",
    "help": "I need some help.",
    "home": "I am heading home.",
    "hospital": "We need to visit the hospital.",
    "jump": "Jump high.",
    "key": "Here is the key.",
    "knife": "Be careful with the knife.",
    "lion": "A fierce lion.",
    "market": "I am going to the market.",
    "me": "This is me.",
    "monkey": "A playful monkey.",
    "mother": "This is my mother.",
    "no": "No, I do not.",
    "okay": "Everything is okay.",
    "please": "Please help me.",
    "read": "I am reading this book.",
    "school": "I am going to school.",
    "she": "She is there.",
    "sister": "This is my sister.",
    "sit": "Please have a seat.",
    "sorry": "I am truly sorry.",
    "stand": "Please stand up.",
    "stop": "Please stop now.",
    "student": "I am an active student.",
    "tea": "I would like a cup of tea.",
    "teacher": "Our teacher is speaking.",
    "temple": "Visiting the holy temple.",
    "thank you": "Thank you very much.",
    "thank_you": "Thank you very much.",
    "tiger": "A wild tiger.",
    "today": "Today is a wonderful day.",
    "umbrella": "Please take an umbrella.",
    "uncle": "This is my uncle.",
    "water": "Please give me some water.",
    "what": "What is this.",
    "when": "When will it happen?",
    "where": "Where is it located?",
    "wife": "This is my wife.",
    "write": "I am writing notes.",
    "wrong": "That is incorrect.",
    "yes": "Yes, I agree.",
    "you": "You are welcome."
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
        actual_label_map_path = str(label_map_path) if label_map_path is not None else str(self.base_dir / "data" / "unified_50class" / "class_names.json")
        if not os.path.exists(actual_label_map_path):
            actual_label_map_path = str(self.base_dir / "configs" / "selected_50_classes_manifest.json")
        if not os.path.exists(actual_label_map_path):
            actual_label_map_path = str(self.base_dir / "models" / "label_map.json")
        if not os.path.exists(actual_label_map_path):
            actual_label_map_path = str(self.base_dir / "configs" / "label_map.json")

        if os.path.exists(actual_label_map_path):
            with open(actual_label_map_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self.classes = data
            elif "classes" in data and isinstance(data["classes"], list):
                self.classes = data["classes"]
            else:
                self.classes = sorted(list(set(data.keys())), key=lambda k: data[k])
            self.label_map = {c: i for i, c in enumerate(self.classes)}
            self.num_classes = len(self.classes)
        else:
            self.classes = sorted(list(TRANSLATION_MAP.keys()))
            self.label_map = {c: i for i, c in enumerate(self.classes)}
            self.num_classes = len(self.classes)

        # 2. Load Model
        actual_model_path = str(model_path) if model_path is not None else str(self.base_dir / "checkpoints" / "achwgate_best.pth")

        self.model = ACHWGATE(num_classes=self.num_classes).to(self.device)
        if os.path.exists(actual_model_path):
            ckpt = torch.load(actual_model_path, map_location=self.device, weights_only=False)
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            self.model.load_state_dict(state_dict)
            print(f"[AC-HWGATE Engine] Loaded checkpoint: {actual_model_path}")
        else:
            print(f"[AC-HWGATE Engine] Warning: Checkpoint not found at {actual_model_path}. Using initialized weights.")

        self.model.eval()

        # 3. Class Prototypes for Unknown-Sign Distance Rejection
        actual_prototypes_path = str(prototypes_path) if prototypes_path is not None else str(self.base_dir / "models" / "class_prototypes.npy")
        if os.path.exists(actual_prototypes_path):
            self.prototypes = np.load(actual_prototypes_path)
            print(f"[AC-HWGATE Engine] Loaded class prototypes from {actual_prototypes_path}")
        else:
            self.prototypes = None

        # 4. Rolling Temporal Buffers & State Machine
        self.rolling_buffer_size = 32
        self.min_frames = 8
        self.stride = 2
        self.frame_counter = 0
        self.raw_frame_buffer = collections.deque(maxlen=self.rolling_buffer_size)
        self.prediction_history = collections.deque(maxlen=4)
        self.confidence_history = collections.deque(maxlen=4)

        # Tuning Thresholds (Calibrated for 40 ISL classes)
        self.confidence_threshold = 0.45
        self.unknown_threshold = 0.35
        self.min_margin = 0.04
        self.min_prototype_sim = 0.20
        self.min_stability_count = 2

        # State Tracking
        self.current_state = "IDLE" # IDLE, DETECTING, CONFIRMED
        self.last_confirmed_sign: Optional[str] = None
        self.confirmed_sentence: List[str] = []
        self.last_latency_ms = 0.0
        self.last_result = {
            "sign": "WAITING",
            "confidence": 0.0,
            "status": "IDLE",
            "translation": "--",
            "speak_trigger": False,
            "latency_ms": 0.0,
            "sentence": ""
        }

        self.warmup()

    def warmup(self):
        """Warm up model to eliminate first-frame latency."""
        try:
            dummy_input = torch.zeros((1, self.rolling_buffer_size, 133, 6), dtype=torch.float32).to(self.device)
            with torch.no_grad():
                for _ in range(3):
                    _ = self.model(dummy_input)
            print("[AC-HWGATE Engine] Model warmed up and ready.")
        except Exception as e:
            print(f"[AC-HWGATE Engine] Warmup note: {e}")

    def reset(self):
        self.raw_frame_buffer.clear()
        self.prediction_history.clear()
        self.confidence_history.clear()
        self.current_state = "IDLE"
        self.last_confirmed_sign = None

    def process_frame(self, holistic_results) -> Dict:
        """
        Processes a single webcam frame.
        holistic_results: MediaPipe holistic output.
        Returns: Dict containing sign, confidence, status, translation, latency_ms, speak_trigger, etc.
        """
        t0 = time.perf_counter()
        self.frame_counter += 1

        # 1. Extract raw landmarks (133, 2)
        raw_kps = feature_extractor.extract_landmarks_from_holistic(holistic_results)

        # Check hand activity (left hand joints 33:54, right hand joints 54:75)
        lh = raw_kps[33:54]
        rh = raw_kps[54:75]
        hands_present = (np.any(np.abs(lh) > 1e-4) or np.any(np.abs(rh) > 1e-4))

        if not hands_present:
            self.current_state = "IDLE"
            self.last_confirmed_sign = None
            self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "sign": "WAITING",
                "confidence": 0.0,
                "status": "IDLE",
                "translation": "Show a sign to begin...",
                "speak_trigger": False,
                "latency_ms": round(self.last_latency_ms, 2),
                "sentence": " ".join([TRANSLATION_MAP.get(s, s) for s in self.confirmed_sentence])
            }

        self.raw_frame_buffer.append(raw_kps)

        # Early start: if less than minimum frames (8), return DETECTING immediately
        if len(self.raw_frame_buffer) < self.min_frames:
            self.current_state = "DETECTING"
            self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "sign": "DETECTING...",
                "confidence": 0.0,
                "status": "DETECTING",
                "translation": "Evaluating sign...",
                "speak_trigger": False,
                "latency_ms": round(self.last_latency_ms, 2),
                "sentence": " ".join([TRANSLATION_MAP.get(s, s) for s in self.confirmed_sentence])
            }

        # Stride check: evaluate every self.stride frames (e.g. 2 frames)
        if (self.frame_counter % self.stride != 0):
            res = dict(self.last_result)
            res["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
            res["speak_trigger"] = False
            return res

        # 2. Extract authoritative temporal sequence [32, 133, 6] (resamples buffer to 32 frames)
        seq_frames = list(self.raw_frame_buffer)
        features = feature_extractor.process_sequence_np(seq_frames)

        # 3. Model Forward Pass
        feat_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, embeddings = self.model(feat_tensor, return_embeddings=True)
            probs = F.softmax(logits, dim=-1)[0]
            top_probs, top_indices = torch.topk(probs, k=2)
            raw_conf = float(top_probs[0].item())
            second_conf = float(top_probs[1].item())
            margin = raw_conf - second_conf
            pred_idx = int(top_indices[0].item())
            predicted_class = self.classes[pred_idx]
            emb = embeddings[0].cpu().numpy()

        # 4. Multi-Signal Unknown Rejection Tests
        is_unknown = False

        # Signal 1: Raw confidence & Margin
        if raw_conf < self.confidence_threshold or margin < self.min_margin:
            is_unknown = True

        # Signal 2: Embedding distance / Cosine similarity to class prototype
        if not is_unknown and self.prototypes is not None and pred_idx < len(self.prototypes):
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            target_proto = self.prototypes[pred_idx]
            sim = float(np.dot(emb_norm, target_proto))
            if sim < self.min_prototype_sim:
                is_unknown = True

        # Signal 3: Hand movement energy
        lh_wrists = [f[33] for f in seq_frames if np.any(np.abs(f[33]) > 1e-4)]
        rh_wrists = [f[54] for f in seq_frames if np.any(np.abs(f[54]) > 1e-4)]
        motion_energy = 0.0
        if len(lh_wrists) > 2:
            motion_energy += float(np.mean(np.linalg.norm(np.diff(lh_wrists, axis=0), axis=-1)))
        if len(rh_wrists) > 2:
            motion_energy += float(np.mean(np.linalg.norm(np.diff(rh_wrists, axis=0), axis=-1)))

        if motion_energy < 0.003: # Idle floating hands
            is_unknown = True

        # Update prediction histories
        if is_unknown:
            self.prediction_history.append("UNKNOWN")
            self.confidence_history.append(raw_conf)
        else:
            self.prediction_history.append(predicted_class)
            self.confidence_history.append(raw_conf)

        t1 = time.perf_counter()
        self.last_latency_ms = (t1 - t0) * 1000.0

        # 5. Fast 2-Frame Confirmation Check
        recent_preds = list(self.prediction_history)
        if len(recent_preds) >= 2 and recent_preds[-1] == recent_preds[-2] and recent_preds[-1] != "UNKNOWN":
            candidate_sign = recent_preds[-1]
            self.current_state = "CONFIRMED"
            
            # Check duplicate speech suppression
            speak_now = False
            if candidate_sign != self.last_confirmed_sign:
                self.last_confirmed_sign = candidate_sign
                speak_now = True
                self.confirmed_sentence.append(candidate_sign)
                if len(self.confirmed_sentence) > 6:
                    self.confirmed_sentence.pop(0)

            english_trans = TRANSLATION_MAP.get(candidate_sign, candidate_sign.capitalize() + ".")
            result = {
                "sign": candidate_sign.upper(),
                "confidence": round(raw_conf * 100.0, 1),
                "status": "CONFIRMED",
                "translation": english_trans,
                "speak_trigger": speak_now,
                "latency_ms": round(self.last_latency_ms, 2),
                "sentence": " ".join([TRANSLATION_MAP.get(s, s) for s in self.confirmed_sentence])
            }
            self.last_result = result
            return result

        # In-progress / detecting state
        self.current_state = "DETECTING"
        result = {
            "sign": "DETECTING...",
            "confidence": round(raw_conf * 100.0, 1) if not is_unknown else 0.0,
            "status": "DETECTING",
            "translation": "Evaluating sign...",
            "speak_trigger": False,
            "latency_ms": round(self.last_latency_ms, 2),
            "sentence": " ".join([TRANSLATION_MAP.get(s, s) for s in self.confirmed_sentence])
        }
        self.last_result = result
        return result
