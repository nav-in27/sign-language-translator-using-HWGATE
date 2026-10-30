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
import hashlib
import uuid
from datetime import datetime, timezone
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
import time
from feature_extractor import feature_extractor
from models.ac_hwgate import ACHWGATE

class RecognitionState:
    INITIALIZING = "INITIALIZING"
    WAITING = "WAITING"
    COLLECTING_FRAMES = "COLLECTING_FRAMES"
    RECOGNIZING = "RECOGNIZING"
    CONFIRMED = "CONFIRMED"
    UNCERTAIN = "UNCERTAIN"
    ERROR = "ERROR"


class PredictionResult(tuple):
    """
    A 3-tuple (label, confidence, is_unknown) with rich metadata attributes for
    state machine, event tracking, sequence IDs, top predictions, and translation.
    Unpacks cleanly as (label, confidence, is_unknown) for 100% backward compatibility.
    """
    label: str
    confidence: float
    is_unknown: bool
    state: str
    status: str
    seq_id: int
    timestamp: float
    event_id: Optional[str]
    provisional_sign: Optional[str]
    confirmed_sign: Optional[str]
    translation: str
    previous_confirmed_sign: Optional[str]
    previous_translation: str
    is_previous_translation: bool
    speech_action: str
    top_predictions: List[Dict[str, Any]]
    rejection_reason: Optional[str]
    reset_reason: Optional[str]
    session_id: Optional[str]
    window_id: str
    buffer_length_before: int
    buffer_length_after: int
    frame_age: float
    model_version: str
    engine: str
    latency_ms: Dict[str, float]

    def __new__(
        cls,
        label: str,
        confidence: float,
        is_unknown: bool,
        state: str = RecognitionState.WAITING,
        seq_id: int = 0,
        timestamp: float = 0.0,
        event_id: Optional[str] = None,
        provisional_sign: Optional[str] = None,
        confirmed_sign: Optional[str] = None,
        translation: str = "",
        previous_confirmed_sign: Optional[str] = None,
        previous_translation: str = "",
        is_previous_translation: bool = False,
        speech_action: str = "none",
        top_predictions: Optional[List[Dict[str, Any]]] = None,
        rejection_reason: Optional[str] = None,
        reset_reason: Optional[str] = None,
        session_id: Optional[str] = None,
        window_id: Optional[str] = None,
        buffer_length_before: int = 0,
        buffer_length_after: int = 0,
        frame_age: float = 0.0,
        model_version: str = "AC-HWGATE-v2-136class-ONNX",
        engine: str = "AC-HWGATE",
        latency_ms: Optional[Dict[str, float]] = None
    ):
        inst = super().__new__(cls, (label, float(confidence), bool(is_unknown)))
        inst.label = str(label)
        inst.confidence = float(confidence)
        inst.is_unknown = bool(is_unknown)
        inst.state = str(state)
        inst.status = str(state) # Alias for backward compatibility
        inst.seq_id = int(seq_id)
        inst.timestamp = float(timestamp)
        inst.event_id = event_id
        inst.provisional_sign = provisional_sign
        inst.confirmed_sign = confirmed_sign
        inst.translation = str(translation)
        inst.previous_confirmed_sign = previous_confirmed_sign
        inst.previous_translation = str(previous_translation)
        inst.is_previous_translation = bool(is_previous_translation)
        inst.speech_action = str(speech_action)
        inst.top_predictions = top_predictions or []
        inst.rejection_reason = rejection_reason
        inst.reset_reason = reset_reason
        inst.session_id = session_id
        inst.window_id = window_id or f"window_{seq_id}"
        inst.buffer_length_before = int(buffer_length_before)
        inst.buffer_length_after = int(buffer_length_after)
        inst.frame_age = float(frame_age)
        inst.model_version = str(model_version)
        inst.engine = str(engine)
        inst.latency_ms = latency_ms or {}
        return inst

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "is_unknown": self.is_unknown,
            "status": self.state,
            "state": self.state,
            "session_id": self.session_id,
            "window_id": self.window_id,
            "seq_id": self.seq_id,
            "timestamp": self.timestamp,
            "frame_age": round(float(self.frame_age), 4),
            "buffer_length_before": self.buffer_length_before,
            "buffer_length_after": self.buffer_length_after,
            "event_id": self.event_id,
            "provisional_sign": self.provisional_sign,
            "confirmed_sign": self.confirmed_sign,
            "translation": self.translation,
            "previous_confirmed_sign": self.previous_confirmed_sign,
            "previous_translation": self.previous_translation,
            "is_previous_translation": self.is_previous_translation,
            "speech_action": self.speech_action,
            "top_predictions": self.top_predictions,
            "reset_reason": self.reset_reason,
            "rejection_reason": self.rejection_reason,
            "model_version": self.model_version,
            "engine": self.engine,
            "latency_ms": self.latency_ms
        }


class ISLInferenceEngine:
    """
    AC-HWGATE High-Speed Real-Time Inference Engine with Consistent Recognition State Machine.
    Supports explicit states (WAITING, COLLECTING_FRAMES, RECOGNIZING, CONFIRMED, UNCERTAIN),
    Sequence ID out-of-order rejection, event-based confirmation, and verified isolated translation.
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Rolling Temporal Buffer Parameters (T = 32 window matching AC-HWGATE 32-frame contract)
        self.window_size = 32
        self.min_frames = 16             # Responsive 16-frame buffer (~0.53s) before classification
        self.stride = 2                  # Rolling stride: predict every 2 frames (~15 Hz updates)
        self.frame_counter = 0
        self.active_frame_counter = 0
        self.last_inference_time = 0.0

        # State Machine Tracking
        self.session_id: str = str(uuid.uuid4())[:8]
        self.window_seq: int = 0
        self.last_frame_timestamp: float = 0.0
        self.state = RecognitionState.INITIALIZING
        self.last_status = RecognitionState.INITIALIZING
        self.last_processed_seq_id = 0
        self.event_counter = 0
        self.current_event_id: Optional[str] = None
        self.last_confirmed_sign: Optional[str] = None
        self.last_confirmed_translation: str = ""
        self.last_confirmed_event_id: Optional[str] = None
        self.current_held_sign: Optional[str] = None
        self.bout_confirmed_sign: Optional[str] = None
        self.provisional_sign: Optional[str] = None
        self.provisional_confidence: float = 0.0
        self.top_k_predictions: List[Dict[str, Any]] = []

        # Debounce and persistence counters for state stability
        self.hand_present_counter = 0
        self.hand_missing_counter = 0
        self.uncertain_eval_counter = 0
        self.stationary_eval_counter = 0
        self.cooldown_frames = 10        # Responsive cooldown (~333ms) to prevent duplicate confirmation without stalling next sign
        self.cooldown_counter = 0
        self.missing_frames_counter = 0
        self.max_tracking_loss_frames = 12  # 12 frames (~400ms at 30fps) before resting hand reset
        self.max_frame_age_sec = 1.0       # >1.0s gap between frames is treated as stream interruption

        # Calibrated thresholds (50-class production: random baseline = 0.02)
        self.temperature_cal = 1.0
        self.min_confidence = 0.42       # Calibrated threshold rejecting transient arm motion noise
        self.min_margin = 0.06           # Minimum difference between top-1 and top-2 candidates
        self.max_entropy = 4.20          # Maximum Shannon entropy gate
        self.min_prototype_sim = 0.35    # Calibrated prototype cosine similarity threshold

        # Hand quality and active signing space thresholds
        self.min_hand_points = 3         # At least 3 hand joints detected
        self.min_hand_conf = 0.10        # Per-joint confidence threshold
        self.min_motion_energy = 0.0065  # Calibrated to reject resting sensor noise (~0.001-0.003) while admitting signs (>0.012)

        # Stability tracking & post-confirmation cooldown
        self.min_consecutive_frames = 3  # Require 3 consecutive matching evaluations (~200ms) for rock-solid stability

        self.raw_frame_buffer = deque(maxlen=self.window_size)
        self.candidate_history = deque(maxlen=8)
        self.last_emitted_sign = None

        # Instrumentation log (stores last 200 pipeline events for trace inspection)
        self.event_log = deque(maxlen=200)

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
        self.loaded_artifact_path: Optional[str] = None
        self.loaded_artifact_sha256: Optional[str] = None
        self.loaded_at_utc: Optional[str] = None
        self.load_error: Optional[str] = None
        if os.path.exists(os.path.join(ROOT_DIR, "checkpoints", "achwgate_50class_production_best.pth")):
            self.production_checkpoint_path = os.path.join(ROOT_DIR, "checkpoints", "achwgate_50class_production_best.pth")
            self.manifest_path = os.path.join(ROOT_DIR, "configs", "selected_50_classes_manifest.json")
        else:
            self.production_checkpoint_path = os.path.join(ROOT_DIR, "checkpoints", "achwgate_136class_production_best.pth")
            self.manifest_path = os.path.join(ROOT_DIR, "configs", "authoritative_136_classes_manifest.json")

        try:
            from translator import nmt_engine
        except ImportError:
            from backend.translator import nmt_engine
        self.translator = nmt_engine

        self.load_model()
        self.warmup()

        self.last_stable_result = PredictionResult("UNKNOWN_SIGN", 0.0, True, state=RecognitionState.WAITING)
        self.state = RecognitionState.WAITING
        self.last_status = RecognitionState.WAITING

    @property
    def model(self):
        """Lazily loads PyTorch AC-HWGATE model if requested."""
        if self.pytorch_model is None:
            self._load_pytorch_model()
        return self.pytorch_model

    def _load_label_map(self) -> Dict[str, int]:
        paths = [
            os.path.join(BASE_DIR, "models", "label_map.json"),
            os.path.join(ROOT_DIR, "configs", "selected_50_classes_manifest.json"),
            os.path.join(ROOT_DIR, "configs", "authoritative_136_classes_manifest.json"),
            os.path.join(ROOT_DIR, "data", "unified_50class", "label_map.json"),
            os.path.join(ROOT_DIR, "data", "unified_136class", "label_map.json"),
            os.path.join(ROOT_DIR, "models", "label_map.json"),
        ]
        for p in paths:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        print(f"[AC-HWGATE] Loaded class names list from {p} ({len(data)} classes)")
                        label_map = {c: i for i, c in enumerate(data)}
                    elif "classes" in data and isinstance(data["classes"], list):
                        cls_list = data["classes"]
                        print(f"[AC-HWGATE] Loaded ground-truth classes from manifest {p} ({len(cls_list)} classes)")
                        label_map = {c: i for i, c in enumerate(cls_list)}
                    elif "label_to_idx" in data:
                        print(f"[AC-HWGATE] Loaded ground-truth label map from {p} ({len(data['label_to_idx'])} classes)")
                        label_map = data["label_to_idx"]
                    elif "idx_to_label" in data:
                        idx_map = {int(k): str(v) for k, v in data["idx_to_label"].items()}
                        print(f"[AC-HWGATE] Loaded ground-truth label map from {p} ({len(idx_map)} classes)")
                        label_map = {v: k for k, v in idx_map.items()}
                    elif isinstance(data, dict):
                        print(f"[AC-HWGATE] Loaded ground-truth label map from {p} ({len(data)} classes)")
                        label_map = data
                    else:
                        continue
                    self._validate_label_map(label_map, p)
                    return {str(k): int(v) for k, v in label_map.items()}
        raise FileNotFoundError("Critical Error: No ground-truth label_map.json found in backend/ or models/!")

    @staticmethod
    def _validate_label_map(label_map: Dict[str, int], source: str) -> None:
        indices = [int(v) for v in label_map.values()]
        if not label_map or len(set(indices)) != len(indices):
            raise ValueError(f"Invalid label map at {source}: indices must be unique")
        if sorted(indices) != list(range(len(indices))):
            raise ValueError(f"Invalid label map at {source}: indices must be contiguous from zero")

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
        self.ort_session = None
        self.pytorch_model = None
        self.load_error = None
        self.label_map = self._load_label_map()
        self.reverse_map = {}
        for k, v in self.label_map.items():
            idx = int(v)
            if idx not in self.reverse_map or "_" in k:
                self.reverse_map[idx] = str(k)
        self.num_classes = len(set(self.label_map.values()))

        onnx_paths = [
            os.path.join(BASE_DIR, "models", "ac_hwgate.onnx")
        ]
        production_onnx_exists = os.path.exists(onnx_paths[0])
        if not production_onnx_exists:
            self.load_error = f"Configured production ONNX artifact is missing: {onnx_paths[0]}"
            raise FileNotFoundError(self.load_error)
        for onnx_path in onnx_paths:
            if HAS_ONNX_RUNTIME and ort is not None and os.path.exists(onnx_path):
                try:
                    opts = ort.SessionOptions()
                    opts.intra_op_num_threads = 4
                    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
                    outputs = session.get_outputs()
                    out_shape = outputs[0].shape
                    if len(out_shape) != 2 or out_shape[1] != self.num_classes:
                        raise ValueError(f"logit output {out_shape} does not match {self.num_classes} labels")
                    if len(outputs) > 1 and outputs[1].shape[-1] != 128:
                        raise ValueError(f"embedding output {outputs[1].shape} is not 128-D")
                    input_shape = session.get_inputs()[0].shape
                    if len(input_shape) != 4 or input_shape[1:] != [self.window_size, 133, 6]:
                        raise ValueError(f"input shape {input_shape} does not match [batch, 32, 133, 6]")
                    self.ort_session = session
                    self.engine_type = f"AC-HWGATE ONNX Runtime ({self.num_classes} classes)"
                    self._set_loaded_artifact(onnx_path)
                    print(f"[AC-HWGATE] Loaded ONNX Runtime model from {onnx_path} ({self.num_classes} classes)")
                    return
                except Exception as e:
                    print(f"[AC-HWGATE] Failed to load ONNX model: {e}")
                    self.ort_session = None

        if production_onnx_exists:
            self.load_error = f"Configured production ONNX artifact could not be loaded: {onnx_paths[0]}"
            raise RuntimeError(self.load_error)

        raise RuntimeError(self.load_error or "Configured production ONNX artifact could not be loaded")

    def _set_loaded_artifact(self, artifact_path: str) -> None:
        digest = hashlib.sha256()
        with open(artifact_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        self.loaded_artifact_path = os.path.abspath(artifact_path)
        self.loaded_artifact_sha256 = digest.hexdigest()
        self.loaded_at_utc = datetime.now(timezone.utc).isoformat()

    def get_model_status(self) -> Dict[str, Any]:
        checkpoint_hash = None
        checkpoint_exists = os.path.exists(self.production_checkpoint_path)
        if checkpoint_exists:
            digest = hashlib.sha256()
            with open(self.production_checkpoint_path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            checkpoint_hash = digest.hexdigest()
        manifest_compatible = self.num_classes in [50, 136] and len(self.reverse_map) == self.num_classes
        return {
            "ready": bool(self.loaded_artifact_path and self.load_error is None and manifest_compatible),
            "engine": self.engine_type,
            "artifact_path": self.loaded_artifact_path,
            "artifact_sha256": self.loaded_artifact_sha256,
            "artifact_kind": "onnx" if self.ort_session is not None else "pytorch",
            "checkpoint_path": os.path.abspath(self.production_checkpoint_path),
            "checkpoint_exists": checkpoint_exists,
            "checkpoint_sha256": checkpoint_hash,
            "manifest_path": os.path.abspath(self.manifest_path),
            "manifest_compatible": manifest_compatible,
            "num_classes": self.num_classes,
            "input_shape": [1, self.window_size, 133, 6],
            "loaded_at_utc": self.loaded_at_utc,
            "load_error": self.load_error,
        }

    def _load_pytorch_model(self):
        weights_paths = [
            os.path.join(ROOT_DIR, "checkpoints", "achwgate_50class_production_best.pth"),
            os.path.join(BASE_DIR, "models", "achwgate_best.pth"),
            os.path.join(ROOT_DIR, "checkpoints", "achwgate_136class_production_best.pth"),
            os.path.join(ROOT_DIR, "checkpoints", "achwgate_best.pth"),
            os.path.join(ROOT_DIR, "checkpoints", "achwgate_104class_best.pth"),
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
                    model.load_state_dict(state_dict, strict=True)
                    model.eval()
                    self.pytorch_model = model
                    self.engine_type = f"AC-HWGATE PyTorch ({self.num_classes} classes)"
                    self._set_loaded_artifact(wpath)
                    print(f"[AC-HWGATE] Loaded PyTorch checkpoint from {wpath} ({self.num_classes} classes)")
                    return
                except Exception as e:
                    print(f"[AC-HWGATE] Failed loading PyTorch weights from {wpath}: {e}")

        raise RuntimeError("No compatible AC-HWGATE artifact could be loaded; refusing random weights")

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

    def _record_trace(
        self,
        res: PredictionResult,
        landmarks_valid: bool = True,
        landmarks_shape: Optional[List[int]] = None,
        hand_pose_status: Optional[Dict[str, Any]] = None,
        buffer_length: int = 0,
        buffer_length_before: int = 0,
        buffer_length_after: int = 0,
        buffer_ready: bool = False,
        model_input_shape: Optional[List[int]] = None,
        model_input_dtype: Optional[str] = None,
        model_input_finite: Optional[bool] = None,
        inference_called: bool = False,
        margin: float = 0.0,
        entropy: float = 0.0,
        prototype_sim: float = 0.0,
        reset_reason: Optional[str] = None,
        rejection_reason: Optional[str] = None
    ):
        """Record complete pipeline stage instrumentation for verification and regression."""
        b_before = buffer_length_before if buffer_length_before else res.buffer_length_before
        b_after = buffer_length_after if buffer_length_after else (buffer_length if buffer_length else res.buffer_length_after)
        self.event_log.append({
            "session_id": res.session_id or self.session_id,
            "seq_id": res.seq_id,
            "window_id": res.window_id or f"window_{res.seq_id}",
            "timestamp": res.timestamp,
            "frame_age": res.frame_age,
            "model_version": res.model_version,
            "landmarks_valid": bool(landmarks_valid),
            "landmarks_shape": landmarks_shape or [133, 2],
            "hand_pose_status": hand_pose_status or {},
            "buffer_length_before": int(b_before),
            "buffer_length": int(b_after),
            "buffer_length_after": int(b_after),
            "buffer_ready": bool(buffer_ready),
            "model_input_shape": model_input_shape or [1, 32, 133, 6],
            "model_input_dtype": model_input_dtype or "float32",
            "model_input_finite": bool(True if model_input_finite is None else model_input_finite),
            "inference_called": bool(inference_called),
            "raw_top_predictions": res.top_predictions,
            "confidence": res.confidence,
            "margin": round(float(margin), 4),
            "entropy": round(float(entropy), 4),
            "prototype_sim": round(float(prototype_sim), 4),
            "reset_reason": reset_reason or res.reset_reason,
            "rejection_reason": rejection_reason or res.rejection_reason,
            "decoded_class": res.label,
            "provisional_sign": res.provisional_sign,
            "confirmation_status": res.state,
            "event_id": res.event_id,
            "translation": res.translation,
            "speech_status": res.speech_action
        })

    def get_latest_event_trace(self) -> List[Dict[str, Any]]:
        """Returns the recent instrumented pipeline event trace."""
        return list(self.event_log)

    def get_live_diagnostics(self) -> Dict[str, Any]:
        """Return non-sensitive runtime facts for each browser prediction response."""
        last_event = self.event_log[-1] if self.event_log else {}
        return {
            "session_id": self.session_id,
            "window_id": last_event.get("window_id", f"win_0_{self.last_processed_seq_id}"),
            "buffer_length": len(self.raw_frame_buffer),
            "buffer_length_before": last_event.get("buffer_length_before", len(self.raw_frame_buffer)),
            "buffer_length_after": len(self.raw_frame_buffer),
            "required_frames": self.min_frames,
            "window_size": self.window_size,
            "last_processed_seq_id": self.last_processed_seq_id,
            "frame_age": last_event.get("frame_age", 0.0),
            "state": self.state,
            "inference_called": bool(last_event.get("inference_called")),
            "reset_reason": last_event.get("reset_reason"),
            "rejection_reason": last_event.get("rejection_reason")
        }

    @property
    def frame_buffer(self) -> deque:
        return self.raw_frame_buffer

    @frame_buffer.setter
    def frame_buffer(self, val: Any) -> None:
        self.raw_frame_buffer.clear()
        if val:
            self.raw_frame_buffer.extend(val)

    def reset(self):
        """Reset internal buffers and state machine counters."""
        self.raw_frame_buffer.clear()
        self.candidate_history.clear()
        self.event_log.clear()
        self.frame_counter = 0
        self.active_frame_counter = 0
        self.last_frame_timestamp = 0.0
        self.window_seq = 0
        self.state = RecognitionState.WAITING
        self.last_status = RecognitionState.WAITING
        self.last_processed_seq_id = 0
        self.cooldown_counter = 0
        self.missing_frames_counter = 0
        self.hand_present_counter = 0
        self.uncertain_eval_counter = 0
        self.stationary_eval_counter = 0
        self.provisional_sign = None
        self.provisional_confidence = 0.0
        self.top_k_predictions = []
        self.last_confirmed_sign = None
        self.last_confirmed_translation = ""
        self.last_confirmed_event_id = None
        self.current_held_sign = None
        self.bout_confirmed_sign = None
        self.last_stable_result = PredictionResult("UNKNOWN_SIGN", 0.0, True, state=RecognitionState.WAITING)

    def preprocess_frame(self, keypoints: Union[List[List[float]], np.ndarray]) -> np.ndarray:
        arr = np.array(keypoints, dtype=np.float32)
        if arr.shape[-1] > 2:
            arr = arr[..., :2]
        return arr

    def predict(
        self,
        keypoints: List[List[float]],
        confidences: Union[List[List[float]], List[float]],
        seq_id: int = 0,
        timestamp: float = 0.0,
        session_id: Optional[str] = None
    ) -> PredictionResult:
        """
        Fast Live Prediction with 32-frame Rolling Buffer, Consistent Recognition State Machine,
        Out-of-Order Sequence Discarding, and Event-Based Confirmation.
        Returns: PredictionResult (unpacks cleanly as (label, confidence, is_unknown))
        """
        self.frame_counter += 1
        current_time = float(timestamp if timestamp > 0.0 else time.time())
        frame_age = float(current_time - self.last_frame_timestamp) if (self.last_frame_timestamp > 0.0 and current_time >= self.last_frame_timestamp) else 0.0
        self.last_frame_timestamp = current_time

        # Session tracking
        if session_id and session_id != self.session_id:
            self.reset()
            self.session_id = str(session_id)

        current_window_id = f"window_{seq_id}"
        buf_len_before = len(self.raw_frame_buffer)

        # Step 1: Discard stale out-of-order sequences
        if seq_id > 0 and seq_id < self.last_processed_seq_id:
            stale_res = PredictionResult(
                "STALE_SEQUENCE", 0.0, True,
                state=self.state,
                seq_id=seq_id,
                timestamp=current_time,
                session_id=self.session_id,
                window_id=current_window_id,
                buffer_length_before=buf_len_before,
                buffer_length_after=buf_len_before,
                frame_age=frame_age,
                previous_confirmed_sign=self.last_confirmed_sign,
                previous_translation=self.last_confirmed_translation,
                is_previous_translation=bool(self.last_confirmed_translation),
                speech_action="none",
                rejection_reason="stale_out_of_order_sequence"
            )
            return stale_res

        if seq_id > 0:
            self.last_processed_seq_id = seq_id

        # Step 2: Stale frame age gap check (Legitimate Reset on stream pause / disconnection)
        if buf_len_before > 0 and frame_age > self.max_frame_age_sec:
            self.raw_frame_buffer.clear()
            self.candidate_history.clear()
            self.provisional_sign = None
            self.state = RecognitionState.WAITING
            self.last_status = RecognitionState.WAITING
            reset_msg = f"stale_frame_gap ({frame_age:.2f}s > {self.max_frame_age_sec}s)"
            res = PredictionResult(
                "UNKNOWN_SIGN", 0.0, True,
                state=RecognitionState.WAITING,
                seq_id=seq_id,
                timestamp=current_time,
                session_id=self.session_id,
                window_id=current_window_id,
                buffer_length_before=buf_len_before,
                buffer_length_after=0,
                frame_age=frame_age,
                reset_reason=reset_msg,
                rejection_reason=reset_msg,
                previous_confirmed_sign=self.last_confirmed_sign,
                previous_translation=self.last_confirmed_translation,
                is_previous_translation=bool(self.last_confirmed_translation),
                speech_action="none"
            )
            self._record_trace(res, reset_reason=reset_msg, rejection_reason=reset_msg)
            return res

        # Step 3: Validate landmarks format
        kp_arr = np.array(keypoints, dtype=np.float32)
        if kp_arr.ndim != 2 or kp_arr.shape[0] != 133 or kp_arr.shape[1] < 2 or not np.isfinite(kp_arr).all():
            res = PredictionResult(
                "UNKNOWN_SIGN", 0.0, True,
                state=RecognitionState.WAITING,
                seq_id=seq_id,
                timestamp=current_time,
                session_id=self.session_id,
                window_id=current_window_id,
                buffer_length_before=buf_len_before,
                buffer_length_after=buf_len_before,
                frame_age=frame_age,
                rejection_reason="invalid_landmarks_shape_or_values"
            )
            self._record_trace(res, landmarks_valid=False, landmarks_shape=list(kp_arr.shape) if kp_arr.ndim else [], rejection_reason=res.rejection_reason)
            return res
        if kp_arr.shape[-1] > 2:
            kp_arr = kp_arr[..., :2]

        conf_arr = np.array(confidences, dtype=np.float32).reshape(-1, 1)
        if conf_arr.shape != (133, 1) or not np.isfinite(conf_arr).all():
            res = PredictionResult(
                "UNKNOWN_SIGN", 0.0, True,
                state=RecognitionState.WAITING,
                seq_id=seq_id,
                timestamp=current_time,
                session_id=self.session_id,
                window_id=current_window_id,
                buffer_length_before=buf_len_before,
                buffer_length_after=buf_len_before,
                frame_age=frame_age,
                rejection_reason="invalid_confidence_shape_or_values"
            )
            self._record_trace(res, landmarks_valid=False, landmarks_shape=list(kp_arr.shape), rejection_reason=res.rejection_reason)
            return res

        # Gate 1: Hand-Quality & Active Signing Space Check
        lh_conf = conf_arr[33:54]
        rh_conf = conf_arr[54:75]
        lh_vis = int(np.sum(lh_conf >= self.min_hand_conf))
        rh_vis = int(np.sum(rh_conf >= self.min_hand_conf))
        lh_mean = float(np.mean(lh_conf))
        rh_mean = float(np.mean(rh_conf))

        has_hands = (lh_vis >= self.min_hand_points and lh_mean >= 0.08) or \
                    (rh_vis >= self.min_hand_points and rh_mean >= 0.08)

        sh_y = 0.45
        if np.sum(conf_arr[11:13] >= 0.25) == 2:
            sh_y = float((kp_arr[11, 1] + kp_arr[12, 1]) / 2.0)
        signing_space_boundary = max(0.95, sh_y + 0.60)

        min_lh_y = float(np.min(kp_arr[33:54, 1])) if lh_vis >= self.min_hand_points else 1.0
        min_rh_y = float(np.min(kp_arr[54:75, 1])) if rh_vis >= self.min_hand_points else 1.0

        hands_in_zone = False
        if lh_vis >= self.min_hand_points and min_lh_y < signing_space_boundary:
            hands_in_zone = True
        if rh_vis >= self.min_hand_points and min_rh_y < signing_space_boundary:
            hands_in_zone = True

        pose_detected = bool(np.sum(conf_arr[11:13] >= 0.25) >= 1)
        hand_pose_status = {
            "pose_detected": pose_detected,
            "has_hands": has_hands,
            "hands_in_zone": hands_in_zone,
            "lh_vis": lh_vis,
            "rh_vis": rh_vis,
            "sh_y": round(sh_y, 4)
        }

        # Step 4: Handle hand absence or hands resting outside signing zone
        if not has_hands or not hands_in_zone:
            self.missing_frames_counter += 1
            if self.missing_frames_counter >= 5:
                self.bout_confirmed_sign = None
            self.hand_present_counter = 0
            self.uncertain_eval_counter = 0

            # Hand-drop stroke completion: if a sign was actively being recognized before hand lowered
            provisional_valid = (
                self.provisional_sign and
                self.provisional_sign not in [
                    "UNKNOWN_SIGN", "UNCERTAIN", "DETECTING...", "WAITING FOR SIGN",
                    "IDLE"
                ]
            )
            candidate_support = sum(1 for c in self.candidate_history if c == self.provisional_sign)

            if (
                self.missing_frames_counter == 1 and
                self.bout_confirmed_sign is None and
                buf_len_before >= self.min_frames and
                provisional_valid and
                candidate_support >= 1 and
                self.candidate_history and
                self.candidate_history[-1] == self.provisional_sign and
                self.provisional_confidence >= self.min_confidence and
                self.state == RecognitionState.RECOGNIZING
            ):
                completed_class = self.provisional_sign
                is_same_held = (completed_class == self.current_held_sign)
                if not is_same_held:
                    self.event_counter += 1
                    new_event_id = f"evt_{self.event_counter}_{int(time.time() * 1000)}"
                    self.current_held_sign = completed_class
                    self.last_confirmed_sign = completed_class
                    self.bout_confirmed_sign = completed_class
                    self.last_confirmed_event_id = new_event_id
                    speech_act = "speak"
                else:
                    new_event_id = self.last_confirmed_event_id
                    speech_act = "none"

                try:
                    from translator import nmt_engine
                except ImportError:
                    from backend.translator import nmt_engine
                confirmed_translation = nmt_engine.translate_sign(completed_class)
                self.last_confirmed_translation = confirmed_translation

                self.state = RecognitionState.CONFIRMED
                self.last_status = RecognitionState.CONFIRMED
                self.cooldown_counter = self.cooldown_frames
                self.candidate_history.clear()
                self.raw_frame_buffer.clear()
                self.provisional_sign = None

                res = PredictionResult(
                    completed_class, self.provisional_confidence, False,
                    state=RecognitionState.CONFIRMED,
                    seq_id=seq_id,
                    timestamp=current_time,
                    session_id=self.session_id,
                    window_id=current_window_id,
                    buffer_length_before=buf_len_before,
                    buffer_length_after=0,
                    frame_age=frame_age,
                    event_id=new_event_id,
                    confirmed_sign=completed_class,
                    translation=confirmed_translation,
                    previous_confirmed_sign=completed_class,
                    previous_translation=confirmed_translation,
                    is_previous_translation=False,
                    speech_action=speech_act,
                    top_predictions=self.top_k_predictions,
                    reset_reason="hand_drop_stroke_completion",
                    model_version="AC-HWGATE-v2-136class-ONNX",
                    engine=self.engine_type
                )
                self.last_stable_result = res
                self._record_trace(res, hand_pose_status=hand_pose_status, reset_reason="hand_drop_stroke_completion")
                return res

            if self.missing_frames_counter >= self.max_tracking_loss_frames:
                # Legitimate reset: Signer has genuinely lowered or rested hands (>= 15 frames / 500ms)
                self.raw_frame_buffer.clear()
                self.candidate_history.clear()
                self.provisional_sign = None
                self.current_held_sign = None
                self.bout_confirmed_sign = None
                self.active_frame_counter = 0
                self.state = RecognitionState.WAITING
                self.last_status = RecognitionState.WAITING
                reset_msg = "hands_lowered_or_rested"
                res = PredictionResult(
                    "UNKNOWN_SIGN", 0.0, True,
                    state=RecognitionState.WAITING,
                    seq_id=seq_id,
                    timestamp=current_time,
                    session_id=self.session_id,
                    window_id=current_window_id,
                    buffer_length_before=buf_len_before,
                    buffer_length_after=0,
                    frame_age=frame_age,
                    reset_reason=reset_msg,
                    rejection_reason=reset_msg,
                    previous_confirmed_sign=self.last_confirmed_sign,
                    previous_translation=self.last_confirmed_translation,
                    is_previous_translation=bool(self.last_confirmed_translation),
                    speech_action="none"
                )
                self.last_stable_result = res
                self._record_trace(res, hand_pose_status=hand_pose_status, reset_reason=reset_msg, rejection_reason=reset_msg)
                return res

            # Transient missing frame (MediaPipe momentary dropout < 15 frames / 500ms):
            # PRESERVE ACCUMULATED BUFFER - DO NOT WIPE PARTIAL PROGRESS!
            res = PredictionResult(
                "UNKNOWN_SIGN", 0.0, True,
                state=self.state,
                seq_id=seq_id,
                timestamp=current_time,
                session_id=self.session_id,
                window_id=current_window_id,
                buffer_length_before=buf_len_before,
                buffer_length_after=buf_len_before,
                frame_age=frame_age,
                previous_confirmed_sign=self.last_confirmed_sign,
                previous_translation=self.last_confirmed_translation,
                is_previous_translation=bool(self.last_confirmed_translation),
                speech_action="none",
                rejection_reason="transient_hand_absence"
            )
            self._record_trace(res, hand_pose_status=hand_pose_status, buffer_length_before=buf_len_before, buffer_length_after=buf_len_before, rejection_reason="transient_hand_absence")
            return res

        # Step 5: Hands are present in active signing zone
        self.missing_frames_counter = 0
        self.hand_present_counter += 1
        self.active_frame_counter += 1

        # Post-confirmation cooldown check
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            is_valid_confirmed = bool(self.last_confirmed_sign and self.last_confirmed_sign != "UNKNOWN_SIGN")
            res = PredictionResult(
                self.last_confirmed_sign if is_valid_confirmed else "UNKNOWN_SIGN",
                float(self.last_stable_result[1]),
                not is_valid_confirmed,
                state=RecognitionState.CONFIRMED if is_valid_confirmed else RecognitionState.WAITING,
                seq_id=seq_id,
                timestamp=current_time,
                session_id=self.session_id,
                window_id=current_window_id,
                buffer_length_before=buf_len_before,
                buffer_length_after=buf_len_before,
                frame_age=frame_age,
                event_id=self.last_confirmed_event_id,
                confirmed_sign=self.last_confirmed_sign,
                translation=self.last_confirmed_translation,
                previous_confirmed_sign=self.last_confirmed_sign,
                previous_translation=self.last_confirmed_translation,
                is_previous_translation=False,
                speech_action="none",
                rejection_reason="cooldown_active"
            )
            self._record_trace(
                res,
                landmarks_valid=True,
                landmarks_shape=list(kp_arr.shape),
                hand_pose_status=hand_pose_status,
                buffer_length_before=buf_len_before,
                buffer_length_after=buf_len_before,
                buffer_ready=False,
                rejection_reason="cooldown_active"
            )
            return res

        # Append new landmark frame to temporal buffer
        self.raw_frame_buffer.append(kp_arr)
        buf_len_after = len(self.raw_frame_buffer)

        # Transition WAITING -> COLLECTING_FRAMES with debounce
        if self.state == RecognitionState.WAITING and self.hand_present_counter >= 2:
            self.state = RecognitionState.COLLECTING_FRAMES
            self.last_status = RecognitionState.COLLECTING_FRAMES
            self.candidate_history.clear()
            self.uncertain_eval_counter = 0
            self.stationary_eval_counter = 0

        # Check temporal buffer accumulation (< min_frames)
        if buf_len_after < self.min_frames:
            if self.state != RecognitionState.COLLECTING_FRAMES:
                self.state = RecognitionState.COLLECTING_FRAMES
                self.last_status = RecognitionState.COLLECTING_FRAMES
            rej_msg = f"buffering_frames ({buf_len_after}/{self.min_frames})"
            res = PredictionResult(
                "UNKNOWN_SIGN", 0.0, True,
                state=RecognitionState.COLLECTING_FRAMES,
                seq_id=seq_id,
                timestamp=current_time,
                session_id=self.session_id,
                window_id=current_window_id,
                buffer_length_before=buf_len_before,
                buffer_length_after=buf_len_after,
                frame_age=frame_age,
                provisional_sign=None,
                previous_confirmed_sign=self.last_confirmed_sign,
                previous_translation=self.last_confirmed_translation,
                is_previous_translation=bool(self.last_confirmed_translation),
                speech_action="none",
                rejection_reason=rej_msg
            )
            self.last_stable_result = res
            self._record_trace(
                res,
                landmarks_valid=True,
                landmarks_shape=list(kp_arr.shape),
                hand_pose_status=hand_pose_status,
                buffer_length_before=buf_len_before,
                buffer_length_after=buf_len_after,
                buffer_ready=False,
                rejection_reason=rej_msg
            )
            return res

        # Buffer is full (>= 16 frames)
        self.window_seq += 1
        current_window_id = f"window_{seq_id}"

        if self.state == RecognitionState.COLLECTING_FRAMES:
            self.state = RecognitionState.RECOGNIZING
            self.last_status = RecognitionState.RECOGNIZING

        # Stride check: evaluate every self.stride frames
        if (self.active_frame_counter % self.stride != 0):
            res = PredictionResult(
                self.last_stable_result[0],
                float(self.last_stable_result[1]),
                bool(self.last_stable_result[2]),
                state=self.state,
                seq_id=seq_id,
                timestamp=current_time,
                session_id=self.session_id,
                window_id=current_window_id,
                buffer_length_before=buf_len_before,
                buffer_length_after=buf_len_after,
                frame_age=frame_age,
                provisional_sign=self.provisional_sign,
                confirmed_sign=self.last_confirmed_sign if self.state == RecognitionState.CONFIRMED else None,
                translation=self.last_confirmed_translation if self.state == RecognitionState.CONFIRMED else "",
                previous_confirmed_sign=self.last_confirmed_sign,
                previous_translation=self.last_confirmed_translation,
                is_previous_translation=(self.state != RecognitionState.CONFIRMED and bool(self.last_confirmed_translation)),
                speech_action="none",
                top_predictions=self.top_k_predictions,
                rejection_reason="stride_skip"
            )
            return res

        # Gate 2: Temporal Hand Motion Dynamics Check
        recent_kps = np.stack(list(self.raw_frame_buffer), axis=0)  # [T, 133, 2]
        lh_energy = float(np.mean(np.std(recent_kps[:, 33:54, :], axis=0))) if lh_vis >= self.min_hand_points else 0.0
        rh_energy = float(np.mean(np.std(recent_kps[:, 54:75, :], axis=0))) if rh_vis >= self.min_hand_points else 0.0
        motion_energy = max(lh_energy, rh_energy)

        if motion_energy < self.min_motion_energy:
            self.stationary_eval_counter += 1
            if self.stationary_eval_counter >= 10:
                # Signer stopped moving hands for sustained period: transition cleanly to WAITING
                self.state = RecognitionState.WAITING
                self.last_status = RecognitionState.WAITING
                self.candidate_history.clear()
                self.raw_frame_buffer.clear()
                self.hand_present_counter = 0
                self.provisional_sign = None
                self.bout_confirmed_sign = None
                reset_msg = "stationary_hands_idle"
                res = PredictionResult(
                    "UNKNOWN_SIGN", 0.0, True,
                    state=RecognitionState.WAITING,
                    seq_id=seq_id,
                    timestamp=current_time,
                    session_id=self.session_id,
                    window_id=current_window_id,
                    buffer_length_before=buf_len_before,
                    buffer_length_after=0,
                    frame_age=frame_age,
                    provisional_sign=None,
                    reset_reason=reset_msg,
                    rejection_reason=reset_msg,
                    previous_confirmed_sign=self.last_confirmed_sign,
                    previous_translation=self.last_confirmed_translation,
                    is_previous_translation=bool(self.last_confirmed_translation),
                    speech_action="none"
                )
                self.last_stable_result = res
                self._record_trace(
                    res,
                    landmarks_valid=bool(np.isfinite(kp_arr).all()),
                    landmarks_shape=list(kp_arr.shape),
                    hand_pose_status=hand_pose_status,
                    buffer_length_before=buf_len_before,
                    buffer_length_after=0,
                    buffer_ready=True,
                    reset_reason=reset_msg,
                    rejection_reason=f"stationary_hand_motion (energy={motion_energy:.5f} < {self.min_motion_energy})"
                )
                return res
            else:
                rej_msg = f"stationary_hand_motion (energy={motion_energy:.5f} < {self.min_motion_energy})"
                res = PredictionResult(
                    "UNKNOWN_SIGN",
                    0.0,
                    True,
                    state=self.state,
                    seq_id=seq_id,
                    timestamp=current_time,
                    session_id=self.session_id,
                    window_id=current_window_id,
                    buffer_length_before=buf_len_before,
                    buffer_length_after=buf_len_after,
                    frame_age=frame_age,
                    provisional_sign=self.provisional_sign,
                    previous_confirmed_sign=self.last_confirmed_sign,
                    previous_translation=self.last_confirmed_translation,
                    is_previous_translation=bool(self.last_confirmed_translation),
                    speech_action="none",
                    rejection_reason=rej_msg
                )
                return res

        self.stationary_eval_counter = 0

        # Feature Extraction & Model Inference (resamples rolling buffer to 32 frames)
        t_infer_start = time.perf_counter()
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
        infer_latency_ms = (time.perf_counter() - t_infer_start) * 1000.0

        # Gate 3: Probabilities, Margin & Entropy
        cal_logits = logits / self.temperature_cal
        exp_logits = np.exp(cal_logits - np.max(cal_logits))
        probs = exp_logits / np.sum(exp_logits)

        sorted_indices = np.argsort(probs)[::-1]
        top1_idx = int(sorted_indices[0])
        top2_idx = int(sorted_indices[1]) if len(sorted_indices) > 1 else 0
        top1_prob = float(probs[top1_idx])
        top2_prob = float(probs[top2_idx])
        prob_margin = top1_prob - top2_prob
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

        # Build raw top-5 candidate predictions
        top_k_preds: List[Dict[str, Any]] = []
        for rank_i in range(min(5, len(sorted_indices))):
            c_idx = int(sorted_indices[rank_i])
            c_name = self.reverse_map.get(c_idx, f"class_{c_idx}")
            top_k_preds.append({"class": c_name, "prob": round(float(probs[c_idx]), 4)})

        raw_class = self.reverse_map.get(top1_idx, "UNKNOWN_SIGN")
        clean_class = raw_class.upper()
        self.top_k_predictions = top_k_preds

        # Gate 3 Evaluation
        gate3_passed = (top1_prob >= self.min_confidence and prob_margin >= self.min_margin and entropy <= self.max_entropy)

        # Gate 4 Evaluation: Prototype Cosine Distance Gate
        gate4_passed = True
        proto_sim = 1.0
        if self.prototypes is not None and embeddings is not None and top1_idx < len(self.prototypes):
            emb_norm = embeddings / (np.linalg.norm(embeddings) + 1e-8)
            proto = self.prototypes[top1_idx]
            proto_sim = float(np.dot(emb_norm, proto))
            if proto_sim < self.min_prototype_sim:
                gate4_passed = False

        # CRITICAL FIX: If Gates 3 or 4 fail, DO NOT LEAK provisional sign (no "GOOD" or "COW")!
        if not gate3_passed or not gate4_passed:
            self.candidate_history.append("UNKNOWN_SIGN")
            self.uncertain_eval_counter += 1
            self.provisional_sign = None  # Strictly clear provisional sign
            self.provisional_confidence = 0.0

            if not gate3_passed:
                rej_reason = f"gate3_failed (conf={top1_prob:.3f}<{self.min_confidence} or margin={prob_margin:.3f}<{self.min_margin} or ent={entropy:.2f}>{self.max_entropy})"
            else:
                rej_reason = f"gate4_failed (proto_sim={proto_sim:.3f}<{self.min_prototype_sim})"

            if self.uncertain_eval_counter >= 6:
                self.state = RecognitionState.UNCERTAIN
                self.last_status = RecognitionState.UNCERTAIN
                res = PredictionResult(
                    "UNCERTAIN", top1_prob, True,
                    state=RecognitionState.UNCERTAIN,
                    seq_id=seq_id,
                    timestamp=current_time,
                    session_id=self.session_id,
                    window_id=current_window_id,
                    buffer_length_before=buf_len_before,
                    buffer_length_after=buf_len_after,
                    frame_age=frame_age,
                    provisional_sign=None,
                    top_predictions=top_k_preds,
                    translation=self.translator.translate_sign("UNCERTAIN"),
                    previous_confirmed_sign=self.last_confirmed_sign,
                    previous_translation=self.last_confirmed_translation,
                    is_previous_translation=False,
                    speech_action="none",
                    rejection_reason=rej_reason,
                    latency_ms={"model_inference_ms": round(infer_latency_ms, 2)}
                )
                self.last_stable_result = res
                self._record_trace(
                    res,
                    landmarks_valid=bool(np.isfinite(kp_arr).all()),
                    landmarks_shape=list(kp_arr.shape),
                    hand_pose_status=hand_pose_status,
                    buffer_length_before=buf_len_before,
                    buffer_length_after=buf_len_after,
                    buffer_ready=True,
                    model_input_shape=list(seq_tensor.shape),
                    model_input_dtype=str(seq_tensor.dtype),
                    model_input_finite=bool(np.isfinite(seq_tensor).all()),
                    inference_called=True,
                    margin=prob_margin,
                    entropy=entropy,
                    prototype_sim=proto_sim,
                    rejection_reason=rej_reason
                )
                return res

            self.state = RecognitionState.RECOGNIZING
            self.last_status = RecognitionState.RECOGNIZING
            # Maintain stable provisional sign if recent history has consistent agreement, preventing 66ms flicker
            prov_to_report = None
            if len(self.candidate_history) >= 2 and self.candidate_history[-1] == self.candidate_history[-2]:
                prov_to_report = self.candidate_history[-1]
            res = PredictionResult(
                "DETECTING...", top1_prob, True,
                state=RecognitionState.RECOGNIZING,
                seq_id=seq_id,
                timestamp=current_time,
                session_id=self.session_id,
                window_id=current_window_id,
                buffer_length_before=buf_len_before,
                buffer_length_after=buf_len_after,
                frame_age=frame_age,
                provisional_sign=prov_to_report,
                top_predictions=top_k_preds,
                previous_confirmed_sign=self.last_confirmed_sign,
                previous_translation=self.last_confirmed_translation,
                is_previous_translation=bool(self.last_confirmed_translation),
                speech_action="none",
                rejection_reason=rej_reason,
                latency_ms={"model_inference_ms": round(infer_latency_ms, 2)}
            )
            self.last_stable_result = res
            self._record_trace(
                res,
                landmarks_valid=bool(np.isfinite(kp_arr).all()),
                landmarks_shape=list(kp_arr.shape),
                hand_pose_status=hand_pose_status,
                buffer_length_before=buf_len_before,
                buffer_length_after=buf_len_after,
                buffer_ready=True,
                model_input_shape=list(seq_tensor.shape),
                model_input_dtype=str(seq_tensor.dtype),
                model_input_finite=bool(np.isfinite(seq_tensor).all()),
                inference_called=True,
                margin=prob_margin,
                entropy=entropy,
                prototype_sim=proto_sim,
                rejection_reason=rej_reason
            )
            return res

        # Candidate passed Gates 3 and 4!
        self.uncertain_eval_counter = max(0, self.uncertain_eval_counter - 2)
        self.provisional_sign = clean_class
        self.provisional_confidence = top1_prob
        self.candidate_history.append(clean_class)
        recent_candidates = list(self.candidate_history)

        # Gate 5: Temporal Consistency Check
        if (
            len(recent_candidates) >= self.min_consecutive_frames and
            len(set(recent_candidates[-self.min_consecutive_frames:])) == 1 and
            recent_candidates[-1] not in ["UNKNOWN_SIGN", "IDLE", "UNCERTAIN", "DETECTING...", "WAITING FOR SIGN"]
        ):
            stable_class = recent_candidates[-1]
            if self.bout_confirmed_sign is not None:
                # Active bout already confirmed a gesture: hold the confirmed sign and prevent rapid duplicate events or flickering
                stable_class = self.bout_confirmed_sign
                is_same_held = True
                new_event_id = self.last_confirmed_event_id
                speech_act = "none"
            else:
                self.bout_confirmed_sign = stable_class
                is_same_held = (stable_class == self.current_held_sign)
                if not is_same_held:
                    self.event_counter += 1
                    new_event_id = f"evt_{self.event_counter}_{int(time.time() * 1000)}"
                    self.current_held_sign = stable_class
                    self.last_confirmed_sign = stable_class
                    self.last_confirmed_event_id = new_event_id
                    speech_act = "speak"
                else:
                    new_event_id = self.last_confirmed_event_id
                    speech_act = "none"

            try:
                from translator import nmt_engine
            except ImportError:
                from backend.translator import nmt_engine
            confirmed_translation = nmt_engine.translate_sign(stable_class)
            self.last_confirmed_translation = confirmed_translation

            self.state = RecognitionState.CONFIRMED
            self.last_status = RecognitionState.CONFIRMED
            self.cooldown_counter = self.cooldown_frames
            self.candidate_history.clear()
            # Retain recent 8 frames of transitional movement to prevent cold 16-frame stall before next sign
            recent_frames = list(self.raw_frame_buffer)[-8:] if len(self.raw_frame_buffer) >= 8 else list(self.raw_frame_buffer)
            self.raw_frame_buffer.clear()
            self.raw_frame_buffer.extend(recent_frames)
            buf_len_after = len(self.raw_frame_buffer)

            res = PredictionResult(
                stable_class, top1_prob, False,
                state=RecognitionState.CONFIRMED,
                seq_id=seq_id,
                timestamp=current_time,
                session_id=self.session_id,
                window_id=current_window_id,
                buffer_length_before=buf_len_before,
                buffer_length_after=buf_len_after,
                frame_age=frame_age,
                event_id=new_event_id,
                confirmed_sign=stable_class,
                translation=confirmed_translation,
                previous_confirmed_sign=stable_class,
                previous_translation=confirmed_translation,
                is_previous_translation=False,
                speech_action=speech_act,
                top_predictions=top_k_preds,
                reset_reason="sign_confirmed_cooldown",
                model_version="AC-HWGATE-v2-136class-ONNX",
                engine=self.engine_type,
                latency_ms={"model_inference_ms": round(infer_latency_ms, 2)}
            )
            self.last_stable_result = res
            self._record_trace(
                res,
                landmarks_valid=bool(np.isfinite(kp_arr).all()),
                landmarks_shape=list(kp_arr.shape),
                hand_pose_status=hand_pose_status,
                buffer_length_before=buf_len_before,
                buffer_length_after=0,
                buffer_ready=True,
                model_input_shape=list(seq_tensor.shape),
                model_input_dtype=str(seq_tensor.dtype),
                model_input_finite=bool(np.isfinite(seq_tensor).all()),
                inference_called=True,
                margin=prob_margin,
                entropy=entropy,
                prototype_sim=proto_sim,
                reset_reason="sign_confirmed_cooldown",
                rejection_reason=None
            )
            return res

        # Passed gates 3 & 4, awaiting consecutive stability match -> RECOGNIZING with high-confidence provisional sign
        self.state = RecognitionState.RECOGNIZING
        self.last_status = RecognitionState.RECOGNIZING
        res = PredictionResult(
            "DETECTING...", top1_prob, True,
            state=RecognitionState.RECOGNIZING,
            seq_id=seq_id,
            timestamp=current_time,
            session_id=self.session_id,
            window_id=current_window_id,
            buffer_length_before=buf_len_before,
            buffer_length_after=buf_len_after,
            frame_age=frame_age,
            provisional_sign=clean_class,
            top_predictions=top_k_preds,
            previous_confirmed_sign=self.last_confirmed_sign,
            previous_translation=self.last_confirmed_translation,
            is_previous_translation=bool(self.last_confirmed_translation),
            speech_action="none",
            rejection_reason="awaiting_temporal_stability",
            latency_ms={"model_inference_ms": round(infer_latency_ms, 2)}
        )
        self.last_stable_result = res
        self._record_trace(
            res,
            landmarks_valid=bool(np.isfinite(kp_arr).all()),
            landmarks_shape=list(kp_arr.shape),
            hand_pose_status=hand_pose_status,
            buffer_length_before=buf_len_before,
            buffer_length_after=buf_len_after,
            buffer_ready=True,
            model_input_shape=list(seq_tensor.shape),
            model_input_dtype=str(seq_tensor.dtype),
            model_input_finite=bool(np.isfinite(seq_tensor).all()),
            inference_called=True,
            margin=prob_margin,
            entropy=entropy,
            prototype_sim=proto_sim,
            rejection_reason="awaiting_temporal_stability"
        )
        return res

    def predict_sequence_features(
        self,
        seq_features: np.ndarray,
        seq_id: int = 0,
        timestamp: float = 0.0
    ) -> PredictionResult:
        """
        Runs model inference directly on pre-extracted [32, 133, 6] feature tensor.
        Applies Gate 3 (margin/entropy), Gate 4 (prototype similarity), and returns PredictionResult.
        """
        current_time = timestamp if timestamp > 0.0 else time.time()
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

        sorted_indices = np.argsort(probs)[::-1]
        top1_idx = int(sorted_indices[0])
        top2_idx = int(sorted_indices[1]) if len(sorted_indices) > 1 else 0
        top1_prob = float(probs[top1_idx])
        top2_prob = float(probs[top2_idx])
        prob_margin = top1_prob - top2_prob
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

        top_k_preds = []
        for rank_i in range(min(5, len(sorted_indices))):
            c_idx = int(sorted_indices[rank_i])
            c_name = self.reverse_map.get(c_idx, f"class_{c_idx}")
            top_k_preds.append({"class": c_name, "prob": round(float(probs[c_idx]), 4)})

        if top1_prob < self.min_confidence or prob_margin < self.min_margin or entropy > self.max_entropy:
            return PredictionResult(
                "UNKNOWN_SIGN", top1_prob, True,
                state=RecognitionState.UNCERTAIN,
                seq_id=seq_id,
                timestamp=current_time,
                top_predictions=top_k_preds
            )

        if self.prototypes is not None and embeddings is not None and top1_idx < len(self.prototypes):
            emb_norm = embeddings / (np.linalg.norm(embeddings) + 1e-8)
            proto = self.prototypes[top1_idx]
            proto_sim = float(np.dot(emb_norm, proto))
            if proto_sim < self.min_prototype_sim:
                return PredictionResult(
                    "UNKNOWN_SIGN", top1_prob, True,
                    state=RecognitionState.UNCERTAIN,
                    seq_id=seq_id,
                    timestamp=current_time,
                    top_predictions=top_k_preds
                )

        raw_class = self.reverse_map.get(top1_idx, "UNKNOWN_SIGN")
        clean_class = raw_class.upper()

        try:
            from translator import nmt_engine
        except ImportError:
            from backend.translator import nmt_engine
        confirmed_translation = nmt_engine.translate_sign(clean_class)

        return PredictionResult(
            clean_class, top1_prob, False,
            state=RecognitionState.CONFIRMED,
            seq_id=seq_id,
            timestamp=current_time,
            confirmed_sign=clean_class,
            translation=confirmed_translation,
            top_predictions=top_k_preds,
            model_version="AC-HWGATE-v2-136class-ONNX",
            engine=self.engine_type
        )


# Singleton engine instance
isl_model = ISLInferenceEngine()
