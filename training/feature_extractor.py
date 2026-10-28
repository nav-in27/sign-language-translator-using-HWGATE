"""
Authoritative Shared Feature Extractor for AC-HWGATE ISL Recognition.
Provides bit-exact parity across Training, Validation, ONNX export, Backend, and Live Webcam.

133 MediaPipe Holistic landmarks:
  - 0..32: Pose (33)
  - 33..53: Left Hand (21)
  - 54..74: Right Hand (21)
  - 75..132: Face Selected (58)

Motion Dynamics per landmark:
  - Position (x, y): 2 channels
  - Velocity (dx, dy): 2 channels
  - Acceleration (ddx, ddy): 2 channels
  Total = 6 channels per joint * 133 joints = 798 features per frame.
  Temporal Sequence: T = 32 frames -> Tensor [B, 32, 133, 6] or [B, 32, 798].
"""

import numpy as np
import torch
import cv2
from typing import Union, List, Tuple, Optional, Any

class FeatureExtractor:
    def __init__(self, num_joints: int = 133, sequence_length: int = 32, include_motion: bool = True):
        self.num_joints = num_joints
        self.sequence_length = sequence_length
        self.include_motion = include_motion
        self.channels_per_joint = 6 if include_motion else 2
        self.total_feature_dim = self.num_joints * self.channels_per_joint # 798

    def extract_landmarks_from_holistic(self, results) -> np.ndarray:
        """
        Extracts 133 2D coordinates (x, y) from MediaPipe Holistic results.
        Returns: np.ndarray of shape (133, 2), dtype float32
        """
        kps = np.zeros((self.num_joints, 2), dtype=np.float32)
        if isinstance(results, np.ndarray):
            if results.shape[0] >= self.num_joints and results.shape[-1] >= 2:
                return results[:self.num_joints, :2].astype(np.float32)
            return kps

        # 1. Pose (33 joints)
        idx = 0
        if results is not None and getattr(results, "pose_landmarks", None):
            for lm in results.pose_landmarks.landmark:
                if idx < 33:
                    kps[idx] = [lm.x, lm.y]
                    idx += 1
        else:
            idx = 33

        # 2. Left Hand (21 joints)
        idx = 33
        if results and results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                if idx < 54:
                    kps[idx] = [lm.x, lm.y]
                    idx += 1
        else:
            idx = 54

        # 3. Right Hand (21 joints)
        idx = 54
        if results and results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                if idx < 75:
                    kps[idx] = [lm.x, lm.y]
                    idx += 1
        else:
            idx = 75

        # 4. Face (first 58 landmarks)
        idx = 75
        if results and results.face_landmarks:
            for lm in list(results.face_landmarks.landmark)[:58]:
                if idx < 133:
                    kps[idx] = [lm.x, lm.y]
                    idx += 1

        return kps

    def normalize_single_frame(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Hierarchical part-aware coordinate normalization:
        - Pose: normalized relative to shoulder midpoint & shoulder width.
        - Left & Right Hand: normalized relative to wrist & scaled by palm size,
          preserving global wrist displacement relative to torso center.
        - Missing hands remain exactly zeros.
        Input: (133, 2) or (133, 3)
        Output: (133, 2) normalized
        """
        kp = np.array(keypoints, dtype=np.float32)
        if kp.shape[-1] > 2:
            kp = kp[..., :2]

        if kp.shape[0] < self.num_joints:
            pad = np.zeros((self.num_joints - kp.shape[0], 2), dtype=np.float32)
            kp = np.vstack([kp, pad])
        elif kp.shape[0] > self.num_joints:
            kp = kp[:self.num_joints, :]

        pose = kp[0:33]    # Pose (33)
        lh   = kp[33:54]   # Left Hand (21)
        rh   = kp[54:75]   # Right Hand (21)
        face = kp[75:133]  # Face (58)

        # Torso reference anchor: Midpoint between shoulders (joints 11 & 12)
        l_sh = pose[11]
        r_sh = pose[12]
        if np.any(np.abs(l_sh) > 1e-4) and np.any(np.abs(r_sh) > 1e-4):
            torso_center = (l_sh + r_sh) * 0.5
            torso_scale = float(np.linalg.norm(l_sh - r_sh))
            if torso_scale < 0.05:
                torso_scale = 0.5
        else:
            torso_center = np.array([0.5, 0.5], dtype=np.float32)
            torso_scale = 0.5

        # 1. Normalize Pose relative to torso anchor
        norm_pose = np.zeros_like(pose)
        valid_pose_mask = np.any(np.abs(pose) > 1e-4, axis=-1)
        if np.any(valid_pose_mask):
            norm_pose[valid_pose_mask] = (pose[valid_pose_mask] - torso_center) / torso_scale

        # 2. Normalize Left Hand (local shape + torso-anchored wrist)
        norm_lh = np.zeros_like(lh)
        if np.any(np.abs(lh) > 1e-4):
            wrist_lh = lh[0]
            centered_lh = lh - wrist_lh
            # Scale by palm span: wrist to middle finger MCP (joint 9)
            palm_span = float(np.linalg.norm(lh[9] - wrist_lh))
            if palm_span < 0.02:
                palm_span = float(np.max(np.abs(centered_lh)))
            if palm_span < 1e-4:
                palm_span = 1.0
            norm_lh = centered_lh / palm_span
            # Store wrist position relative to torso center so sign height is preserved
            norm_lh[0] = (wrist_lh - torso_center) / torso_scale

        # 3. Normalize Right Hand (local shape + torso-anchored wrist)
        norm_rh = np.zeros_like(rh)
        if np.any(np.abs(rh) > 1e-4):
            wrist_rh = rh[0]
            centered_rh = rh - wrist_rh
            palm_span = float(np.linalg.norm(rh[9] - wrist_rh))
            if palm_span < 0.02:
                palm_span = float(np.max(np.abs(centered_rh)))
            if palm_span < 1e-4:
                palm_span = 1.0
            norm_rh = centered_rh / palm_span
            norm_rh[0] = (wrist_rh - torso_center) / torso_scale

        # 4. Normalize Face relative to nose bridge
        norm_face = np.zeros_like(face)
        if np.any(np.abs(face) > 1e-4):
            nose = face[1] if np.any(np.abs(face[1]) > 1e-4) else torso_center
            norm_face = (face - nose) / (torso_scale * 0.5)

        return np.concatenate([norm_pose, norm_lh, norm_rh, norm_face], axis=0).astype(np.float32)

    def compute_motion_features(self, positions: np.ndarray) -> np.ndarray:
        """
        Computes velocity and acceleration for a temporal sequence.
        Input: positions [T, 133, 2]
        Output: [T, 133, 6] (Position, Velocity, Acceleration)
        """
        T, J, C = positions.shape
        # Velocity V_t = P_t - P_{t-1}
        velocities = np.zeros_like(positions)
        velocities[1:] = positions[1:] - positions[:-1]

        # Acceleration A_t = V_t - V_{t-1}
        accelerations = np.zeros_like(velocities)
        accelerations[1:] = velocities[1:] - velocities[:-1]

        return np.concatenate([positions, velocities, accelerations], axis=-1).astype(np.float32)

    def process_sequence_np(self, sequence_frames: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Authoritative processing:
        Input: sequence of frames [T_in, 133, 2/3]
        Output: [sequence_length, 133, 6]
        """
        raw_seq = np.array(sequence_frames, dtype=np.float32)
        T_in = len(raw_seq)

        if T_in == 0:
            return np.zeros((self.sequence_length, self.num_joints, self.channels_per_joint), dtype=np.float32)

        # 1. Normalize each frame spatially
        norm_frames = [self.normalize_single_frame(f) for f in raw_seq]
        norm_arr = np.stack(norm_frames, axis=0) # [T_in, 133, 2]

        # 2. Resample or interpolate temporally to sequence_length (32)
        if T_in != self.sequence_length:
            indices = np.linspace(0, T_in - 1, self.sequence_length)
            floor_idx = np.floor(indices).astype(int)
            ceil_idx = np.ceil(indices).astype(int)
            alpha = (indices - floor_idx)[:, None, None]
            
            resampled = (1.0 - alpha) * norm_arr[floor_idx] + alpha * norm_arr[ceil_idx]
            norm_arr = resampled.astype(np.float32)

        # 3. Add Motion Dynamics (P, V, A)
        if self.include_motion:
            return self.compute_motion_features(norm_arr)
        return norm_arr

    def process_sequence_torch(self, sequence_tensor: torch.Tensor) -> torch.Tensor:
        """PyTorch wrapper for batch processing."""
        device = sequence_tensor.device
        if sequence_tensor.dim() == 3: # [T, 133, C]
            np_data = sequence_tensor.detach().cpu().numpy()
            processed = self.process_sequence_np(np_data)
            return torch.from_numpy(processed).to(device)
        elif sequence_tensor.dim() == 4: # [B, T, 133, C]
            B = sequence_tensor.size(0)
            np_data = sequence_tensor.detach().cpu().numpy()
            batch_out = [self.process_sequence_np(np_data[b]) for b in range(B)]
            return torch.from_numpy(np.stack(batch_out, axis=0)).to(device)
        return sequence_tensor

    def process_keypoints_torch(self, keypoint_tensor: torch.Tensor) -> torch.Tensor:
        """Processes keypoints torch tensor (single frame [133, 2] or sequence)."""
        device = keypoint_tensor.device
        if keypoint_tensor.dim() == 2:
            np_data = keypoint_tensor.detach().cpu().numpy()
            norm = self.normalize_single_frame(np_data)
            return torch.from_numpy(norm).to(device)
        return self.process_sequence_torch(keypoint_tensor)

    def extract_from_video(self, video_path: str, holistic_instance=None) -> np.ndarray:
        """
        Extracts temporal landmark sequence directly from a video file.
        Returns: [32, 133, 6] tensor ready for AC-HWGATE.
        """
        import mediapipe as mp
        close_holistic = False
        if holistic_instance is None:
            mp_solutions: Any = mp.solutions
            mp_holistic = getattr(mp_solutions, "holistic", None)
            if mp_holistic is None:
                raise ImportError("MediaPipe holistic solution unavailable")
            holistic_instance = mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            close_holistic = True

        cap = cv2.VideoCapture(video_path)
        raw_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic_instance.process(frame_rgb)
            kps = self.extract_landmarks_from_holistic(results)
            raw_frames.append(kps)
        cap.release()

        if close_holistic:
            holistic_instance.close()

        return self.process_sequence_np(raw_frames)

# Shared singleton instance
feature_extractor = FeatureExtractor(num_joints=133, sequence_length=32, include_motion=True)
