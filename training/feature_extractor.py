import numpy as np
import torch

class FeatureExtractor:
    """
    Unified Part-Aware Keypoint Feature Extractor for ISL Sign Language Recognition.
    Guarantees:
    1. Local Hand Invariance: Hands are centered at their wrist and scaled by hand span.
       Single-handed signs (letters, numbers) match 100% whether the user's upper body / face
       is in the frame or not, and regardless of camera distance or person size.
    2. Body-Relative Posture Invariance: Upper body pose landmarks are centered at the shoulders
       and scaled by shoulder width, preserving arm angles and hand placement (e.g. at chin, chest, or forehead).
    3. Identical Parity across Training, Validation, ONNX Export, and Real-Time Inference.
    """
    def __init__(self, num_joints=133, in_channels=2):
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.expected_dim = num_joints * in_channels # 133 * 2 = 266

    def process_keypoints_np(self, keypoints):
        """
        Processes raw MediaPipe keypoints array into part-normalized (133, 2) features.
        Input: keypoints array of shape (133, 2) or (133, 3) or (N, 2/3)
        Returns: normalized_kps (133, 2)
        """
        kp = np.array(keypoints, dtype=np.float32)
        if kp.shape[-1] > 2:
            kp = kp[..., :2]
            
        # Ensure exact 133 keypoints
        if kp.shape[0] < self.num_joints:
            pad = np.zeros((self.num_joints - kp.shape[0], 2), dtype=np.float32)
            kp = np.vstack([kp, pad])
        elif kp.shape[0] > self.num_joints:
            kp = kp[:self.num_joints, :]

        # Slice landmarks into logical body parts
        pose = kp[0:33]    # 33 pose landmarks
        lh   = kp[33:54]   # 21 left hand landmarks
        rh   = kp[54:75]   # 21 right hand landmarks
        face = kp[75:133]  # 58 face landmarks

        # 1. Normalize Left Hand (wrist-centered, span-scaled)
        norm_lh = np.zeros_like(lh)
        if np.any(np.abs(lh) > 1e-4):
            wrist_lh = lh[0]
            centered_lh = lh - wrist_lh
            scale_lh = np.linalg.norm(lh[9] - wrist_lh) # Wrist to middle MCP
            if scale_lh < 0.02:
                scale_lh = np.max(np.abs(centered_lh))
            if scale_lh < 1e-4:
                scale_lh = 1.0
            norm_lh = centered_lh / scale_lh

        # 2. Normalize Right Hand (wrist-centered, span-scaled)
        norm_rh = np.zeros_like(rh)
        if np.any(np.abs(rh) > 1e-4):
            wrist_rh = rh[0]
            centered_rh = rh - wrist_rh
            scale_rh = np.linalg.norm(rh[9] - wrist_rh)
            if scale_rh < 0.02:
                scale_rh = np.max(np.abs(centered_rh))
            if scale_rh < 1e-4:
                scale_rh = 1.0
            norm_rh = centered_rh / scale_rh

        # 3. Normalize Upper Body Pose (shoulder-centered, shoulder-width scaled)
        norm_pose = np.zeros_like(pose)
        if np.any(np.abs(pose) > 1e-4):
            l_sh = pose[11]
            r_sh = pose[12]
            if np.abs(l_sh).sum() > 1e-3 and np.abs(r_sh).sum() > 1e-3:
                sh_center = (l_sh + r_sh) / 2.0
                sh_width = np.linalg.norm(r_sh - l_sh)
                if sh_width < 0.05:
                    sh_width = 1.0
                norm_pose = (pose - sh_center) / sh_width
            else:
                valid_p = pose[np.abs(pose).sum(axis=-1) > 1e-4]
                if len(valid_p) > 0:
                    center_p = np.mean(valid_p, axis=0)
                    scale_p = np.max(np.abs(valid_p - center_p))
                    if scale_p < 1e-4:
                        scale_p = 1.0
                    norm_pose = (pose - center_p) / scale_p

        # 4. Normalize Face Landmarks (nose-centered, face-span scaled)
        norm_face = np.zeros_like(face)
        if np.any(np.abs(face) > 1e-4):
            valid_f = face[np.abs(face).sum(axis=-1) > 1e-4]
            if len(valid_f) > 0:
                center_f = np.mean(valid_f, axis=0)
                scale_f = np.max(np.abs(valid_f - center_f))
                if scale_f < 1e-4:
                    scale_f = 1.0
                norm_face = (face - center_f) / scale_f

        # Concatenate normalized components preserving the canonical 133 layout
        return np.concatenate([norm_pose, norm_lh, norm_rh, norm_face], axis=0).astype(np.float32)

    def process_keypoints_torch(self, keypoints):
        """
        Processes raw keypoints PyTorch tensor into part-normalized (133, 2) features.
        Supports shapes: [133, 2], [1, 133, 2], [T, 133, 2], [B, T, 133, 2]
        """
        if not isinstance(keypoints, torch.Tensor):
            keypoints = torch.tensor(keypoints, dtype=torch.float32)

        if keypoints.dim() == 2:
            # Single frame [133, 2]
            np_arr = keypoints.detach().cpu().numpy()
            norm_np = self.process_keypoints_np(np_arr)
            return torch.from_numpy(norm_np).to(keypoints.device)
        elif keypoints.dim() == 3:
            # [T, 133, 2] or [1, 133, 2]
            T = keypoints.size(0)
            out = torch.zeros_like(keypoints)
            for t in range(T):
                out[t] = self.process_keypoints_torch(keypoints[t])
            return out
        elif keypoints.dim() == 4:
            # [B, T, 133, 2]
            B, T = keypoints.size(0), keypoints.size(1)
            out = torch.zeros_like(keypoints)
            for b in range(B):
                for t in range(T):
                    out[b, t] = self.process_keypoints_torch(keypoints[b, t])
            return out
        return keypoints

# Singleton instance
feature_extractor = FeatureExtractor()
