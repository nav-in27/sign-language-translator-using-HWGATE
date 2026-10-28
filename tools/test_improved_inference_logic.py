import os
import sys
import json
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from backend.inference import isl_model
from training.feature_extractor import feature_extractor
from backend.translator import nmt_engine

class CalibratedTestEngine:
    def __init__(self):
        self.window_size = 60
        self.min_frames = 20
        self.stride = 2
        self.min_confidence = 0.38
        self.min_margin = 0.04
        self.max_entropy = 4.50
        self.min_prototype_sim = 0.45
        self.min_motion_energy = 0.006
        self.min_consecutive_frames = 3
        self.cooldown_frames = 8
        self.cooldown_counter = 0
        
        self.raw_frame_buffer = deque(maxlen=self.window_size)
        self.candidate_history = deque(maxlen=6)
        self.last_confirmed_sign = None
        self.frame_counter = 0
        self.last_status = "WAITING"
        
        with open('configs/authoritative_136_classes_manifest.json') as mf:
            self.classes = json.load(mf)['classes']
            
    def predict_frame(self, kps, confs):
        self.frame_counter += 1
        kp_arr = np.array(kps, dtype=np.float32)
        if kp_arr.shape[-1] > 2:
            kp_arr = kp_arr[..., :2]
        conf_arr = np.array(confs, dtype=np.float32).reshape(-1, 1)
        
        # 1. Hand quality & elevation check
        lh_conf = conf_arr[33:54]
        rh_conf = conf_arr[54:75]
        lh_vis = int(np.sum(lh_conf >= 0.10))
        rh_vis = int(np.sum(rh_conf >= 0.10))
        has_hands = (lh_vis >= 3 or rh_vis >= 3)
        
        # Check shoulder reference
        sh_y = 0.45
        if np.sum(conf_arr[11:13] >= 0.3) == 2:
            sh_y = float((kp_arr[11, 1] + kp_arr[12, 1]) / 2.0)
        signing_space_boundary = sh_y + 0.32
        
        lw_y = kp_arr[33, 1]
        rw_y = kp_arr[54, 1]
        hands_in_zone = (lh_vis >= 3 and lw_y < signing_space_boundary) or \
                        (rh_vis >= 3 and rw_y < signing_space_boundary)
                        
        if not has_hands or not hands_in_zone:
            # Hands are resting or missing
            if len(self.raw_frame_buffer) > 0:
                self.raw_frame_buffer.popleft()
            self.last_status = "WAITING"
            return "WAITING FOR SIGN", 0.0, True
            
        self.raw_frame_buffer.append(kp_arr)
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return self.last_confirmed_sign or "DETECTING...", 0.0, False
            
        if len(self.raw_frame_buffer) < self.min_frames:
            self.last_status = "DETECTING"
            return "DETECTING...", 0.0, True
            
        if self.frame_counter % self.stride != 0:
            return self.last_status, 0.0, True
            
        # 2. Motion Energy Gate
        buf_kps = np.stack(list(self.raw_frame_buffer), axis=0)
        hand_pts = buf_kps[:, 33:75, :]
        motion_energy = float(np.mean(np.std(hand_pts, axis=0)))
        if motion_energy < self.min_motion_energy:
            self.last_status = "WAITING"
            return "WAITING FOR SIGN", 0.0, True
            
        # 3. Model Inference
        feat = feature_extractor.process_sequence_np(list(self.raw_frame_buffer))
        input_name = isl_model.ort_session.get_inputs()[0].name
        outs = isl_model.ort_session.run(None, {input_name: feat[None, ...].astype(np.float32)})
        logits = outs[0][0]
        embeddings = outs[1][0] if len(outs) > 1 else None
        
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / np.sum(exp_l)
        
        sorted_probs = np.sort(probs)[::-1]
        top1_prob = float(sorted_probs[0])
        top2_prob = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
        prob_margin = top1_prob - top2_prob
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        predicted_idx = int(np.argmax(probs))
        
        # 4. Gating
        if top1_prob < self.min_confidence or prob_margin < self.min_margin or entropy > self.max_entropy:
            self.last_status = "UNCERTAIN" if motion_energy > 0.015 else "DETECTING"
            return "UNCERTAIN" if self.last_status == "UNCERTAIN" else "DETECTING...", top1_prob, True
            
        if isl_model.prototypes is not None and embeddings is not None:
            emb_norm = embeddings / (np.linalg.norm(embeddings) + 1e-8)
            proto_sim = float(np.dot(emb_norm, isl_model.prototypes[predicted_idx]))
            if proto_sim < self.min_prototype_sim:
                self.last_status = "UNCERTAIN" if motion_energy > 0.015 else "DETECTING"
                return "UNCERTAIN" if self.last_status == "UNCERTAIN" else "DETECTING...", top1_prob, True
                
        pred_label = self.classes[predicted_idx].upper()
        self.candidate_history.append(pred_label)
        
        # 5. Stability Gate (3 consecutive matches)
        recent = list(self.candidate_history)
        if len(recent) >= self.min_consecutive_frames and len(set(recent[-self.min_consecutive_frames:])) == 1:
            stable_sign = recent[-1]
            self.last_confirmed_sign = stable_sign
            self.last_status = "CONFIRMED"
            self.cooldown_counter = self.cooldown_frames
            self.candidate_history.clear()
            while len(self.raw_frame_buffer) > 10:
                self.raw_frame_buffer.popleft()
            return stable_sign, top1_prob, False
            
        self.last_status = "DETECTING"
        return "DETECTING...", top1_prob, True

def test_engine_on_video(video_path, gt_label):
    print(f"\n--- Testing on {os.path.basename(video_path)} (GT: {gt_label}) ---")
    cap = cv2.VideoCapture(video_path)
    mp_h = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1)
    engine = CalibratedTestEngine()
    
    confirmed = []
    f_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        f_idx += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_h.process(rgb)
        
        kps, confs = [], []
        # Pose
        if res.pose_landmarks:
            for lm in res.pose_landmarks.landmark[:33]:
                kps.append([lm.x, lm.y]); confs.append([getattr(lm, "visibility", 1.0)])
        else:
            for _ in range(33): kps.append([0, 0]); confs.append([0])
        # LH
        if res.left_hand_landmarks:
            for lm in res.left_hand_landmarks.landmark[:21]:
                kps.append([lm.x, lm.y]); confs.append([1.0])
        else:
            for _ in range(21): kps.append([0, 0]); confs.append([0])
        # RH
        if res.right_hand_landmarks:
            for lm in res.right_hand_landmarks.landmark[:21]:
                kps.append([lm.x, lm.y]); confs.append([1.0])
        else:
            for _ in range(21): kps.append([0, 0]); confs.append([0])
        # Face
        if res.face_landmarks:
            for lm in res.face_landmarks.landmark[:58]:
                kps.append([lm.x, lm.y]); confs.append([1.0])
        else:
            for _ in range(58): kps.append([0, 0]); confs.append([0])
            
        label, conf, is_unk = engine.predict_frame(kps, confs)
        if engine.last_status == "CONFIRMED" and not is_unk:
            confirmed.append((f_idx, label, conf))
            print(f"  Frame {f_idx:03d}: [CONFIRMED] '{label}' (Conf: {conf:.4f})")
            
    cap.release()
    mp_h.close()
    print(f"Total confirmed events: {len(confirmed)}")
    for ev in confirmed:
        print(f"  Frame {ev[0]}: Sign='{ev[1]}' (Conf: {ev[2]:.4f}) -> {'CORRECT' if ev[1].lower() == gt_label.lower() else 'WRONG'}")

if __name__ == "__main__":
    v1 = r"C:\Users\navee\Downloads\archive (2)\Video_Dataset\Video_Dataset\Thank you\WIN_20231103_13_12_51_Pro.mp4"
    v2 = r"C:\Users\navee\Downloads\archive (2)\Video_Dataset\Video_Dataset\Good Morning\WIN_20231103_13_17_04_Pro.mp4"
    test_engine_on_video(v1, "thank_you")
    test_engine_on_video(v2, "good_morning")
