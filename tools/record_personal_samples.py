"""
Personal Webcam Recording Tool for AC-HWGATE Indian Sign Language.
Allows recording high-quality temporal sign sequences (T=32 frames, 133 joints, 6 channels)
directly from the user's webcam for any of the 40 vocabulary classes.

Prioritized classes for live demonstration:
  NO, YES, HELLO, THANK YOU, SORRY, PLEASE, HELP, STOP, WATER, EAT,
  DRINK, GO, COME, WHAT, WHERE, TODAY, FRIEND, SCHOOL, TEACHER, HOME.
"""

import os
import sys
import time
import json
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Any

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from training.feature_extractor import FeatureExtractor, feature_extractor

PRIORITY_SIGNS = [
    "no", "yes", "hello", "thank_you", "sorry", "please", "help", "stop",
    "water", "eat", "drink", "go", "come", "what", "where", "today",
    "goodbye", "okay", "friend", "school", "teacher", "home"
]

ALL_40_CLASSES = [
    "brother", "come", "drink", "eat", "father", "food", "friend", "go",
    "goodbye", "he", "hello", "help", "home", "hospital", "market", "me",
    "mother", "no", "okay", "please", "read", "school", "she", "sister",
    "sit", "sorry", "stand", "stop", "student", "tea", "teacher", "thank_you",
    "today", "water", "what", "when", "where", "write", "yes", "you"
]

PERSONAL_DATA_DIR = BASE_DIR / "data" / "personal"

def draw_hud(frame, current_sign, sample_count, state, progress=0.0):
    h, w, _ = frame.shape
    # Draw dark header bar
    cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 20), -1)
    cv2.line(frame, (0, 80), (w, 80), (0, 200, 255), 2)

    # Title
    cv2.putText(frame, "AC-HWGATE PERSONAL SAMPLE RECORDER", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Sign info
    text = f"TARGET SIGN: [{current_sign.upper()}]  |  Recorded: {sample_count}"
    cv2.putText(frame, text, (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    # State indicator
    if state == "RECORDING":
        color = (0, 0, 255)
        state_text = f"RECORDING... ({int(progress * 100)}%)"
        # Draw red recording border
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 4)
    elif state == "COUNTDOWN":
        color = (0, 165, 255)
        state_text = f"GET READY: {int(progress)}"
    else:
        color = (0, 255, 0)
        state_text = "READY  [SPACE = Record | N = Next | P = Prev | Q = Quit]"

    # Footer
    cv2.rectangle(frame, (0, h - 45), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, state_text, (20, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

def record_samples():
    PERSONAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Order priority signs first, followed by others
    sign_list = PRIORITY_SIGNS + [s for s in ALL_40_CLASSES if s not in PRIORITY_SIGNS]
    sign_idx = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam (device 0)!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mp_solutions: Any = mp.solutions
    mp_holistic = mp_solutions.holistic
    mp_drawing = mp_solutions.drawing_utils
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    state = "READY"
    state_start_time = 0.0
    recorded_frames = []
    target_frames = 36 # collect ~36 frames, then resample to 32

    print("============================================================")
    print("      AC-HWGATE PERSONAL WEBCAM SAMPLE RECORDER             ")
    print("============================================================")
    print("Controls:")
    print("  SPACE : Start 3-second countdown and record 1 sign sample")
    print("  N     : Next sign in vocabulary")
    print("  P     : Previous sign in vocabulary")
    print("  Q     : Quit and save")
    print("============================================================")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame = cv2.flip(frame, 1) # Mirror for intuitive user experience
        display_frame = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        current_sign = sign_list[sign_idx]
        sign_dir = PERSONAL_DATA_DIR / current_sign.replace(' ', '_')
        sign_dir.mkdir(parents=True, exist_ok=True)
        existing_samples = len(list(sign_dir.glob("sample_*.npz")))

        results = holistic.process(frame_rgb)

        # Draw skeleton landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(display_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(display_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(display_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        curr_time = time.time()

        if state == "COUNTDOWN":
            elapsed = curr_time - state_start_time
            remaining = 3.0 - elapsed
            if remaining <= 0:
                state = "RECORDING"
                state_start_time = curr_time
                recorded_frames = []
            draw_hud(display_frame, current_sign, existing_samples, state, progress=max(1, int(remaining + 1)))

        elif state == "RECORDING":
            # Extract landmarks for this frame
            kps = feature_extractor.extract_landmarks_from_holistic(results)
            recorded_frames.append(kps)
            prog = min(1.0, len(recorded_frames) / target_frames)
            draw_hud(display_frame, current_sign, existing_samples, state, progress=prog)

            if len(recorded_frames) >= target_frames:
                # Sequence complete: Process with FeatureExtractor to [32, 133, 6]
                processed_seq = feature_extractor.process_sequence_np(recorded_frames)
                out_filename = f"sample_{existing_samples + 1:03d}.npz"
                out_path = sign_dir / out_filename
                np.savez_compressed(
                    out_path,
                    features=processed_seq.astype(np.float32),
                    word=current_sign,
                    timestamp=time.time()
                )
                print(f"[+] Saved {out_path} ({processed_seq.shape})")
                state = "READY"

        else: # READY
            draw_hud(display_frame, current_sign, existing_samples, state)

        cv2.imshow("AC-HWGATE Personal Recorder", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord(' ') and state == "READY":
            state = "COUNTDOWN"
            state_start_time = time.time()
        elif (key == ord('n') or key == ord('N')) and state == "READY":
            sign_idx = (sign_idx + 1) % len(sign_list)
        elif (key == ord('p') or key == ord('P')) and state == "READY":
            sign_idx = (sign_idx - 1) % len(sign_list)

    cap.release()
    cv2.destroyAllWindows()
    holistic.close()
    print("[*] Recording session finished.")

if __name__ == "__main__":
    record_samples()
