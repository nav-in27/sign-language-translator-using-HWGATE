"""
Live Interactive Demonstration for AC-HWGATE Indian Sign Language Translation System.
Shows real-time recognition, unknown sign rejection, sentence assembly, and TTS.
"""

import os
import sys
import time
import threading
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

# Add root directory
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.live_inference import LiveISLInferenceEngine, TRANSLATION_MAP

# Asynchronous Text-To-Speech helper
def speak_text(text: str):
    def _speak():
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass
    threading.Thread(target=_speak, daemon=True).start()

def draw_demo_ui(frame, res, fps):
    h, w, _ = frame.shape
    # Dark modern HUD overlay panel on the right
    hud_w = 340
    hud_x = w - hud_w
    
    # Semi-transparent sidebar background
    overlay = frame.copy()
    cv2.rectangle(overlay, (hud_x, 0), (w, h), (18, 18, 24), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.line(frame, (hud_x, 0), (hud_x, h), (0, 200, 255), 2)

    # Header
    cv2.putText(frame, "INDIAN SIGN LANGUAGE", (hud_x + 15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, "REAL-TIME TRANSLATOR", (hud_x + 15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
    cv2.line(frame, (hud_x + 15, 75), (w - 15, 75), (60, 60, 80), 1)

    # Model info
    cv2.putText(frame, "MODEL:", (hud_x + 15, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 170), 1)
    cv2.putText(frame, "AC-HWGATE", (hud_x + 15, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 150), 2)

    # Sign Card
    sign = res["sign"]
    conf = res["confidence"]
    status = res["status"]

    cv2.rectangle(frame, (hud_x + 15, 150), (w - 15, 245), (30, 30, 40), -1)
    cv2.rectangle(frame, (hud_x + 15, 150), (w - 15, 245), (80, 80, 100), 1)

    cv2.putText(frame, "DETECTED SIGN:", (hud_x + 25, 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 190), 1)
    
    if status == "CONFIRMED":
        sign_color = (0, 255, 0)
    elif status == "DETECTING":
        sign_color = (0, 200, 255)
    else:
        sign_color = (130, 130, 140)

    cv2.putText(frame, sign, (hud_x + 25, 215),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, sign_color, 2)

    # Actual Instantaneous Confidence (Not an animated progress bar)
    if status == "CONFIRMED":
        conf_text = f"CONFIDENCE: {conf:.1f}%"
        conf_color = (0, 255, 150)
    elif status == "DETECTING":
        conf_text = "CONFIDENCE: EVALUATING..."
        conf_color = (0, 200, 255)
    else:
        conf_text = "CONFIDENCE: --"
        conf_color = (130, 130, 140)

    cv2.putText(frame, conf_text, (hud_x + 15, 275),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, conf_color, 1)

    # Status
    cv2.putText(frame, "STATUS:", (hud_x + 15, 315),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 170), 1)
    status_color = (0, 255, 0) if status == "CONFIRMED" else ((0, 200, 255) if status == "DETECTING" else (160, 160, 180))
    cv2.putText(frame, status, (hud_x + 15, 340),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Translation
    cv2.putText(frame, "TRANSLATION:", (hud_x + 15, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 170), 1)
    translation = res["translation"]
    # Wrap translation if long
    if len(translation) > 24:
        cv2.putText(frame, translation[:24], (hud_x + 15, 405),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        cv2.putText(frame, translation[24:], (hud_x + 15, 425),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
    else:
        cv2.putText(frame, translation, (hud_x + 15, 410),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Performance & Latency
    cv2.putText(frame, f"FPS: {fps:.1f}  |  Latency: {res['latency_ms']:.1f} ms", (hud_x + 15, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 160), 1)

def main():
    print("============================================================")
    print("           AC-HWGATE LIVE DEMONSTRATION MODE                ")
    print("============================================================")

    engine = LiveISLInferenceEngine()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam device 0!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    prev_time = time.time()
    fps_smooth = 30.0

    print("[*] Live demo running. Press 'Q' to quit, 'C' to clear sentence.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame = cv2.flip(frame, 1) # Mirror webcam
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. MediaPipe Holistic
        results = holistic.process(frame_rgb)

        # 2. Draw landmarks on video side
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # 3. Engine Inference
        res = engine.process_frame(results)

        # 4. Trigger TTS if confirmed
        if res.get("speak_trigger") and res["translation"] != "--":
            speak_text(res["translation"])

        # 5. Measure FPS
        curr_time = time.time()
        fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / max(0.001, curr_time - prev_time))
        prev_time = curr_time

        # 6. Draw sleek HUD
        draw_demo_ui(frame, res, fps_smooth)

        cv2.imshow("AC-HWGATE ISL Translator", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('c') or key == ord('C'):
            engine.reset()

    cap.release()
    cv2.destroyAllWindows()
    holistic.close()

if __name__ == "__main__":
    main()
