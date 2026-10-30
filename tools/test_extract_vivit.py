import cv2
import mediapipe as mp
import numpy as np

vpath = r'C:\Users\navee\Downloads\archive\ProcessedData_vivit\beautiful\MVI_9569.MOV'
cap = cv2.VideoCapture(vpath)
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frames.append(frame)
cap.release()

print(f"Loaded {len(frames)} frames from {vpath}")

# Test 1: Standard Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=True, # Evaluate every frame independently
    model_complexity=2,
    min_detection_confidence=0.1,
    min_tracking_confidence=0.1
)

detected_rh = 0
detected_lh = 0
for i, f in enumerate(frames):
    # Try upscaling 2x
    f_up = cv2.resize(f, (448, 448))
    f_rgb = cv2.cvtColor(f_up, cv2.COLOR_BGR2RGB)
    res = holistic.process(f_rgb)
    if res.right_hand_landmarks:
        detected_rh += 1
    if res.left_hand_landmarks:
        detected_lh += 1

holistic.close()
print(f"Holistic (complexity=2, static=True, 448x448): RH={detected_rh}, LH={detected_lh} / {len(frames)}")

# Test 2: Inpainting or removing the white line skeleton
# Let's see what happens if we mask out near-white lines or blur
detected_rh_blur = 0
hands = mp.solutions.hands.Hands(static_image_mode=True, min_detection_confidence=0.05)
for i, f in enumerate(frames):
    f_up = cv2.resize(f, (448, 448))
    # Slight median blur to soften stick-figure edges
    f_blur = cv2.medianBlur(f_up, 3)
    f_rgb = cv2.cvtColor(f_blur, cv2.COLOR_BGR2RGB)
    res = hands.process(f_rgb)
    if res.multi_hand_landmarks:
        detected_rh_blur += 1
hands.close()
print(f"Hands with medianBlur: {detected_rh_blur} / {len(frames)}")
