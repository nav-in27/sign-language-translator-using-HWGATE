import os
import cv2
import mediapipe as mp
import torch
from tqdm import tqdm

def get_133_keypoints(results):
    keypoints = torch.zeros(133, 2)
    confidences = torch.zeros(133, 1)
    idx = 0
    
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            keypoints[idx] = torch.tensor([lm.x, lm.y])
            confidences[idx] = torch.tensor([lm.visibility])
            idx += 1
    else: idx += 33
        
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            keypoints[idx] = torch.tensor([lm.x, lm.y])
            confidences[idx] = torch.tensor([1.0])
            idx += 1
    else: idx += 21
        
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            keypoints[idx] = torch.tensor([lm.x, lm.y])
            confidences[idx] = torch.tensor([1.0])
            idx += 1
    else: idx += 21
        
    if results.face_landmarks:
        for lm in list(results.face_landmarks.landmark)[:58]:
            keypoints[idx] = torch.tensor([lm.x, lm.y])
            confidences[idx] = torch.tensor([1.0])
            idx += 1
    else: idx += 58
        
    return keypoints, confidences

def get_class_folders(base_path):
    # Walk through base_path, look for folders that contain images directly
    class_folders = {}
    for root, dirs, files in os.walk(base_path):
        image_files = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]
        if len(image_files) > 0:
            cls_name = os.path.basename(root)
            if cls_name not in class_folders:
                class_folders[cls_name] = []
            class_folders[cls_name].append((root, image_files))
    return class_folders

def main():
    archives = [
        r"C:\Users\navee\Downloads\archive (6)",
        r"C:\Users\navee\Downloads\archive (7)",
        r"C:\Users\navee\Downloads\archive (8)"
    ]
    
    output_dir = r"C:\Users\navee\.gemini\antigravity-ide\scratch\isl-translator\training\data_extracted_full"
    os.makedirs(output_dir, exist_ok=True)
    
    from typing import Any
    mp_solutions: Any = mp.solutions
    mp_holistic = mp_solutions.holistic
    holistic = mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.5)
    
    total_processed = 0
    for archive in archives:
        if not os.path.exists(archive):
            print(f"Skipping {archive}, not found.")
            continue
            
        print(f"Scanning {archive}...")
        class_folders = get_class_folders(archive)
        
        for cls_name, paths in class_folders.items():
            cls_name_clean = cls_name.lower().replace(" ", "_")
            out_cls_dir = os.path.join(output_dir, cls_name_clean)
            os.makedirs(out_cls_dir, exist_ok=True)
            
            for folder_path, image_files in paths:
                # Cap at 50 images per class per folder to keep it fast but effective (reduced from 250)
                image_files = image_files[:50]
                
                for img_name in tqdm(image_files, desc=f"Class {cls_name_clean} in {os.path.basename(archive)}"):
                    out_path = os.path.join(out_cls_dir, f"{os.path.basename(folder_path)}_{img_name}.pt")
                    if os.path.exists(out_path):
                        continue
                        
                    img_path = os.path.join(folder_path, img_name)
                    image = cv2.imread(img_path)
                    if image is None: continue
                        
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image_rgb)
                    
                    kp, conf = get_133_keypoints(results)
                    
                    data = {
                        'keypoints': kp.unsqueeze(0),
                        'confidences': conf.unsqueeze(0),
                        'label': cls_name_clean
                    }
                    torch.save(data, out_path)
                    total_processed += 1
                    
    holistic.close()
    print(f"Extraction complete! Total samples: {total_processed}")

if __name__ == "__main__":
    main()
