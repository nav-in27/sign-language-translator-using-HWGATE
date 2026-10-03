import os
import cv2
import mediapipe as mp
import torch
import sys
import time

def extract_hand_classes(max_samples_per_class=80):
    input_base = r"C:\Users\navee\Downloads\archive (3)\isl_dataset"
    output_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_unified")
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.15
    )
    
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] + [chr(c) for c in range(ord('A'), ord('Z')+1)]
    print(f"[+] Starting extraction for {len(classes)} hand classes from {input_base}...")
    start_time = time.time()
    
    total_extracted = 0
    for cls in classes:
        cls_lower = cls.lower()
        src_dir = os.path.join(input_base, cls)
        dst_dir = os.path.join(output_base, cls_lower)
        os.makedirs(dst_dir, exist_ok=True)
        
        # Remove old low-quality samples that had zero hands
        for f in os.listdir(dst_dir):
            if f.startswith("clean_hand_"):
                os.remove(os.path.join(dst_dir, f))
                
        files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        extracted_count = 0
        for fname in files:
            if extracted_count >= max_samples_per_class:
                break
                
            img_path = os.path.join(src_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = hands.process(img_rgb)
            
            if not res.multi_hand_landmarks:
                continue
                
            hand_lm = res.multi_hand_landmarks[0]
            pts = torch.tensor([[lm.x, lm.y] for lm in hand_lm.landmark], dtype=torch.float32) # (21, 2)
            confs = torch.ones(21, 1, dtype=torch.float32)
            
            # Variant 1: Right Hand (indices 54..74 in 133 layout)
            kp_rh = torch.zeros(133, 2)
            conf_rh = torch.zeros(133, 1)
            kp_rh[54:75] = pts
            conf_rh[54:75] = confs
            
            out_file_rh = os.path.join(dst_dir, f"clean_hand_rh_{extracted_count}.pt")
            torch.save({
                'keypoints': kp_rh.unsqueeze(0),
                'confidences': conf_rh.unsqueeze(0),
                'label': cls_lower
            }, out_file_rh)
            
            # Variant 2: Left Hand (mirrored x-coordinate, indices 33..53)
            pts_mirrored = pts.clone()
            pts_mirrored[:, 0] = 1.0 - pts_mirrored[:, 0]
            
            kp_lh = torch.zeros(133, 2)
            conf_lh = torch.zeros(133, 1)
            kp_lh[33:54] = pts_mirrored
            conf_lh[33:54] = confs
            
            out_file_lh = os.path.join(dst_dir, f"clean_hand_lh_{extracted_count}.pt")
            torch.save({
                'keypoints': kp_lh.unsqueeze(0),
                'confidences': conf_lh.unsqueeze(0),
                'label': cls_lower
            }, out_file_lh)
            
            extracted_count += 1
            total_extracted += 2
            
        print(f"  Class '{cls_lower}': extracted {extracted_count*2} pristine samples ({extracted_count} RH + {extracted_count} LH)")
        
    hands.close()
    elapsed = time.time() - start_time
    print(f"[+] Finished extracting {total_extracted} pristine hand samples in {elapsed:.1f}s ({elapsed/60:.2f} min).")

if __name__ == "__main__":
    extract_hand_classes(max_samples_per_class=80)
