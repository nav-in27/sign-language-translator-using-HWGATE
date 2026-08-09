import os
import cv2
import mediapipe as mp
import torch
from tqdm import tqdm

def get_133_keypoints(results):
    """
    Extracts keypoints and normalizes them to a 133-joint format similar to RTMPose.
    MediaPipe: Pose(33) + LeftHand(21) + RightHand(21) + Face(468).
    We take Pose, both Hands, and the first 58 face landmarks to get 133.
    Returns: keypoints (133, 2), confidences (133, 1)
    """
    keypoints = torch.zeros(133, 2)
    confidences = torch.zeros(133, 1)
    
    idx = 0
    # 1. Pose (33)
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            keypoints[idx] = torch.tensor([lm.x, lm.y])
            confidences[idx] = torch.tensor([lm.visibility])
            idx += 1
    else:
        idx += 33
        
    # 2. Left Hand (21)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            keypoints[idx] = torch.tensor([lm.x, lm.y])
            confidences[idx] = torch.tensor([1.0]) # MediaPipe hands don't have visibility, assume 1.0 if detected
            idx += 1
    else:
        idx += 21
        
    # 3. Right Hand (21)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            keypoints[idx] = torch.tensor([lm.x, lm.y])
            confidences[idx] = torch.tensor([1.0])
            idx += 1
    else:
        idx += 21
        
    # 4. Face (Take first 58 out of 468) -> 33 + 21 + 21 + 58 = 133
    if results.face_landmarks:
        for lm in list(results.face_landmarks.landmark)[:58]:
            keypoints[idx] = torch.tensor([lm.x, lm.y])
            confidences[idx] = torch.tensor([1.0])
            idx += 1
    else:
        idx += 58
        
    return keypoints, confidences

def process_dataset(root_dir, output_dir):
    """
    Reads images from root_dir (which has subfolders for each class),
    extracts keypoints, and saves them as .pt files in output_dir.
    """
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.5)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    classes = os.listdir(root_dir)
    print(f"Processing classes in {root_dir}: {classes}")
    
    for cls in classes:
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
            
        out_cls_dir = os.path.join(output_dir, cls)
        if not os.path.exists(out_cls_dir):
            os.makedirs(out_cls_dir)
            
        images = [f for f in os.listdir(cls_dir) if f.endswith('.jpg') or f.endswith('.png')]
        
        # Limit the number of images per class for fast demonstration
        if args.max_images > 0:
            images = images[:args.max_images]
            
        for img_name in tqdm(images, desc=f"Class {cls}"):
            img_path = os.path.join(cls_dir, img_name)
            out_path = os.path.join(out_cls_dir, img_name.replace('.jpg', '.pt').replace('.png', '.pt'))
            
            # Skip if already processed
            if os.path.exists(out_path):
                continue
                
            image = cv2.imread(img_path)
            if image is None:
                continue
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            
            kp, conf = get_133_keypoints(results)
            
            # Save as a single sequence of length 1: (1, 133, 2) and (1, 133, 1)
            data = {
                'keypoints': kp.unsqueeze(0),
                'confidences': conf.unsqueeze(0),
                'label': cls
            }
            torch.save(data, out_path)
            
    holistic.close()
    print(f"Finished processing {root_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--output', type=str, required=True, help='Path to output extracted keypoints')
    parser.add_argument('--max_images', type=int, default=5, help='Max images per class to process (for quick demo)')
    args = parser.parse_args()
    
    process_dataset(args.input, args.output)
