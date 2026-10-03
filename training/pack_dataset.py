import os
import sys
import json
import time
import math
import random
import torch
import numpy as np
from collections import defaultdict

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import feature_extractor

def augment_sample(kp, conf, num_augmentations=1):
    """Generates realistic variations with scale, rotation, and micro-jitter."""
    augmented = []
    for _ in range(num_augmentations):
        k = kp.clone()
        c = conf.clone()
        
        # Jitter
        noise = torch.randn_like(k) * 0.015
        mask = (k.abs().sum(dim=-1, keepdim=True) > 0.01).float()
        k = k + noise * mask
        
        # Scaling
        scale = random.uniform(0.92, 1.08)
        k = k * scale
        
        # Subtle Rotation (-10 to +10 deg)
        angle = random.uniform(-10.0, 10.0)
        theta = math.radians(angle)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        rot = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]], dtype=k.dtype)
        active = (k.abs().sum(dim=-1) > 0.01)
        if active.any():
            center = k[active].mean(dim=0)
            k_centered = k - center
            k = (k_centered @ rot.T) + center
            
        augmented.append((k, c))
    return augmented

def collect_body_templates(data_dir):
    """Collects realistic upper-body pose & face templates from video gesture signers."""
    templates = []
    sample_words = ['hello', 'thank_you', 'good_morning', 'bear', 'cook', 'drink', 'elephant', 'jump', 'clean', 'tea']
    for word in sample_words:
        d = os.path.join(data_dir, word)
        if not os.path.exists(d):
            continue
        files = [f for f in os.listdir(d) if f.endswith('.pt')]
        for f in files[:4]:
            try:
                data = torch.load(os.path.join(d, f), weights_only=False)
                kp = data['keypoints'] if isinstance(data, dict) else data
                if kp.dim() == 3:
                    kp = kp[0]
                kp_np = kp.numpy() if isinstance(kp, torch.Tensor) else kp
                pose = kp_np[0:33, :2]
                face = kp_np[75:133, :2]
                if np.abs(pose).sum() > 5.0 and np.abs(face).sum() > 5.0:
                    templates.append((pose.copy(), face.copy()))
            except Exception:
                continue
    print(f"[+] Harvested {len(templates)} diverse body templates from real signers.")
    return templates

def pack_unified_dataset(target_samples_per_class=120):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data_unified")
    
    # Use clean_label_map.json (the 103 verified classes)
    clean_map_path = os.path.join(base_dir, "clean_label_map.json")
    if os.path.exists(clean_map_path):
        with open(clean_map_path, "r") as f:
            label_map = json.load(f)
    else:
        with open(os.path.join(base_dir, "label_map.json"), "r") as f:
            label_map = json.load(f)
            
    # Save as canonical label_map.json in training/ and backend/
    with open(os.path.join(base_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)
    backend_map_path = os.path.join(os.path.dirname(base_dir), "backend", "label_map.json")
    with open(backend_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"[+] Synchronized ground-truth label_map.json ({len(label_map)} classes) to training/ and backend/")
        
    packed_output_path = os.path.join(base_dir, "dataset_packed.pt")
    classes = sorted(label_map.keys(), key=lambda k: label_map[k])
    num_classes = len(classes)
    print(f"[+] Packing pristine dataset for {num_classes} classes from {data_dir}...")
    
    body_templates = collect_body_templates(data_dir)
    
    all_kps = []
    all_confs = []
    all_labels = []
    
    stats = defaultdict(int)
    t0 = time.time()
    
    for cls_idx, cls_name in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.exists(cls_dir):
            print(f"[!] Warning: Class directory missing: {cls_dir}")
            continue
            
        is_isolated_hand = (len(cls_name) == 1)
        all_files = os.listdir(cls_dir)
        
        if is_isolated_hand:
            target_files = [f for f in all_files if f.startswith("clean_hand_")]
            if not target_files:
                target_files = [f for f in all_files if f.endswith(".pt")]
        else:
            target_files = [f for f in all_files if f.endswith(".pt")]
            
        loaded_kps = []
        loaded_confs = []
        
        for f_idx, fname in enumerate(target_files):
            fpath = os.path.join(cls_dir, fname)
            try:
                data = torch.load(fpath, map_location="cpu", weights_only=False)
                if isinstance(data, dict):
                    kp = data['keypoints']
                    conf = data.get('confidences', torch.ones_like(kp[..., :1]))
                else:
                    kp = data
                    conf = torch.ones(kp.shape[0], kp.shape[1], 1) if kp.dim() == 3 else torch.ones(kp.shape[0], 1)
                    
                if kp.dim() == 3:
                    kp = kp[0]
                if conf.dim() == 3:
                    conf = conf[0]
                elif conf.dim() == 1:
                    conf = conf.unsqueeze(-1)
                    
                # Filter out samples with no active hands
                if kp[33:75].abs().sum() < 0.05:
                    continue
                    
                if is_isolated_hand:
                    # In isolated hand signs:
                    # 50% remain isolated hand (pose=0, face=0)
                    # 50% are augmented with realistic body templates (pose!=0, face!=0)
                    # so the model recognizes the hand whether user has webcam body in view or not!
                    if f_idx % 2 == 1 and len(body_templates) > 0:
                        tpl_pose, tpl_face = body_templates[f_idx % len(body_templates)]
                        kp_aug = kp.clone()
                        conf_aug = conf.clone()
                        kp_aug[0:33] = torch.from_numpy(tpl_pose).float()
                        conf_aug[0:33] = 1.0
                        kp_aug[75:133] = torch.from_numpy(tpl_face).float()
                        conf_aug[75:133] = 1.0
                        
                        kp_norm = feature_extractor.process_keypoints_torch(kp_aug)
                        loaded_kps.append(kp_norm)
                        loaded_confs.append(conf_aug)
                    else:
                        kp_norm = feature_extractor.process_keypoints_torch(kp)
                        loaded_kps.append(kp_norm)
                        loaded_confs.append(conf)
                else:
                    kp_norm = feature_extractor.process_keypoints_torch(kp)
                    loaded_kps.append(kp_norm)
                    loaded_confs.append(conf)
            except Exception:
                continue
                
        # Augment under-represented classes to reach target_samples_per_class
        if len(loaded_kps) > 0 and len(loaded_kps) < target_samples_per_class:
            orig_kps = list(loaded_kps)
            orig_confs = list(loaded_confs)
            needed = target_samples_per_class - len(loaded_kps)
            idx = 0
            while needed > 0:
                k, c = orig_kps[idx % len(orig_kps)], orig_confs[idx % len(orig_confs)]
                aug_list = augment_sample(k, c, num_augmentations=1)
                loaded_kps.append(aug_list[0][0])
                loaded_confs.append(aug_list[0][1])
                needed -= 1
                idx += 1
                
        # Limit max samples to target_samples_per_class for balanced training
        if len(loaded_kps) > target_samples_per_class:
            combined = list(zip(loaded_kps, loaded_confs))
            random.seed(42 + cls_idx)
            random.shuffle(combined)
            combined = combined[:target_samples_per_class]
            loaded_kps, loaded_confs = zip(*combined)
            
        for k, c in zip(loaded_kps, loaded_confs):
            all_kps.append(k.float())
            all_confs.append(c.float())
            all_labels.append(cls_idx)
            
        stats[cls_name] = len(loaded_kps)
        if (cls_idx + 1) % 25 == 0 or (cls_idx + 1) == num_classes:
            print(f"  Processed [{cls_idx+1}/{num_classes}] classes... ({len(all_labels)} total samples so far)")
            
    kps_tensor = torch.stack(all_kps)     # [N, 133, 2]
    confs_tensor = torch.stack(all_confs) # [N, 133, 1]
    labels_tensor = torch.tensor(all_labels, dtype=torch.long) # [N]
    
    print("\n" + "=" * 70)
    print(f"DATASET PACKING COMPLETED in {time.time() - t0:.1f}s")
    print(f"Total Packed Samples: {kps_tensor.shape[0]}")
    print(f"Keypoints Tensor Shape: {kps_tensor.shape}, dtype: {kps_tensor.dtype}")
    print(f"Confidences Tensor Shape: {confs_tensor.shape}, dtype: {confs_tensor.dtype}")
    print(f"Labels Tensor Shape: {labels_tensor.shape}, dtype: {labels_tensor.dtype}")
    print(f"Number of Unique Classes: {len(torch.unique(labels_tensor))}")
    print("=" * 70)
    
    # Save packed dataset
    torch.save({
        'keypoints': kps_tensor,
        'confidences': confs_tensor,
        'labels': labels_tensor,
        'classes': classes,
        'label_map': label_map,
        'num_classes': num_classes
    }, packed_output_path)
    
    print(f"[+] Saved consolidated dataset to {packed_output_path} ({os.path.getsize(packed_output_path) / (1024*1024):.2f} MB)")
    return packed_output_path

if __name__ == "__main__":
    pack_unified_dataset(target_samples_per_class=120)
