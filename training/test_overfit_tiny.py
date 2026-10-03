import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.cwa_hwgate import CWA_HWGATE
from feature_extractor import feature_extractor

class TinyDataset(Dataset):
    def __init__(self, root_dir, samples_per_class=10, num_target_classes=10):
        self.samples = []
        self.labels = []
        classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')])[:num_target_classes]
        self.classes = classes
        
        for idx, cls in enumerate(classes):
            cls_dir = os.path.join(root_dir, cls)
            files = [f for f in os.listdir(cls_dir) if f.endswith('.pt')][:samples_per_class]
            for f in files:
                self.samples.append(os.path.join(cls_dir, f))
                self.labels.append(idx)
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        path = self.samples[idx]
        sample = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(sample, dict):
            kp = sample['keypoints']
            conf = sample.get('confidences', torch.ones_like(kp[..., :1]))
        else:
            kp = sample
            conf = torch.ones(kp.shape[0], kp.shape[1], 1) if kp.dim() == 3 else torch.ones(kp.shape[0], 1)
            
        # Apply deterministic feature normalization via FeatureExtractor
        kp = feature_extractor.process_keypoints_torch(kp)
        
        if kp.dim() == 2:
            kp = kp.unsqueeze(0) # [1, 133, 2]
            conf = conf.unsqueeze(0)
            
        return kp.float(), conf.float(), self.labels[idx]

def run_sanity_overfit_test():
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_unified")
    dataset = TinyDataset(dataset_dir, samples_per_class=10, num_target_classes=10)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    print(f"[+] Running Sanity Overfit Test on {len(dataset)} samples across 10 classes with FeatureExtractor...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CWA_HWGATE(num_classes=10, dropout=0.0).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    criterion = nn.NLLLoss()
    
    for epoch in range(1, 31):
        model.train()
        correct = 0
        total = 0
        total_loss = 0.0
        
        for kp, conf, target in dataloader:
            kp, conf, target = kp.to(device), conf.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(kp, conf) # [1, B, num_classes]
            logits = output[0] # [B, num_classes]
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * target.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == target).sum().item()
            total += target.size(0)
            
        acc = (correct / total) * 100.0
        avg_loss = total_loss / total
        if epoch % 5 == 0 or epoch == 1 or acc >= 98.0:
            print(f"Epoch [{epoch:02d}/30] Loss: {avg_loss:.4f} | Train Acc: {acc:.2f}%")
            
        if acc >= 98.0:
            print(f"\n[+] SANITY OVERFIT TEST PASSED! Reached {acc:.2f}% Training Accuracy on tiny subset.")
            return True
            
    if acc < 95.0:
        raise RuntimeError(f"[X] SANITY OVERFIT TEST FAILED! Model achieved only {acc:.2f}% (Required >= 95.0%). Bug detected in pipeline.")
        
    return True

if __name__ == "__main__":
    run_sanity_overfit_test()
