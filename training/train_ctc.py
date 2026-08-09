import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from models.cwa_hwgate import CWA_HWGATE

def normalize_keypoints(keypoints):
    # keypoints is [133, 2]
    valid_mask = (keypoints[:, 0] != 0) | (keypoints[:, 1] != 0)
    if not valid_mask.any():
        return keypoints
        
    valid_kps = keypoints[valid_mask]
    center = valid_kps.mean(dim=0)
    
    # Bone-length Normalization (Shoulder to Shoulder)
    # MediaPipe Pose: 11 (left shoulder), 12 (right shoulder)
    left_shoulder = keypoints[11]
    right_shoulder = keypoints[12]
    
    # Check if shoulders are detected
    if (left_shoulder[0] != 0 or left_shoulder[1] != 0) and (right_shoulder[0] != 0 or right_shoulder[1] != 0):
        scale = torch.norm(left_shoulder - right_shoulder)
    else:
        # Fallback to max bounding box distance
        scale = torch.max(torch.abs(valid_kps - center))
        
    if scale == 0:
        scale = 1.0
        
    normalized = torch.zeros_like(keypoints)
    normalized[valid_mask] = (valid_kps - center) / scale
    return normalized

class ISLSequenceDataset(Dataset):
    """
    Synthetic Continuous Sequence Dataset.
    Combines isolated frames into temporal sequences to train with CTC Loss.
    """
    def __init__(self, root_dir, seq_len=5):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.classes = sorted(os.listdir(root_dir))
        # Add a blank class for CTC at index 0
        if "<blank>" not in self.classes:
            self.classes.insert(0, "<blank>")
            
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for cls in self.classes:
            if cls == "<blank>": continue
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for f in os.listdir(cls_dir):
                if f.endswith('.pt'):
                    self.samples.append((os.path.join(cls_dir, f), self.class_to_idx[cls]))
                    
        print(f"Loaded {len(self.samples)} samples across {len(self.classes)} classes (including blank).")
        
    def __len__(self):
        # Arbitrary epoch length for synthetic data
        return 1000
        
    def __getitem__(self, idx):
        seq_keypoints = []
        seq_confidences = []
        seq_labels = []
        
        # Build a sequence of `seq_len` frames
        for _ in range(self.seq_len):
            sample_idx = random.randint(0, len(self.samples)-1)
            pt_path, label = self.samples[sample_idx]
            data = torch.load(pt_path)
            
            keypoints = data['keypoints'].squeeze(0) # [133, 2]
            confidences = data['confidences'].squeeze(0) # [133, 1]
            
            keypoints = normalize_keypoints(keypoints)
            
            seq_keypoints.append(keypoints)
            seq_confidences.append(confidences)
            seq_labels.append(label)
            
        keypoints_tensor = torch.stack(seq_keypoints) # [seq_len, 133, 2]
        confidences_tensor = torch.stack(seq_confidences) # [seq_len, 133, 1]
        labels_tensor = torch.tensor(seq_labels, dtype=torch.long)
        
        return keypoints_tensor, confidences_tensor, labels_tensor

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for keypoints, confidences, labels in dataloader:
        keypoints, confidences, labels = keypoints.to(device), confidences.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Output shape: [seq_len, batch_size, num_classes]
        log_probs = model(keypoints, confidences)
        
        # Prepare inputs for CTCLoss
        T, B, C = log_probs.size()
        input_lengths = torch.full((B,), T, dtype=torch.long).to(device)
        target_lengths = torch.full((B,), labels.size(1), dtype=torch.long).to(device)
        
        loss = criterion(log_probs, labels, input_lengths, target_lengths)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training with CTC Loss on device: {device}")
    
    dataset_dir = r"C:\Users\navee\.gemini\antigravity-ide\scratch\isl-translator\training\data_extracted_full"
    
    dataset = ISLSequenceDataset(dataset_dir, seq_len=5)
    num_classes = len(dataset.classes) # Includes <blank>
    print(f"Number of classes: {num_classes}")
    
    # Initialize Model
    model = CWA_HWGATE(num_classes=num_classes).to(device)
    
    # CTCLoss expects blank=0
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    epochs = 15
    for epoch in range(epochs):
        loss = train_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} | CTC Loss: {loss:.4f}")
        
    torch.save(model.state_dict(), "cwa_hwgate_ctc.pth")
    print("CTC Model saved to cwa_hwgate_ctc.pth")
    
    # Save the classes list so inference knows index 0 is blank
    import json
    with open("ctc_classes.json", "w") as f:
        json.dump(dataset.classes, f)

if __name__ == "__main__":
    main()
