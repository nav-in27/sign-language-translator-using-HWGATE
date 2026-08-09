import os
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
    
    # Scale by maximum distance from center
    scale = torch.max(torch.abs(valid_kps - center))
    if scale == 0:
        scale = 1.0
        
    normalized = torch.zeros_like(keypoints)
    normalized[valid_mask] = (valid_kps - center) / scale
    return normalized

class ISLAlphabetDataset(Dataset):
    """
    Dataset for ISL Alphabets and Numbers.
    Loads pre-extracted 133-keypoint tensors from .pt files.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for f in os.listdir(cls_dir):
                if f.endswith('.pt'):
                    self.samples.append((os.path.join(cls_dir, f), self.class_to_idx[cls]))
                    
        print(f"Found {len(self.samples)} samples across {len(self.classes)} classes.")
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        pt_path, label = self.samples[idx]
        data = torch.load(pt_path)
        
        # data['keypoints'] is [1, 133, 2] (Seq=1, Joints=133, Coord=2)
        keypoints = data['keypoints'].squeeze(0) # [133, 2]
        confidences = data['confidences'].squeeze(0) # [133, 1]
        
        # Apply Spatial Normalization
        keypoints = normalize_keypoints(keypoints)
        
        # We need seq_len=1 for the model, so we wrap them back
        # The model expects [seq_len, num_joints, 2]
        keypoints = keypoints.unsqueeze(0) # [1, 133, 2]
        confidences = confidences.unsqueeze(0) # [1, 133, 1]
        
        return keypoints, confidences, torch.tensor(label, dtype=torch.long)

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for keypoints, confidences, labels in dataloader:
        keypoints, confidences, labels = keypoints.to(device), confidences.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Output shape: [seq_len, batch_size, num_classes]
        # For sequence of length 1, output is [1, batch_size, num_classes]
        log_probs = model(keypoints, confidences)
        
        # We only care about the single frame's prediction
        logits = log_probs[0] # [batch_size, num_classes]
        
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    acc = correct / total if total > 0 else 0
    return total_loss / len(dataloader), acc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training CWA-HWGATE on device: {device}")
    
    # Path to the extracted dataset
    dataset_dir = r"C:\Users\navee\.gemini\antigravity-ide\scratch\isl-translator\training\data_extracted_full"
    
    if not os.path.exists(dataset_dir):
        print(f"ERROR: Dataset directory {dataset_dir} does not exist.")
        print("Please run extract_keypoints.py first to generate the .pt files.")
        return
        
    dataset = ISLAlphabetDataset(dataset_dir)
    num_classes = len(dataset.classes)
    print(f"Number of classes: {num_classes}")
    
    if num_classes == 0:
        print("No classes found. Aborting.")
        return
    
    # 1. Initialize Model
    model = CWA_HWGATE(num_classes=num_classes).to(device)
    
    # 2. Setup Loss (Standard CrossEntropy for isolated frames)
    # We use NLLLoss because CWA_HWGATE outputs log_softmax
    criterion = nn.NLLLoss()
    
    # 3. Setup Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # 4. DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 5. Training Loop
    epochs = 30
    for epoch in range(epochs):
        loss, acc = train_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Acc: {acc*100:.2f}%")
        
    print("Pre-training complete. Model spatial layers have learned ISL alphabets.")
    torch.save(model.state_dict(), "cwa_hwgate_pretrained.pth")
    print("Model saved to cwa_hwgate_pretrained.pth")

if __name__ == "__main__":
    main()
