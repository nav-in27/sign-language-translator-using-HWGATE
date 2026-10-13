import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    getattr(sys.stdout, "reconfigure")(line_buffering=True)

class ISLUnifiedDataset(Dataset):
    def __init__(self, data_dir, augment=False):
        self.data_dir = data_dir
        self.augment = augment
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]) if os.path.exists(data_dir) else []
        self.samples = []
        for idx, cls_name in enumerate(self.classes):
            cls_folder = os.path.join(data_dir, cls_name)
            for f in os.listdir(cls_folder):
                if f.endswith('.pt'):
                    self.samples.append((os.path.join(cls_folder, f), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        fpath, label = self.samples[index]
        data = torch.load(fpath, map_location='cpu', weights_only=False)
        kps = data['keypoints']
        confs = data.get('confidences', torch.ones(kps.shape[0], 133, 1))
        return (kps.squeeze(0) if kps.dim() == 3 else kps,
                confs.squeeze(0) if confs.dim() == 3 else confs,
                label)

class HWGATE_Classifier(nn.Module):
    """
    Optimized CWA-HWGATE Spatial-Temporal Graph Architecture for Sign Classification.
    Extracts joint-level spatial features and sequence temporal context.
    """
    def __init__(self, num_classes=180, num_joints=133, in_channels=2, embed_dim=256, dropout=0.1):
        super().__init__()
        in_dim = num_joints * in_channels # 133 * 2 = 266
        
        # Joint-level Spatial Feature Extractor
        self.joint_embed = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.SiLU(),
        )
        
        # Residual Graph Attention Blocks
        self.res1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )
        
        self.res2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )
        
        # Confidence Gating Branch
        self.conf_gate = nn.Sequential(
            nn.Linear(num_joints, 64),
            nn.SiLU(),
            nn.Linear(64, embed_dim),
            nn.Sigmoid()
        )
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, keypoints, confidences):
        # keypoints: [B, 1, 133, 2] or [B, T, 133, 2]
        # confidences: [B, 1, 133, 1]
        B = keypoints.size(0)
        
        # Average across temporal dimension if needed
        if keypoints.dim() == 4 and keypoints.size(1) > 1:
            kp_flat = keypoints.mean(dim=1).view(B, -1)
            conf_flat = confidences.mean(dim=1).squeeze(-1)
        else:
            kp_flat = keypoints.view(B, -1)
            conf_flat = confidences.squeeze(-1).squeeze(1) if confidences.dim() == 4 else confidences.view(B, -1)
            
        # 1. Joint Embedding
        feat = self.joint_embed(kp_flat) # [B, embed_dim]
        
        # 2. Confidence Gating
        gate = self.conf_gate(conf_flat) # [B, embed_dim]
        feat = feat * gate
        
        # 3. Residual Blocks
        feat = feat + F.silu(self.res1(feat))
        feat = feat + F.silu(self.res2(feat))
        
        # 4. Logits
        logits = self.classifier(feat) # [B, num_classes]
        
        return F.log_softmax(logits, dim=-1)

if __name__ == "__main__":
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_unified")
    dataset = ISLUnifiedDataset(dataset_dir, augment=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(dataset.classes)
    
    model = HWGATE_Classifier(num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.NLLLoss()
    
    print(f"Testing model on {num_classes} classes, {len(dataset)} samples...")
    for epoch in range(1, 10):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for step, (kps, confs, labels) in enumerate(loader):
            kps, confs, labels = kps.to(device), confs.to(device), labels.to(device)
            optimizer.zero_grad()
            log_probs = model(kps, confs)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(log_probs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (step + 1) % 100 == 0 or (step + 1) == len(loader):
                print(f"  [Epoch {epoch}] Batch [{step+1}/{len(loader)}] Loss: {loss.item():.4f} Acc: {(correct/total)*100:.2f}%", flush=True)
            
        print(f"Epoch {epoch}: Loss = {total_loss/len(loader):.4f}, Train Acc = {(correct/total)*100:.2f}%", flush=True)
