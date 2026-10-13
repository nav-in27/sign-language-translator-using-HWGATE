"""
Retrain the ISL model with a simpler, more appropriate architecture.

Problem diagnosis:
- 264 classes, only ~4284 samples (~16 per class, some as low as 4)
- HWGATE transformer is way too heavy for this data size
- Original training produced a model with near-uniform outputs (~0.5% per class)

Solution:
- Use a lightweight LSTM + Attention model that can learn from small data
- Add gradient clipping, label smoothing, and warmup
- Use mixed precision for speed
- Remove the time.sleep(0.1) throttle that was killing training speed
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ============================================================
# 1. LIGHTWEIGHT MODEL: Bidirectional LSTM + Temporal Attention
# ============================================================
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, lstm_out):
        # lstm_out: (B, T, H)
        attn_weights = self.attention(lstm_out)  # (B, T, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (B, H)
        return context

class ISLModelLite(nn.Module):
    def __init__(self, num_classes, input_dim=225, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention pooling
        self.attention = TemporalAttention(hidden_dim * 2)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        # x: (B, T, 75, 3) or (B, T, 80, 3) — only use first 75 keypoints
        B, T = x.shape[0], x.shape[1]
        
        # Flatten keypoints: (B, T, 75*3) = (B, T, 225)
        x = x[:, :, :75, :3].reshape(B, T, -1)
        
        # Project
        x = self.input_proj(x)  # (B, T, hidden_dim)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (B, T, hidden_dim*2)
        
        # Attention pooling
        context = self.attention(lstm_out)  # (B, hidden_dim*2)
        
        # Classify
        return self.classifier(context)  # (B, num_classes)


# ============================================================
# 2. DATASET (same logic as original dataset.py)
# ============================================================
class ISLDataset(Dataset):
    def __init__(self, data_paths, labels, label_map, max_frames=64, augment=False):
        self.data_paths = data_paths
        self.labels = labels
        self.label_map = label_map
        self.max_frames = max_frames
        self.augment = augment

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        sequence = np.load(self.data_paths[idx])
        num_frames = sequence.shape[0]
        
        # Extract pose (drop visibility), left hand, right hand
        pose = sequence[:, 0:132].reshape(num_frames, 33, 4)[:, :, :3]
        lh = sequence[:, 132:195].reshape(num_frames, 21, 3)
        rh = sequence[:, 195:258].reshape(num_frames, 21, 3)
        combined = np.concatenate([pose, lh, rh], axis=1)  # (T, 75, 3)
        
        # Pad to 80 keypoints for compatibility with HWGATE if needed
        pad_kps = np.zeros((num_frames, 5, 3))
        sequence = np.concatenate([combined, pad_kps], axis=1)  # (T, 80, 3)
        
        # Pad or truncate to max_frames
        if sequence.shape[0] > self.max_frames:
            if self.augment:
                start = np.random.randint(0, sequence.shape[0] - self.max_frames + 1)
                sequence = sequence[start:start + self.max_frames]
            else:
                # Center crop for eval
                start = (sequence.shape[0] - self.max_frames) // 2
                sequence = sequence[start:start + self.max_frames]
        elif sequence.shape[0] < self.max_frames:
            pad_len = self.max_frames - sequence.shape[0]
            padding = np.zeros((pad_len, 80, 3))
            sequence = np.vstack((sequence, padding))
        
        if self.augment:
            # Spatial noise
            noise = np.random.normal(0, 0.003, sequence.shape)
            sequence = sequence + noise
            # Random scaling
            scale = np.random.uniform(0.95, 1.05)
            sequence = sequence * scale
            # Random temporal speed change (drop or duplicate some frames)
            if np.random.random() < 0.3:
                speed = np.random.uniform(0.8, 1.2)
                indices = np.linspace(0, self.max_frames - 1, int(self.max_frames * speed))
                indices = np.clip(np.round(indices).astype(int), 0, self.max_frames - 1)
                sequence_new = sequence[indices]
                # Re-pad/truncate
                if sequence_new.shape[0] > self.max_frames:
                    sequence = sequence_new[:self.max_frames]
                elif sequence_new.shape[0] < self.max_frames:
                    pad_len = self.max_frames - sequence_new.shape[0]
                    padding = np.zeros((pad_len, 80, 3))
                    sequence = np.vstack((sequence_new, padding))
                else:
                    sequence = sequence_new
        
        sequence = sequence.astype(np.float32)
        label_idx = self.label_map[self.labels[idx]]
        return torch.tensor(sequence), torch.tensor(label_idx)


# ============================================================
# 3. TRAINING
# ============================================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 32
    num_epochs = 80
    max_frames = 64
    
    # Load data
    dataset_dir = r"C:\Users\navee\Downloads\archive (9)"
    npy_paths = glob.glob(os.path.join(dataset_dir, '**', '*.npy'), recursive=True)
    
    if len(npy_paths) == 0:
        raise ValueError("No .npy files found!")
    
    labels = [os.path.basename(os.path.dirname(p)) for p in npy_paths]
    unique_labels = sorted(list(set(labels)))
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    num_classes = len(label_map)
    
    print(f"Found {len(npy_paths)} samples across {num_classes} classes")
    
    # Stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        npy_paths, labels, test_size=0.15, random_state=42, stratify=labels
    )
    
    train_dataset = ISLDataset(X_train, y_train, label_map, max_frames=max_frames, augment=True)
    val_dataset = ISLDataset(X_val, y_val, label_map, max_frames=max_frames, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")
    
    # Save label map
    # __file__ is scratch/isl-translator/backend/retrain_model.py
    # We want scratch/isl_recognition
    output_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(output_dir, "isl_recognition")
    
    with open(os.path.join(output_dir, 'label_map.json'), 'w') as f:
        json.dump(label_map, f)
    
    # Model
    model = ISLModelLite(num_classes=num_classes, input_dim=225, hidden_dim=256, num_layers=2, dropout=0.3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Loss with label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    
    # Cosine annealing with warmup
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    best_val_acc = 0.0
    patience = 20
    patience_counter = 0
    
    # Mixed precision for speed
    scaler = torch.cuda.amp.GradScaler(enabled=True) if device.type == 'cuda' else None
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            if scaler:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    outputs = model(X)
                    loss = criterion(outputs, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_loss += loss.item() * X.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{train_correct/train_total:.4f}"
            })
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item() * X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1}/{num_epochs}] LR: {current_lr:.6f} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  >> New best val acc: {best_val_acc:.4f}. Saving model...")
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model_lite.pth'))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping! Best Val Acc: {best_val_acc:.4f}")
                break
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {os.path.join(output_dir, 'best_model_lite.pth')}")

if __name__ == '__main__':
    train()
