import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=45, num_joints=133, in_channels=2):
        super().__init__()
        self.conv1 = nn.Conv1d(num_joints * in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, keypoints, confidences):
        # keypoints: [B, T, J, C]
        B, T, J, C = keypoints.size()
        x = keypoints.view(B, T, J * C).transpose(1, 2) # [B, features, T]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1) # [B, 128]
        logits = self.fc(x) # [B, num_classes]
        return F.log_softmax(logits, dim=-1).unsqueeze(0) # [1, B, num_classes]

class SimpleLSTM(nn.Module):
    def __init__(self, num_classes=45, num_joints=133, in_channels=2):
        super().__init__()
        self.lstm = nn.LSTM(num_joints * in_channels, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, keypoints, confidences):
        B, T, J, C = keypoints.size()
        x = keypoints.view(B, T, J * C)
        out, _ = self.lstm(x)
        logits = self.fc(out[:, -1, :]) # Take last timestep
        return F.log_softmax(logits, dim=-1).unsqueeze(0)

class SimpleGRU(nn.Module):
    def __init__(self, num_classes=45, num_joints=133, in_channels=2):
        super().__init__()
        self.gru = nn.GRU(num_joints * in_channels, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, keypoints, confidences):
        B, T, J, C = keypoints.size()
        x = keypoints.view(B, T, J * C)
        out, _ = self.gru(x)
        logits = self.fc(out[:, -1, :])
        return F.log_softmax(logits, dim=-1).unsqueeze(0)

class ST_GCN(nn.Module):
    def __init__(self, num_classes=45, num_joints=133, in_channels=2):
        super().__init__()
        self.fc1 = nn.Linear(num_joints * in_channels, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, keypoints, confidences):
        B, T, J, C = keypoints.size()
        x = keypoints.view(B, T, J * C)
        x = F.relu(self.fc1(x))
        x = x.mean(dim=1) # average over time
        logits = self.fc2(x)
        return F.log_softmax(logits, dim=-1).unsqueeze(0)
