import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=45, num_joints=133, in_channels=2):
        super(CNN, self).__init__()
        # Input: [batch, seq_len, num_joints, in_channels]
        # Since we use single frames, seq_len=1. We'll flatten seq_len into batch or just squeeze it.
        # Shape after squeeze: [batch, num_joints, in_channels]
        # CNN expects [batch, channels, length], so we permute to [batch, in_channels, num_joints]
        
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        # After 3 pools of size 2, the spatial dimension 133 becomes:
        # 133 -> 66 -> 33 -> 16
        self.fc1 = nn.Linear(256 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, conf=None):
        # x: [batch, seq_len, num_joints, 2]
        x = x.squeeze(1) # [batch, num_joints, 2]
        x = x.permute(0, 2, 1) # [batch, 2, num_joints]
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Output shape should be [seq_len, batch_size, num_classes] to match CWA_HWGATE interface
        # For our benchmark, seq_len is always 1
        x = x.unsqueeze(0)
        return F.log_softmax(x, dim=-1)
