import torch
import torch.nn as nn
import torch.nn.functional as F

class ISL_GRU(nn.Module):
    """Bidirectional GRU Baseline Model."""
    def __init__(self, num_classes=179, input_dim=266, hidden_dim=256, num_layers=2, dropout=0.15):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, keypoints, confidences=None):
        if keypoints.dim() == 2:
            keypoints = keypoints.unsqueeze(1)
        B, T = keypoints.size(0), keypoints.size(1)
        x = keypoints.reshape(B, T, -1)
        x = self.input_proj(x)
        gru_out, _ = self.gru(x)
        feat = gru_out.mean(dim=1)
        logits = self.classifier(feat)
        return F.log_softmax(logits, dim=-1).unsqueeze(0)


class ISL_LSTM(nn.Module):
    """Bidirectional LSTM Baseline Model with Attention."""
    def __init__(self, num_classes=179, input_dim=266, hidden_dim=256, num_layers=2, dropout=0.15):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, keypoints, confidences=None):
        if keypoints.dim() == 2:
            keypoints = keypoints.unsqueeze(1)
        B, T = keypoints.size(0), keypoints.size(1)
        x = keypoints.reshape(B, T, -1)
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        attn_w = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(lstm_out * attn_w, dim=1)
        logits = self.classifier(context)
        return F.log_softmax(logits, dim=-1).unsqueeze(0)


class ISL_STGCN(nn.Module):
    """Spatial-Temporal Graph Convolutional Network (ST-GCN) Baseline."""
    def __init__(self, num_classes=179, num_joints=133, in_channels=2, embed_dim=256, dropout=0.15):
        super().__init__()
        in_dim = num_joints * in_channels
        self.gcn_conv1 = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        self.tcn_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.gcn_conv2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, keypoints, confidences=None):
        if keypoints.dim() == 2:
            keypoints = keypoints.unsqueeze(1)
        B, T = keypoints.size(0), keypoints.size(1)
        x = keypoints.reshape(B * T, -1)
        feat = self.gcn_conv1(x).view(B, T, -1)
        if T > 1:
            feat = feat + self.tcn_conv(feat.transpose(1, 2)).transpose(1, 2)
        feat = self.gcn_conv2(feat.mean(dim=1))
        logits = self.classifier(feat)
        return F.log_softmax(logits, dim=-1).unsqueeze(0)
