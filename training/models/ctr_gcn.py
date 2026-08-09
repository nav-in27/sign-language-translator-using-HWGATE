import torch
import torch.nn as nn
import torch.nn.functional as F

class CTRGC(nn.Module):
    """
    Channel-wise Topology Refinement Graph Convolution
    """
    def __init__(self, in_channels, out_channels, rel_reduction=8):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # We learn the topology dynamically via channel-wise correlations
        self.conv_q = nn.Conv2d(in_channels, in_channels // rel_reduction, 1)
        self.conv_k = nn.Conv2d(in_channels, in_channels // rel_reduction, 1)
        self.conv_v = nn.Conv2d(in_channels, out_channels, 1)
        
        self.alpha = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # x shape: [B, C, T, V]
        N, C, T, V = x.size()
        
        q = self.conv_q(x).view(N, -1, T * V).transpose(1, 2) # [N, T*V, C']
        k = self.conv_k(x).view(N, -1, T * V) # [N, C', T*V]
        
        # Compute dynamic adjacency
        attn = torch.matmul(q, k) # [N, T*V, T*V]
        attn = F.softmax(attn, dim=-1)
        
        v = self.conv_v(x).view(N, -1, T * V) # [N, C_out, T*V]
        
        # Apply dynamic adjacency
        out = torch.matmul(v, attn.transpose(1, 2))
        out = out.view(N, self.out_channels, T, V)
        
        return out * self.alpha + self.conv_v(x)


class CTR_GCN(nn.Module):
    """
    CTR-GCN for Continuous Sign Language Recognition
    """
    def __init__(self, num_classes=45, num_joints=133, in_channels=2, embed_dim=256):
        super(CTR_GCN, self).__init__()
        
        # Input Data BN
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        
        # Spatio-Temporal Layers
        self.layer1 = nn.Sequential(
            CTRGC(in_channels, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            CTRGC(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(2, 1)) # temporal pooling
        )
        self.layer3 = nn.Sequential(
            CTRGC(128, embed_dim),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(2, 1))
        )
        
        # Temporal Convolution Network (MS-TCN) to capture sequence context
        self.tcn = nn.Sequential(
            nn.Conv1d(embed_dim * num_joints, embed_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2, dilation=2),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, keypoints, confidences=None):
        # keypoints shape: [B, T, V, C] (V=num_joints, C=channels)
        B, T, V, C = keypoints.size()
        
        # Permute for BN: [B, C*V, T]
        x = keypoints.permute(0, 3, 2, 1).contiguous().view(B, C * V, T)
        x = self.data_bn(x)
        
        # Reshape for CTRGC: [B, C, T, V]
        x = x.view(B, C, V, T).permute(0, 1, 3, 2).contiguous()
        
        x = self.layer1(x)
        if x.size(2) > 1: x = self.layer2(x)
        if x.size(2) > 1: x = self.layer3(x)
        
        # x is now [B, embed_dim, T', V]
        B, C_out, T_out, V_out = x.size()
        
        # Flatten spatial dimension for TCN
        x = x.permute(0, 1, 3, 2).contiguous().view(B, C_out * V_out, T_out)
        
        # Pass through MS-TCN
        x = self.tcn(x) # [B, embed_dim, T_out]
        
        # Map to classes
        x = x.transpose(1, 2) # [B, T_out, embed_dim]
        logits = self.fc(x) # [B, T_out, num_classes]
        
        # CTC expects [T_out, B, num_classes]
        logits = logits.transpose(0, 1)
        
        return F.log_softmax(logits, dim=-1)
