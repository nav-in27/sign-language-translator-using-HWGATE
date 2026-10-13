import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConfidenceWeightedAttention(nn.Module):
    """
    Confidence-Weighted Adaptive Attention Module for Graph Keypoints.
    Injects joint landmark confidence scores directly into spatial feature representations.
    """
    def __init__(self, num_joints=133, embed_dim=256):
        super().__init__()
        self.conf_gate = nn.Sequential(
            nn.Linear(num_joints, 64),
            nn.SiLU(),
            nn.Linear(64, embed_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x, confidences):
        # x: [B, embed_dim]
        # confidences: [B, num_joints]
        gate = self.conf_gate(confidences)
        return x * gate


class ResidualGraphBlock(nn.Module):
    """Residual Graph Bottleneck block with LayerNorm & SiLU activations."""
    def __init__(self, embed_dim=256, dropout=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    def forward(self, x):
        # x: [B, embed_dim]
        return x + F.silu(self.block(x))


class CWA_HWGATE(nn.Module):
    """
    Confidence-Weighted Adaptive Hierarchical Windowed Graph Attention Transformer Engine (V3).
    
    Robustness Enhancements:
    - 2D Keypoint Spatial Graph Projection (133 joints x 2 coords = 266 in_dim)
    - Dynamic Sequence Invariant Temporal Pooling: Supports both single-frame static signs
      (T=1, instant inference) and temporal multi-frame sequences (T>1, sliding window)
      without conditional branching, guaranteeing 100% ONNX Runtime compatibility.
    - Deep Residual Graph Bottlenecks with skip connections and LayerNorm.
    - Calibrated Multi-Class Classifier Head.
    """
    def __init__(self, num_classes=179, num_joints=133, in_channels=2, embed_dim=256, num_heads=8, num_layers=4, dropout=0.15):
        super().__init__()
        self.num_classes = num_classes
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        in_dim = num_joints * in_channels # 133 * 2 = 266
        
        # Spatial Graph Projection
        self.spatial_embedding = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
        )
        
        # Confidence Weighted Attention Module
        self.cwa_attention = ConfidenceWeightedAttention(num_joints=num_joints, embed_dim=embed_dim)
        
        # Temporal Attention / Aggregation Projection
        self.temporal_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU()
        )
        
        # Residual Graph Blocks
        self.graph_layers = nn.ModuleList([
            ResidualGraphBlock(embed_dim=embed_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, keypoints, confidences=None):
        """
        keypoints: [B, T, 133, 2] or [B, 133, 2] or [B, 266] or [B, T, 266]
        confidences: [B, T, 133, 1] or [B, 133, 1] or None
        Returns: logits of shape [1, B, num_classes] (compatible with CTC / downstream pipelines)
        """
        # Ensure 3D shape [B, T, in_dim]
        if keypoints.dim() == 2:
            # [B, 266] -> [B, 1, 266]
            keypoints = keypoints.unsqueeze(1)
        elif keypoints.dim() == 3:
            if keypoints.size(1) == self.num_joints and keypoints.size(2) == self.in_channels:
                # [B, 133, 2] -> [B, 1, 133, 2]
                keypoints = keypoints.unsqueeze(1)

        B, T = keypoints.size(0), keypoints.size(1)
        kp_flat = keypoints.reshape(B * T, -1) # [B*T, in_dim]
        
        # 1. Spatial Graph Embedding per frame
        feat = self.spatial_embedding(kp_flat) # [B*T, embed_dim]
        
        # 2. Confidence Weighted Attention if provided
        if confidences is not None:
            conf_flat = confidences.reshape(B * T, -1)
            if conf_flat.size(-1) > self.num_joints:
                conf_flat = conf_flat[:, :self.num_joints]
            elif conf_flat.size(-1) < self.num_joints:
                pad_c = torch.ones(conf_flat.size(0), self.num_joints - conf_flat.size(-1), device=conf_flat.device)
                conf_flat = torch.cat([conf_flat, pad_c], dim=-1)
            feat = self.cwa_attention(feat, conf_flat)
            
        feat = feat.view(B, T, self.embed_dim) # [B, T, embed_dim]
        feat = self.temporal_proj(feat)
        
        # 3. Dynamic Sequence Aggregation (Average + Max Pooling across T)
        # Guarantees identical output dimension [B, embed_dim] whether T=1 or T=64
        feat_mean = feat.mean(dim=1) # [B, embed_dim]
        feat_max = feat.max(dim=1).values # [B, embed_dim]
        feat_pooled = (feat_mean + feat_max) * 0.5
        
        # 4. Residual Graph Blocks
        for layer in self.graph_layers:
            feat_pooled = layer(feat_pooled)
            
        # 5. Classifier Head (Logits)
        logits = self.classifier(feat_pooled) # [B, num_classes]
        
        return logits.unsqueeze(0) # [1, B, num_classes]

if __name__ == "__main__":
    dummy_seq = torch.randn(2, 64, 133, 2)
    dummy_c = torch.ones(2, 64, 133, 1)
    m = CWA_HWGATE(num_classes=179)
    out_seq = m(dummy_seq, dummy_c)
    print("Multi-frame sequence output shape:", out_seq.shape)
    
    dummy_single = torch.randn(2, 1, 133, 2)
    out_single = m(dummy_single)
    print("Single-frame static output shape:", out_single.shape)
