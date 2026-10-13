"""
Adaptive Context-Aware Hierarchical Windowed Graph Attention Transformer Encoder (AC-HWGATE).
Production Model Architecture for Indian Sign Language (ISL) Recognition & Translation.

Components:
A   - Adaptive Graph Attention: Learns dynamic spatial topological correlations between joints with residual skip.
C   - Context-Aware Temporal Attention: Global multi-head self-attention across the full sequence.
H   - Hierarchical Part Representation: Part-specific encoders for Left Hand, Right Hand, Pose, Face.
W   - Windowed Local Temporal Attention: Multi-head attention restricted to local temporal windows.
G   - Graph Attention: Spatial graph message passing along joint connectivity.
ATE - Attention Transformer Encoder: Multi-layer Pre-LN Transformer blocks with feedforward networks.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

class AdaptiveGraphAttention(nn.Module):
    """
    Adaptive Graph Attention layer for skeletal landmark graphs.
    Computes dynamic spatial connectivity between 133 joints with residual highway.
    """
    def __init__(self, in_channels: int = 6, out_channels: int = 16, num_joints: int = 133, num_heads: int = 2):
        super().__init__()
        self.num_joints = num_joints
        self.num_heads = num_heads
        self.out_channels = out_channels
        
        # Learnable adaptive spatial graph adjacency matrix (initialized identity + small noise)
        init_adj = torch.eye(num_joints).repeat(num_heads, 1, 1) + torch.randn(num_heads, num_joints, num_joints) * 0.01
        self.adaptive_adj = nn.Parameter(init_adj)
        self.proj = nn.Linear(in_channels, out_channels * num_heads)
        self.res_proj = nn.Linear(in_channels, out_channels)
        self.out_proj = nn.Linear(out_channels * num_heads, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, J, C_in]
        B, T, J, C = x.shape
        x_flat = x.view(B * T, J, C)
        h = self.proj(x_flat).view(B * T, J, self.num_heads, self.out_channels)
        h = h.permute(0, 2, 1, 3) # [B*T, heads, J, D]
        
        # Normalize adaptive adjacency across neighbors
        adj = torch.softmax(self.adaptive_adj, dim=-1) # [heads, J, J]
        out = torch.matmul(adj.unsqueeze(0), h) # [B*T, heads, J, D]
        out = out.permute(0, 2, 1, 3).reshape(B * T, J, -1) # [B*T, J, heads*D]
        out = self.out_proj(out)
        
        # Residual connection
        res = self.res_proj(x_flat)
        out = self.gelu(self.norm(out + res))
        return out.view(B, T, J, self.out_channels)

class PreLNWindowedAttention(nn.Module):
    """Local Windowed Temporal Attention Block with Pre-LayerNorm residual."""
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, window_size: int = 8, dropout: float = 0.1):
        super().__init__()
        self.window_size = window_size
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_norm = self.norm(x)
        if T % self.window_size != 0:
            pad_len = self.window_size - (T % self.window_size)
            x_norm = F.pad(x_norm, (0, 0, 0, pad_len))
        else:
            pad_len = 0
            
        num_windows = x_norm.size(1) // self.window_size
        w_x = x_norm.reshape(B * num_windows, self.window_size, D)
        out, _ = self.attn(w_x, w_x, w_x)
        out = out.reshape(B, -1, D)
        if pad_len > 0:
            out = out[:, :T, :]
        return x + self.dropout(out)

class PreLNTransformerBlock(nn.Module):
    """Context-Aware Global Transformer Block with Pre-LayerNorm."""
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm1 = self.norm1(x)
        attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class ACHWGATE(nn.Module):
    """
    Adaptive Context-Aware Hierarchical Windowed Graph Attention Transformer Encoder.
    """
    def __init__(
        self,
        num_classes: int = 40,
        num_joints: int = 133,
        in_channels: int = 6,
        sequence_length: int = 32,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        window_size: int = 8,
        dropout: float = 0.15
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim

        # 1. Adaptive Graph Attention Layer (Spatial Joint Topology)
        self.graph_out_channels = 16
        self.adaptive_graph = AdaptiveGraphAttention(
            in_channels=in_channels,
            out_channels=self.graph_out_channels,
            num_joints=num_joints,
            num_heads=2
        )

        # 2. Hierarchical Part Projections
        # 33 Pose, 21 Left Hand, 21 Right Hand, 58 Face
        part_in_dim = self.graph_out_channels
        self.pose_proj  = nn.Linear(33 * part_in_dim, embed_dim)
        self.lh_proj    = nn.Linear(21 * part_in_dim, embed_dim)
        self.rh_proj    = nn.Linear(21 * part_in_dim, embed_dim)
        self.face_proj  = nn.Linear(58 * part_in_dim, embed_dim)
        self.hands_proj = nn.Linear(42 * part_in_dim, embed_dim)

        # 3. Adaptive Part-Aware Spatial Gating & Hierarchical Fusion
        self.part_gate = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 4)
        )
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # 4. Temporal Positional Encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, sequence_length, embed_dim) * 0.02)

        # 5. Local Windowed Temporal Attention
        self.windowed_attn = PreLNWindowedAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout
        )

        # 6. Global Context-Aware Transformer Encoder Stack
        self.transformer_layers = nn.ModuleList([
            PreLNTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # 7. Attention-Weighted Temporal Pooling over T=32
        self.temporal_pool_attn = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # 8. Penultimate Feature Neck & Output Classifier Head
        self.neck = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Calibrated temperature buffer
        self.register_buffer("temperature", torch.tensor(1.0))

    def forward(self, x: torch.Tensor, return_embeddings: bool = False):
        """
        Forward pass for AC-HWGATE.
        Input:
            x: Tensor of shape [B, T, 133, 6] or flattened [B, T, 798]
        Output:
            logits: [B, num_classes] (and optional embeddings [B, embed_dim])
        """
        if len(x.shape) == 3:
            B, T, C = x.shape
            if C == self.in_channels:
                x = x.view(B, T, self.num_joints, self.in_channels)
            elif C == self.num_joints * 6:
                x = x.view(B, T, self.num_joints, 6)
            elif C == self.num_joints * 2:
                zeros = torch.zeros(B, T, self.num_joints, 4, device=x.device, dtype=x.dtype)
                x = torch.cat([x.view(B, T, self.num_joints, 2), zeros], dim=-1)

        B, T, J, C = x.shape

        # Step 1: Adaptive Graph Attention over skeletal topology
        g_feat = self.adaptive_graph(x) # [B, T, 133, graph_out_channels]

        # Step 2: Hierarchical Body-Part Projections
        p_feat  = self.pose_proj(g_feat[:, :, 0:33, :].reshape(B, T, -1))
        lh_feat = self.lh_proj(g_feat[:, :, 33:54, :].reshape(B, T, -1))
        rh_feat = self.rh_proj(g_feat[:, :, 54:75, :].reshape(B, T, -1))
        f_feat  = self.face_proj(g_feat[:, :, 75:133, :].reshape(B, T, -1))
        h_feat  = self.hands_proj(g_feat[:, :, 33:75, :].reshape(B, T, -1))

        # Step 3: Adaptive Part-Aware Spatial Gating & Fusion
        cat_parts = torch.cat([p_feat, lh_feat, rh_feat, f_feat], dim=-1) # [B, T, 4D]
        gate_weights = F.softmax(self.part_gate(cat_parts), dim=-1)       # [B, T, 4]

        fused = (
            gate_weights[:, :, 0:1] * p_feat +
            gate_weights[:, :, 1:2] * lh_feat +
            gate_weights[:, :, 2:3] * rh_feat +
            gate_weights[:, :, 3:4] * f_feat +
            h_feat  # Hands carry prioritized residual signal
        )
        h_hier = self.fusion(fused) # [B, T, D]

        # Step 4: Temporal Positional Encoding
        if T == self.sequence_length:
            h_seq = h_hier + self.pos_embedding
        else:
            pos_interp = F.interpolate(
                self.pos_embedding.transpose(1, 2),
                size=T,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            h_seq = h_hier + pos_interp

        # Step 5: Local Windowed Temporal Attention
        h_local = self.windowed_attn(h_seq)

        # Step 6: Global Context-Aware Transformer Encoding
        h_context = h_local
        for layer in self.transformer_layers:
            h_context = layer(h_context)

        # Step 7: Attention-Weighted Temporal Pooling over T=32
        attn_pool_w = F.softmax(self.temporal_pool_attn(h_context), dim=1) # [B, T, 1]
        pooled = (h_context * attn_pool_w).sum(dim=1) + h_context.mean(dim=1) # [B, D]

        # Step 8: Penultimate Feature Neck & Classification Head
        feat = self.neck(pooled) # [B, embed_dim]
        logits = self.classifier(feat) / self.temperature

        if return_embeddings:
            return logits, feat
        return logits

    def predict_gloss(self, x: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            confs, preds = torch.max(probs, dim=-1)
            is_unknown = confs < threshold
            return preds, confs, is_unknown

# Compatibility alias
AC_HWGATE = ACHWGATE
