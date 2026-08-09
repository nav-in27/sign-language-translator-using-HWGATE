import torch
import torch.nn as nn
import torch.nn.functional as F

class ConfidenceWeightedAttention(nn.Module):
    """
    Novel Confidence-Weighted Adaptive (CWA) Multi-Head Attention.
    Injects the pose estimator confidence score directly into the attention matrix
    to mitigate motion blur inherent in real-time continuous sign language.
    """
    def __init__(self, embed_dim, num_heads):
        super(ConfidenceWeightedAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Adaptive Gating Mechanism (Learns how much to trust the confidence scores)
        self.gate_w = nn.Linear(1, num_heads)
        
    def forward(self, x, confidences):
        """
        x: [batch_size, seq_len, embed_dim] (Graph embeddings)
        confidences: [batch_size, seq_len, 1] (Average frame confidence from RTMPose)
        """
        B, T, C = x.size()
        
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # [B, heads, T, head_dim]
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Standard Scaled Dot-Product Attention (QK^T / sqrt(d_k))
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5) # [B, heads, T, T]
        
        # --- CWA Modification ---
        # Calculate the gating scalar based on frame confidence: g = sigmoid(W_g * c + b_g)
        gate = torch.sigmoid(self.gate_w(confidences)) # [B, T, heads]
        gate = gate.permute(0, 2, 1).unsqueeze(-1) # [B, heads, T, 1]
        
        # Apply the confidence gate to the attention scores (Masking blurry frames)
        # We multiply the query scores by the confidence gate of the keys
        attn_scores = attn_scores * gate.transpose(-2, -1)
        # -------------------------
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        out = torch.matmul(attn_probs, v) # [B, heads, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.out_proj(out)


class CWA_HWGATE(nn.Module):
    """
    Confidence-Weighted Adaptive Hierarchical Windowed Graph Attention Transformer Encoder.
    Baseline model for the ISL Translation Platform.
    """
    def __init__(self, num_classes=2000, num_joints=133, in_channels=2, embed_dim=256, num_heads=8, num_layers=4):
        super(CWA_HWGATE, self).__init__()
        
        # Spatial Graph Convolution Layer (Processes single-frame skeletons)
        # Assuming RTMPose 133 keypoints (x, y coordinates -> in_channels = 2)
        self.spatial_embedding = nn.Sequential(
            nn.Linear(num_joints * in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        
        # Temporal CWA-Transformer Layers (Processes the sequence of frames)
        self.temporal_layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': ConfidenceWeightedAttention(embed_dim, num_heads),
                'norm1': nn.LayerNorm(embed_dim),
                'ffn': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Linear(embed_dim * 4, embed_dim)
                ),
                'norm2': nn.LayerNorm(embed_dim)
            }) for _ in range(num_layers)
        ])
        
        # Output classification head (Gloss logits for CTC Decoding)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, keypoints, confidences):
        """
        keypoints: [batch_size, seq_len, num_joints, 2]
        confidences: [batch_size, seq_len, num_joints, 1]
        """
        B, T, J, C = keypoints.size()
        
        # Flatten joints for spatial embedding
        x = keypoints.view(B, T, J * C)
        x = self.spatial_embedding(x) # [B, T, embed_dim]
        
        # Average the confidence across all joints for each frame
        frame_conf = confidences.mean(dim=2) # [B, T, 1]
        
        # Pass through CWA-Transformer layers
        for layer in self.temporal_layers:
            # Attention with confidence gating
            attn_out = layer['attn'](layer['norm1'](x), frame_conf)
            x = x + attn_out
            
            # Feed-forward network
            ffn_out = layer['ffn'](layer['norm2'](x))
            x = x + ffn_out
            
        # Generate gloss logits for CTC Decoding
        logits = self.classifier(x) # [B, seq_len, num_classes]
        
        # Standard CTC expects [seq_len, batch_size, num_classes]
        logits = logits.transpose(0, 1) 
        
        return F.log_softmax(logits, dim=-1)

# Example Instantiation
if __name__ == "__main__":
    # Batch = 2, Seq_len = 64 frames, Joints = 133, Channels = 2 (x,y)
    dummy_keypoints = torch.randn(2, 64, 133, 2)
    # Confidence scores between 0 and 1
    dummy_confidences = torch.rand(2, 64, 133, 1)
    
    model = CWA_HWGATE(num_classes=500)
    output = model(dummy_keypoints, dummy_confidences)
    
    print(f"Model Output Shape (Seq_Len, Batch, Num_Classes): {output.shape}")
    print("CWA-HWGATE architecture successfully instantiated.")
