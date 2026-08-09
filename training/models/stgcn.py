import torch
import torch.nn as nn
import torch.nn.functional as F

# A lightweight Spatial-Temporal GCN approximation for 133 nodes
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        # x: [batch, num_joints, in_features]
        # adj: [num_joints, num_joints] or implicit
        x = self.fc(x) # [batch, num_joints, out_features]
        # A simple spectral approximation (just global average pooling over joints as a fully connected graph)
        # For true ST-GCN, we would define the physical connections of MediaPipe.
        # But this is a benchmarking baseline. We'll use a learnable adjacency matrix.
        if adj is not None:
            x = torch.matmul(adj, x)
        return x

class STGCN(nn.Module):
    def __init__(self, num_classes=45, num_joints=133, in_channels=2):
        super(STGCN, self).__init__()
        
        self.num_joints = num_joints
        
        # Learnable adjacency matrix since MediaPipe full 133 graph is complex
        self.A = nn.Parameter(torch.randn(num_joints, num_joints))
        
        self.gcn1 = GraphConvolution(in_channels, 64)
        self.gcn2 = GraphConvolution(64, 128)
        self.gcn3 = GraphConvolution(128, 256)
        
        self.fc = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, conf=None):
        # x: [batch, seq_len, num_joints, 2]
        # Since we use static frames or seq_len=1 for this benchmark:
        x = x.squeeze(1) # [batch, num_joints, 2]
        
        # Normalize adjacency matrix
        adj = F.softmax(self.A, dim=1)
        
        x = F.relu(self.gcn1(x, adj))
        x = F.relu(self.gcn2(x, adj))
        x = F.relu(self.gcn3(x, adj))
        
        # Global Average Pooling over joints
        x = x.mean(dim=1) # [batch, 256]
        
        x = self.dropout(x)
        x = self.fc(x) # [batch, num_classes]
        
        x = x.unsqueeze(0) # [seq_len=1, batch, num_classes]
        return F.log_softmax(x, dim=-1)
