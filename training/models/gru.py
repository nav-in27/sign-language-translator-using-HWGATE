import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUModel(nn.Module):
    def __init__(self, num_classes=45, num_joints=133, in_channels=2):
        super(GRUModel, self).__init__()
        self.input_dim = num_joints * in_channels
        self.hidden_dim = 256
        
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=False,
            dropout=0.3
        )
        self.fc = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, x, conf=None):
        # x: [batch, seq_len, num_joints, 2]
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        x = x.view(batch_size, seq_len, -1) # [batch, seq_len, 266]
        x = x.permute(1, 0, 2) # [seq_len, batch, 266]
        
        out, h_n = self.gru(x)
        
        out = self.fc(out) # [seq_len, batch, num_classes]
        
        return F.log_softmax(out, dim=-1)
