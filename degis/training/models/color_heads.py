import torch
import torch.nn as nn
import torch.nn.functional as F

class ColorHead(nn.Module):
    """
    Enhanced color head with 4 hidden layers and better disentanglement.
    1024 → 1024 → 512 → 256 → 128 → hist_dim, plus projection head for ortho-loss.
    """
    def __init__(self, clip_dim=1024, hist_dim=514, hidden1=1024, hidden2=512, hidden3=256, hidden4=128):
        super().__init__()
        # Main color prediction path with 4 hidden layers
        self.fc1 = nn.Linear(clip_dim, hidden1)
        self.act1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.1)
        
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.act2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.1)
        
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.act3 = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(0.1)
        
        self.fc4 = nn.Linear(hidden3, hidden4)
        self.act4 = nn.ReLU(inplace=True)
        self.dropout4 = nn.Dropout(0.1)
        
        self.fc5 = nn.Linear(hidden4, hist_dim)  # histogram logits
        
        # Projection head for orthogonality loss
        self.to_emb = nn.Linear(hidden4, clip_dim)
        
        # Batch normalization for better training stability
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.bn3 = nn.BatchNorm1d(hidden3)
        self.bn4 = nn.BatchNorm1d(hidden4)

    def forward(self, x):
        x1 = self.act1(self.bn1(self.fc1(x)))
        x1 = self.dropout1(x1)
        
        x2 = self.act2(self.bn2(self.fc2(x1)))
        x2 = self.dropout2(x2)
        
        x3 = self.act3(self.bn3(self.fc3(x2)))
        x3 = self.dropout3(x3)
        
        x4 = self.act4(self.bn4(self.fc4(x3)))
        x4 = self.dropout4(x4)
        
        logits = self.fc5(x4)
        probs = torch.softmax(logits, dim=1)
        c_emb = self.to_emb(x4)
        
        return logits, probs, c_emb


class RestHead(nn.Module):
    """
    Enhanced rest head with residual connections and better disentanglement.
    """
    def __init__(self, clip_dim=1024, hidden1=512, hidden2=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clip_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden2, clip_dim)
        )
        self.residual = nn.Linear(clip_dim, clip_dim)
        
    def forward(self, x):
        return self.net(x) + self.residual(x)
