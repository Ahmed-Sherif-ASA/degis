# models/heads.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ColorHead(nn.Module):
    """
    Same as notebook:
    1024 → 1024 → 512 → hist_dim, plus a projection head (to_emb) for ortho-loss.
    """
    def __init__(self, clip_dim=1024, hist_dim=514, hidden1=1024, hidden2=512):
        super().__init__()
        self.fc1   = nn.Linear(clip_dim, hidden1)
        self.act1  = nn.ReLU(inplace=True)
        self.fc2   = nn.Linear(hidden1, hidden2)
        self.act2  = nn.ReLU(inplace=True)
        self.fc3   = nn.Linear(hidden2, hist_dim)     # histogram logits
        self.to_emb = nn.Linear(hidden2, clip_dim)    # optional head for ortho loss

    def forward(self, x):
        x1     = self.act1(self.fc1(x))
        x2     = self.act2(self.fc2(x1))
        logits = self.fc3(x2)
        probs  = torch.softmax(logits, dim=1)
        c_emb  = self.to_emb(x2)
        return logits, probs, c_emb


class RestHead(nn.Module):
    def __init__(self, clip_dim=1024, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clip_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, clip_dim), nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)
