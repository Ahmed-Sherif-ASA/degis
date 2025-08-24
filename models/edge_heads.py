import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeHead(nn.Module):
    """
    CLIP (1024) → shared MLP → 7x7x256 seed → 5x deconvs → 224x224 edges.
    Output is passed through sigmoid (0–1). We return flattened [B, 224*224].
    """
    def __init__(self, clip_dim=1024, edge_dim=224*224, hidden=1024):
        super().__init__()
        self.edge_dim = edge_dim

        self.shared = nn.Sequential(
            nn.Linear(clip_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),   nn.ReLU(inplace=True),
        )

        self.to_grid = nn.Linear(hidden, 256 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),  # 7→14
            nn.ConvTranspose2d(128,  64, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),  # 14→28
            nn.ConvTranspose2d( 64,  32, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),  # 28→56
            nn.ConvTranspose2d( 32,  16, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),  # 56→112
            nn.ConvTranspose2d( 16,   1, kernel_size=4, stride=2, padding=1),                          # 112→224
        )

        self.to_emb = nn.Linear(hidden, clip_dim)

    def forward(self, x):
        # x: [B, clip_dim]
        h = self.shared(x)                                 # [B, hidden]
        seed = self.to_grid(h).view(-1, 256, 7, 7)         # [B,256,7,7]
        logits = self.decoder(seed)                        # [B,1,224,224]
        edge_pred = torch.sigmoid(logits).flatten(1)       # [B, 224*224]
        c_emb = self.to_emb(h)                             # [B, clip_dim]
        return h, edge_pred, c_emb


class RestHead(nn.Module):
    """Small MLP ‘rest’ head for orthogonality regularization."""
    def __init__(self, clip_dim=1024, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clip_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),   nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)
