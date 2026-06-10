"""Small from-scratch CNN for vessel bbox crops (camera arm of P1).

From-scratch (no pretrained download) since crops are tiny and the dataset is
modest. Input: (B,3,64,64) in [0,1]. Output: (B,emb) raw embedding (cosine applied
at scoring, matching the radar arm's convention).
"""
from __future__ import annotations

import torch.nn as nn


class ImageEncoder(nn.Module):
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.GELU())
        self.body = nn.Sequential(
            block(3, 32),    # 64->32
            block(32, 64),   # 32->16
            block(64, 128),  # 16->8
            block(128, 128), # 8->4
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.proj = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, emb_dim))

    def forward(self, x):
        return self.proj(self.body(x))
