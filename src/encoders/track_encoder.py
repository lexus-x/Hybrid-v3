"""Kinematic track encoder — shared design for radar tracks and AIS trajectories.

A tracklet is a length-L sequence of per-point features; the encoder maps it to
an L2-normalised embedding. Two independent instances are used (radar vs AIS) so
each modality learns its own weights while sharing the architecture and output
space. Deliberately small — the datasets are modest, so capacity is kept low to
avoid overfitting.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TrackEncoder(nn.Module):
    def __init__(self, in_dim: int = 6, d_model: int = 128, nhead: int = 4,
                 layers: int = 2, emb_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input = nn.Linear(in_dim, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,   # pre-norm: stabilises training (post-norm caused stochastic NaNs)
        )
        self.in_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, emb_dim))

    def forward(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """feats (B,L,F); mask (B,L) True=valid. Returns (B,emb) L2-normalised."""
        x = self.in_norm(self.input(feats))
        pad = ~mask  # TransformerEncoder expects True = position to IGNORE
        x = self.encoder(x, src_key_padding_mask=pad)
        m = mask.unsqueeze(-1).float()
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1.0)  # masked mean
        return self.proj(pooled)   # raw embedding; cosine (eps-stable) is applied at scoring
