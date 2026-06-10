"""Camera open-set matcher (BONK arm): image-crop <-> AIS-attribute matching.

Mirrors the radar OpenSetMatcher but the two encoders are: a CNN over the crop
(appearance) and an MLP over the AIS attribute vector [range, bearing, dims].
"Dark" = a visual detection that matches no AIS above threshold (open-set reject).
Cosine scoring + learned absent logit (same head as the radar arm).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders.image_encoder import ImageEncoder


class CameraOpenSetMatcher(nn.Module):
    def __init__(self, attr_dim: int = 5, emb: int = 128):
        super().__init__()
        self.img_enc = ImageEncoder(emb_dim=emb)
        self.ais_enc = nn.Sequential(
            nn.Linear(attr_dim, 128), nn.GELU(), nn.LayerNorm(128), nn.Linear(128, emb)
        )
        self.temperature = 0.1
        self.absent = nn.Parameter(torch.tensor(-1.0))

    def encode_img(self, crops):       # (B,3,64,64) -> (B,E)
        return self.img_enc(crops)

    def encode_ais(self, attrs):       # (B,attr) -> (B,E)
        return self.ais_enc(attrs)

    def sim_matrix(self, R, A):        # (N,E),(M,E) -> (N,M) cosine/temp
        Rn = F.normalize(R, dim=-1, eps=1e-6)
        An = F.normalize(A, dim=-1, eps=1e-6)
        return (Rn @ An.t()) / self.temperature

    def logits_with_absent(self, sims):  # (N,M) -> (N,M+1)
        absent = self.absent.expand(sims.size(0), 1)
        return torch.cat([sims, absent], dim=1)

    @torch.no_grad()
    def dark_scores(self, R, A, cand_mask=None):
        """P(dark) per row = softmax mass on absent. cand_mask (N,M) True=valid candidate."""
        sims = self.sim_matrix(R, A)
        if cand_mask is not None:
            sims = sims.masked_fill(~cand_mask, -1e4)
        p = F.softmax(self.logits_with_absent(sims), dim=1)
        return p[:, -1]
