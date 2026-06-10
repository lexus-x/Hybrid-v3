"""Open-set dark-vessel matcher.

radar/sensor track --encoder--> r ; each candidate AIS track --encoder--> a_k.
Match score = cosine(r, a_k) / temperature. A track is "dark" when NO candidate
scores above the learned threshold: dark_score = 1 - max_k softmax-weighted sim.
Training: InfoNCE pulls a track to its true AIS and pushes the rest (open-set
negatives include "no match" via a learned absent-logit).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders.track_encoder import TrackEncoder


class OpenSetMatcher(nn.Module):
    def __init__(self, sensor_in: int = 6, ais_in: int = 6, emb: int = 128):
        super().__init__()
        self.sensor_enc = TrackEncoder(in_dim=sensor_in, emb_dim=emb)
        self.ais_enc = TrackEncoder(in_dim=ais_in, emb_dim=emb)
        # fixed temperature (learnable exp-temp was the source of late-training NaNs)
        self.temperature = 0.1
        # learned "no-AIS / absent" logit — the open-set reject option
        self.absent = nn.Parameter(torch.tensor(-1.0))

    def temp(self):
        return self.temperature

    def encode_sensor(self, feats, mask):
        return self.sensor_enc(feats, mask)

    def encode_ais(self, feats, mask):
        return self.ais_enc(feats, mask)

    def match_logits(self, r: torch.Tensor, cand: torch.Tensor, cand_mask: torch.Tensor):
        """r (E,); cand (K,E); cand_mask (K,) True=real candidate.
        Returns logits over [K candidates + 1 absent] for this single track."""
        # cosine sim (eps-stable backward; encoder no longer L2-normalises)
        sims = F.cosine_similarity(cand, r.unsqueeze(0).expand_as(cand), dim=-1, eps=1e-6) / self.temp()
        # finite large-negative (NOT -inf: -inf yields NaN grads via 0*-inf in backward)
        sims = sims.masked_fill(~cand_mask, -1e4)
        absent = self.absent.expand(1)
        return torch.cat([sims, absent], dim=0)          # (K+1,)

    @torch.no_grad()
    def dark_score(self, r, cand, cand_mask):
        """P(dark) = softmax over [cands, absent] mass on the absent slot."""
        logits = self.match_logits(r, cand, cand_mask)
        p = F.softmax(logits, dim=0)
        return p[-1].item()
