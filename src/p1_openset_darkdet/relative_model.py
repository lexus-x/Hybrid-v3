"""Joint Relative Kinematic Matcher.

Processes relative kinematics trajectories directly to output match logits and a
learned open-set reject option.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders.track_encoder import TrackEncoder


class JointRelativeMatcher(nn.Module):
    def __init__(self, in_dim: int = 6, emb: int = 128):
        super().__init__()
        # Encodes the 6D relative kinematics trajectory
        self.encoder = TrackEncoder(in_dim=in_dim, emb_dim=emb)
        # Linear layer to map relative embedding to scalar match logit
        self.classifier = nn.Linear(emb, 1)
        # Learned absent logit (open-set reject option)
        self.absent = nn.Parameter(torch.tensor(-1.0))

    def match_logits(self, cand_relative_feats: torch.Tensor, cand_mask: torch.Tensor, cand_valid: torch.Tensor):
        """
        cand_relative_feats: (K, L, 6) relative trajectory features for K candidates
        cand_mask: (K, L) valid points mask within each candidate tracklet
        cand_valid: (K,) mask indicating which candidate slots are populated
        
        Returns logits: (K+1,) over [K candidates + 1 absent]
        """
        K = cand_relative_feats.size(0)
        
        # Avoid all-padded rows causing NaNs in Transformer key_padding_mask
        safe_mask = cand_mask.clone()
        safe_mask[~cand_valid, 0] = True
        
        # Encode relative trajectories: (K, L, 6) -> (K, emb)
        embs = self.encoder(cand_relative_feats, safe_mask)
        
        # Classify each candidate to get match logit: (K, 1) -> (K,)
        logits = self.classifier(embs).squeeze(-1)
        
        # Mask out invalid candidate slots (e.g. padded candidates)
        logits = logits.masked_fill(~cand_valid, -1e4)
        
        # Append absent logit
        absent = self.absent.expand(1)
        return torch.cat([logits, absent], dim=0) # (K+1,)

    @torch.no_grad()
    def dark_score(self, cand_relative_feats: torch.Tensor, cand_mask: torch.Tensor, cand_valid: torch.Tensor):
        """P(dark) = softmax over [cands, absent] mass on the absent slot."""
        logits = self.match_logits(cand_relative_feats, cand_mask, cand_valid)
        p = F.softmax(logits, dim=0)
        return p[-1].item()
