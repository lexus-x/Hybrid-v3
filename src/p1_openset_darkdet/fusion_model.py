"""Han.md fusion matcher: Time2Vec + cross-attention radar<->AIS interaction.

Upgrades the cosine open-set matcher with two of Han's proposals (the feasible
"can-add" ones — NOT the foundation-model / text-path / supervised-classifier):
  - Time2Vec: a learnable linear+periodic encoding of the per-point time channel,
    giving the encoder a richer temporal signal than a single normalised Δt.
  - Cross-attention mid-fusion: the radar tracklet attends to each candidate AIS
    tracklet; the fused representation -> a learned match logit. This lets the
    model detect micro-inconsistencies (radar wobble vs smooth AIS) instead of a
    pure cosine on pooled embeddings.

Keeps the open-set reject (learned `absent` logit) so it trains with the same
controlled-AIS-dropout protocol and plugs into the same eval harness via
`score(...) -> logits[K+1]`. Stability lessons retained: pre-norm, finite -1e4
mask (never -inf), masked-mean pooling, all-pad guard for attention.

Flags `use_t2v` / `use_cross` enable the ablation: cosine baseline vs +Time2Vec
vs +cross-attn vs both.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Time2Vec(nn.Module):
    """t2v(tau)[0] = w0*tau + b0 ; t2v(tau)[i>=1] = sin(wi*tau + bi)."""
    def __init__(self, out_dim: int = 8):
        super().__init__()
        self.lin = nn.Linear(1, out_dim)

    def forward(self, tau):                       # (...,1) -> (...,out_dim)
        z = self.lin(tau)
        return torch.cat([z[..., :1], torch.sin(z[..., 1:])], dim=-1)


class TokenEncoder(nn.Module):
    """Tracklet -> token sequence (pre-norm Transformer); optional Time2Vec on time."""
    def __init__(self, in_dim=6, d=128, nhead=4, layers=2, t2v_dim=8,
                 use_t2v=True, time_idx=5, dropout=0.1):
        super().__init__()
        self.use_t2v, self.time_idx = use_t2v, time_idx
        self.t2v = Time2Vec(t2v_dim) if use_t2v else None
        self.input = nn.Linear(in_dim + (t2v_dim if use_t2v else 0), d)
        self.in_norm = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(d, nhead, d * 2, dropout=dropout,
                                           batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)

    def forward(self, feats, mask):               # (B,L,F),(B,L) -> (B,L,d)
        x = feats
        if self.use_t2v:
            tau = feats[..., self.time_idx:self.time_idx + 1]
            x = torch.cat([feats, self.t2v(tau)], dim=-1)
        x = self.in_norm(self.input(x))
        return self.encoder(x, src_key_padding_mask=~mask)


class FusionOpenSetMatcher(nn.Module):
    def __init__(self, in_dim=6, d=128, use_t2v=True, use_cross=True):
        super().__init__()
        self.radar_enc = TokenEncoder(in_dim, d, use_t2v=use_t2v)
        self.ais_enc = TokenEncoder(in_dim, d, use_t2v=use_t2v)
        self.use_cross = use_cross
        self.temperature = 0.1
        self.absent = nn.Parameter(torch.tensor(-1.0))
        if use_cross:
            self.cross = nn.MultiheadAttention(d, 4, dropout=0.1, batch_first=True)
            self.fuse_norm = nn.LayerNorm(d)
            self.match_mlp = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d),
                                           nn.GELU(), nn.Linear(d, 1))

    @staticmethod
    def _pool(tok, mask):                          # (B,L,d),(B,L) -> (B,d) masked mean
        m = mask.unsqueeze(-1).float()
        return (tok * m).sum(1) / m.sum(1).clamp(min=1.0)

    def score(self, s_feat, s_mask, cand_feats, cand_pmask, cand_valid):
        """All tensors on device: (L,F),(L),(K,L,F),(K,L),(K). Returns logits (K+1,)."""
        R = self.radar_enc(s_feat.unsqueeze(0), s_mask.unsqueeze(0))     # (1,Lr,d)
        Kn = cand_feats.size(0)
        safe = cand_pmask.clone()
        safe[~cand_valid, 0] = True                                      # all-pad guard
        A = self.ais_enc(cand_feats, safe)                              # (K,La,d)
        if self.use_cross:
            Rb = R.expand(Kn, -1, -1)                                    # (K,Lr,d)
            attn, _ = self.cross(Rb, A, A, key_padding_mask=~safe)       # (K,Lr,d)
            fr = self._pool(self.fuse_norm(attn + Rb), s_mask.unsqueeze(0).expand(Kn, -1))
            pa = self._pool(A, safe)                                     # (K,d)
            logit = self.match_mlp(torch.cat([fr, pa], dim=-1)).squeeze(-1) / self.temperature
        else:
            r = self._pool(R, s_mask.unsqueeze(0))[0]                    # (d,)
            a = self._pool(A, safe)                                      # (K,d)
            logit = F.cosine_similarity(a, r.unsqueeze(0).expand_as(a), dim=-1, eps=1e-6) / self.temperature
        logit = logit.masked_fill(~cand_valid, -1e4)
        return torch.cat([logit, self.absent.expand(1)], dim=0)         # (K+1,)

    @torch.no_grad()
    def dark_score(self, *a):
        return float(F.softmax(self.score(*a), dim=0)[-1].item())
