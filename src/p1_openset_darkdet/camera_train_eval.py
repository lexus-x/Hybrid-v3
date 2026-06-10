"""Camera arm (BONK) train + eval: appearance crop <-> AIS-attribute open-set matching.

Batch-negative contrastive training (each crop matches its own AIS vs the other
AIS in the minibatch + a learned 'absent' option). Training-time AIS-dropout
teaches the reject option. Eval = controlled AIS-dropout on held-out test
detections -> dark AUROC (mirrors the radar arm). Multi-seed for honest CIs.

Run:  python -m p1_openset_darkdet.camera_train_eval <bonk_root> [cache.npz]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from p1_openset_darkdet.camera_dataset import build
from p1_openset_darkdet.camera_model import CameraOpenSetMatcher

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def train_one(data, seed=0, epochs=40, bs=64, lr=3e-4, p_drop=0.5):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    tr = np.where(data["split"] == "train")[0]
    crops = torch.as_tensor(data["crops"]).to(DEV)
    attrs = torch.as_tensor(data["attrs"]).to(DEV)
    model = CameraOpenSetMatcher(attr_dim=attrs.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for ep in range(epochs):
        rng.shuffle(tr)
        tot = 0.0
        for i in range(0, len(tr), bs):
            idx = tr[i:i + bs]
            if len(idx) < 2:
                continue
            R = model.encode_img(crops[idx]); A = model.encode_ais(attrs[idx])
            sims = model.sim_matrix(R, A)                       # (B,B)
            B = sims.size(0)
            tgt = torch.arange(B, device=DEV)                   # default: own AIS = diagonal
            drop = torch.as_tensor(rng.random(B) < p_drop, device=DEV)
            sims = sims.clone()
            sims[drop, torch.arange(B, device=DEV)[drop]] = -1e4  # remove own AIS -> must pick absent
            tgt = torch.where(drop, torch.full_like(tgt, B), tgt)  # absent index = B
            logits = model.logits_with_absent(sims)
            loss = F.cross_entropy(logits, tgt)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item()
    model.eval()
    return model


def evaluate(model, data, seed=0, K=7):
    """Controlled AIS-dropout with a REALISTIC small candidate pool (own + K confusers),
    mirroring the radar arm's local gating pool. Returns dark AUROC."""
    rng = np.random.default_rng(1000 + seed)
    te = np.where(data["split"] == "test")[0]
    crops = torch.as_tensor(data["crops"][te]).to(DEV)
    attrs = torch.as_tensor(data["attrs"][te]).to(DEV)
    M = len(te)
    withheld = rng.random(M) < 0.5
    with torch.no_grad():
        R = F.normalize(model.encode_img(crops), dim=-1, eps=1e-6)
        A = F.normalize(model.encode_ais(attrs), dim=-1, eps=1e-6)
        absent = model.absent
        ds = np.empty(M, np.float32)
        for i in range(M):
            others = rng.choice(np.delete(np.arange(M), i), size=min(K, M - 1), replace=False)
            cand = others if withheld[i] else np.concatenate([[i], others])
            sims = (A[cand] @ R[i]) / model.temperature
            logits = torch.cat([sims, absent.view(1)])
            ds[i] = torch.softmax(logits, 0)[-1].item()
    y = withheld.astype(int)
    if len(set(y.tolist())) < 2:
        return None
    return dict(n_test=int(M), n_dark=int(y.sum()), pool=K + 1,
                dark_auroc=float(roc_auc_score(y, ds)),
                dark_auprc=float(average_precision_score(y, ds)))


def main():
    root = sys.argv[1]
    cache = sys.argv[2] if len(sys.argv) > 2 else os.path.join(OUT, "bonk_cache.npz")
    data = build(root, cache=cache)
    print(f"detections {len(data['crops'])} | images {len(set(data['img']))} | "
          f"train {int((data['split']=='train').sum())} test {int((data['split']=='test').sum())}")
    aurocs = []
    for seed in range(5):
        m = train_one(data, seed=seed)
        r = evaluate(m, data, seed=seed)
        if r:
            aurocs.append(r["dark_auroc"]); print(f"seed {seed}: dark AUROC {r['dark_auroc']:.3f} (n_test {r['n_test']})")
    res = {"camera_dark_auroc_mean": float(np.mean(aurocs)) if aurocs else None,
           "camera_dark_auroc_std": float(np.std(aurocs)) if aurocs else None,
           "all": aurocs, "n_test": r["n_test"] if aurocs else 0,
           "note": "BONK camera arm: appearance crop <-> AIS-attribute open-set matching; "
                   "controlled AIS-dropout dark labels; batch-negative contrastive; 5 seeds."}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "bonk_camera.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
