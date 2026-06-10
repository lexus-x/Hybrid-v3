"""W2 pipeline on the Busan radar+AIS arm: train the open-set matcher
(contrastive, with a learned absent/reject option) and evaluate the dark score.

Tiny-data regime (≈89 tracks) — PIPELINE SANITY (loss→0 on train, separability
on test), not the headline. Scaled numbers come from the camera+AIS arm; the
learned model's real edge over gating is the robustness eval (eval/robustness_busan).
Run:  python -m p1_openset_darkdet.train_eval_busan
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from data.busan_loader import load_ais, load_radar
from data.pair_builder import GateCfg, match_radar_to_ais, track_labels
from data.dropout_splitter import make_splits
from p1_openset_darkdet.dataset import MAX_CAND, build_samples
from p1_openset_darkdet.model import OpenSetMatcher

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _t(a):
    return torch.as_tensor(a).to(DEV)


def encode_sample(model, s):
    r = model.encode_sensor(_t(s["s_feat"]).unsqueeze(0).float(), _t(s["s_mask"]).unsqueeze(0))[0]
    cf = _t(s["cand_feats"]).float(); cm = _t(s["cand_pmask"])
    valid = _t(s["cand_valid"])
    safe = cm.clone(); safe[~valid, 0] = True   # avoid all-padded rows -> NaN attention
    a = model.encode_ais(cf, safe)              # (K,E)
    return r, a, valid


def logits_for(model, s):
    r, a, valid = encode_sample(model, s)
    return model.match_logits(r, a, valid), valid


def train_model(epochs=150, lr=3e-4, seed=0, p_drop=0.5):
    """Train and return (model, context) where context has ais/rad/tl/samples.

    p_drop: training-time AIS-dropout — fraction of matched tracks for which the
    true AIS is withheld and the target becomes the 'absent' slot, so the model
    learns to REJECT (flag dark), not just match.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cfg = GateCfg()
    ais, rad = load_ais(), load_radar()
    pts = match_radar_to_ais(rad, ais, cfg)
    tl = make_splits(track_labels(pts, cfg))
    samples = build_samples(ais, rad, tl)
    train = [s for s in samples if s["split"] == "train" and s["pos_idx"] >= 0]

    model = OpenSetMatcher().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    last = {}
    for ep in range(epochs):
        rng.shuffle(train)
        tot = 0.0; correct = 0
        for s in train:
            r, a, valid = encode_sample(model, s)
            valid = valid.clone()
            pos = s["pos_idx"]
            if pos >= 0 and rng.random() < p_drop:     # training-time AIS dropout -> teach reject
                valid[pos] = False
                target = MAX_CAND                       # 'absent'/dark slot
            else:
                target = pos
            logits = model.match_logits(r, a, valid)
            loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor(target, device=DEV).unsqueeze(0))
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); correct += int(logits.argmax().item() == target)
        if ep % 25 == 0 or ep == epochs - 1:
            last = {"epoch": ep, "train_loss": tot / max(len(train), 1),
                    "train_match_acc": correct / max(len(train), 1)}
            print(last)
    model.eval()
    return model, dict(ais=ais, rad=rad, cfg=cfg, tl=tl, samples=samples, train_log=last)


def main():
    model, ctx = train_model()
    test = [s for s in ctx["samples"] if s["split"] == "test" and s["dark_label"] is not None
            and not (isinstance(s["dark_label"], float) and np.isnan(s["dark_label"]))]
    y, score = [], []
    for s in test:
        r, a, valid = encode_sample(model, s)
        ds = model.dark_score(r, a, valid)
        score.append(0.5 if not np.isfinite(ds) else float(ds))
        y.append(1 if bool(s["dark_label"]) else 0)
    res = {"train": ctx["train_log"], "n_test": len(test), "n_dark": int(sum(y))}
    if len(set(y)) == 2:
        res["dark_auroc"] = float(roc_auc_score(y, score))
        res["dark_auprc"] = float(average_precision_score(y, score))
    res["note"] = "tiny-data PIPELINE SANITY (Busan ~89 tracks); headline needs camera arm + robustness eval."
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "busan_w2.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
