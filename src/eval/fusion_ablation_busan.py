"""Han.md fusion ablation on the Busan radar arm: does Time2Vec + cross-attention
lower the learned false-alarm floor and/or raise clean AUROC?

Ablation configs (all trained with the SAME controlled-AIS-dropout open-set protocol):
  cosine_baseline : no Time2Vec, no cross-attn  (re-implements the current matcher)
  t2v             : + Time2Vec only
  cross           : + cross-attention only
  han_full        : + Time2Vec + cross-attention

For each config x 5 seeds: learned clean AUROC, hybrid clean AUROC, learned
robustness curve, hybrid robustness curve. Geometric baseline computed once.
Writes outputs/busan_fusion_ablation.json + fig_fusion_ablation.png.

Run:        python -m eval.fusion_ablation_busan
Smoke test: python -m eval.fusion_ablation_busan smoke
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from data.busan_loader import load_ais, load_radar
from data.pair_builder import GateCfg, match_radar_to_ais, track_labels
from data.dropout_splitter import make_splits
from eval.hybrid_v2_busan import _confident_match
from eval.robustness_busan import (DIRS, OFFSETS_M, _build_cands, _cand_mmsis,
                                   _geom_match)
from eval.strengthen_busan import geometric_dark_score
from p1_openset_darkdet.dataset import MAX_CAND, _featurize, build_samples
from p1_openset_darkdet.fusion_model import FusionOpenSetMatcher

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
K_SEEDS = 5
CONFIGS = [("cosine_baseline", False, False), ("t2v", True, False),
           ("cross", False, True), ("han_full", True, True)]


def _t(a):
    return torch.as_tensor(a).to(DEV)


def prep():
    cfg = GateCfg()
    ais, rad = load_ais(), load_radar()
    pts = match_radar_to_ais(rad, ais, cfg)
    tl = make_splits(track_labels(pts, cfg))
    samples = build_samples(ais, rad, tl)
    return dict(ais=ais, rad=rad, cfg=cfg, tl=tl, samples=samples)


def train_fusion(ctx, use_t2v, use_cross, seed=0, epochs=150, lr=3e-4, p_drop=0.5):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    train = [s for s in ctx["samples"] if s["split"] == "train" and s["pos_idx"] >= 0]
    model = FusionOpenSetMatcher(use_t2v=use_t2v, use_cross=use_cross).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for ep in range(epochs):
        rng.shuffle(train)
        for s in train:
            sf, sm = _t(s["s_feat"]).float(), _t(s["s_mask"])
            cf, cm = _t(s["cand_feats"]).float(), _t(s["cand_pmask"])
            cv = _t(s["cand_valid"]).clone()
            pos = s["pos_idx"]
            if pos >= 0 and rng.random() < p_drop:
                cv[pos] = False; target = MAX_CAND
            else:
                target = pos
            logits = model.score(sf, sm, cf, cm, cv)
            loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor(target, device=DEV).unsqueeze(0))
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
    return model


def _ldark(model, sf, sm, cf, cm, cv):
    ds = model.dark_score(_t(sf).float(), _t(sm), _t(cf).float(), _t(cm), _t(cv))
    return 0.5 if not np.isfinite(ds) else float(ds)


def clean(model, ctx):
    ais, rad, cfg, tl = ctx["ais"], ctx["rad"], ctx["cfg"], ctx["tl"]
    rad_by_t = rad.sort_values("t")
    dom_by_id = dict(zip(tl["targetId"], tl["dom_mmsi"]))
    test = [s for s in ctx["samples"] if s["split"] == "test" and pd.notna(s["dark_label"])]
    y, lsc, hsc = [], [], []
    for s in test:
        ds = _ldark(model, s["s_feat"], s["s_mask"], s["cand_feats"], s["cand_pmask"], s["cand_valid"])
        sub = rad_by_t[rad_by_t["targetId"] == s["targetId"]]
        dom = dom_by_id.get(s["targetId"])
        rt, rlon, rlat = sub["t"].to_numpy(), sub["lon"].to_numpy(), sub["lat"].to_numpy()
        rsog, rcog = sub["sog"].to_numpy(), sub["cog"].to_numpy()
        t0, t1 = float(rt.min()), float(rt.max())
        cand = [m for m in _cand_mmsis(ais, rlon, rlat, t0, t1) if not (s["withheld"] and m == dom)]
        conf = _confident_match(ais, cand, rlon, rlat, rsog, rcog, rt, {}, float(rlat.mean()), cfg)
        y.append(1 if bool(s["dark_label"]) else 0)
        lsc.append(ds); hsc.append(0.0 if conf else ds)
    if len(set(y)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, lsc)), float(roc_auc_score(y, hsc))


def robustness(model, ctx):
    ais, rad, cfg, tl = ctx["ais"], ctx["rad"], ctx["cfg"], ctx["tl"]
    rad_by_t = rad.sort_values("t")
    ho = tl[(tl["split"].isin(["val", "test"])) & tl["track_matched"] & tl["dom_mmsi"].notna()]
    learned = {o: [] for o in OFFSETS_M}; hybrid = {o: [] for o in OFFSETS_M}
    for _, row in ho.iterrows():
        sub = rad_by_t[rad_by_t["targetId"] == row["targetId"]]
        if len(sub) < 2:
            continue
        rt, rlon, rlat = sub["t"].to_numpy(), sub["lon"].to_numpy(), sub["lat"].to_numpy()
        rsog, rcog = sub["sog"].to_numpy(), sub["cog"].to_numpy()
        ref_lon, ref_lat = float(rlon.mean()), float(rlat.mean())
        t0, t1 = float(rt.min()), float(rt.max()); win = max(t1 - t0, 1.0)
        dom = row["dom_mmsi"]
        cands = [dom] + [m for m in _cand_mmsis(ais, rlon, rlat, t0, t1) if m != dom]
        sf, sm = _featurize(rt, rlon, rlat, rsog, rcog, ref_lon, ref_lat, t0, win)
        for off_m in OFFSETS_M:
            lfd, hfd = [], []
            for d in range(DIRS):
                ang = 2 * np.pi * d / DIRS
                off = (off_m * np.cos(ang), off_m * np.sin(ang))
                cf, cm, cv = _build_cands(ais, cands, ref_lon, ref_lat, t0, t1, win, {dom: off})
                ds = _ldark(model, sf, sm, cf, cm, cv)
                ddark = 1 if ds > 0.5 else 0
                conf = _confident_match(ais, cands, rlon, rlat, rsog, rcog, rt, {dom: off}, ref_lat, cfg)
                lfd.append(ddark); hfd.append(0 if conf else ddark)
            learned[off_m].append(np.mean(lfd)); hybrid[off_m].append(np.mean(hfd))
    return ([float(np.mean(learned[o])) for o in OFFSETS_M],
            [float(np.mean(hybrid[o])) for o in OFFSETS_M])


def geom_baseline(ctx):
    ais, rad, cfg, tl = ctx["ais"], ctx["rad"], ctx["cfg"], ctx["tl"]
    rad_by_t = rad.sort_values("t")
    ho = tl[(tl["split"].isin(["val", "test"])) & tl["track_matched"] & tl["dom_mmsi"].notna()]
    geom = {o: [] for o in OFFSETS_M}
    for _, row in ho.iterrows():
        sub = rad_by_t[rad_by_t["targetId"] == row["targetId"]]
        if len(sub) < 2:
            continue
        rt, rlon, rlat = sub["t"].to_numpy(), sub["lon"].to_numpy(), sub["lat"].to_numpy()
        rcog = sub["cog"].to_numpy(); ref_lat = float(rlat.mean())
        dom = row["dom_mmsi"]
        for off_m in OFFSETS_M:
            gfd = []
            for d in range(DIRS):
                ang = 2 * np.pi * d / DIRS
                off = (off_m * np.cos(ang), off_m * np.sin(ang))
                gfd.append(0 if _geom_match(ais, dom, rlon, rlat, rcog, rt, off, ref_lat, cfg) else 1)
            geom[off_m].append(np.mean(gfd))
    fd = [float(np.mean(geom[o])) for o in OFFSETS_M]
    dom_by_id = dict(zip(tl["targetId"], tl["dom_mmsi"]))
    test = [s for s in ctx["samples"] if s["split"] == "test" and pd.notna(s["dark_label"])]
    y, g = [], []
    for s in test:
        sub = rad_by_t[rad_by_t["targetId"] == s["targetId"]]
        g.append(geometric_dark_score(ais, sub, dom_by_id.get(s["targetId"]), s["withheld"], cfg))
        y.append(1 if bool(s["dark_label"]) else 0)
    auc = float(roc_auc_score(y, g)) if len(set(y)) == 2 else float("nan")
    return fd, auc


def main():
    ctx = prep()
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        m = train_fusion(ctx, True, True, seed=0, epochs=20)
        la, ha = clean(m, ctx)
        print(f"SMOKE han_full(20ep): learned_clean={la:.3f} hybrid_clean={ha:.3f}")
        return
    geom_fd, geom_auc = geom_baseline(ctx)
    print(f"geometric: clean AUROC {geom_auc:.3f} | FD@500m {geom_fd[-1]:.2f}")
    results = {}
    for name, ut, uc in CONFIGS:
        la_, ha_, lc_, hc_ = [], [], [], []
        for seed in range(K_SEEDS):
            model = train_fusion(ctx, ut, uc, seed=seed)
            la, ha = clean(model, ctx); lc, hc = robustness(model, ctx)
            la_.append(la); ha_.append(ha); lc_.append(lc); hc_.append(hc)
            print(f"[{name}] seed{seed}: learned_clean={la:.3f} hybrid_clean={ha:.3f} | "
                  f"learnedFD@500={lc[-1]:.2f} hybridFD@500={hc[-1]:.2f} "
                  f"learnedFD@0={lc[0]:.2f} hybridFD@0={hc[0]:.2f}")
        L, H = np.array(lc_), np.array(hc_)
        results[name] = dict(
            learned_clean_auroc_mean=float(np.nanmean(la_)), learned_clean_auroc_std=float(np.nanstd(la_)),
            hybrid_clean_auroc_mean=float(np.nanmean(ha_)), hybrid_clean_auroc_std=float(np.nanstd(ha_)),
            learned_fd_mean=L.mean(0).tolist(), learned_fd_std=L.std(0).tolist(),
            hybrid_fd_mean=H.mean(0).tolist(), hybrid_fd_std=H.std(0).tolist())
    res = {"k_seeds": K_SEEDS, "offsets_m": OFFSETS_M,
           "geometric_fd": geom_fd, "geometric_clean_auroc": geom_auc,
           "configs": results,
           "reference_prev": {"cosine_multiseed_clean_auroc": 0.80, "hybrid_v2_clean_auroc": 0.944,
                              "hybrid_v2_fd_500m": 0.14, "learned_fd_floor": 0.19},
           "summary": ("Ablation of Han's Time2Vec + cross-attention on the open-set matcher. "
                       "Question: does fusion lower the learned false-alarm floor (~0.19) and/or "
                       "raise clean AUROC, pushing the hybrid below the 0.10 target? n_test=19/seed.")}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "busan_fusion_ablation.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(OFFSETS_M, geom_fd, "o--", color="#888", label="Geometric")
        cols = {"cosine_baseline": "#999", "t2v": "#e67e22", "cross": "#2471a3", "han_full": "#1e8449"}
        for name in cols:
            ax[0].plot(OFFSETS_M, results[name]["learned_fd_mean"], "s-", color=cols[name], label=f"{name} (learned)")
        ax[0].set_xlabel("AIS offset (m)"); ax[0].set_ylabel("Learned false-dark rate")
        ax[0].set_title("Learned-arm robustness by config"); ax[0].legend(fontsize=7)
        names = list(cols)
        lac = [results[n]["learned_clean_auroc_mean"] for n in names]
        hac = [results[n]["hybrid_clean_auroc_mean"] for n in names]
        x = np.arange(len(names))
        ax[1].bar(x - 0.2, lac, 0.4, label="learned", color="#2471a3")
        ax[1].bar(x + 0.2, hac, 0.4, label="hybrid", color="#1e8449")
        ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=20, fontsize=8)
        ax[1].set_ylim(0, 1.05); ax[1].set_ylabel("Clean dark AUROC"); ax[1].legend()
        ax[1].set_title("Clean AUROC by config")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "fig_fusion_ablation.png"), dpi=150); plt.close()
        print("figure: outputs/fig_fusion_ablation.png")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
