"""Multi-seed rigor for the Busan radar arm (no new data, pure compute).

Trains the open-set matcher over K seeds and reports, with mean +/- std error
bands: (a) the robustness curve (learned false-dark vs injected AIS offset) and
(b) the clean-split dark-detection AUROC. Geometric baseline is model-independent
so it is computed once. Turns the single-run, wide-CI numbers into averaged,
reproducible evidence. Writes outputs/busan_multiseed.json + fig_robustness_multiseed.png.

Run:  python -m eval.multiseed_busan
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from eval.robustness_busan import (DIRS, OFFSETS_M, _build_cands, _cand_mmsis,
                                   _geom_match)
from eval.strengthen_busan import geometric_dark_score
from p1_openset_darkdet.dataset import _featurize
from p1_openset_darkdet.train_eval_busan import DEV, encode_sample, train_model

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
K_SEEDS = 5


def heldout_tracks(ctx):
    tl = ctx["tl"]
    return tl[(tl["split"].isin(["val", "test"])) & tl["track_matched"] & tl["dom_mmsi"].notna()]


def robustness_for_model(model, ctx, want_geom=False):
    ais, rad, cfg = ctx["ais"], ctx["rad"], ctx["cfg"]
    rad_by_t = rad.sort_values("t")
    ho = heldout_tracks(ctx)
    learned = {o: [] for o in OFFSETS_M}
    geom = {o: [] for o in OFFSETS_M}
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
        r = model.encode_sensor(torch.as_tensor(sf).unsqueeze(0).float().to(DEV),
                                torch.as_tensor(sm).unsqueeze(0).to(DEV))[0]
        for off_m in OFFSETS_M:
            lfd, gfd = [], []
            for d in range(DIRS):
                ang = 2 * np.pi * d / DIRS
                off = (off_m * np.cos(ang), off_m * np.sin(ang))
                cf, cm, cv = _build_cands(ais, cands, ref_lon, ref_lat, t0, t1, win, {dom: off})
                safe = cm.copy(); safe[~cv, 0] = True
                a = model.encode_ais(torch.as_tensor(cf).float().to(DEV), torch.as_tensor(safe).to(DEV))
                ds = model.dark_score(r, a, torch.as_tensor(cv).to(DEV))
                lfd.append(1 if (np.isfinite(ds) and ds > 0.5) else 0)
                if want_geom:
                    gfd.append(0 if _geom_match(ais, dom, rlon, rlat, rcog, rt, off, ref_lat, cfg) else 1)
            learned[off_m].append(np.mean(lfd))
            if want_geom:
                geom[off_m].append(np.mean(gfd))
    lc = {o: float(np.mean(v)) for o, v in learned.items()}
    gc = {o: float(np.mean(geom[o])) for o in OFFSETS_M} if want_geom else None
    return lc, gc


def clean_auroc(model, ctx):
    ais, rad, cfg = ctx["ais"], ctx["rad"], ctx["cfg"]
    rad_by_t = rad.sort_values("t")
    dom_by_id = dict(zip(ctx["tl"]["targetId"], ctx["tl"]["dom_mmsi"]))
    test = [s for s in ctx["samples"] if s["split"] == "test" and s["dark_label"] is not None
            and not (isinstance(s["dark_label"], float) and np.isnan(s["dark_label"]))]
    y, learned, geom = [], [], []
    for s in test:
        r, a, valid = encode_sample(model, s)
        ds = model.dark_score(r, a, valid)
        learned.append(0.5 if not np.isfinite(ds) else float(ds))
        sub = rad_by_t[rad_by_t["targetId"] == s["targetId"]]
        geom.append(geometric_dark_score(ais, sub, dom_by_id.get(s["targetId"]), s["withheld"], cfg))
        y.append(1 if bool(s["dark_label"]) else 0)
    la = roc_auc_score(y, learned) if len(set(y)) == 2 else float("nan")
    ga = roc_auc_score(y, geom) if len(set(y)) == 2 else float("nan")
    return float(la), float(ga)


def main():
    learned_curves, learned_aurocs = [], []
    geom_curve, geom_auroc = None, None
    for seed in range(K_SEEDS):
        model, ctx = train_model(seed=seed)
        lc, gc = robustness_for_model(model, ctx, want_geom=(seed == 0))
        la, ga = clean_auroc(model, ctx)
        learned_curves.append([lc[o] for o in OFFSETS_M])
        learned_aurocs.append(la)
        if seed == 0:
            geom_curve = [gc[o] for o in OFFSETS_M]; geom_auroc = ga
        print(f"seed {seed}: clean AUROC learned={la:.3f} | learned FD@500m={lc[OFFSETS_M[-1]]:.3f}")

    arr = np.array(learned_curves)
    res = {
        "offsets_m": OFFSETS_M, "k_seeds": K_SEEDS,
        "learned_false_dark_mean": arr.mean(0).tolist(),
        "learned_false_dark_std": arr.std(0).tolist(),
        "geometric_false_dark": geom_curve,
        "learned_clean_auroc_mean": float(np.mean(learned_aurocs)),
        "learned_clean_auroc_std": float(np.std(learned_aurocs)),
        "learned_clean_auroc_all": learned_aurocs,
        "geometric_clean_auroc": geom_auroc,
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "busan_multiseed.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        m, sd = arr.mean(0), arr.std(0)
        plt.figure(figsize=(5.2, 3.6))
        plt.plot(OFFSETS_M, geom_curve, "o-", color="#c0392b", label="Geometric gating")
        plt.plot(OFFSETS_M, m, "s-", color="#2471a3", label=f"Learned open-set (mean of {K_SEEDS} seeds)")
        plt.fill_between(OFFSETS_M, np.clip(m - sd, 0, 1), np.clip(m + sd, 0, 1), color="#2471a3", alpha=0.2)
        plt.xlabel("Injected AIS registration offset (m)"); plt.ylabel("False-dark rate (AIS present)")
        plt.title(f"Robustness (Busan, {K_SEEDS} seeds)"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(OUT, "fig_robustness_multiseed.png"), dpi=150); plt.close()
        print("figure: outputs/fig_robustness_multiseed.png")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
