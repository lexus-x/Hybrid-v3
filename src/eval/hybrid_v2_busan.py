"""W5 v2: congestion-robust hybrid router + unified score + pooled calibration.

Fixes the three v1 failures (see outputs/busan_hybrid.json):
 1. Congestion fooled the residual-only router at 500 m -> v2 confidence requires a
    TIGHT multi-cue match (distance<=r_close AND angle<=ang_tight AND |Δsog|<=sog_tol).
    A random nearby boat in a crowd no longer counts as "confident".
 2. Score-scale mixing hurt clean AUROC -> v2 uses ONE coherent score: confident =>
    0 (a tight match was found, not dark); uncertain => the learned dark probability.
 3. Temperature fit on the tiny per-seed val set was degenerate (T=1) -> v2 pools
    val logits across ALL seeds, fits one T, evaluates ECE on pooled test.

Router rule per track: if some AIS candidate has an in-gate point that is close,
course-aligned AND speed-consistent -> trust "matched / not dark"; else defer to
the offset-invariant learned matcher. Expectation: hybrid ~ min(geometric, learned)
at every offset (best-of-both envelope), and clean AUROC above the pure learned model.

Run:  python -m eval.hybrid_v2_busan
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from common.geo import angdiff_deg, haversine_m
from eval.robustness_busan import (DIRS, OFFSETS_M, _build_cands, _cand_mmsis,
                                   _geom_match, _m_to_deg)
from eval.strengthen_busan import ece, geometric_dark_score
from p1_openset_darkdet.dataset import _featurize
from p1_openset_darkdet.train_eval_busan import (DEV, encode_sample, logits_for,
                                                 train_model)

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
K_SEEDS = 5
R_CLOSE = 80.0    # m   (true-match P50 46 m, P90 106 m -> 80 m is a tight band)
ANG_TIGHT = 25.0  # deg (full gate is 60; a TRUE match is much tighter)
SOG_TOL = 3.0     # kn  (speed must agree, not just position/heading)


def _confident_match(ais, cand_mmsis, rlon, rlat, rsog, rcog, rt, offsets, ref_lat, cfg):
    """True iff SOME candidate has an in-gate point that is close + course-aligned +
    speed-consistent with the radar track -> the association is unambiguous."""
    for m in cand_mmsis:
        sel = (ais["mmsi"] == m)
        if not sel.any():
            continue
        al = ais.loc[sel, "lon"].to_numpy().copy()
        aa = ais.loc[sel, "lat"].to_numpy().copy()
        ac = ais.loc[sel, "cog"].to_numpy()
        asog = ais.loc[sel, "sog"].to_numpy()
        at = ais.loc[sel, "t"].to_numpy()
        if offsets and m in offsets:
            de, dn = _m_to_deg(offsets[m][0], offsets[m][1], ref_lat)
            al = al + de; aa = aa + dn
        for i in range(len(rlon)):
            dt = np.abs(rt[i] - at)
            mk = dt <= cfg.dt_tol
            if not mk.any():
                continue
            d = haversine_m(rlon[i], rlat[i], al[mk], aa[mk])
            ang = angdiff_deg(rcog[i], ac[mk])
            dsog = np.abs(rsog[i] - asog[mk])
            if ((d <= R_CLOSE) & (ang <= ANG_TIGHT) & (dsog <= SOG_TOL)).any():
                return True
    return False


def curves_for_model(model, ctx):
    ais, rad, cfg, tl = ctx["ais"], ctx["rad"], ctx["cfg"], ctx["tl"]
    rad_by_t = rad.sort_values("t")
    ho = tl[(tl["split"].isin(["val", "test"])) & tl["track_matched"] & tl["dom_mmsi"].notna()]
    geom = {o: [] for o in OFFSETS_M}
    learned = {o: [] for o in OFFSETS_M}
    hybrid = {o: [] for o in OFFSETS_M}
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
            lfd, gfd, hfd = [], [], []
            for d in range(DIRS):
                ang = 2 * np.pi * d / DIRS
                off = (off_m * np.cos(ang), off_m * np.sin(ang))
                cf, cm, cv = _build_cands(ais, cands, ref_lon, ref_lat, t0, t1, win, {dom: off})
                safe = cm.copy(); safe[~cv, 0] = True
                a = model.encode_ais(torch.as_tensor(cf).float().to(DEV), torch.as_tensor(safe).to(DEV))
                ds = model.dark_score(r, a, torch.as_tensor(cv).to(DEV))
                ddark = 1 if (np.isfinite(ds) and ds > 0.5) else 0
                gdark = 0 if _geom_match(ais, dom, rlon, rlat, rcog, rt, off, ref_lat, cfg) else 1
                conf = _confident_match(ais, cands, rlon, rlat, rsog, rcog, rt, {dom: off}, ref_lat, cfg)
                hdark = 0 if conf else ddark   # tight match found => not dark; else defer to learned
                lfd.append(ddark); gfd.append(gdark); hfd.append(hdark)
            learned[off_m].append(np.mean(lfd))
            geom[off_m].append(np.mean(gfd))
            hybrid[off_m].append(np.mean(hfd))
    g = [float(np.mean(geom[o])) for o in OFFSETS_M]
    l = [float(np.mean(learned[o])) for o in OFFSETS_M]
    h = [float(np.mean(hybrid[o])) for o in OFFSETS_M]
    return g, l, h


def clean_eval(model, ctx):
    ais, rad, cfg, tl = ctx["ais"], ctx["rad"], ctx["cfg"], ctx["tl"]
    rad_by_t = rad.sort_values("t")
    dom_by_id = dict(zip(tl["targetId"], tl["dom_mmsi"]))
    test = [s for s in ctx["samples"] if s["split"] == "test" and pd.notna(s["dark_label"])]
    y, gsc, lsc, hsc, logits_list = [], [], [], [], []
    for s in test:
        r, a, valid = encode_sample(model, s)
        ds = model.dark_score(r, a, valid)
        ds = 0.5 if not np.isfinite(ds) else float(ds)
        sub = rad_by_t[rad_by_t["targetId"] == s["targetId"]]
        dom = dom_by_id.get(s["targetId"])
        gs = geometric_dark_score(ais, sub, dom, s["withheld"], cfg)
        rt, rlon, rlat = sub["t"].to_numpy(), sub["lon"].to_numpy(), sub["lat"].to_numpy()
        rsog, rcog = sub["sog"].to_numpy(), sub["cog"].to_numpy()
        t0, t1 = float(rt.min()), float(rt.max())
        cand = [m for m in _cand_mmsis(ais, rlon, rlat, t0, t1)
                if not (s["withheld"] and m == dom)]
        conf = _confident_match(ais, cand, rlon, rlat, rsog, rcog, rt, {}, float(rlat.mean()), cfg)
        hs = 0.0 if conf else ds   # tight match => not dark; else learned probability
        y.append(1 if bool(s["dark_label"]) else 0)
        gsc.append(gs); lsc.append(ds); hsc.append(hs)
        lg, _ = logits_for(model, s)
        logits_list.append(lg.detach().cpu().numpy())
    return np.array(y), np.array(gsc), np.array(lsc), np.array(hsc), logits_list


def _val_logits(model, ctx):
    val = [s for s in ctx["samples"] if s["split"] == "val" and pd.notna(s["dark_label"])]
    L, y = [], []
    for s in val:
        lg, _ = logits_for(model, s)
        L.append(lg.detach().cpu().numpy()); y.append(1 if bool(s["dark_label"]) else 0)
    return L, y


def _p_dark(logits, T):
    z = torch.as_tensor(logits, dtype=torch.float32) / T
    return float(F.softmax(z, dim=0)[-1].item())


def fit_temperature(val_logits, val_y):
    val_y = np.asarray(val_y)
    if len(val_y) == 0 or len(set(val_y.tolist())) < 2:
        return 1.0
    best_T, best_nll = 1.0, np.inf
    for T in np.linspace(0.3, 8.0, 78):
        p = np.clip([_p_dark(lg, T) for lg in val_logits], 1e-6, 1 - 1e-6)
        nll = -np.mean(val_y * np.log(p) + (1 - val_y) * np.log(1 - p))
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T


def main():
    geomc, learnc, hybc = [], [], []
    g_auc, l_auc, h_auc = [], [], []
    pool_val_L, pool_val_y = [], []          # pooled across seeds for calibration
    pool_test_L, pool_test_y = [], []
    for seed in range(K_SEEDS):
        model, ctx = train_model(seed=seed)
        g, l, h = curves_for_model(model, ctx)
        geomc.append(g); learnc.append(l); hybc.append(h)
        y, gsc, lsc, hsc, logits_list = clean_eval(model, ctx)
        if len(set(y.tolist())) == 2:
            g_auc.append(roc_auc_score(y, gsc))
            l_auc.append(roc_auc_score(y, lsc))
            h_auc.append(roc_auc_score(y, hsc))
        vL, vy = _val_logits(model, ctx)
        pool_val_L += vL; pool_val_y += list(vy)
        pool_test_L += logits_list; pool_test_y += list(y)
        print(f"seed {seed}: clean AUROC geom={g_auc[-1]:.3f} learned={l_auc[-1]:.3f} "
              f"HYBRID={h_auc[-1]:.3f} | FD@500m geom={g[-1]:.2f} learned={l[-1]:.2f} HYBRID={h[-1]:.2f}")

    # pooled temperature scaling: fit on pooled val, evaluate ECE on pooled test
    T = fit_temperature(pool_val_L, pool_val_y)
    ty = np.asarray(pool_test_y)
    ece_before = ece(ty, np.array([_p_dark(lg, 1.0) for lg in pool_test_L])) if len(set(ty.tolist())) == 2 else None
    ece_after = ece(ty, np.array([_p_dark(lg, T) for lg in pool_test_L])) if len(set(ty.tolist())) == 2 else None

    G, L, H = np.array(geomc), np.array(learnc), np.array(hybc)
    res = {
        "version": "v2", "k_seeds": K_SEEDS,
        "router": {"r_close_m": R_CLOSE, "ang_tight_deg": ANG_TIGHT, "sog_tol_kn": SOG_TOL},
        "offsets_m": OFFSETS_M,
        "false_dark": {
            "geometric_mean": G.mean(0).tolist(),
            "learned_mean": L.mean(0).tolist(), "learned_std": L.std(0).tolist(),
            "hybrid_mean": H.mean(0).tolist(), "hybrid_std": H.std(0).tolist(),
        },
        "clean_auroc": {
            "geometric_mean": float(np.mean(g_auc)), "geometric_std": float(np.std(g_auc)),
            "learned_mean": float(np.mean(l_auc)), "learned_std": float(np.std(l_auc)),
            "hybrid_mean": float(np.mean(h_auc)), "hybrid_std": float(np.std(h_auc)),
        },
        "calibration_pooled": {
            "n_test_pooled": int(len(ty)), "temperature": float(T),
            "ece_before": ece_before, "ece_after": ece_after,
        },
        "summary": ("v2 tight multi-cue router (close+course+speed). Hybrid aims for the best-of-both "
                    "envelope: ~geometric at low AIS error, ~learned at high error, never worse than "
                    "either; clean AUROC above pure learned. Calibration fit on pooled val. n_test=19/seed."),
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "busan_hybrid_v2.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
        ax[0].plot(OFFSETS_M, G.mean(0), "o-", color="#c0392b", label="Geometric")
        ax[0].plot(OFFSETS_M, L.mean(0), "s-", color="#2471a3", label="Learned")
        ax[0].plot(OFFSETS_M, H.mean(0), "^-", color="#1e8449", lw=2.4, label="Hybrid v2 (ours)")
        ax[0].fill_between(OFFSETS_M, np.clip(H.mean(0) - H.std(0), 0, 1),
                           np.clip(H.mean(0) + H.std(0), 0, 1), color="#1e8449", alpha=0.18)
        ax[0].set_xlabel("Injected AIS registration offset (m)")
        ax[0].set_ylabel("False-dark rate (AIS present)")
        ax[0].set_title(f"Robustness v2 ({K_SEEDS} seeds)"); ax[0].legend()
        names = ["Geometric", "Learned", "Hybrid v2"]
        vals = [res["clean_auroc"]["geometric_mean"], res["clean_auroc"]["learned_mean"],
                res["clean_auroc"]["hybrid_mean"]]
        errs = [res["clean_auroc"]["geometric_std"], res["clean_auroc"]["learned_std"],
                res["clean_auroc"]["hybrid_std"]]
        ax[1].bar(names, vals, yerr=errs, capsize=6, color=["#c0392b", "#2471a3", "#1e8449"])
        ax[1].set_ylim(0, 1.05); ax[1].set_ylabel("Clean-split dark AUROC")
        ax[1].set_title("Clean-split head-to-head (n=19)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "fig_hybrid_v2.png"), dpi=150); plt.close()
        print("figure: outputs/fig_hybrid_v2.png")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
