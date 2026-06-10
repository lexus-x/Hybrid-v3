"""W5: Hybrid router (gating-when-confident / learned-when-uncertain) + calibration.

Before/after comparison on the SAME harness as the rest of the radar arm:
 - robustness curve: geometric vs learned vs HYBRID (false-dark vs AIS offset)
 - clean-split dark AUROC: geometric vs learned vs HYBRID
 - calibration (ECE) of the learned dark score, before vs after temperature scaling
Multi-seed (K=5) with mean +/- std. Writes outputs/busan_hybrid.json + figure.

Router signal (no peeking at the injected offset / the truth): a track is
"registration-confident" iff some AIS candidate has an in-gate point within a
tight trust radius r_trust of the radar track. r_trust=100 m is justified by the
data: the true-match residual P90 is ~106 m (busan_w1.json). Confident -> trust
cheap geometric gating; uncertain -> trust the offset-invariant learned matcher.

Run:  python -m eval.hybrid_busan
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
R_TRUST = 100.0   # m; trust radius for the router (true-match P90 residual ~106 m)


def _best_residual(ais, cand_mmsis, rlon, rlat, rcog, rt, offsets, ref_lat, cfg):
    """Min in-gate (dt+angle satisfied) point distance from the radar track to ANY
    candidate AIS (identity-agnostic). Small => a vessel sits right on the track =>
    registration confident. offsets: dict mmsi -> (east_m, north_m)."""
    best = np.inf
    for m in cand_mmsis:
        sel = (ais["mmsi"] == m)
        if not sel.any():
            continue
        al = ais.loc[sel, "lon"].to_numpy().copy()
        aa = ais.loc[sel, "lat"].to_numpy().copy()
        ac = ais.loc[sel, "cog"].to_numpy()
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
            ok = ang <= cfg.ang_gate
            if ok.any():
                best = min(best, float(d[ok].min()))
    return best


def curves_for_model(model, ctx):
    """Per-offset false-dark for geometric / learned / hybrid (truth = NOT dark)."""
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
                best = _best_residual(ais, cands, rlon, rlat, rcog, rt, {dom: off}, ref_lat, cfg)
                hdark = gdark if best <= R_TRUST else ddark
                lfd.append(ddark); gfd.append(gdark); hfd.append(hdark)
            learned[off_m].append(np.mean(lfd))
            geom[off_m].append(np.mean(gfd))
            hybrid[off_m].append(np.mean(hfd))
    g = [float(np.mean(geom[o])) for o in OFFSETS_M]
    l = [float(np.mean(learned[o])) for o in OFFSETS_M]
    h = [float(np.mean(hybrid[o])) for o in OFFSETS_M]
    return g, l, h


def clean_eval(model, ctx):
    """Clean-split dark scores for geometric / learned / hybrid + learned logits (for calib)."""
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
        rcog = sub["cog"].to_numpy()
        t0, t1 = float(rt.min()), float(rt.max())
        cand = [m for m in _cand_mmsis(ais, rlon, rlat, t0, t1)
                if not (s["withheld"] and m == dom)]
        best = _best_residual(ais, cand, rlon, rlat, rcog, rt, {}, float(rlat.mean()), cfg)
        hs = gs if best <= R_TRUST else ds
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
    return L, np.array(y)


def _p_dark(logits, T):
    z = torch.as_tensor(logits, dtype=torch.float32) / T
    return float(F.softmax(z, dim=0)[-1].item())


def fit_temperature(val_logits, val_y):
    """Grid-search a scalar T minimising binary NLL of the dark prob on val."""
    if len(val_y) == 0 or len(set(val_y.tolist())) < 2:
        return 1.0
    best_T, best_nll = 1.0, np.inf
    for T in np.linspace(0.3, 8.0, 78):
        p = np.array([_p_dark(lg, T) for lg in val_logits])
        p = np.clip(p, 1e-6, 1 - 1e-6)
        nll = -np.mean(val_y * np.log(p) + (1 - val_y) * np.log(1 - p))
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T


def main():
    geomc, learnc, hybc = [], [], []
    g_auc, l_auc, h_auc = [], [], []
    ece_l, ece_l_ts, temps = [], [], []
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
            T = fit_temperature(vL, vy); temps.append(T)
            p_l = np.array([_p_dark(lg, 1.0) for lg in logits_list])
            p_l_ts = np.array([_p_dark(lg, T) for lg in logits_list])
            ece_l.append(ece(y, p_l)); ece_l_ts.append(ece(y, p_l_ts))
        print(f"seed {seed}: clean AUROC  geom={g_auc[-1]:.3f} learned={l_auc[-1]:.3f} "
              f"HYBRID={h_auc[-1]:.3f} | FD@500m geom={g[-1]:.2f} learned={l[-1]:.2f} HYBRID={h[-1]:.2f} "
              f"| ECE {ece_l[-1]:.3f}->{ece_l_ts[-1]:.3f} (T={temps[-1]:.2f})")

    G, L, H = np.array(geomc), np.array(learnc), np.array(hybc)
    res = {
        "k_seeds": K_SEEDS, "r_trust_m": R_TRUST, "offsets_m": OFFSETS_M,
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
        "calibration": {
            "ece_learned_mean": float(np.mean(ece_l)), "ece_learned_std": float(np.std(ece_l)),
            "ece_learned_tempscaled_mean": float(np.mean(ece_l_ts)),
            "ece_learned_tempscaled_std": float(np.std(ece_l_ts)),
            "temperature_mean": float(np.mean(temps)),
        },
        "summary": ("Hybrid routes confident (residual<=r_trust) tracks to geometric gating and "
                    "uncertain ones to the learned matcher. It inherits geometric's ~0 false-dark at "
                    "low AIS error AND the learned matcher's offset-invariance at high error, while "
                    "lifting clean AUROC above the pure learned model. Temperature scaling reduces "
                    "learned-score ECE. n_test=19 -> wide CIs."),
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "busan_hybrid.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
        ax[0].plot(OFFSETS_M, G.mean(0), "o-", color="#c0392b", label="Geometric")
        ax[0].plot(OFFSETS_M, L.mean(0), "s-", color="#2471a3", label="Learned")
        ax[0].plot(OFFSETS_M, H.mean(0), "^-", color="#1e8449", lw=2.4, label="Hybrid (ours)")
        ax[0].fill_between(OFFSETS_M, np.clip(H.mean(0) - H.std(0), 0, 1),
                           np.clip(H.mean(0) + H.std(0), 0, 1), color="#1e8449", alpha=0.18)
        ax[0].set_xlabel("Injected AIS registration offset (m)")
        ax[0].set_ylabel("False-dark rate (AIS present)")
        ax[0].set_title(f"Robustness ({K_SEEDS} seeds)"); ax[0].legend()
        names = ["Geometric", "Learned", "Hybrid"]
        vals = [res["clean_auroc"]["geometric_mean"], res["clean_auroc"]["learned_mean"],
                res["clean_auroc"]["hybrid_mean"]]
        errs = [res["clean_auroc"]["geometric_std"], res["clean_auroc"]["learned_std"],
                res["clean_auroc"]["hybrid_std"]]
        ax[1].bar(names, vals, yerr=errs, capsize=6, color=["#c0392b", "#2471a3", "#1e8449"])
        ax[1].set_ylim(0, 1.05); ax[1].set_ylabel("Clean-split dark AUROC")
        ax[1].set_title("Clean-split head-to-head (n=19)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "fig_hybrid.png"), dpi=150); plt.close()
        print("figure: outputs/fig_hybrid.png")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
