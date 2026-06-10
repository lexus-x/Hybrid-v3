"""W5 v3: joint relative kinematics hybrid router + unified score + pooled calibration.

Upgrades the router and underlying learned matcher to use translation-invariant 
relative ENU kinematics features.
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
from eval.robustness_busan import (DIRS, OFFSETS_M, _cand_mmsis, _geom_match, _m_to_deg)
from eval.strengthen_busan import ece, geometric_dark_score
from p1_openset_darkdet.train_relative import DEV, train_relative_model
from p1_openset_darkdet.relative_features import compute_relative_features
from p1_openset_darkdet.dataset import MAX_CAND, MAX_LEN, CAND_DT

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


def build_relative_cands(ais, mmsi_list, rt, rlon, rlat, rsog, rcog, t0, t1, ref_lat, offsets=None):
    """offsets: dict mmsi-> (east_m, north_m) registration offset to inject."""
    cf = np.zeros((MAX_CAND, MAX_LEN, 6), np.float32)
    cm = np.zeros((MAX_CAND, MAX_LEN), bool)
    cv = np.zeros(MAX_CAND, bool)
    for k, m in enumerate(mmsi_list[:MAX_CAND]):
        sel = (ais["mmsi"] == m) & (ais["t"] >= t0 - CAND_DT) & (ais["t"] <= t1 + CAND_DT)
        if sel.sum() < 1:
            continue
        lon = ais.loc[sel, "lon"].to_numpy().copy()
        lat = ais.loc[sel, "lat"].to_numpy().copy()
        if offsets and m in offsets:
            de, dn = _m_to_deg(offsets[m][0], offsets[m][1], ref_lat)
            lon = lon + de; lat = lat + dn
        f, msk = compute_relative_features(
            rt, rlon, rlat, rsog, rcog,
            ais.loc[sel, "t"].to_numpy(), lon, lat,
            ais.loc[sel, "sog"].to_numpy(), ais.loc[sel, "cog"].to_numpy()
        )
        cf[k], cm[k], cv[k] = f, msk, True
    return cf, cm, cv


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
        
        for off_m in OFFSETS_M:
            lfd, gfd, hfd = [], [], []
            for d in range(DIRS):
                ang = 2 * np.pi * d / DIRS
                off = (off_m * np.cos(ang), off_m * np.sin(ang))
                cf, cm, cv = build_relative_cands(ais, cands, rt, rlon, rlat, rsog, rcog, t0, t1, ref_lat, {dom: off})
                ds = model.dark_score(
                    torch.as_tensor(cf).float().to(DEV),
                    torch.as_tensor(cm).to(DEV),
                    torch.as_tensor(cv).to(DEV)
                )
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
        cf = torch.as_tensor(s["cand_feats"]).float().to(DEV)
        cm = torch.as_tensor(s["cand_pmask"]).to(DEV)
        valid = torch.as_tensor(s["cand_valid"]).to(DEV)
        
        ds = model.dark_score(cf, cm, valid)
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
        lg = model.match_logits(cf, cm, valid)
        logits_list.append(lg.detach().cpu().numpy())
    return np.array(y), np.array(gsc), np.array(lsc), np.array(hsc), logits_list


def _val_logits(model, ctx):
    val = [s for s in ctx["samples"] if s["split"] == "val" and pd.notna(s["dark_label"])]
    L, y = [], []
    for s in val:
        cf = torch.as_tensor(s["cand_feats"]).float().to(DEV)
        cm = torch.as_tensor(s["cand_pmask"]).to(DEV)
        valid = torch.as_tensor(s["cand_valid"]).to(DEV)
        lg = model.match_logits(cf, cm, valid)
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


def run_seed_worker(seed):
    print(f"[Worker] Starting seed {seed}")
    model, ctx = train_relative_model(seed=seed)
    print(f"[Worker] Evaluating curves for seed {seed}")
    g, l, h = curves_for_model(model, ctx)
    print(f"[Worker] Evaluating clean split for seed {seed}")
    y, gsc, lsc, hsc, logits_list = clean_eval(model, ctx)
    print(f"[Worker] Gathering validation logits for seed {seed}")
    vL, vy = _val_logits(model, ctx)
    print(f"[Worker] Finished seed {seed}")
    return {
        "seed": seed,
        "g": g,
        "l": l,
        "h": h,
        "y": y.tolist() if hasattr(y, 'tolist') else list(y),
        "gsc": gsc.tolist() if hasattr(gsc, 'tolist') else list(gsc),
        "lsc": lsc.tolist() if hasattr(lsc, 'tolist') else list(lsc),
        "hsc": hsc.tolist() if hasattr(hsc, 'tolist') else list(hsc),
        "logits_list": [lg.tolist() if hasattr(lg, 'tolist') else list(lg) for lg in logits_list],
        "vL": [lg.tolist() if hasattr(lg, 'tolist') else list(lg) for lg in vL],
        "vy": vy
    }


def main():
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print(f"Starting parallel evaluation over {K_SEEDS} seeds using {K_SEEDS} processes...")
    with multiprocessing.Pool(processes=K_SEEDS) as pool:
        results = pool.map(run_seed_worker, range(K_SEEDS))

    # Sort results by seed just in case
    results = sorted(results, key=lambda x: x["seed"])

    geomc, learnc, hybc = [], [], []
    g_auc, l_auc, h_auc = [], [], []
    pool_val_L, pool_val_y = [], []          # pooled across seeds for calibration
    pool_test_L, pool_test_y = [], []

    for r in results:
        seed = r["seed"]
        g, l, h = r["g"], r["l"], r["h"]
        y = np.array(r["y"])
        gsc = np.array(r["gsc"])
        lsc = np.array(r["lsc"])
        hsc = np.array(r["hsc"])
        logits_list = [np.array(lg) for lg in r["logits_list"]]
        vL = [np.array(lg) for lg in r["vL"]]
        vy = r["vy"]

        geomc.append(g); learnc.append(l); hybc.append(h)
        if len(set(y.tolist())) == 2:
            g_auc.append(roc_auc_score(y, gsc))
            l_auc.append(roc_auc_score(y, lsc))
            h_auc.append(roc_auc_score(y, hsc))
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
        "version": "v3", "k_seeds": K_SEEDS,
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
        "summary": ("v3 joint relative kinematics router. Robustness eval over 5 seeds."),
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "busan_hybrid_v3.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
        ax[0].plot(OFFSETS_M, G.mean(0), "o-", color="#c0392b", label="Geometric")
        ax[0].plot(OFFSETS_M, L.mean(0), "s-", color="#2471a3", label="Learned")
        ax[0].plot(OFFSETS_M, H.mean(0), "^-", color="#1e8449", lw=2.4, label="Hybrid v3 (ours)")
        ax[0].fill_between(OFFSETS_M, np.clip(H.mean(0) - H.std(0), 0, 1),
                           np.clip(H.mean(0) + H.std(0), 0, 1), color="#1e8449", alpha=0.18)
        ax[0].set_xlabel("Injected AIS registration offset (m)")
        ax[0].set_ylabel("False-dark rate (AIS present)")
        ax[0].set_title(f"Robustness v3 ({K_SEEDS} seeds)"); ax[0].legend()
        names = ["Geometric", "Learned", "Hybrid v3"]
        vals = [res["clean_auroc"]["geometric_mean"], res["clean_auroc"]["learned_mean"],
                res["clean_auroc"]["hybrid_mean"]]
        errs = [res["clean_auroc"]["geometric_std"], res["clean_auroc"]["learned_std"],
                res["clean_auroc"]["hybrid_std"]]
        ax[1].bar(names, vals, yerr=errs, capsize=6, color=["#c0392b", "#2471a3", "#1e8449"])
        ax[1].set_ylim(0, 1.05); ax[1].set_ylabel("Clean-split dark AUROC")
        ax[1].set_title("Clean-split head-to-head (n=19)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "fig_hybrid_v3.png"), dpi=150); plt.close()
        print("figure: outputs/fig_hybrid_v3.png")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
