"""Strengthen the Busan radar+AIS result before adding the camera arm:
 - clean-split dark-detection head-to-head (learned vs geometric baseline)
 - calibration (ECE) of the learned dark score
 - bootstrap 95% CIs (critical at n=19)
 - figures: robustness curve + head-to-head AUROC

Run:  python -m eval.strengthen_busan
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from common.geo import angdiff_deg, haversine_m
from p1_openset_darkdet.dataset import CAND_DT
from p1_openset_darkdet.train_eval_busan import DEV, encode_sample, train_model

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")


def geometric_dark_score(ais, sub, dom, withheld, cfg):
    """1 - (fraction of the track's radar points with ANY in-gate AIS match).
    Mirrors the rule baseline's track-level matched fraction, with dropout applied."""
    rt, rlon, rlat, rcog = (sub[c].to_numpy() for c in ("t", "lon", "lat", "cog"))
    m = (ais["t"] >= rt.min() - cfg.dt_tol) & (ais["t"] <= rt.max() + cfg.dt_tol)
    if withheld:
        m = m & (ais["mmsi"] != dom)
    at = ais.loc[m, "t"].to_numpy(); alon = ais.loc[m, "lon"].to_numpy()
    alat = ais.loc[m, "lat"].to_numpy(); acog = ais.loc[m, "cog"].to_numpy()
    if len(at) == 0:
        return 1.0
    order = np.argsort(at); at, alon, alat, acog = at[order], alon[order], alat[order], acog[order]
    hit = 0
    for i in range(len(rt)):
        lo = np.searchsorted(at, rt[i] - cfg.dt_tol); hi = np.searchsorted(at, rt[i] + cfg.dt_tol)
        if hi <= lo:
            continue
        d = haversine_m(rlon[i], rlat[i], alon[lo:hi], alat[lo:hi])
        ad = angdiff_deg(rcog[i], acog[lo:hi])
        if ((d <= cfg.dist_gate) & (ad <= cfg.ang_gate)).any():
            hit += 1
    return 1.0 - hit / len(rt)


def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for b in range(bins):
        m = (p >= edges[b]) & (p < edges[b + 1] if b < bins - 1 else p <= edges[b + 1])
        if m.sum() == 0:
            continue
        e += (m.sum() / len(y)) * abs(p[m].mean() - y[m].mean())
    return float(e)


def boot_auroc(y, s, n=2000, seed=0):
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(set(y[b].tolist())) < 2:
            continue
        vals.append(roc_auc_score(y[b], s[b]))
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def main():
    model, ctx = train_model()
    ais, rad, cfg, tl = ctx["ais"], ctx["rad"], ctx["cfg"], ctx["tl"]
    rad_by_t = rad.sort_values("t")
    dom_by_id = dict(zip(tl["targetId"], tl["dom_mmsi"]))
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

    res = {"n_test": len(y), "n_dark": int(sum(y))}
    if len(set(y)) == 2:
        res["learned"] = {"auroc": float(roc_auc_score(y, learned)),
                          "auprc": float(average_precision_score(y, learned)),
                          "auroc_ci95": boot_auroc(y, learned), "ece": ece(y, learned)}
        res["geometric"] = {"auroc": float(roc_auc_score(y, geom)),
                            "auprc": float(average_precision_score(y, geom)),
                            "auroc_ci95": boot_auroc(y, geom)}
    res["takeaway"] = ("On the CLEAN dropout split geometry is near-ideal (withholding removes its only "
                       "in-gate match) so it can match/beat the learned model here; the learned model's "
                       "advantage is the robustness regime (see busan_w4_robustness.json). n=19 -> wide CIs.")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "busan_strengthen.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))

    # ---- figures ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        wf = os.path.join(OUT, "busan_w4_robustness.json")
        if os.path.exists(wf):
            rob = json.load(open(wf))
            xs = [r["offset_m"] for r in rob["rows"]]
            plt.figure(figsize=(5, 3.4))
            plt.plot(xs, [r["geometric_false_dark_rate"] for r in rob["rows"]], "o-", label="Geometric gating")
            plt.plot(xs, [r["learned_false_dark_rate"] for r in rob["rows"]], "s-", label="Learned open-set")
            plt.axvline(cfg.dist_gate, ls=":", c="gray", lw=1, label="dist gate")
            plt.xlabel("Injected AIS registration offset (m)"); plt.ylabel("False-dark rate (AIS present)")
            plt.title("Robustness: false darks vs AIS offset (Busan)"); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(OUT, "fig_robustness.png"), dpi=150); plt.close()
        if "learned" in res:
            plt.figure(figsize=(3.6, 3.4))
            names = ["Geometric", "Learned"]
            aurocs = [res["geometric"]["auroc"], res["learned"]["auroc"]]
            cis = [res["geometric"]["auroc_ci95"], res["learned"]["auroc_ci95"]]
            err = [[a - c[0] for a, c in zip(aurocs, cis)], [c[1] - a for a, c in zip(aurocs, cis)]]
            plt.bar(names, aurocs, yerr=err, capsize=6, color=["#bbb", "#4a90d9"])
            plt.ylim(0, 1); plt.ylabel("Dark-detection AUROC (clean split)")
            plt.title("Clean-split head-to-head (n=19)"); plt.tight_layout()
            plt.savefig(os.path.join(OUT, "fig_clean_auroc.png"), dpi=150); plt.close()
        print("figures written to outputs/: fig_robustness.png, fig_clean_auroc.png")
    except Exception as e:
        print("figure step skipped:", e)


if __name__ == "__main__":
    main()
