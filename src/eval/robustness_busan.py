"""W4 robustness eval (Busan): the thesis test.

For held-out tracks whose true AIS IS present (truth = NOT dark), inject a
position registration offset of growing magnitude into that AIS, then ask each
method "is this track dark?". Geometric gating produces FALSE DARKS once the
offset approaches the distance gate; a learned matcher that has learned tolerance
should stay matched longer. We report false-dark rate vs offset for both.

Run:  python -m eval.robustness_busan
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from common.geo import angdiff_deg, haversine_m
from p1_openset_darkdet.dataset import (CAND_DIST, CAND_DT, F_DIM, MAX_CAND,
                                        MAX_LEN, _featurize)
from p1_openset_darkdet.train_eval_busan import DEV, train_model

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
OFFSETS_M = [0, 50, 100, 200, 300, 500]   # injected AIS registration offset (m)
DIRS = 4                                    # average over N offset directions


def _m_to_deg(east_m, north_m, lat):
    return (east_m / (111320.0 * np.cos(np.radians(lat))), north_m / 110540.0)


def _cand_mmsis(ais, rlon, rlat, t0, t1):
    sub = ais[(ais["t"] >= t0 - CAND_DT) & (ais["t"] <= t1 + CAND_DT)]
    out = set()
    sl, sa, sm = sub["lon"].to_numpy(), sub["lat"].to_numpy(), sub["mmsi"].to_numpy()
    for i in range(len(rlon)):
        d = haversine_m(rlon[i], rlat[i], sl, sa)
        out.update(sm[d <= CAND_DIST].tolist())
    return out


def _build_cands(ais, mmsi_list, ref_lon, ref_lat, t0, t1, win, offsets=None):
    """offsets: dict mmsi-> (east_m, north_m) registration offset to inject."""
    cf = np.zeros((MAX_CAND, MAX_LEN, F_DIM), np.float32)
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
        f, msk = _featurize(ais.loc[sel, "t"].to_numpy(), lon, lat,
                            ais.loc[sel, "sog"].to_numpy(), ais.loc[sel, "cog"].to_numpy(),
                            ref_lon, ref_lat, t0, win)
        cf[k], cm[k], cv[k] = f, msk, True
    return cf, cm, cv


def _geom_match(ais, mmsi, rlon, rlat, rcog, rt, off, ref_lat, cfg):
    sel = (ais["mmsi"] == mmsi)
    al, aa = ais.loc[sel, "lon"].to_numpy().copy(), ais.loc[sel, "lat"].to_numpy().copy()
    ac, at = ais.loc[sel, "cog"].to_numpy(), ais.loc[sel, "t"].to_numpy()
    de, dn = _m_to_deg(off[0], off[1], ref_lat)
    al, aa = al + de, aa + dn
    for i in range(len(rlon)):
        dt = np.abs(rt[i] - at)
        m = dt <= cfg.dt_tol
        if not m.any():
            continue
        d = haversine_m(rlon[i], rlat[i], al[m], aa[m])
        ang = angdiff_deg(rcog[i], ac[m])
        if ((d <= cfg.dist_gate) & (ang <= cfg.ang_gate)).any():
            return True   # matched
    return False


def main():
    model, ctx = train_model()
    ais, rad, cfg, tl = ctx["ais"], ctx["rad"], ctx["cfg"], ctx["tl"]
    rad_by_t = rad.sort_values("t")
    heldout = tl[(tl["split"].isin(["val", "test"])) & tl["track_matched"] & tl["dom_mmsi"].notna()]
    print(f"held-out matched tracks for robustness: {len(heldout)} | dev {DEV}")

    rng = np.random.default_rng(0)
    rows = []
    for off_m in OFFSETS_M:
        learned_fd, geom_fd, learned_scores = [], [], []
        for _, row in heldout.iterrows():
            sub = rad_by_t[rad_by_t["targetId"] == row["targetId"]]
            if len(sub) < 2:
                continue
            rt, rlon, rlat = sub["t"].to_numpy(), sub["lon"].to_numpy(), sub["lat"].to_numpy()
            rsog, rcog = sub["sog"].to_numpy(), sub["cog"].to_numpy()
            ref_lon, ref_lat = float(rlon.mean()), float(rlat.mean())
            t0, t1 = float(rt.min()), float(rt.max()); win = max(t1 - t0, 1.0)
            dom = row["dom_mmsi"]
            cands = list(_cand_mmsis(ais, rlon, rlat, t0, t1))
            cands = [dom] + [m for m in cands if m != dom]      # dom first
            s_feat, s_mask = _featurize(rt, rlon, rlat, rsog, rcog, ref_lon, ref_lat, t0, win)
            r = model.encode_sensor(torch.as_tensor(s_feat).unsqueeze(0).float().to(DEV),
                                    torch.as_tensor(s_mask).unsqueeze(0).to(DEV))[0]
            for d in range(DIRS):
                ang = 2 * np.pi * d / DIRS
                off = (off_m * np.cos(ang), off_m * np.sin(ang))
                cf, cm, cv = _build_cands(ais, cands, ref_lon, ref_lat, t0, t1, win, {dom: off})
                safe = cm.copy(); safe[~cv, 0] = True
                a = model.encode_ais(torch.as_tensor(cf).float().to(DEV), torch.as_tensor(safe).to(DEV))
                ds = model.dark_score(r, a, torch.as_tensor(cv).to(DEV))
                ds = 0.5 if not np.isfinite(ds) else float(ds)
                learned_scores.append(ds)
                learned_fd.append(1 if ds > 0.5 else 0)
                geom_fd.append(0 if _geom_match(ais, dom, rlon, rlat, rcog, rt, off, ref_lat, cfg) else 1)
        rows.append({"offset_m": off_m,
                     "learned_false_dark_rate": float(np.mean(learned_fd)),
                     "learned_mean_dark_score": float(np.mean(learned_scores)),
                     "geometric_false_dark_rate": float(np.mean(geom_fd))})

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "busan_w4_robustness.json"), "w") as f:
        json.dump({"offsets_m": OFFSETS_M, "rows": rows, "n_tracks": int(len(heldout)),
                   "note": "False-dark rate (truth=AIS present) vs injected AIS registration offset. "
                           "Geometric gating breaks near the 300m gate; learned matcher should stay lower."},
                  f, indent=2)
    print(f"{'offset_m':>9} {'geom_FD':>9} {'learned_FD':>11} {'learn_dark':>11}")
    for r in rows:
        print(f"{r['offset_m']:>9} {r['geometric_false_dark_rate']:>9.3f} "
              f"{r['learned_false_dark_rate']:>11.3f} {r['learned_mean_dark_score']:>11.3f}")


if __name__ == "__main__":
    main()
