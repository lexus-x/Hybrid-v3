"""Build per-track open-set samples from the Busan radar+AIS arm.

For each radar track: its kinematic tracklet + a shortlist of candidate AIS
tracklets (MMSIs near it in space/time) featurised in the SAME local ENU frame,
plus the index of the true match (dom_mmsi) or -1. The controlled-dropout test
split removes the true AIS from candidates when `ais_withheld` -> known dark.
"""
from __future__ import annotations

import numpy as np

from common.geo import haversine_m

F_DIM = 6
MAX_LEN = 32          # max points per tracklet
MAX_CAND = 12         # candidate AIS shortlist size
CAND_DIST = 800.0     # m  (loose spatial gate for candidate generation)
CAND_DT = 30.0        # s  (time pad around the radar track span)


def _featurize(t, lon, lat, sog, cog, ref_lon, ref_lat, t0, win):
    """Return (MAX_LEN, F_DIM) feats + (MAX_LEN,) bool mask, in a shared ENU frame."""
    order = np.argsort(t)
    t, lon, lat, sog, cog = (a[order][:MAX_LEN] for a in (t, lon, lat, sog, cog))
    east = (lon - ref_lon) * 111320.0 * np.cos(np.radians(ref_lat)) / 1000.0  # km
    north = (lat - ref_lat) * 110540.0 / 1000.0
    feats = np.zeros((MAX_LEN, F_DIM), np.float32)
    L = len(t)
    feats[:L, 0] = east
    feats[:L, 1] = north
    feats[:L, 2] = np.clip(sog / 20.0, 0, 3)
    feats[:L, 3] = np.sin(np.radians(cog))
    feats[:L, 4] = np.cos(np.radians(cog))
    feats[:L, 5] = np.clip((t - t0) / max(win, 1.0), -2, 2)
    mask = np.zeros(MAX_LEN, bool)
    mask[:L] = True
    return feats, mask


def build_samples(ais, rad, track_labels):
    """ais, rad: point DataFrames; track_labels: per-track rows incl split/dark_label/dom_mmsi.
    Returns list of sample dicts (numpy arrays)."""
    at = ais["t"].to_numpy()
    order = np.argsort(at)
    at = at[order]
    alon, alat = ais["lon"].to_numpy()[order], ais["lat"].to_numpy()[order]
    asog, acog = ais["sog"].to_numpy()[order], ais["cog"].to_numpy()[order]
    ammsi = ais["mmsi"].to_numpy()[order]

    rad_by_t = rad.sort_values("t")
    samples = []
    for _, row in track_labels.iterrows():
        sub = rad_by_t[rad_by_t["targetId"] == row["targetId"]]
        if len(sub) < 2:
            continue
        rt = sub["t"].to_numpy(); rlon = sub["lon"].to_numpy(); rlat = sub["lat"].to_numpy()
        rsog = sub["sog"].to_numpy(); rcog = sub["cog"].to_numpy()
        ref_lon, ref_lat = float(rlon.mean()), float(rlat.mean())
        t0, t1 = float(rt.min()), float(rt.max())
        win = max(t1 - t0, 1.0)

        # candidate MMSIs: AIS in window + within CAND_DIST of any radar point
        lo = np.searchsorted(at, t0 - CAND_DT); hi = np.searchsorted(at, t1 + CAND_DT)
        cand_mmsi = set()
        if hi > lo:
            wl, wa, wo = alon[lo:hi], alat[lo:hi], alat[lo:hi]
            for i in range(len(rt)):
                d = haversine_m(rlon[i], rlat[i], alon[lo:hi], alat[lo:hi])
                near = ammsi[lo:hi][d <= CAND_DIST]
                cand_mmsi.update(near.tolist())

        withheld = bool(row.get("ais_withheld", False))
        dom = row.get("dom_mmsi")
        if withheld and dom in cand_mmsi:
            cand_mmsi.discard(dom)
        cand_mmsi = list(cand_mmsi)[:MAX_CAND]

        s_feat, s_mask = _featurize(rt, rlon, rlat, rsog, rcog, ref_lon, ref_lat, t0, win)
        cand_feats = np.zeros((MAX_CAND, MAX_LEN, F_DIM), np.float32)
        cand_pmask = np.zeros((MAX_CAND, MAX_LEN), bool)
        cand_valid = np.zeros(MAX_CAND, bool)
        pos_idx = -1
        for k, m in enumerate(cand_mmsi):
            sel = (ammsi == m) & (at >= t0 - CAND_DT) & (at <= t1 + CAND_DT)
            if sel.sum() < 1:
                continue
            cf, cm = _featurize(at[sel], alon[sel], alat[sel], asog[sel], acog[sel],
                                ref_lon, ref_lat, t0, win)
            cand_feats[k], cand_pmask[k], cand_valid[k] = cf, cm, True
            if dom is not None and m == dom:
                pos_idx = k
        samples.append(dict(
            targetId=row["targetId"], split=row["split"],
            dark_label=row.get("dark_label"), withheld=withheld,
            s_feat=s_feat, s_mask=s_mask,
            cand_feats=cand_feats, cand_pmask=cand_pmask, cand_valid=cand_valid,
            pos_idx=pos_idx,
        ))
    return samples
