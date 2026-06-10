"""Geometric radar<->AIS matching (the rule/gating baseline) + bias stats.

Reproduces the PDF pipeline: time/distance/angle gating + cost-min assignment
J = d + LAMBDA * dpsi. Produces per-radar-point matches, per-track labels, and
the systematic sensor-bias estimate. This is the BASELINE the learned open-set
detector (P1) must beat, and the source of confident matches for dropout labels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from common.geo import angdiff_deg, enu_offset_m, haversine_m


@dataclass
class GateCfg:
    dt_tol: float = 8.0       # s   (max |radar-AIS| time gap)
    dist_gate: float = 300.0  # m
    ang_gate: float = 60.0    # deg
    lam: float = 2.0          # cost J = d + lam*dpsi  (reverse-engineered from PDF examples)
    track_frac: float = 0.5   # >= this fraction matched => track is "matched"


def match_radar_to_ais(rad: pd.DataFrame, ais: pd.DataFrame, cfg: GateCfg = GateCfg()) -> pd.DataFrame:
    """Add per-radar-point match columns: matched, mmsi, dist_m, dpsi, dt_s, east_m, north_m."""
    at = ais["t"].to_numpy()
    alon, alat, acog = ais["lon"].to_numpy(), ais["lat"].to_numpy(), ais["cog"].to_numpy()
    ammsi = ais["mmsi"].to_numpy()
    n = len(rad)
    out = {k: np.full(n, np.nan) for k in ("dist_m", "dpsi", "dt_s", "east_m", "north_m")}
    matched = np.zeros(n, bool)
    mmsi = np.empty(n, object)
    rt, rlon, rlat, rcog = (rad[c].to_numpy() for c in ("t", "lon", "lat", "cog"))
    for i in range(n):
        lo = np.searchsorted(at, rt[i] - cfg.dt_tol)
        hi = np.searchsorted(at, rt[i] + cfg.dt_tol)
        if hi <= lo:
            continue
        d = haversine_m(rlon[i], rlat[i], alon[lo:hi], alat[lo:hi])
        ad = angdiff_deg(rcog[i], acog[lo:hi])
        ok = (d <= cfg.dist_gate) & (ad <= cfg.ang_gate)
        if not ok.any():
            continue
        cost = np.where(ok, d + cfg.lam * ad, np.inf)
        j = int(np.argmin(cost))
        k = lo + j
        matched[i] = True
        mmsi[i] = ammsi[k]
        out["dist_m"][i], out["dpsi"][i], out["dt_s"][i] = d[j], ad[j], rt[i] - at[k]
        e, nth = enu_offset_m(rlon[i], rlat[i], alon[k], alat[k])
        out["east_m"][i], out["north_m"][i] = e, nth
    res = rad.copy()
    res["matched"] = matched
    res["mmsi"] = mmsi
    for k, v in out.items():
        res[k] = v
    return res


def track_labels(matched_pts: pd.DataFrame, cfg: GateCfg = GateCfg()) -> pd.DataFrame:
    """Per-track: matched fraction, dominant mmsi, matched flag (>=track_frac)."""
    g = matched_pts.groupby("targetId")
    frac = g["matched"].mean().rename("match_frac")
    npts = g.size().rename("n_pts")
    dom = g.apply(lambda d: d.loc[d["matched"], "mmsi"].mode().iat[0]
                  if d["matched"].any() else None, include_groups=False).rename("dom_mmsi")
    tl = pd.concat([npts, frac, dom], axis=1).reset_index()
    tl["track_matched"] = tl["match_frac"] >= cfg.track_frac
    return tl


def summary(matched_pts: pd.DataFrame, tl: pd.DataFrame) -> dict:
    m = matched_pts["matched"]
    md = matched_pts.loc[m, "dist_m"]
    bias_e = matched_pts.loc[m, "east_m"].mean()
    bias_n = matched_pts.loc[m, "north_m"].mean()
    return {
        "radar_points": int(len(matched_pts)),
        "point_match_rate": float(m.mean()),
        "tracks": int(len(tl)),
        "track_match_rate": float(tl["track_matched"].mean()),
        "matched_tracks": int(tl["track_matched"].sum()),
        "dark_candidate_tracks": int((~tl["track_matched"]).sum()),
        "dist_p50_m": float(np.percentile(md, 50)),
        "dist_p90_m": float(np.percentile(md, 90)),
        "bias_east_m": float(bias_e),
        "bias_north_m": float(bias_n),
        "bias_mag_m": float(np.hypot(bias_e, bias_n)),
    }
