"""Train joint relative kinematics matcher.

Constructs training samples using relative ENU kinematics features and trains the
single-tower JointRelativeMatcher with translation offset data augmentation.
"""
from __future__ import annotations

import json
import os
import numpy as np
import torch
import torch.nn.functional as F

from data.busan_loader import load_ais, load_radar
from data.pair_builder import GateCfg, match_radar_to_ais, track_labels
from data.dropout_splitter import make_splits
from p1_openset_darkdet.relative_model import JointRelativeMatcher
from p1_openset_darkdet.relative_features import compute_relative_features
from p1_openset_darkdet.dataset import CAND_DIST, CAND_DT, MAX_CAND, MAX_LEN
from common.geo import haversine_m

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _m_to_deg(east_m, north_m, lat):
    return (east_m / (111320.0 * np.cos(np.radians(lat))), north_m / 110540.0)


def build_relative_samples(ais, rad, track_labels):
    """ais, rad: point DataFrames; track_labels: per-track rows.
    Returns list of relative sample dicts."""
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
        rsog, rcog = sub["sog"].to_numpy(), sub["cog"].to_numpy()
        t0, t1 = float(rt.min()), float(rt.max())

        # candidate MMSIs: AIS in window + within CAND_DIST of any radar point
        lo = np.searchsorted(at, t0 - CAND_DT); hi = np.searchsorted(at, t1 + CAND_DT)
        cand_msi = set()
        if hi > lo:
            for i in range(len(rt)):
                d = haversine_m(rlon[i], rlat[i], alon[lo:hi], alat[lo:hi])
                near = ammsi[lo:hi][d <= CAND_DIST]
                cand_msi.update(near.tolist())

        withheld = bool(row.get("ais_withheld", False))
        dom = row.get("dom_mmsi")
        if withheld and dom in cand_msi:
            cand_msi.discard(dom)
        cand_msi = list(cand_msi)[:MAX_CAND]

        cand_feats = np.zeros((MAX_CAND, MAX_LEN, 6), np.float32)
        cand_pmask = np.zeros((MAX_CAND, MAX_LEN), bool)
        cand_valid = np.zeros(MAX_CAND, bool)
        pos_idx = -1
        for k, m in enumerate(cand_msi):
            sel = (ammsi == m) & (at >= t0 - CAND_DT) & (at <= t1 + CAND_DT)
            if sel.sum() < 1:
                continue
            cf, cm = compute_relative_features(
                rt, rlon, rlat, rsog, rcog,
                at[sel], alon[sel], alat[sel], asog[sel], acog[sel]
            )
            cand_feats[k], cand_pmask[k], cand_valid[k] = cf, cm, True
            if dom is not None and m == dom:
                pos_idx = k
        samples.append(dict(
            targetId=row["targetId"], split=row["split"],
            dark_label=row.get("dark_label"), withheld=withheld,
            cand_feats=cand_feats, cand_pmask=cand_pmask, cand_valid=cand_valid,
            pos_idx=pos_idx,
            rt=rt, rlon=rlon, rlat=rlat, rsog=rsog, rcog=rcog,
            cand_msi=cand_msi, dom_mmsi=dom
        ))
    return samples


def build_feats_for_sample(ais, cand_msi, dom, rt, rlon, rlat, rsog, rcog, t0, t1, ref_lat, offset=None):
    """Build candidate relative features for a sample with an optional offset injected to the true target."""
    cand_feats = np.zeros((MAX_CAND, MAX_LEN, 6), np.float32)
    cand_pmask = np.zeros((MAX_CAND, MAX_LEN), bool)
    cand_valid = np.zeros(MAX_CAND, bool)
    pos_idx = -1
    
    at = ais["t"].to_numpy()
    alon, alat = ais["lon"].to_numpy(), ais["lat"].to_numpy()
    asog, acog = ais["sog"].to_numpy(), ais["cog"].to_numpy()
    ammsi = ais["mmsi"].to_numpy()
    
    for k, m in enumerate(cand_msi):
        sel = (ammsi == m) & (at >= t0 - CAND_DT) & (at <= t1 + CAND_DT)
        if sel.sum() < 1:
            continue
        lon = alon[sel].copy()
        lat = alat[sel].copy()
        if offset and m == dom:
            de, dn = _m_to_deg(offset[0], offset[1], ref_lat)
            lon = lon + de; lat = lat + dn
        cf, cm = compute_relative_features(
            rt, rlon, rlat, rsog, rcog,
            at[sel], lon, lat, asog[sel], acog[sel]
        )
        cand_feats[k], cand_pmask[k], cand_valid[k] = cf, cm, True
        if dom is not None and m == dom:
            pos_idx = k
    return cand_feats, cand_pmask, cand_valid, pos_idx


def _t(a):
    return torch.as_tensor(a).to(DEV)


def train_relative_model(epochs=150, lr=3e-4, seed=0, p_drop=0.5):
    """Train JointRelativeMatcher model using contrastive dropout + offset augmentation."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cfg = GateCfg()
    ais, rad = load_ais(), load_radar()
    pts = match_radar_to_ais(rad, ais, cfg)
    tl = make_splits(track_labels(pts, cfg))
    samples = build_relative_samples(ais, rad, tl)
    train_base = [s for s in samples if s["split"] == "train" and s["pos_idx"] >= 0]

    # Data augmentation: Expand training set with offset perturbations
    train = []
    for s in train_base:
        # 1. Clean copy
        train.append(s)
        
        # 2. Add augmented copies with random offsets injected to the true target
        dom = s["dom_mmsi"]
        cand_msi = s["cand_msi"]
        rt, rlon, rlat = s["rt"], s["rlon"], s["rlat"]
        rsog, rcog = s["rsog"], s["rcog"]
        t0, t1 = float(rt.min()), float(rt.max())
        ref_lat = float(rlat.mean())
        
        # We add 9 augmented copies to make 10x training size
        for _ in range(9):
            off_dist = rng.uniform(0.0, 500.0) # Up to 500 meters offset
            off_ang = rng.uniform(0.0, 2 * np.pi)
            off_x = off_dist * np.cos(off_ang)
            off_y = off_dist * np.sin(off_ang)
            
            cf, cm, cv, pos = build_feats_for_sample(
                ais, cand_msi, dom, rt, rlon, rlat, rsog, rcog, t0, t1, ref_lat, (off_x, off_y)
            )
            train.append(dict(
                targetId=s["targetId"], split=s["split"],
                dark_label=s["dark_label"], withheld=s["withheld"],
                cand_feats=cf, cand_pmask=cm, cand_valid=cv,
                pos_idx=pos
            ))

    model = JointRelativeMatcher(in_dim=6, emb=128).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    last = {}
    for ep in range(epochs):
        rng.shuffle(train)
        tot = 0.0; correct = 0
        for s in train:
            cf = _t(s["cand_feats"]).float(); cm = _t(s["cand_pmask"])
            valid = _t(s["cand_valid"])
            
            pos = s["pos_idx"]
            valid = valid.clone()
            if pos >= 0 and rng.random() < p_drop:     # training-time AIS dropout -> teach reject
                valid[pos] = False
                target = MAX_CAND                       # 'absent'/dark slot
            else:
                target = pos
                
            logits = model.match_logits(cf, cm, valid)
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
