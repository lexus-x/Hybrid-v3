"""Relative kinematic feature extraction.

Computes translation-invariant features between a radar tracklet and an
interpolated candidate AIS tracklet.
"""
from __future__ import annotations

import numpy as np
from common.geo import angdiff_deg, enu_offset_m

MAX_LEN = 32

def compute_relative_features(rt, rlon, rlat, rsog, rcog, at, alon, alat, asog, acog):
    """
    Interpolates candidate AIS tracklet onto radar timestamps `rt`, 
    then computes translation-invariant relative kinematic features.
    
    Returns:
      feats: (MAX_LEN, 6) relative feature array:
        0: r_along (km) - relative position projected along AIS heading
        1: r_cross (km) - relative position projected cross AIS heading
        2: dr_along (km/s) - derivative of r_along
        3: dr_cross (km/s) - derivative of r_cross
        4: dsog (knots / 20) - difference in SOG
        5: dcog (deg / 180) - difference in COG
      mask: (MAX_LEN,) boolean mask indicating valid points
    """
    L = min(len(rt), MAX_LEN)
    feats = np.zeros((MAX_LEN, 6), dtype=np.float32)
    mask = np.zeros(MAX_LEN, dtype=bool)
    mask[:L] = True
    
    if L == 0:
        return feats, mask

    # Trim radar inputs to MAX_LEN
    rt_trim = rt[:L]
    rlon_trim = rlon[:L]
    rlat_trim = rlat[:L]
    rsog_trim = rsog[:L]
    rcog_trim = rcog[:L]

    # Interpolate AIS values to radar timestamps `rt_trim`
    if len(at) == 0:
        # No AIS data available at all for this candidate
        return feats, mask
    elif len(at) == 1:
        alon_interp = np.full_like(rt_trim, alon[0])
        alat_interp = np.full_like(rt_trim, alat[0])
        asog_interp = np.full_like(rt_trim, asog[0])
        acog_interp = np.full_like(rt_trim, acog[0])
    else:
        alon_interp = np.interp(rt_trim, at, alon)
        alat_interp = np.interp(rt_trim, at, alat)
        asog_interp = np.interp(rt_trim, at, asog)
        # Interpolate COG using sin and cos to avoid 0/360 wrap-around errors
        asin = np.sin(np.radians(acog))
        acos = np.cos(np.radians(acog))
        asin_interp = np.interp(rt_trim, at, asin)
        acos_interp = np.interp(rt_trim, at, acos)
        acog_interp = np.degrees(np.arctan2(asin_interp, acos_interp)) % 360.0

    # East-North position offset of radar relative to AIS (meters)
    rx, ry = enu_offset_m(alon_interp, alat_interp, rlon_trim, rlat_trim)
    
    # Convert meters to kilometers
    rx = rx / 1000.0
    ry = ry / 1000.0

    # Compute relative velocity in ENU frame (finite differences)
    drx = np.zeros_like(rx)
    dry = np.zeros_like(ry)
    
    if L > 1:
        dt = np.diff(rt_trim)
        dt = np.clip(dt, 1e-3, None)
        drx[1:] = np.diff(rx) / dt
        dry[1:] = np.diff(ry) / dt
        drx[0] = drx[1]
        dry[0] = dry[1]

    # Project relative position and velocity into AIS heading frame
    rad_acog = np.radians(acog_interp)
    r_along = rx * np.sin(rad_acog) + ry * np.cos(rad_acog)
    r_cross = rx * np.cos(rad_acog) - ry * np.sin(rad_acog)
    
    dr_along = drx * np.sin(rad_acog) + dry * np.cos(rad_acog)
    dr_cross = drx * np.cos(rad_acog) - dry * np.sin(rad_acog)

    # SOG and COG difference
    dsog = (rsog_trim - asog_interp) / 20.0
    dcog = angdiff_deg(rcog_trim, acog_interp) / 180.0

    # Populate feature matrix
    feats[:L, 0] = r_along
    feats[:L, 1] = r_cross
    feats[:L, 2] = dr_along
    feats[:L, 3] = dr_cross
    feats[:L, 4] = dsog
    feats[:L, 5] = dcog

    return feats, mask
