"""Geospatial + temporal helpers shared across datasets.

Kept dependency-light (numpy only) so loaders/baselines reuse one implementation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_R = 6371000.0  # metres
KNOTS_TO_MS = 0.514444


def parse_quoted_dt(series: pd.Series) -> pd.Series:
    """Parse Busan-style timestamps like ="2025-08-01 15:10:55.831311" to epoch seconds (float)."""
    cleaned = (
        series.astype(str)
        .str.replace('="', "", regex=False)
        .str.replace('"', "", regex=False)
        .str.strip()
    )
    dt = pd.to_datetime(cleaned, format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
    return dt.astype("int64") / 1e9


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres. Scalars or numpy arrays (broadcasting)."""
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dlmb = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2.0) ** 2
    return 2.0 * EARTH_R * np.arcsin(np.sqrt(a))


def angdiff_deg(a, b):
    """Smallest absolute circular difference between two bearings in degrees."""
    d = np.abs(np.asarray(a) - np.asarray(b)) % 360.0
    return np.minimum(d, 360.0 - d)


def enu_offset_m(lon_ref, lat_ref, lon, lat):
    """Approx east/north offset (metres) of (lon,lat) relative to (lon_ref,lat_ref)."""
    east = (np.asarray(lon) - lon_ref) * 111320.0 * np.cos(np.radians(lat_ref))
    north = (np.asarray(lat) - lat_ref) * 110540.0
    return east, north
