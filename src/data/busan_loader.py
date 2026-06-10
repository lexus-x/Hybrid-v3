"""Load the Busan tri-source snapshot (AIS stream + radar-target tracks).

Confidential dataset; only the AIS + Radar-Target CSVs are needed for the
radar+AIS arm of P1. Position messages: AIS types 1/3/18/19. sog/cog for AIS
live in the JSON `data` column; radar target CSVs carry them as plain columns.
"""
from __future__ import annotations

import glob
import json
import os

import pandas as pd

from common.geo import parse_quoted_dt

POS_TYPES = {"1", "3", "18", "19"}
DEFAULT_RAW = os.path.join(os.path.dirname(__file__), "raw", "busan")


def load_ais(raw_dir: str = DEFAULT_RAW) -> pd.DataFrame:
    """Return AIS position points: columns [mmsi, lon, lat, cog, t] sorted by t."""
    frames = []
    for f in sorted(glob.glob(os.path.join(raw_dir, "ais_*.csv"))):
        df = pd.read_csv(f, dtype=str)
        df = df[df["message_type"].isin(POS_TYPES)]
        df = df[(df["longitude"] != "0") & (df["latitude"] != "0")].copy()
        j = df["data"].map(json.loads)
        df["cog"] = j.map(lambda d: d.get("cog"))
        df["sog"] = j.map(lambda d: d.get("sog"))
        frames.append(df[["mmsi", "longitude", "latitude", "cog", "sog", "date_time"]])
    ais = pd.concat(frames, ignore_index=True)
    ais["t"] = parse_quoted_dt(ais["date_time"])
    for c in ("longitude", "latitude", "cog", "sog"):
        ais[c] = pd.to_numeric(ais[c], errors="coerce")
    ais["sog"] = ais["sog"].fillna(0.0)
    ais = ais.dropna(subset=["longitude", "latitude", "cog", "t"])
    return ais.rename(columns={"longitude": "lon", "latitude": "lat"}).sort_values("t").reset_index(drop=True)


def load_radar(raw_dir: str = DEFAULT_RAW) -> pd.DataFrame:
    """Return radar target points: columns [targetId, lon, lat, cog, sog, t]."""
    frames = []
    for f in sorted(glob.glob(os.path.join(raw_dir, "radartarget_*.csv"))):
        df = pd.read_csv(f, dtype=str)
        frames.append(df[["targetId", "longitude", "latitude", "cog", "sog", "dateTime"]])
    rad = pd.concat(frames, ignore_index=True)
    rad["t"] = parse_quoted_dt(rad["dateTime"])
    for c in ("longitude", "latitude", "cog", "sog"):
        rad[c] = pd.to_numeric(rad[c], errors="coerce")
    rad = rad.dropna(subset=["longitude", "latitude", "cog", "t"])
    return rad.rename(columns={"longitude": "lon", "latitude": "lat"}).reset_index(drop=True)


if __name__ == "__main__":
    a, r = load_ais(), load_radar()
    print(f"AIS points {len(a)} | MMSI {a.mmsi.nunique()}")
    print(f"Radar points {len(r)} | targets {r.targetId.nunique()}")
