"""FVessel (video+AIS) loader — the public anchor for P1's camera arm.

Pre-written against FVessel's documented schema so it is ready the moment the
data is downloaded (OneDrive/Baidu — see README). Produces a track-label table
compatible with data.dropout_splitter.make_splits, so the SAME controlled
AIS-dropout protocol used for Busan applies to the camera setting.

Expected layout (per video folder, e.g. `01_Video+AIS/`):
  ais/<...>.csv          cols: Number, MMSI, Lon, Lat, Speed, Course, Heading, Type, Timestamp
  gt/<Video>_gt_tracking.txt   cols: second, id, bb_left, bb_top, bb_w, bb_h, conf, x, y, z
  gt/<Video>_gt_fusion.txt     cols: second, mmsi, bb_left, bb_top, bb_w, bb_h, conf, x, y, z
A visual track `id` is "known" (has AIS) if it appears in gt_fusion (linked MMSI),
else it is a real-world dark candidate.
"""
from __future__ import annotations

import glob
import os

import pandas as pd

FUSION_COLS = ["second", "mmsi", "bb_left", "bb_top", "bb_w", "bb_h", "conf", "x", "y", "z"]
TRACK_COLS = ["second", "id", "bb_left", "bb_top", "bb_w", "bb_h", "conf", "x", "y", "z"]
AIS_COLS = ["Number", "MMSI", "Lon", "Lat", "Speed", "Course", "Heading", "Type", "Timestamp"]


def _read_gt(path: str, cols) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    return pd.read_csv(path, header=None, names=cols)


def load_video_tracklabels(video_dir: str, video_name: str) -> pd.DataFrame:
    """One row per visual track id in a video: n_pts, track_matched, dom_mmsi, match_frac.

    Mirrors data.pair_builder.track_labels output so dropout_splitter works unchanged.
    """
    gt = os.path.join(video_dir, "gt")
    trk = _read_gt(os.path.join(gt, f"{video_name}_gt_tracking.txt"), TRACK_COLS)
    fus = _read_gt(os.path.join(gt, f"{video_name}_gt_fusion.txt"), FUSION_COLS)
    if trk.empty:
        return pd.DataFrame(columns=["targetId", "n_pts", "match_frac", "dom_mmsi", "track_matched"])

    # frames where a track id is linked to an MMSI (fusion) — approx by (second,bbox) overlap on id rows
    npts = trk.groupby("id").size().rename("n_pts")
    # a track id is "matched" if any fusion row shares its bbox-second; approximate via second-overlap per id
    fused_seconds = set(zip(fus["second"], fus["mmsi"])) if not fus.empty else set()
    # map: for each id, fraction of its seconds that have ANY fusion row at that second
    fus_secs = set(fus["second"]) if not fus.empty else set()
    frac = (trk.assign(f=trk["second"].isin(fus_secs)).groupby("id")["f"].mean().rename("match_frac"))
    # dominant mmsi: most frequent mmsi among fusion rows whose second co-occurs with the id's seconds
    dom = {}
    for tid, sub in trk.groupby("id"):
        secs = set(sub["second"])
        cand = fus[fus["second"].isin(secs)]["mmsi"] if not fus.empty else pd.Series([], dtype=object)
        dom[tid] = cand.mode().iat[0] if len(cand) else None
    out = pd.concat([npts, frac], axis=1).reset_index().rename(columns={"id": "targetId"})
    out["dom_mmsi"] = out["targetId"].map(dom)
    out["track_matched"] = out["dom_mmsi"].notna() & (out["match_frac"] > 0)
    out.insert(0, "video", video_name)
    return out


def load_all(fvessel_root: str) -> pd.DataFrame:
    """Scan all video folders under an extracted FVessel root -> combined track labels."""
    rows = []
    for d in sorted(glob.glob(os.path.join(fvessel_root, "*Video*AIS*"))):
        if not os.path.isdir(d):
            continue
        for gt_path in glob.glob(os.path.join(d, "gt", "*_gt_tracking.txt")):
            vname = os.path.basename(gt_path).replace("_gt_tracking.txt", "")
            rows.append(load_video_tracklabels(d, vname))
    if not rows:
        raise FileNotFoundError(
            f"No FVessel videos found under {fvessel_root}. Download FVessel first (see README)."
        )
    df = pd.concat(rows, ignore_index=True)
    df["targetId"] = df["video"] + ":" + df["targetId"].astype(str)  # globally unique
    return df


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "raw", "fvessel")
    tl = load_all(root)
    print(f"FVessel tracks: {len(tl)} | matched(has-AIS): {int(tl.track_matched.sum())} | "
          f"videos: {tl.video.nunique()}")
