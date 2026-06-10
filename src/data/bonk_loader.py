"""BONK-Pose (camera+AIS) loader — the camera arm of P1.

Each 6D label gives, per detected vessel: a stable `identifier` (anonymised
vessel id ≈ MMSI), a 2D bbox `bbImage2d` [x1,y1,x2,y2], AIS-derived dimensions
(h,w,l) and a 3D `position.centroid` [X,Y,Z] in the camera frame (Z = range, m).
`calib/<id>.txt` is the 3x3 intrinsic K. Because every annotation is AIS-derived,
all detections are AIS-linked ("known"); "dark" cases are produced at train/eval
time by controlled AIS-dropout (mirroring the Busan radar arm) — withhold a
detection's AIS from the scene pool.

bbox->bearing uses K so visual detections and AIS positions live in one bearing
frame; the appearance crop is what makes matching non-trivial (loaded separately).
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

DEFAULT_ROOT = os.path.join(os.path.dirname(__file__), "raw", "bonk",
                            "datasets", "6D_pose_estimation")


def _read_calib(path):
    K = np.loadtxt(path)
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])  # fx,fy,cx,cy


def load_6d(root: str = DEFAULT_ROOT) -> pd.DataFrame:
    """Return one row per detection across all images.

    Columns: img_id, image_path, identifier, x1,y1,x2,y2, X,Y,Z (camera-frame m),
    h,w,l, fx,fy,cx,cy, det_bearing (from bbox via K), ais_bearing (from centroid).
    """
    rows = []
    for lp in sorted(glob.glob(os.path.join(root, "label", "*.json"))):
        img_id = os.path.splitext(os.path.basename(lp))[0]
        cp = os.path.join(root, "calib", img_id + ".txt")
        ip = os.path.join(root, "image", img_id + ".jpg")
        if not os.path.exists(cp):
            continue
        fx, fy, cx, cy = _read_calib(cp)
        d = json.load(open(lp))
        for o in d.get("objects", []):
            bb = o.get("bbImage2d")
            pos = (o.get("position") or {}).get("centroid")
            if not bb or not pos:
                continue
            u = 0.5 * (bb[0] + bb[2])                          # bbox horizontal centre
            det_bearing = float(np.arctan2((u - cx) / fx, 1.0))
            X, Y, Z = float(pos[0]), float(pos[1]), float(pos[2])
            ais_bearing = float(np.arctan2(X, Z)) if Z != 0 else 0.0
            rows.append(dict(
                img_id=img_id, image_path=ip, identifier=o.get("identifier"),
                x1=bb[0], y1=bb[1], x2=bb[2], y2=bb[3],
                X=X, Y=Y, Z=Z, h=o.get("height"), w=o.get("width"), l=o.get("length"),
                fx=fx, fy=fy, cx=cx, cy=cy,
                det_bearing=det_bearing, ais_bearing=ais_bearing,
            ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
    df = load_6d(root)
    print(f"detections: {len(df)} | images: {df.img_id.nunique() if len(df) else 0} | "
          f"unique vessels: {df.identifier.nunique() if len(df) else 0}")
    if len(df):
        print(df[["img_id", "identifier", "Z", "det_bearing", "ais_bearing"]].head(8).to_string(index=False))
        print(f"bearing residual (det-ais) deg: median "
              f"{np.degrees(np.abs(df.det_bearing - df.ais_bearing)).median():.2f}")
