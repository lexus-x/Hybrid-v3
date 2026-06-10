"""Build camera (BONK) per-detection samples: bbox crop + AIS attribute vector.

Crops each annotated vessel from its image (resized 64x64) and pairs it with an
AIS attribute vector [range, bearing, h, w, l] (normalised). Splits by image id.
Caches to .npz so the (slow) image-cropping pass runs once.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from data.bonk_loader import load_6d

CROP = 64
ATTR_DIM = 5


def _attr(row):
    return np.array([
        np.clip(row["Z"] / 500.0, 0, 4),
        row["ais_bearing"] / np.pi,
        np.clip((row["h"] or 0) / 30.0, 0, 3),
        np.clip((row["w"] or 0) / 15.0, 0, 3),
        np.clip((row["l"] or 0) / 100.0, 0, 3),
    ], np.float32)


def build(root, cache=None, seed=0, ratios=(0.6, 0.2, 0.2)):
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return dict(crops=z["crops"], attrs=z["attrs"], ids=z["ids"], img=z["img"], split=z["split"])
    df = load_6d(root)
    crops, attrs, ids, imgs = [], [], [], []
    cur_img, im = None, None
    for _, r in df.sort_values("img_id").iterrows():
        if r["img_id"] != cur_img:
            im = cv2.imread(r["image_path"]); cur_img = r["img_id"]
        if im is None:
            continue
        H, W = im.shape[:2]
        x1, y1 = max(0, int(r["x1"])), max(0, int(r["y1"]))
        x2, y2 = min(W, int(r["x2"])), min(H, int(r["y2"]))
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        crop = cv2.resize(im[y1:y2, x1:x2], (CROP, CROP))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        crops.append(crop.transpose(2, 0, 1))
        attrs.append(_attr(r)); ids.append(str(r["identifier"])); imgs.append(r["img_id"])
    crops = np.asarray(crops, np.float32); attrs = np.asarray(attrs, np.float32)
    ids = np.asarray(ids); imgs = np.asarray(imgs)

    # split by image id (disjoint images across splits)
    rng = np.random.default_rng(seed)
    uimg = np.unique(imgs); rng.shuffle(uimg)
    n = len(uimg); a, b = int(ratios[0] * n), int((ratios[0] + ratios[1]) * n)
    part = {**{u: "train" for u in uimg[:a]}, **{u: "val" for u in uimg[a:b]},
            **{u: "test" for u in uimg[b:]}}
    split = np.array([part[i] for i in imgs])
    out = dict(crops=crops, attrs=attrs, ids=ids, img=imgs, split=split)
    if cache:
        np.savez_compressed(cache, **out)
    return out


if __name__ == "__main__":
    import sys
    root = sys.argv[1]
    d = build(root, cache=sys.argv[2] if len(sys.argv) > 2 else None)
    print(f"detections {len(d['crops'])} | images {len(set(d['img']))} | "
          f"train {int((d['split']=='train').sum())} val {int((d['split']=='val').sum())} "
          f"test {int((d['split']=='test').sum())} | crop {d['crops'].shape} attr {d['attrs'].shape}")
