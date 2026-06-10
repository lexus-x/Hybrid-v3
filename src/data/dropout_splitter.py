"""Controlled AIS-dropout protocol (the eval that avoids circularity).

Confidently-matched radar tracks are the only reliable ground truth we have.
We split tracks into train/val/test, then in the TEST set withhold the AIS for
a random half of the matched tracks: those become KNOWN-POSITIVE dark episodes
(answer = dark) while the rest stay negatives (AIS present). Genuinely unmatched
tracks are kept aside as real-world (unlabeled) dark candidates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_splits(track_labels: pd.DataFrame, seed: int = 0,
                min_match_frac: float = 0.6,
                ratios=(0.6, 0.2, 0.2), dropout_frac: float = 0.5) -> pd.DataFrame:
    """Return track_labels + columns [split, ais_withheld, dark_label].

    dark_label is defined only for eval rows: True when a matched track has its
    AIS withheld (known dark), False for matched tracks keeping AIS. Unmatched
    tracks get split='unmatched_pool' (qualitative, dark_label=NaN).
    """
    rng = np.random.default_rng(seed)
    tl = track_labels.copy()

    eligible = tl["track_matched"] & (tl["match_frac"] >= min_match_frac)
    tl["split"] = "unmatched_pool"
    tl["ais_withheld"] = False
    tl["dark_label"] = pd.Series(pd.NA, index=tl.index, dtype="object")

    idx = tl.index[eligible].to_numpy()
    rng.shuffle(idx)
    n = len(idx)
    n_tr = int(ratios[0] * n)
    n_va = int(ratios[1] * n)
    parts = {"train": idx[:n_tr], "val": idx[n_tr:n_tr + n_va], "test": idx[n_tr + n_va:]}
    for name, ix in parts.items():
        tl.loc[ix, "split"] = name

    test_ix = parts["test"]
    withhold = rng.random(len(test_ix)) < dropout_frac
    tl.loc[test_ix[withhold], "ais_withheld"] = True
    tl.loc[test_ix, "dark_label"] = tl.loc[test_ix, "ais_withheld"]
    return tl


def split_summary(tl: pd.DataFrame) -> dict:
    test = tl[tl["split"] == "test"]
    return {
        "eligible_tracks": int((tl["split"].isin(["train", "val", "test"])).sum()),
        "train": int((tl["split"] == "train").sum()),
        "val": int((tl["split"] == "val").sum()),
        "test": int((tl["split"] == "test").sum()),
        "test_dark_pos": int((test["dark_label"] == True).sum()),
        "test_neg": int((test["dark_label"] == False).sum()),
        "unmatched_pool": int((tl["split"] == "unmatched_pool").sum()),
    }
