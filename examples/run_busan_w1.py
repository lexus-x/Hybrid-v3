"""W1 gate (Busan radar+AIS arm): load -> geometric match -> track labels ->
controlled-dropout splits -> save reproducible artifacts.

Run from the repository root:
    PYTHONPATH=src python examples/run_busan_w1.py
"""
from __future__ import annotations

import json
import os

from data.busan_loader import load_ais, load_radar
from data.pair_builder import GateCfg, match_radar_to_ais, summary, track_labels
from data.dropout_splitter import make_splits, split_summary

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = GateCfg()
    ais, rad = load_ais(), load_radar()
    pts = match_radar_to_ais(rad, ais, cfg)
    tl = track_labels(pts, cfg)
    summ = summary(pts, tl)
    splits = make_splits(tl)
    summ["splits"] = split_summary(splits)
    summ["gate_cfg"] = cfg.__dict__

    splits.to_csv(os.path.join(OUT, "busan_track_labels.csv"), index=False)
    with open(os.path.join(OUT, "busan_w1.json"), "w") as f:
        json.dump(summ, f, indent=2)

    print(json.dumps(summ, indent=2))
    print("\nGEOMETRIC BASELINE (Busan radar+AIS) — point match rate "
          f"{summ['point_match_rate']:.3f}, P90 dist {summ['dist_p90_m']:.1f} m, "
          f"bias {summ['bias_mag_m']:.1f} m ({100*summ['bias_mag_m']/cfg.dist_gate:.1f}% of gate)")
    print(f"Dark eval positives available (test, AIS withheld): {summ['splits']['test_dark_pos']}")


if __name__ == "__main__":
    main()
