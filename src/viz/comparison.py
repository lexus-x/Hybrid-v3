"""Comparison figures for the README: baselines vs v2-hybrid vs MFM-fusion.

Reads the result JSONs in outputs/ and renders:
  - cmp_robustness.png : false-dark rate vs AIS offset (geometric / learned / hybrid / MFM)
  - cmp_clean_auroc.png: clean-split dark AUROC bars across all methods
  - scoreboard.png     : headline numbers card
Run: python -m viz.comparison
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
ASSETS = os.path.join(OUT, "assets")
RADAR = "#c0392b"; AIS = "#2471a3"; LEARN = "#1e8449"; HYB = "#1e8449"; MFM = "#7d3c98"; GEO = "#c0392b"


def _load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def robustness_fig():
    hv2 = _load("busan_hybrid_v2.json")
    fab = _load("busan_fusion_ablation.json")
    if not hv2:
        print("no hybrid_v2 json yet"); return None
    off = hv2["offsets_m"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4)); fig.patch.set_facecolor("white")
    ax.plot(off, hv2["false_dark"]["geometric_mean"], "o-", color=GEO, lw=2, label="Geometric gate (rule)")
    ax.plot(off, hv2["false_dark"]["learned_mean"], "s-", color=AIS, lw=2, label="Learned open-set")
    ax.plot(off, hv2["false_dark"]["hybrid_mean"], "^-", color=HYB, lw=2.6, label="Hybrid v2 (ours)")
    if fab and "han_full" in fab.get("configs", {}):
        ax.plot(off, fab["configs"]["han_full"]["hybrid_fd_mean"], "D--", color=MFM, lw=2.2,
                label="MFM hybrid (Time2Vec+X-attn)")
    ax.axhline(0.10, ls=":", color="#888", lw=1.2); ax.text(off[1], 0.115, "0.10 target", fontsize=8, color="#888")
    ax.set_xlabel("Injected AIS registration offset (m)"); ax.set_ylabel("False-dark rate (lower = better)")
    ax.set_title("Robustness to AIS registration error (Busan, 5 seeds)")
    ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout(); p = os.path.join(ASSETS, "cmp_robustness.png")
    fig.savefig(p, dpi=170); plt.close(fig); return p


def auroc_fig():
    hv2 = _load("busan_hybrid_v2.json"); fab = _load("busan_fusion_ablation.json")
    if not hv2:
        return None
    names, vals, errs, cols = [], [], [], []
    names.append("Geometric"); vals.append(hv2["clean_auroc"]["geometric_mean"]); errs.append(0); cols.append(GEO)
    names.append("Learned"); vals.append(hv2["clean_auroc"]["learned_mean"]); errs.append(hv2["clean_auroc"]["learned_std"]); cols.append(AIS)
    names.append("Hybrid v2"); vals.append(hv2["clean_auroc"]["hybrid_mean"]); errs.append(hv2["clean_auroc"]["hybrid_std"]); cols.append(HYB)
    if fab and "han_full" in fab.get("configs", {}):
        c = fab["configs"]["han_full"]
        names.append("MFM learned"); vals.append(c["learned_clean_auroc_mean"]); errs.append(c["learned_clean_auroc_std"]); cols.append(MFM)
        names.append("MFM hybrid"); vals.append(c["hybrid_clean_auroc_mean"]); errs.append(c["hybrid_clean_auroc_std"]); cols.append("#5b2c6f")
    fig, ax = plt.subplots(figsize=(7.2, 4.2)); fig.patch.set_facecolor("white")
    x = np.arange(len(names))
    ax.bar(x, vals, yerr=errs, capsize=5, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=12, fontsize=9)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Clean-split dark AUROC (n=19)")
    ax.set_title("Dark-detection AUROC by method")
    fig.tight_layout(); p = os.path.join(ASSETS, "cmp_clean_auroc.png")
    fig.savefig(p, dpi=170); plt.close(fig); return p


def scoreboard():
    hv2 = _load("busan_hybrid_v2.json")
    if not hv2:
        return None
    fig, ax = plt.subplots(figsize=(9, 2.4)); ax.axis("off"); fig.patch.set_facecolor("white")
    cards = [
        ("Baseline reproduced", "0.740", "point-match (PDF 0.744)", RADAR),
        ("Clean AUROC", f"{hv2['clean_auroc']['hybrid_mean']:.2f}", "hybrid vs 0.76 learned", HYB),
        ("False-dark @500 m", f"{hv2['false_dark']['hybrid_mean'][-1]:.2f}", "hybrid vs 0.98 rule", AIS),
        ("Robustness gain", "7×", "fewer false darks @500 m", MFM),
    ]
    for i, (t, big, sub, c) in enumerate(cards):
        x = 0.02 + i * 0.245
        ax.add_patch(plt.Rectangle((x, 0.1), 0.225, 0.8, transform=ax.transAxes,
                                   fc="#f8f9f9", ec=c, lw=2))
        ax.text(x + 0.112, 0.72, t, transform=ax.transAxes, ha="center", fontsize=9.5, color="#555")
        ax.text(x + 0.112, 0.45, big, transform=ax.transAxes, ha="center", fontsize=22, weight="bold", color=c)
        ax.text(x + 0.112, 0.22, sub, transform=ax.transAxes, ha="center", fontsize=8, color="#777")
    fig.tight_layout(); p = os.path.join(ASSETS, "scoreboard.png")
    fig.savefig(p, dpi=170); plt.close(fig); return p


if __name__ == "__main__":
    os.makedirs(ASSETS, exist_ok=True)
    for f in (robustness_fig, auroc_fig, scoreboard):
        r = f(); print("wrote", r) if r else print(f.__name__, "skipped (missing json)")
