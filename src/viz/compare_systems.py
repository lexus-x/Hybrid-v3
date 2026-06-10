"""Explicit v2-vs-MFM comparison assets for the README.

cmp_v2_vs_mfm.png : head-to-head bars (detection quality + false-dark), v2 vs MFM.
arch_comparison.png: the two architecture diagrams stacked into one image + verdict.

Run: python -m viz.compare_systems
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
ASSETS = os.path.join(OUT, "assets")
V2 = "#2471a3"; MFM = "#7d3c98"


def head_to_head():
    hv2 = json.load(open(os.path.join(OUT, "busan_hybrid_v2.json")))
    fab = json.load(open(os.path.join(OUT, "busan_fusion_ablation.json")))
    m = fab["configs"]["han_full"]
    v2 = dict(la=hv2["clean_auroc"]["learned_mean"], ha=hv2["clean_auroc"]["hybrid_mean"],
              fd0=hv2["false_dark"]["hybrid_mean"][0], fd5=hv2["false_dark"]["hybrid_mean"][-1])
    mf = dict(la=m["learned_clean_auroc_mean"], ha=m["hybrid_clean_auroc_mean"],
              fd0=m["hybrid_fd_mean"][0], fd5=m["hybrid_fd_mean"][-1])

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4)); fig.patch.set_facecolor("white")
    fig.suptitle("v2  (cosine matcher + hybrid router)   vs   MFM  (Time2Vec + cross-attention + hybrid)",
                 fontsize=12.5, weight="bold", y=1.0)

    # panel A: detection quality (higher = better)
    g = ["Learned\nAUROC", "Hybrid\nAUROC"]; x = np.arange(2); w = 0.36
    a.bar(x - w / 2, [v2["la"], v2["ha"]], w, color=V2, label="v2")
    a.bar(x + w / 2, [mf["la"], mf["ha"]], w, color=MFM, label="MFM")
    for i, (v, mv) in enumerate(zip([v2["la"], v2["ha"]], [mf["la"], mf["ha"]])):
        a.text(i - w / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=9, weight="bold")
        a.text(i + w / 2, mv + 0.015, f"{mv:.2f}", ha="center", fontsize=9, weight="bold")
    a.set_xticks(x); a.set_xticklabels(g); a.set_ylim(0, 1.08)
    a.set_ylabel("AUROC (higher = better)"); a.set_title("Detection quality"); a.legend()

    # panel B: false-dark (lower = better)
    g2 = ["False-dark\n@ 0 m", "False-dark\n@ 500 m"]
    b.bar(x - w / 2, [v2["fd0"], v2["fd5"]], w, color=V2, label="v2")
    b.bar(x + w / 2, [mf["fd0"], mf["fd5"]], w, color=MFM, label="MFM")
    for i, (v, mv) in enumerate(zip([v2["fd0"], v2["fd5"]], [mf["fd0"], mf["fd5"]])):
        b.text(i - w / 2, v + 0.006, f"{v:.2f}", ha="center", fontsize=9, weight="bold")
        b.text(i + w / 2, mv + 0.006, f"{mv:.2f}", ha="center", fontsize=9, weight="bold")
    b.set_xticks(x); b.set_xticklabels(g2); b.set_ylim(0, max(v2["fd5"], mf["fd5"]) * 1.5)
    b.set_ylabel("False-dark rate (lower = better)"); b.set_title("Robustness"); b.legend()

    fig.text(0.5, -0.03, "Verdict: at n=19 the heavier MFM matches v2 — no statistically meaningful gain. "
             "Fusion's value is a scale hypothesis, to be tested on a public dataset.",
             ha="center", fontsize=10, style="italic", color="#444")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    p = os.path.join(ASSETS, "cmp_v2_vs_mfm.png"); fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    return p


def arch_side_by_side():
    p_v2 = os.path.join(ASSETS, "arch_v2_hybrid.png")
    p_mfm = os.path.join(ASSETS, "arch_mfm_fusion.png")
    if not (os.path.exists(p_v2) and os.path.exists(p_mfm)):
        return None
    im1, im2 = mpimg.imread(p_v2), mpimg.imread(p_mfm)
    fig, (a, b) = plt.subplots(2, 1, figsize=(12, 11)); fig.patch.set_facecolor("white")
    a.imshow(im1); a.axis("off"); b.imshow(im2); b.axis("off")
    fig.suptitle("Architecture of both detectors", fontsize=16, weight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    p = os.path.join(ASSETS, "arch_comparison.png"); fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p


if __name__ == "__main__":
    os.makedirs(ASSETS, exist_ok=True)
    print("wrote", head_to_head())
    print("wrote", arch_side_by_side())
