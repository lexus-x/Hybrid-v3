"""Han.md (Dark-MFM) vs Hybrid-router-v2 — focused visual comparison.

Builds outputs/han_vs_hybrid/ with four standalone figures + one combined dashboard:
  01_detection_quality.png   clean-AUROC bars, Han.md learned vs Hybrid
  02_false_dark_vs_offset.png  false-dark vs AIS positional error (the robustness story)
  03_mfm_ablation.png        the 4 MFM configs vs the hybrid line (components don't help)
  04_verdict_dashboard.png   one-glance scoreboard + verdict text

Data: outputs/busan_hybrid_v2.json, outputs/busan_fusion_ablation.json
Run:  /home/user/anaconda3/bin/python -m viz.han_vs_hybrid
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
DST = os.path.join(OUT, "han_vs_hybrid")

HAN = "#7d3c98"      # Han.md / Dark-MFM (purple)
HYB = "#1a7a4c"      # Hybrid router v2 (green)
GEO = "#e67e22"      # geometric rule baseline (orange)
COS = "#2471a3"      # plain cosine matcher (blue)
GREY = "#555555"

plt.rcParams.update({"font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True})


def _load():
    hv2 = json.load(open(os.path.join(OUT, "busan_hybrid_v2.json")))
    fab = json.load(open(os.path.join(OUT, "busan_fusion_ablation.json")))
    return hv2, fab


# ---------------------------------------------------------------- fig 1
def detection_quality(hv2, fab):
    han = fab["configs"]["han_full"]
    han_la = han["learned_clean_auroc_mean"]; han_ls = han["learned_clean_auroc_std"]
    hyb_a = hv2["clean_auroc"]["hybrid_mean"]
    cos_a = fab["reference_prev"]["cosine_multiseed_clean_auroc"]
    geo_a = hv2["clean_auroc"]["geometric_mean"]

    labels = ["Geometric\nrule", "Cosine\nmatcher", "Han.md\nDark-MFM", "Hybrid\nrouter v2"]
    vals   = [geo_a, cos_a, han_la, hyb_a]
    errs   = [0, 0, han_ls, 0]
    cols   = [GEO, COS, HAN, HYB]

    fig, ax = plt.subplots(figsize=(8.4, 5.0)); fig.patch.set_facecolor("white")
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, yerr=errs, capsize=6, color=cols, edgecolor="white", linewidth=1.5)
    for xi, v, e in zip(x, vals, errs):
        tag = f"{v:.2f}" + (f"\n±{e:.2f}" if e else "")
        ax.text(xi, v + (e if e else 0) + 0.02, tag, ha="center", va="bottom",
                fontsize=10.5, weight="bold")
    ax.axhline(0.5, color=GREY, ls=":", lw=1)
    ax.text(len(labels) - 0.5, 0.515, "chance", color=GREY, fontsize=8.5, ha="right")
    ax.set_ylim(0, 1.12); ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Clean dark-vessel AUROC  (higher = better)")
    ax.set_title("Detection quality on clean Busan tracks\nHan.md's end-to-end fusion vs the hybrid router",
                 weight="bold")
    fig.text(0.5, -0.02,
             "Geometric is perfect on clean data but brittle (see false-dark). The Han.md MFM (0.72) does "
             "not beat the\nplain cosine matcher (0.80); the hybrid router reaches 0.94 with zero learned variance.",
             ha="center", fontsize=9.2, style="italic", color="#444")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, "01_detection_quality.png")


# ---------------------------------------------------------------- fig 2
def false_dark_vs_offset(hv2, fab):
    off = np.array(hv2["offsets_m"], float)
    geo = np.array(hv2["false_dark"]["geometric_mean"])
    han = np.array(fab["configs"]["han_full"]["learned_fd_mean"])
    han_s = np.array(fab["configs"]["han_full"]["learned_fd_std"])
    hyb = np.array(hv2["false_dark"]["hybrid_mean"])
    hyb_s = np.array(hv2["false_dark"]["hybrid_std"])

    fig, ax = plt.subplots(figsize=(8.8, 5.2)); fig.patch.set_facecolor("white")
    ax.plot(off, geo, "-o", color=GEO, lw=2.2, label="Geometric rule")
    ax.plot(off, han, "-s", color=HAN, lw=2.2, label="Han.md Dark-MFM (learned)")
    ax.fill_between(off, han - han_s, han + han_s, color=HAN, alpha=0.13)
    ax.plot(off, hyb, "-^", color=HYB, lw=2.6, label="Hybrid router v2")
    ax.fill_between(off, hyb - hyb_s, hyb + hyb_s, color=HYB, alpha=0.15)

    ax.axhline(0.10, color="red", ls="--", lw=1.3)
    ax.text(off[-1], 0.115, "0.10 target", color="red", ha="right", fontsize=9)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("AIS positional error injected (metres)")
    ax.set_ylabel("False-dark rate  (lower = better)")
    ax.set_title("Robustness: false alarms as AIS drifts off the radar return", weight="bold")
    ax.legend(loc="center left")
    fig.text(0.5, -0.02,
             "The geometric rule collapses (0.98 false-dark at 500 m). Han.md's learned model is flat but "
             "stuck at a ~0.21 floor.\nThe hybrid stays ≤0.14 at every offset — best of both, which is why it ships.",
             ha="center", fontsize=9.2, style="italic", color="#444")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, "02_false_dark_vs_offset.png")


# ---------------------------------------------------------------- fig 3
def mfm_ablation(hv2, fab):
    order = [("cosine_baseline", "Cosine\n(no fusion)"), ("t2v", "+ Time2Vec"),
             ("cross", "+ Cross-attn"), ("han_full", "Han.md full\n(T2V+cross)")]
    means = [fab["configs"][k]["learned_clean_auroc_mean"] for k, _ in order]
    stds  = [fab["configs"][k]["learned_clean_auroc_std"] for k, _ in order]
    labels = [lbl for _, lbl in order]
    cols = [COS, "#9b59b6", "#8e44ad", HAN]
    hyb = hv2["clean_auroc"]["hybrid_mean"]

    fig, ax = plt.subplots(figsize=(8.6, 5.0)); fig.patch.set_facecolor("white")
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=6, color=cols, edgecolor="white", linewidth=1.5)
    for xi, m, s in zip(x, means, stds):
        ax.text(xi, m + s + 0.012, f"{m:.2f}\n±{s:.2f}", ha="center", va="bottom",
                fontsize=9.5, weight="bold")
    ax.axhline(hyb, color=HYB, ls="--", lw=2)
    ax.text(len(labels) - 0.5, hyb + 0.012, f"Hybrid router = {hyb:.2f}",
            color=HYB, ha="right", fontsize=10, weight="bold")
    ax.set_ylim(0, 1.05); ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Learned clean AUROC  (5 seeds, n=19/seed)")
    ax.set_title("Adding Han.md's components does not raise accuracy", weight="bold")
    fig.text(0.5, -0.02,
             "Stacking Time2Vec and cross-attention onto the cosine matcher lowers mean AUROC (0.77→0.72) and "
             "raises variance\n(±0.05→±0.09). At n=19 the heavier MFM is data-starved — the simple hybrid wins.",
             ha="center", fontsize=9.2, style="italic", color="#444")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, "03_mfm_ablation.png")


# ---------------------------------------------------------------- fig 4
def verdict_dashboard(hv2, fab):
    han = fab["configs"]["han_full"]
    rows = [
        ("Metric", "Han.md  Dark-MFM", "Hybrid router v2", "Winner"),
        ("Clean AUROC", f"{han['learned_clean_auroc_mean']:.2f} ± {han['learned_clean_auroc_std']:.2f}",
         f"{hv2['clean_auroc']['hybrid_mean']:.2f} ± {hv2['clean_auroc']['hybrid_std']:.2f}", "Hybrid"),
        ("False-dark @ 0 m", f"{han['learned_fd_mean'][0]:.2f}",
         f"{hv2['false_dark']['hybrid_mean'][0]:.2f}", "Hybrid"),
        ("False-dark @ 500 m", f"{han['learned_fd_mean'][-1]:.2f}",
         f"{hv2['false_dark']['hybrid_mean'][-1]:.2f}", "Hybrid"),
        ("Seed variance", "high (±0.09)", "zero (deterministic)", "Hybrid"),
        ("Hand-tuned rules", "none (end-to-end)", "3 router cues", "Han.md"),
        ("Verdict @ n=19", "data-starved", "ships", "Hybrid"),
    ]

    fig, ax = plt.subplots(figsize=(10.5, 4.8)); fig.patch.set_facecolor("white")
    ax.axis("off")
    ncol = 4; nrow = len(rows)
    cw = [0.30, 0.27, 0.27, 0.16]; cx = np.cumsum([0] + cw)
    for r, row in enumerate(rows):
        y = 1 - (r + 0.5) / nrow
        for c, cell in enumerate(row):
            xc = cx[c] + cw[c] / 2
            if r == 0:
                ax.add_patch(plt.Rectangle((cx[c], 1 - (r + 1) / nrow), cw[c], 1 / nrow,
                             color="#2c3e50", zorder=0))
                ax.text(xc, y, cell, ha="center", va="center", color="white",
                        weight="bold", fontsize=11)
            else:
                bg = "#f4f6f7" if r % 2 else "#eaeded"
                ax.add_patch(plt.Rectangle((cx[c], 1 - (r + 1) / nrow), cw[c], 1 / nrow,
                             color=bg, zorder=0))
                col = HAN if (c == 3 and cell == "Han.md") else (HYB if c == 3 else "#222")
                wt = "bold" if c in (0, 3) else "normal"
                ax.text(xc, y, cell, ha="center", va="center", color=col,
                        weight=wt, fontsize=10.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.suptitle("Han.md (Dark-MFM)  vs  Hybrid router v2  —  scoreboard",
                 weight="bold", fontsize=14, y=1.02)
    fig.text(0.5, -0.04,
             "Bottom line: Han.md was tested fairly and logged as an honest negative at this data scale. "
             "Fusion's value is a\nscale hypothesis — re-run on a larger public dataset before retiring the hybrid.",
             ha="center", fontsize=9.5, style="italic", color="#444")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    _save(fig, "04_verdict_dashboard.png")


def _save(fig, name):
    p = os.path.join(DST, name)
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    print("wrote", os.path.relpath(p, OUT))


if __name__ == "__main__":
    os.makedirs(DST, exist_ok=True)
    hv2, fab = _load()
    detection_quality(hv2, fab)
    false_dark_vs_offset(hv2, fab)
    mfm_ablation(hv2, fab)
    verdict_dashboard(hv2, fab)
    print("done ->", DST)
