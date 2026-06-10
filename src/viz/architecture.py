"""Architecture diagrams for the README: v2 hybrid router + MFM fusion matcher.

Pure matplotlib (no graphviz). Clean org-style: soft palette, rounded boxes,
labelled arrows. Outputs PNGs to outputs/assets/.
Run: python -m viz.architecture
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "assets")

RADAR = "#c0392b"; AIS = "#2471a3"; LEARN = "#1e8449"; GATE = "#b9770e"
INK = "#222222"; SOFT = "#5d6d7e"; BG = "#ffffff"
FILL = "#f4f6f7"; FILL2 = "#eaf2f8"; FILL3 = "#e9f7ef"; FILL4 = "#fbeee6"


def box(ax, x, y, w, h, text, fc=FILL, ec=SOFT, tc=INK, fs=11, bold=False, lw=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.03",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=3, weight="bold" if bold else "normal")


def arrow(ax, p1, p2, color=SOFT, lw=2.0, label=None, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16,
                                 color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.12, label, ha="center", va="bottom", fontsize=8.5, color=color, style="italic")


def _canvas(w=12, h=6.4):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, 12); ax.set_ylim(0, 6.4)
    ax.axis("off"); fig.patch.set_facecolor(BG)
    return fig, ax


def diagram_v2():
    fig, ax = _canvas()
    ax.text(0.2, 6.05, "v2 — Calibrated Open-Set Dark-Vessel Detection (Hybrid Router)",
            fontsize=14.5, weight="bold", color=INK)
    # inputs
    box(ax, 0.3, 4.55, 2.3, 0.9, "Radar tracklet\n[Δlat,Δlon,sog,cog,sig,Δt]", FILL4, RADAR, RADAR, 9.5)
    box(ax, 0.3, 1.0, 2.3, 0.9, "Time-aligned\nAIS window", FILL2, AIS, AIS, 9.5)
    # encoders
    box(ax, 3.1, 4.55, 2.4, 0.9, "Track Encoder\n(pre-norm Transformer)", FILL4, RADAR, INK, 9.5)
    box(ax, 3.1, 1.0, 2.4, 0.9, "AIS Encoder\n(Transformer)", FILL2, AIS, INK, 9.5)
    arrow(ax, (2.6, 5.0), (3.1, 5.0), RADAR); arrow(ax, (2.6, 1.45), (3.1, 1.45), AIS)
    # learned matcher
    box(ax, 6.0, 2.55, 2.6, 1.3, "Open-Set Matcher\ncosine + learned\n‘absent’ logit → P(dark)", FILL3, LEARN, INK, 9.8, bold=True)
    arrow(ax, (5.5, 5.0), (6.2, 3.85), RADAR, rad=-0.15)
    arrow(ax, (5.5, 1.45), (6.2, 2.55), AIS, rad=0.15)
    # geometric gate
    box(ax, 6.0, 4.7, 2.6, 0.95, "Geometric gate\n(time/dist/angle)", FILL, GATE, INK, 9.8)
    arrow(ax, (2.6, 5.25), (6.0, 5.2), GATE, lw=1.6, ls=(0, (4, 3)))
    # router
    box(ax, 9.0, 3.3, 2.6, 1.5,
        "Confidence Router\n‘tight match?’\nclose+course+speed", "#fdf2e9", GATE, INK, 9.8, bold=True)
    arrow(ax, (8.6, 5.15), (9.2, 4.8), GATE, label="residual")
    arrow(ax, (8.6, 3.2), (9.2, 3.6), LEARN, label="P(dark)")
    # output
    box(ax, 9.4, 1.0, 1.9, 1.0, "Dark decision\n+ calibrated score", "#f5eef8", "#7d3c98", "#7d3c98", 9.8, bold=True)
    arrow(ax, (10.3, 3.3), (10.3, 2.0), "#7d3c98")
    ax.text(9.05, 2.35, "confident → gate\nuncertain → learned", fontsize=8, color=SOFT, style="italic")
    fig.tight_layout()
    p = os.path.join(OUT, "arch_v2_hybrid.png"); fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    return p


def diagram_mfm():
    fig, ax = _canvas()
    ax.text(0.2, 6.05, "MFM — Multimodal Fusion Matcher (Time2Vec + Cross-Attention)",
            fontsize=14.5, weight="bold", color=INK)
    box(ax, 0.3, 4.55, 2.2, 0.9, "Radar tracklet", FILL4, RADAR, RADAR, 10)
    box(ax, 0.3, 1.0, 2.2, 0.9, "AIS candidates\n(K tracklets)", FILL2, AIS, AIS, 10)
    # time2vec
    box(ax, 2.9, 4.55, 1.9, 0.9, "Time2Vec\n+ Token Enc.", FILL4, RADAR, INK, 9.5)
    box(ax, 2.9, 1.0, 1.9, 0.9, "Time2Vec\n+ Token Enc.", FILL2, AIS, INK, 9.5)
    arrow(ax, (2.5, 5.0), (2.9, 5.0), RADAR); arrow(ax, (2.5, 1.45), (2.9, 1.45), AIS)
    # tokens
    box(ax, 5.2, 4.55, 1.7, 0.9, "Radar tokens", "#fdf2e9", RADAR, INK, 9.5)
    box(ax, 5.2, 1.0, 1.7, 0.9, "AIS tokens", "#eaf2f8", AIS, INK, 9.5)
    arrow(ax, (4.8, 5.0), (5.2, 5.0), RADAR); arrow(ax, (4.8, 1.45), (5.2, 1.45), AIS)
    # cross-attention
    box(ax, 7.3, 2.7, 2.5, 1.5, "Cross-Attention\nradar ⟶ AIS\n(detects conflict)", FILL3, LEARN, INK, 10, bold=True)
    arrow(ax, (6.9, 5.0), (7.6, 4.2), RADAR, rad=-0.2, label="query")
    arrow(ax, (6.9, 1.45), (7.6, 2.7), AIS, rad=0.2, label="key/value")
    # match mlp + absent
    box(ax, 10.1, 3.5, 1.7, 1.0, "Match MLP\n→ logit/cand", FILL3, LEARN, INK, 9.5)
    box(ax, 10.1, 2.1, 1.7, 0.85, "learned\n‘absent’ logit", "#f5eef8", "#7d3c98", "#7d3c98", 9)
    arrow(ax, (9.8, 3.6), (10.1, 3.9), LEARN)
    # softmax -> p(dark)
    box(ax, 10.25, 0.6, 1.45, 0.85, "softmax\n→ P(dark)", "#f9ebea", "#922b21", "#922b21", 9.5, bold=True)
    arrow(ax, (10.95, 3.5), (10.95, 1.45), "#922b21", rad=0.0)
    fig.tight_layout()
    p = os.path.join(OUT, "arch_mfm_fusion.png"); fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    return p


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("wrote", diagram_v2())
    print("wrote", diagram_mfm())
