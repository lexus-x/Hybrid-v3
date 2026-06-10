"""Animated explainer (GIF + MP4) for the README — SLOW, readable version.

Left panel  : a scene — radar track (fixed) + its true AIS drifting away by the
              offset; the geometric gate (300 m) and the router trust-radius (80 m).
              A large verdict line flips RULE: DARK (false alarm) vs matched.
Right panel : the measured false-dark curves with a marker sweeping the offset.

Paced with hold frames at 0 / 200 / 300 / 500 m and a low frame-rate so each
state is readable. Run: python -m viz.animation
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
ASSETS = os.path.join(OUT, "assets")
GEO = "#c0392b"; AIS = "#2471a3"; HYB = "#1e8449"

GATE = 300.0; TRUST = 80.0
GIF_FPS = 5            # slow: ~12 s clip
MP4_FPS = 6


def _curves():
    d = json.load(open(os.path.join(OUT, "busan_hybrid_v2.json")))
    return (d["offsets_m"], d["false_dark"]["geometric_mean"],
            d["false_dark"]["learned_mean"], d["false_dark"]["hybrid_mean"])


def _schedule():
    """Offset value per frame: ease in, pause at key offsets, long hold at the end."""
    seq = [0.0] * 8                                    # hold at clean
    seq += list(np.linspace(0, 200, 18))              # ramp to the crossover region
    seq += [200.0] * 6                                # pause: still fine
    seq += list(np.linspace(200, 300, 8))
    seq += [300.0] * 6                                # pause: gate boundary
    seq += list(np.linspace(300, 500, 14))
    seq += [500.0] * 12                               # long hold: rule has collapsed
    return seq


def build():
    off_x, g, l, h = _curves()
    seq = _schedule()
    fig, (axS, axC) = plt.subplots(1, 2, figsize=(12, 5.2)); fig.patch.set_facecolor("white")
    fig.suptitle("Why the rule breaks and the hybrid holds, as AIS drifts off the radar track",
                 fontsize=13.5, weight="bold", y=0.98)

    # ---- left scene ----
    axS.set_xlim(-600, 600); axS.set_ylim(-420, 380); axS.set_aspect("equal"); axS.axis("off")
    axS.add_patch(plt.Circle((0, 0), GATE, fc="none", ec=GEO, ls="--", lw=1.8))
    axS.add_patch(plt.Circle((0, 0), TRUST, fc="#e9f7ef", ec=HYB, ls=":", lw=1.8))
    axS.text(0, GATE + 18, "geometric gate (300 m)", color=GEO, fontsize=9.5, ha="center")
    axS.text(0, TRUST + 14, "router trust (80 m)", color=HYB, fontsize=9.5, ha="center")
    axS.scatter([0], [0], s=170, color="#222", zorder=5)
    axS.text(0, -40, "radar track", fontsize=10, ha="center")
    ais_line, = axS.plot([], [], color=AIS, lw=1.4, ls="-", alpha=0.6)
    ais_pt = axS.scatter([0], [0], s=150, color=AIS, zorder=6)
    ais_txt = axS.text(0, 0, "", fontsize=10, color=AIS, ha="left", weight="bold")
    off_txt = axS.text(0, 345, "", fontsize=13, ha="center", weight="bold", color="#333")
    verdict = axS.text(0, -390, "", fontsize=13.5, ha="center", weight="bold")

    # ---- right curves ----
    axC.plot(off_x, g, "o-", color=GEO, lw=2.2, label="Geometric (rule)")
    axC.plot(off_x, l, "s-", color=AIS, lw=2.2, label="Learned")
    axC.plot(off_x, h, "^-", color=HYB, lw=3.0, label="Hybrid (ours)")
    axC.set_xlim(0, 500); axC.set_ylim(-0.03, 1.03)
    axC.set_xlabel("AIS registration offset (m)", fontsize=11)
    axC.set_ylabel("False-dark rate (lower = better)", fontsize=11)
    axC.set_title("Measured false-dark rate (5 seeds)", fontsize=11.5)
    axC.legend(fontsize=10, loc="center left"); axC.grid(alpha=0.25)
    vline = axC.axvline(0, color="#444", lw=1.6)
    mk_g = axC.scatter([0], [g[0]], s=80, color=GEO, zorder=6, edgecolor="white")
    mk_h = axC.scatter([0], [h[0]], s=80, color=HYB, zorder=6, edgecolor="white")
    readout = axC.text(0.97, 0.5, "", transform=axC.transAxes, ha="right", fontsize=10.5,
                       bbox=dict(boxstyle="round", fc="white", ec="#ccc"))

    def frame(i):
        d = seq[i]
        ang = np.radians(35)
        x, y = d * np.cos(ang), d * np.sin(ang)
        ais_pt.set_offsets([[x, y]])
        ais_line.set_data([0, x], [0, y])
        ais_txt.set_position((x + 16, y + 10)); ais_txt.set_text(f"true AIS")
        off_txt.set_text(f"AIS registration offset:  {d:.0f} m")
        geo_dark = d > GATE
        verdict.set_text(("RULE  →  DARK  ✗  (false alarm)" if geo_dark else "RULE  →  matched  ✓")
                         + "        HYBRID  →  matched  ✓")
        verdict.set_color(GEO if geo_dark else HYB)
        vline.set_xdata([d, d])
        gi = float(np.interp(d, off_x, g)); hi = float(np.interp(d, off_x, h))
        mk_g.set_offsets([[d, gi]]); mk_h.set_offsets([[d, hi]])
        readout.set_text(f"@ {d:.0f} m\nrule:  {gi:.2f}\nhybrid: {hi:.2f}")
        return ais_pt, ais_line, ais_txt, off_txt, verdict, vline, mk_g, mk_h, readout

    anim = FuncAnimation(fig, frame, frames=len(seq), interval=1000 / GIF_FPS, blit=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(ASSETS, exist_ok=True)
    gif = os.path.join(ASSETS, "robustness_explainer.gif")
    anim.save(gif, writer=PillowWriter(fps=GIF_FPS)); print("wrote", gif, f"({len(seq)} frames @ {GIF_FPS}fps)")
    try:
        from matplotlib.animation import FFMpegWriter
        mp4 = os.path.join(ASSETS, "robustness_explainer.mp4")
        anim.save(mp4, writer=FFMpegWriter(fps=MP4_FPS, bitrate=2600)); print("wrote", mp4)
    except Exception as e:
        print("mp4 skipped:", e)
    plt.close(fig)


if __name__ == "__main__":
    build()
