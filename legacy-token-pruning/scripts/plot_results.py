#!/usr/bin/env python3
"""
Generate the publication-quality result figures for the CIFAR-10 token-pruning
project. Reads measured data from:
  * outputs/benchmark_results.json            (accuracy / FLOPs / throughput / VRAM)
  * outputs/sota_pruned_small_atp_final/metrics.csv  (training curve)

Writes PNGs to outputs/figures/ (and these are copied into docs/figures/ at
repo-assembly time).
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]          # repo root
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Consistent professional palette
C_DENSE = "#4C72B0"
C_PRUNED = "#DD8452"
C_ACCENT = "#55A868"
C_MUTED = "#8C8C8C"
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.axisbelow": True,
})


def load_results():
    with (ROOT / "outputs" / "benchmark_results.json").open() as f:
        return json.load(f)


def load_curve():
    # The CSV contains a duplicate header (training was restarted after 2 epochs).
    # Keep only numeric rows and dedupe by epoch, keeping the last occurrence so
    # the abandoned 2-epoch attempt is overwritten by the real 100-epoch run.
    p = ROOT / "outputs" / "sota_pruned_small_atp_final" / "metrics.csv"
    by_epoch = {}
    with p.open() as f:
        for r in csv.DictReader(f):
            try:
                e = int(r["Epoch"])
            except (ValueError, TypeError):
                continue
            by_epoch[e] = r
    return [by_epoch[e] for e in sorted(by_epoch)]


def fig_accuracy(r):
    # Same-arch ablation plus larger dense references for context.
    labels = ["Teacher\nViT-L/384", "Dense ref\nViT-B/224",
              "Student OFF\n(keep=1.0)", "Student ON\n(pruned)"]
    vals = [99.34, 99.16, r["accuracy"]["pruned_off"] * 100,
            r["accuracy"]["pruned_on"] * 100]
    colors = [C_MUTED, C_MUTED, C_DENSE, C_PRUNED]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labels, vals, color=colors, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(98.5, 99.6)
    ax.set_ylabel("CIFAR-10 test accuracy (%)")
    ax.set_title("Accuracy: token pruning costs only 0.08 pt\n(same weights, pruning toggled)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_accuracy.png", bbox_inches="tight")
    plt.close(fig)


def fig_token_budget(r):
    pp = r["accuracy"]["num_patches"]
    stages = ["Input\n(L0)"] + [f"After L{d['layer']}" for d in r["accuracy"]["kept_tokens_per_layer"]]
    toks = [pp] + [round(d["avg_kept_tokens"]) for d in r["accuracy"]["kept_tokens_per_layer"]]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(stages, toks, color=[C_DENSE] + [C_PRUNED] * (len(toks) - 1), width=0.6)
    for b, t in zip(bars, toks):
        pct = t / pp * 100
        ax.text(b.get_x() + b.get_width() / 2, t + 6, f"{t}\n({pct:.0f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Patch tokens processed")
    ax.set_ylim(0, pp * 1.18)
    ax.set_title("Token budget compounds across stages: 576 → 87 (≈15% kept)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_token_budget.png", bbox_inches="tight")
    plt.close(fig)


def fig_flops(r):
    g = r["gflops"]
    fig, ax = plt.subplots(figsize=(5, 4.2))
    bars = ax.bar(["Dense", "Pruned (ATP)"], [g["dense"], g["pruned"]],
                  color=[C_DENSE, C_PRUNED], width=0.55)
    for b, v in zip(bars, [g["dense"], g["pruned"]]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}",
                ha="center", va="bottom", fontweight="bold")
    red = (1 - g["pruned"] / g["dense"]) * 100
    ax.set_ylabel("GFLOPs (batch size 1)")
    ax.set_ylim(0, g["dense"] * 1.18)
    ax.set_title(f"Compute: −{red:.0f}% FLOPs")
    fig.tight_layout()
    fig.savefig(OUT / "fig_flops.png", bbox_inches="tight")
    plt.close(fig)


def fig_throughput(r):
    thr = r["throughput"]
    bss = sorted(int(k) for k in thr["dense"].keys())
    d = [thr["dense"][str(b)]["throughput_img_s"] for b in bss]
    p = [thr["pruned"][str(b)]["throughput_img_s"] for b in bss]
    x = range(len(bss))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.2))
    b1 = ax.bar([i - w / 2 for i in x], d, w, label="Dense", color=C_DENSE)
    b2 = ax.bar([i + w / 2 for i in x], p, w, label="Pruned (ATP)", color=C_PRUNED)
    for i, b in enumerate(bss):
        su = p[i] / d[i]
        y = max(d[i], p[i])
        ax.text(i, y + 8, f"{su:.2f}×", ha="center", va="bottom",
                fontsize=10, fontweight="bold",
                color=C_ACCENT if su >= 1 else "#C0392B")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"bs={b}" for b in bss])
    ax.set_ylabel("Throughput (images / s)")
    ax.set_title("Throughput: 1.35–1.40× at server batch sizes,\nbut slower at batch size 1 (overhead dominates)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_throughput.png", bbox_inches="tight")
    plt.close(fig)


def fig_vram(r):
    thr = r["throughput"]
    bss = sorted(int(k) for k in thr["dense"].keys())
    d = [thr["dense"][str(b)]["peak_vram_mb"] for b in bss]
    p = [thr["pruned"][str(b)]["peak_vram_mb"] for b in bss]
    x = range(len(bss))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar([i - w / 2 for i in x], d, w, label="Dense", color=C_DENSE)
    ax.bar([i + w / 2 for i in x], p, w, label="Pruned (ATP)", color=C_PRUNED)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"bs={b}" for b in bss])
    ax.set_ylabel("Peak VRAM (MB)")
    ax.set_title("Limitation: ATP-Lite uses MORE memory\n(asymmetric attention keeps full-length K/V)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_vram.png", bbox_inches="tight")
    plt.close(fig)


def fig_training_curve(rows):
    ep = [int(r["Epoch"]) for r in rows]
    tr = [float(r["Train_Acc"]) * 100 for r in rows]
    te = [float(r["Test_Acc"]) * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ep, te, color=C_PRUNED, lw=2.2, label="Test accuracy")
    ax.plot(ep, tr, color=C_DENSE, lw=1.6, alpha=0.8, label="Train accuracy")
    best = max(te)
    bi = ep[te.index(best)]
    ax.scatter([bi], [best], color=C_ACCENT, zorder=5)
    ax.annotate(f"best {best:.2f}% @ ep {bi}", (bi, best),
                textcoords="offset points", xytext=(-10, -18), fontsize=9)
    ax.axvspan(1, 6, color=C_MUTED, alpha=0.12)
    ax.text(3.5, ax.get_ylim()[0] + 2, "pruning\nwarmup+ramp", ha="center",
            fontsize=8, color=C_MUTED)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training curve — pruned ViT-Small/384 with KD (100 epochs)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_training_curve.png", bbox_inches="tight")
    plt.close(fig)


def main():
    r = load_results()
    fig_accuracy(r)
    fig_token_budget(r)
    fig_flops(r)
    fig_throughput(r)
    fig_vram(r)
    try:
        fig_training_curve(load_curve())
    except Exception as e:
        print(f"training curve skipped: {e}", file=sys.stderr)
    print("wrote figures to", OUT)
    for p in sorted(OUT.glob("*.png")):
        print(" ", p.name)


if __name__ == "__main__":
    main()
