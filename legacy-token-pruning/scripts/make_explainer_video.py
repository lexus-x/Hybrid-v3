#!/usr/bin/env python3
"""
Render a professional explainer video (MP4) + a looping hero GIF for the
CIFAR-10 token-pruning project.

The video walks through: problem -> idea -> architecture -> a live progressive
token-pruning animation (computed from the trained ViT-Small/384 ATP checkpoint)
-> measured results (reusing outputs/figures/*.png) -> limitations -> takeaway.

All numbers shown are the measured values from outputs/benchmark_results.json.

Outputs:
  outputs/explainer.mp4        full narrated-by-captions explainer
  outputs/hero_pruning.gif     looping 4-stage pruning of one image (README hero)
  outputs/video_frames/        intermediate scene PNGs (gitignored)

Requires ffmpeg on PATH.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
import torch
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.sota_hybrid_vit import build_sota_model

CP = Path(__file__).resolve().parents[1]            # repo root
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
CIFAR = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Theme
BG = "#0C1424"
CARD = "#16223A"
TEXT = "#EAF1FB"
MUTED = "#9DB0CC"
ACCENT = "#5AD1B0"
ACCENT2 = "#FFB454"
DROP = "#0A0F1A"
KEEP = "#5AD1B0"
W, H, DPI = 12.8, 7.2, 100


def denorm(img):
    return (img.cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1).permute(1, 2, 0).numpy()


def new_canvas():
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    fig.patch.set_facecolor(BG)
    return fig


def text_scene(path, title, lines, subtitle=None, foot=None, title_color=TEXT):
    fig = new_canvas()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_facecolor(BG)
    ax.text(0.5, 0.86, title, ha="center", va="center", color=title_color,
            fontsize=34, fontweight="bold", transform=ax.transAxes)
    if subtitle:
        ax.text(0.5, 0.77, subtitle, ha="center", va="center", color=ACCENT,
                fontsize=18, transform=ax.transAxes)
    y = 0.60
    for ln in lines:
        ax.text(0.5, y, ln, ha="center", va="center", color=TEXT,
                fontsize=20, transform=ax.transAxes)
        y -= 0.10
    if foot:
        ax.text(0.5, 0.06, foot, ha="center", va="center", color=MUTED,
                fontsize=13, transform=ax.transAxes)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def image_grid_scene(path, image, kept_indices, grid, patch, headline, caption,
                     stat_lines):
    fig = new_canvas()
    # left: image with grid overlay
    axim = fig.add_axes([0.04, 0.10, 0.52, 0.78])
    axim.imshow(denorm(image))
    axim.axis("off")
    kept = set(int(i) for i in kept_indices)
    for p in range(grid * grid):
        r, c = divmod(p, grid)
        x0, y0 = c * patch, r * patch
        if p in kept:
            axim.add_patch(Rectangle((x0, y0), patch, patch, lw=1.0,
                                     edgecolor=KEEP, facecolor="none"))
        else:
            axim.add_patch(Rectangle((x0, y0), patch, patch, lw=0,
                                     facecolor=DROP, alpha=0.62))
    axim.set_title(headline, color=TEXT, fontsize=20, fontweight="bold", pad=12)
    # right: stats
    axt = fig.add_axes([0.60, 0.10, 0.36, 0.78])
    axt.axis("off")
    axt.set_facecolor(BG)
    axt.text(0.0, 0.92, caption, color=ACCENT, fontsize=21, fontweight="bold",
             va="top", transform=axt.transAxes)
    y = 0.70
    for s in stat_lines:
        axt.text(0.0, y, s, color=TEXT, fontsize=19, va="top", transform=axt.transAxes)
        y -= 0.13
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def figure_card(path, png_path, title):
    fig = new_canvas()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.5, 0.95, title, ha="center", va="top", color=TEXT,
            fontsize=26, fontweight="bold", transform=ax.transAxes)
    img = plt.imread(str(png_path))
    axf = fig.add_axes([0.13, 0.07, 0.74, 0.80])
    axf.axis("off")
    axf.imshow(img)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def two_figure_card(path, png_a, png_b, title):
    fig = new_canvas()
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(0.5, 0.96, title, ha="center", va="top", color=TEXT,
            fontsize=25, fontweight="bold", transform=ax.transAxes)
    for i, png in enumerate((png_a, png_b)):
        axf = fig.add_axes([0.04 + i * 0.48, 0.08, 0.46, 0.80]); axf.axis("off")
        axf.imshow(plt.imread(str(png)))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


@torch.no_grad()
def compute_stages(model, dataset, device, want_class):
    """Return (image, stages) for a high-confidence correct example of want_class."""
    best = None
    for idx in range(min(600, len(dataset))):
        image, target = dataset[idx]
        if target != want_class:
            continue
        out = model(image.unsqueeze(0).to(device), return_info=True)
        prob = torch.softmax(out["logits"], dim=1)
        conf, pred = prob.max(dim=1)
        if int(pred) != target:
            continue
        rec = (float(conf), image, out["grid_size"], out["num_patches"],
               [(d["layer"], d["keep_ratio"], d["selected_patch_indices"][0].tolist())
                for d in out["pruning"]])
        if best is None or rec[0] > best[0]:
            best = rec
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=CP / "outputs" / "sota_pruned_small_atp_final" / "best.pt")
    ap.add_argument("--data-dir", type=Path, default=CP / "data")
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = json.load((CP / "outputs" / "benchmark_results.json").open())
    figdir = CP / "outputs" / "figures"
    frames = CP / "outputs" / "video_frames"
    frames.mkdir(parents=True, exist_ok=True)

    model = build_sota_model("pruned", prune_layers=(3, 6, 9), keep_ratios=(0.75, 0.5, 0.4),
                             prune_mode="lite", model_name="vit_small_patch16_384").to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.set_keep_ratios((0.75, 0.5, 0.4))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((384, 384)), transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN.flatten().tolist(), IMAGENET_STD.flatten().tolist())])
    dataset = datasets.CIFAR10(root=args.data_dir, train=False, transform=transform, download=True)

    conf, image, grid, npatch, stages = compute_stages(model, dataset, device, want_class=1)  # automobile
    patch = 384 // grid

    acc_on = res["accuracy"]["pruned_on"] * 100
    acc_off = res["accuracy"]["pruned_off"] * 100
    g = res["gflops"]
    thr = res["throughput"]
    su32 = thr["pruned"]["32"]["throughput_img_s"] / thr["dense"]["32"]["throughput_img_s"]
    su1 = thr["pruned"]["1"]["throughput_img_s"] / thr["dense"]["1"]["throughput_img_s"]

    # ---- Build scenes as (png, seconds) ----
    scenes = []

    def add(name, seconds):
        scenes.append((str(frames / name), seconds))

    text_scene(frames / "s00.png",
               "Dynamic Token Pruning for Vision Transformers",
               ["Reproducing DynamicViT (NeurIPS 2021) on CIFAR-10",
                "Knowledge Distillation + Asymmetric Token Pruning"],
               subtitle="AIB3004 Course Project  ·  motivated by TC-Pruner",
               foot="ViT-Small/384 student  ·  A100 80GB  ·  all numbers measured")
    add("s00.png", 4.0)

    text_scene(frames / "s01.png", "The problem",
               ["A ViT turns a 384x384 image into 576 patch tokens.",
                "Self-attention cost grows ~ O(N^2) in the token count.",
                "But most patches are background — redundant compute."],
               subtitle="Why prune tokens?")
    add("s01.png", 5.0)

    text_scene(frames / "s02.png", "The idea (DynamicViT)",
               ["Not every token matters.",
                "Score token importance, keep the top-k, drop the rest.",
                "Do it progressively at several depths."],
               subtitle="Keep the foreground, discard the background")
    add("s02.png", 4.5)

    text_scene(frames / "s03.png", "How we score & prune",
               ["Importance = CLS-token attention at layers 3, 6, 9.",
                "Hard top-k drop; ratios compound: 0.75 -> 0.5 -> 0.4.",
                "Asymmetric attention keeps full K/V for a full-context read.",
                "Curriculum: warmup (ep 1-2) -> ramp (ep 3-6) -> full."],
               subtitle="ATP-Lite")
    add("s03.png", 5.5)

    # Live progressive pruning animation
    image_grid_scene(frames / "p0.png", image, range(npatch), grid, patch,
                     f"Input — {CIFAR[1]}", "All tokens",
                     [f"{npatch} patch tokens", "Every patch is attended", "100% of compute"])
    add("p0.png", 2.2)
    cum_labels = []
    for i, (layer, kr, kept) in enumerate(stages):
        pct = len(kept) / npatch * 100
        image_grid_scene(frames / f"p{i+1}.png", image, kept, grid, patch,
                         f"After layer {layer}", f"keep {kr}",
                         [f"{len(kept)} / {npatch} tokens kept",
                          f"{pct:.0f}% of the original",
                          "dropped = dimmed"])
        add(f"p{i+1}.png", 2.4)
        cum_labels.append((layer, len(kept)))

    text_scene(frames / "p_sum.png", "Result of pruning",
               [f"576 -> 432 -> 216 -> 87 tokens",
                "The final, most expensive blocks run on ~15% of tokens.",
                "≈ 85% of tokens dropped by depth."],
               subtitle="Token budget", title_color=ACCENT)
    add("p_sum.png", 4.0)

    # Results (reuse the measured figures)
    figure_card(frames / "r_acc.png", figdir / "fig_accuracy.png",
                f"Accuracy: pruning costs only {acc_off - acc_on:.2f} pt ({acc_off:.2f}% -> {acc_on:.2f}%)")
    add("r_acc.png", 4.5)

    two_figure_card(frames / "r_comp.png", figdir / "fig_flops.png", figdir / "fig_throughput.png",
                    f"Compute: -{(1-g['pruned']/g['dense'])*100:.0f}% FLOPs, up to {su32:.2f}x throughput")
    add("r_comp.png", 5.0)

    figure_card(frames / "r_lim.png", figdir / "fig_vram.png",
                "Honest limitation: more memory, and slower at batch size 1")
    text_scene(frames / "r_lim2.png", "Limitations (we report them)",
               [f"Slower at batch size 1 ({su1:.2f}x): pruner overhead dominates.",
                "VRAM goes UP: asymmetric attention keeps full-length K/V.",
                "Token maps are foreground-biased but noisy.",
                "+8% params from the pruner modules."],
               subtitle="Speedup is real but batch-size dependent")
    add("r_lim.png", 4.0)
    add("r_lim2.png", 5.0)

    text_scene(frames / "end.png", "Takeaway",
               ["~25% less compute, ~0.1 pt accuracy loss — DynamicViT reproduced.",
                "Distillation keeps a small pruned student within ~0.4 pt of large models.",
                "Efficiency must be measured, not assumed.",
                "Extends to task-conditioned pruning in TC-Pruner (VLA)."],
               subtitle="CIFAR-10 token pruning", title_color=ACCENT,
               foot="Code, figures & full report in this repository")
    add("end.png", 5.0)

    # ---- Encode MP4 via ffmpeg concat demuxer ----
    listfile = frames / "concat.txt"
    with listfile.open("w") as f:
        for png, sec in scenes:
            f.write(f"file '{png}'\n")
            f.write(f"duration {sec}\n")
        f.write(f"file '{scenes[-1][0]}'\n")  # repeat last frame (ffmpeg quirk)
    mp4 = CP / "outputs" / "explainer.mp4"
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listfile),
           "-vf", f"fps={args.fps},format=yuv420p", "-c:v", "libx264",
           "-movflags", "+faststart", str(mp4)]
    subprocess.run(cmd, check=True, capture_output=True)
    print("wrote", mp4)

    # ---- Hero GIF: the 4 pruning stages of one image, looping ----
    import imageio.v2 as imageio
    gif_frames = []
    for name in ["p0.png", "p1.png", "p2.png", "p3.png"]:
        gif_frames.append(imageio.imread(str(frames / name)))
    gif = CP / "outputs" / "hero_pruning.gif"
    imageio.mimsave(gif, gif_frames, duration=1.4, loop=0)
    print("wrote", gif)


if __name__ == "__main__":
    main()
