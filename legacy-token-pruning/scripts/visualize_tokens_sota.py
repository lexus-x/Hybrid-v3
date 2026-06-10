#!/usr/bin/env python3
"""
Token-pruning visualisations for the trained SOTA-Hybrid ViT-Small/384 checkpoint.

Produces, for a handful of high-confidence correctly-classified CIFAR-10 test
images:
  * one per-image figure showing progressive pruning
    (original grid -> kept after layer 3 -> 6 -> 9), dropped patches dimmed
  * one summary grid (original vs final kept tokens) across several classes,
    suitable for a slide.

Works directly with the SOTA model's forward(return_info=True), which returns
`pruning[i]["selected_patch_indices"]` (original patch ids surviving each stage)
and `final_patch_indices`.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sota_hybrid_vit import build_sota_model

IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


def parse_floats(v):
    return tuple(float(x.strip()) for x in v.split(","))


def parse_ints(v):
    return tuple(int(x.strip()) for x in v.split(","))


def denorm(img):
    return (img.cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)


def draw_overlay(ax, image, kept_indices, grid_size, patch_size, title):
    ax.imshow(denorm(image).permute(1, 2, 0).numpy())
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    kept = set(int(i) for i in kept_indices)
    for p in range(grid_size * grid_size):
        r, c = divmod(p, grid_size)
        x0, y0 = c * patch_size, r * patch_size
        if p in kept:
            ax.add_patch(Rectangle((x0, y0), patch_size, patch_size,
                                   linewidth=1.0, edgecolor="lime", facecolor="none"))
        else:
            ax.add_patch(Rectangle((x0, y0), patch_size, patch_size,
                                   linewidth=0.0, facecolor="black", alpha=0.55))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=ROOT / "outputs" / "sota_pruned_small_atp_final" / "best.pt")
    ap.add_argument("--model-name", type=str, default="vit_small_patch16_384")
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--prune-layers", type=str, default="3,6,9")
    ap.add_argument("--keep-ratios", type=str, default="0.75,0.5,0.4")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--output-dir", type=Path,
                    default=ROOT / "outputs" / "token_viz")
    ap.add_argument("--num-images", type=int, default=6)
    ap.add_argument("--scan", type=int, default=400, help="how many test images to scan for good examples")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prune_layers = parse_ints(args.prune_layers)
    keep_ratios = parse_floats(args.keep_ratios)

    model = build_sota_model("pruned", prune_layers=prune_layers, keep_ratios=keep_ratios,
                             prune_mode="lite", model_name=args.model_name).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.set_keep_ratios(keep_ratios)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN.flatten().tolist(), IMAGENET_STD.flatten().tolist()),
    ])
    dataset = datasets.CIFAR10(root=args.data_dir, train=False, transform=transform, download=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Scan for one high-confidence, correctly-classified example per distinct class.
    picked = {}
    for idx in range(min(args.scan, len(dataset))):
        image, target = dataset[idx]
        out = model(image.unsqueeze(0).to(device), return_info=True)
        prob = torch.softmax(out["logits"], dim=1)
        conf, pred = prob.max(dim=1)
        pred = int(pred)
        conf = float(conf)
        if pred != target:
            continue
        if target not in picked or conf > picked[target]["conf"]:
            picked[target] = {
                "idx": idx, "conf": conf, "pred": pred,
                "grid_size": out["grid_size"],
                "num_patches": out["num_patches"],
                "stages": [(d["layer"], d["selected_patch_indices"][0].tolist())
                           for d in out["pruning"]],
                "final": out["final_patch_indices"][0].tolist(),
            }

    chosen = sorted(picked.values(), key=lambda r: r["conf"], reverse=True)[: args.num_images]
    patch_size = args.img_size // chosen[0]["grid_size"]

    # 1) Per-image progressive-pruning figures
    for k, rec in enumerate(chosen):
        image, target = dataset[rec["idx"]]
        grid = rec["grid_size"]
        n_panels = 1 + len(rec["stages"])
        fig, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 3.4))
        draw_overlay(axes[0], image, range(rec["num_patches"]), grid, patch_size,
                     f"Original\n{rec['num_patches']} tokens")
        for j, (layer, kept) in enumerate(rec["stages"]):
            draw_overlay(axes[j + 1], image, kept, grid, patch_size,
                         f"After layer {layer}\n{len(kept)} tokens")
        fig.suptitle(f"{CIFAR10_CLASSES[target]}  (conf {rec['conf']:.3f})",
                     fontsize=12, y=1.02)
        fig.tight_layout()
        out_path = args.output_dir / f"progressive_{k:02d}_{CIFAR10_CLASSES[target]}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_path}")

    # 2) Summary grid: original vs final kept, one column per image
    n = len(chosen)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.4))
    if n == 1:
        axes = axes.reshape(2, 1)
    for col, rec in enumerate(chosen):
        image, target = dataset[rec["idx"]]
        grid = rec["grid_size"]
        draw_overlay(axes[0, col], image, range(rec["num_patches"]), grid, patch_size,
                     CIFAR10_CLASSES[target])
        draw_overlay(axes[1, col], image, rec["final"], grid, patch_size,
                     f"{len(rec['final'])}/{rec['num_patches']} kept")
    axes[0, 0].set_ylabel("Original", fontsize=11)
    axes[1, 0].set_ylabel("Pruned (layer 9)", fontsize=11)
    fig.suptitle("Surviving tokens after pruning (CLS-attention importance, "
                 f"keep ratios {','.join(str(k) for k in keep_ratios)})", fontsize=13)
    fig.tight_layout()
    summary_path = args.output_dir / "summary_grid.png"
    fig.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
