#!/usr/bin/env python3
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch, Rectangle
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import ToyDynamicViTConfig, build_model  # noqa: E402


CIFAR10_MEAN = torch.tensor((0.4914, 0.4822, 0.4465)).view(3, 1, 1)
CIFAR10_STD = torch.tensor((0.2470, 0.2435, 0.2616)).view(3, 1, 1)
BACKGROUND = "#0C1424"
CARD = "#131F33"
TEXT = "#E8EEF7"
MUTED = "#A4B4CB"
ACCENT = "#50E3C2"
ACCENT_2 = "#FFB703"
GRID_ON = "#55E26A"
GRID_OFF = "#000000"


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_checkpoint_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = ToyDynamicViTConfig.from_dict(checkpoint["config"])
    model_type = checkpoint.get("model_type", "pruned" if config.prune_layers else "dense")
    model = build_model(
        model_type=model_type,
        num_classes=config.num_classes,
        image_size=config.image_size,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        dropout=config.dropout,
        prune_layers=config.prune_layers,
        keep_ratios=config.keep_ratios,
        scorer_hidden_dim=config.scorer_hidden_dim,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def build_dataset(data_dir: Path, image_size: int):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN.flatten().tolist(), CIFAR10_STD.flatten().tolist()),
        ]
    )
    return datasets.CIFAR10(root=data_dir, train=False, transform=transform, download=True)


def build_loader(dataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )


def denormalize(image: torch.Tensor) -> np.ndarray:
    image = image.cpu() * CIFAR10_STD + CIFAR10_MEAN
    image = image.clamp(0.0, 1.0)
    return image.permute(1, 2, 0).numpy()


@torch.no_grad()
def select_diverse_examples(
    model,
    dataset,
    device: torch.device,
    num_samples: int,
    max_per_class: int,
    batch_size: int,
) -> List[Dict]:
    loader = build_loader(dataset, batch_size)
    candidates_by_class: Dict[int, List[Dict]] = defaultdict(list)
    sample_offset = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        probs = torch.softmax(model(images), dim=1)
        confidence, preds = probs.max(dim=1)

        for idx in range(images.size(0)):
            if int(preds[idx].item()) != int(targets[idx].item()):
                continue
            class_id = int(targets[idx].item())
            candidates_by_class[class_id].append(
                {
                    "dataset_index": sample_offset + idx,
                    "target": class_id,
                    "prediction": int(preds[idx].item()),
                    "confidence": float(confidence[idx].item()),
                }
            )
        sample_offset += images.size(0)

    for class_id in candidates_by_class:
        candidates_by_class[class_id].sort(key=lambda item: item["confidence"], reverse=True)

    class_order = sorted(
        candidates_by_class,
        key=lambda class_id: candidates_by_class[class_id][0]["confidence"] if candidates_by_class[class_id] else -1.0,
        reverse=True,
    )

    selected = []
    class_counts = defaultdict(int)
    rank = 0
    while len(selected) < num_samples:
        added = False
        for class_id in class_order:
            class_candidates = candidates_by_class[class_id]
            if class_counts[class_id] >= max_per_class:
                continue
            if rank >= len(class_candidates):
                continue
            selected.append(class_candidates[rank])
            class_counts[class_id] += 1
            added = True
            if len(selected) >= num_samples:
                break
        if not added:
            break
        rank += 1

    if not selected:
        raise ValueError("No correct predictions were found for video export.")
    return selected


@torch.no_grad()
def enrich_examples(model, dataset, selected: Sequence[Dict], class_names: Sequence[str], device: torch.device) -> List[Dict]:
    enriched = []
    for sample in selected:
        image, target = dataset[sample["dataset_index"]]
        outputs = model(image.unsqueeze(0).to(device), return_info=True)
        probs = torch.softmax(outputs["logits"], dim=1)
        confidence, pred = probs.max(dim=1)
        stages = [
            {
                "title": f"All Tokens ({outputs['num_patches']})",
                "kept_indices": list(range(outputs["num_patches"])),
                "kept_count": outputs["num_patches"],
                "subtitle": "No pruning yet",
            }
        ]
        for stage_idx, stage in enumerate(outputs["pruning"], start=1):
            stages.append(
                {
                    "title": f"Stage {stage_idx} ({stage['kept_token_count']})",
                    "kept_indices": stage["selected_patch_indices"][0].tolist(),
                    "kept_count": stage["kept_token_count"],
                    "subtitle": f"Prune layer {stage['layer']}",
                }
            )

        class_id = int(target)
        pred_id = int(pred.item())
        enriched.append(
            {
                "dataset_index": sample["dataset_index"],
                "image": image,
                "target": class_id,
                "prediction": pred_id,
                "confidence": float(confidence.item()),
                "true_name": class_names[class_id] if class_names else str(class_id),
                "pred_name": class_names[pred_id] if class_names else str(pred_id),
                "grid_size": outputs["grid_size"],
                "patch_size": model.config.patch_size,
                "stages": stages,
            }
        )
    return enriched


def draw_card(ax, facecolor: str = CARD) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="round,pad=0.015,rounding_size=0.04",
            linewidth=0,
            facecolor=facecolor,
            transform=ax.transAxes,
            zorder=0,
        )
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def draw_overlay(ax, image: torch.Tensor, kept_indices: Iterable[int], grid_size: int, patch_size: int, title: str, subtitle: str, active: bool) -> None:
    np_image = denormalize(image)
    ax.imshow(np_image)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(CARD)
    border_color = ACCENT if active else "#314766"
    border_width = 5 if active else 2
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(border_color)
        spine.set_linewidth(border_width)

    alpha = 0.18 if active else 0.34
    kept_set = set(int(idx) for idx in kept_indices)
    for patch_idx in range(grid_size * grid_size):
        row = patch_idx // grid_size
        col = patch_idx % grid_size
        x0 = col * patch_size
        y0 = row * patch_size
        if patch_idx in kept_set:
            rect = Rectangle((x0, y0), patch_size, patch_size, linewidth=1.3, edgecolor=GRID_ON, facecolor="none")
        else:
            rect = Rectangle((x0, y0), patch_size, patch_size, linewidth=0.35, edgecolor=GRID_OFF, facecolor=GRID_OFF, alpha=alpha)
        ax.add_patch(rect)

    ax.set_title(title, color=TEXT, fontsize=13, fontweight="bold", pad=10)
    ax.text(0.5, -0.09, subtitle, transform=ax.transAxes, ha="center", va="top", color=MUTED, fontsize=10)


def figure_to_frame(fig) -> np.ndarray:
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def render_intro_frame(progress: float, summary_rows: Sequence[Dict], plot_image: np.ndarray) -> np.ndarray:
    dense_row = next(row for row in summary_rows if row["name"] == "dense")
    best_pruned = max((row for row in summary_rows if row["name"] != "dense"), key=lambda row: row["accuracy"])
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BACKGROUND)
    ax_bg = fig.add_axes([0, 0, 1, 1])
    ax_bg.set_facecolor(BACKGROUND)
    ax_bg.axis("off")

    left = fig.add_axes([0.05, 0.08, 0.40, 0.84])
    draw_card(left)
    left.text(0.07, 0.86, "Dynamic Token Pruning Demo", color=TEXT, fontsize=24, fontweight="bold")
    left.text(0.07, 0.78, "Toy DynamicViT reproduction on CIFAR-10", color=MUTED, fontsize=13)
    left.text(0.07, 0.63, f"Dense accuracy: {dense_row['accuracy'] * 100:.2f}%", color=TEXT, fontsize=20, fontweight="bold")
    left.text(0.07, 0.53, f"Best pruned: {best_pruned['name']} at {best_pruned['accuracy'] * 100:.2f}%", color=TEXT, fontsize=18)
    left.text(
        0.07,
        0.42,
        f"Final kept tokens: {int(best_pruned['avg_kept_tokens_per_stage'][-1])} / 64",
        color=ACCENT,
        fontsize=18,
        fontweight="bold",
    )
    left.text(
        0.07,
        0.24,
        "Honest result: token count drops a lot.\nAccuracy mostly holds.\nWall-clock speed is flat on this tiny model.",
        color=MUTED,
        fontsize=13,
        linespacing=1.6,
    )

    right = fig.add_axes([0.49, 0.08, 0.46, 0.84])
    draw_card(right)
    right.text(0.05, 0.93, "Accuracy / Speed Tradeoff", color=TEXT, fontsize=16, fontweight="bold")
    right.imshow(plot_image, extent=(0.06, 0.94, 0.12, 0.86), aspect="auto", zorder=2)
    right.axis("off")

    bar_ax = fig.add_axes([0.05, 0.03, 0.90, 0.02])
    bar_ax.set_facecolor("#1B2A43")
    bar_ax.barh([0], [progress], height=1.0, color=ACCENT)
    bar_ax.set_xlim(0.0, 1.0)
    bar_ax.axis("off")
    return figure_to_frame(fig)


def render_sample_frame(sample: Dict, active_stage: int, local_progress: float, clip_index: int, total_clips: int) -> np.ndarray:
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BACKGROUND)
    ax_bg = fig.add_axes([0, 0, 1, 1])
    ax_bg.set_facecolor(BACKGROUND)
    ax_bg.axis("off")

    header = fig.add_axes([0.04, 0.82, 0.92, 0.13])
    draw_card(header)
    header.text(0.04, 0.62, f"Sample {clip_index + 1} / {total_clips}: {sample['true_name']}", color=TEXT, fontsize=22, fontweight="bold")
    header.text(
        0.04,
        0.24,
        f"true={sample['true_name']} | pred={sample['pred_name']} | confidence={sample['confidence']:.3f} | test index={sample['dataset_index']}",
        color=MUTED,
        fontsize=12,
    )

    panel_positions = [
        [0.04, 0.25, 0.28, 0.48],
        [0.36, 0.25, 0.28, 0.48],
        [0.68, 0.25, 0.28, 0.48],
    ]
    for stage_idx, (stage, pos) in enumerate(zip(sample["stages"], panel_positions)):
        ax = fig.add_axes(pos)
        draw_overlay(
            ax=ax,
            image=sample["image"],
            kept_indices=stage["kept_indices"],
            grid_size=sample["grid_size"],
            patch_size=sample["patch_size"],
            title=stage["title"],
            subtitle=stage["subtitle"],
            active=stage_idx == active_stage,
        )

    footer = fig.add_axes([0.04, 0.07, 0.92, 0.11])
    draw_card(footer)
    kept_count = sample["stages"][active_stage]["kept_count"]
    footer.text(
        0.04,
        0.62,
        f"Active stage: {sample['stages'][active_stage]['title']} | retained {kept_count / sample['stages'][0]['kept_count'] * 100:.1f}% of patch tokens",
        color=TEXT,
        fontsize=15,
        fontweight="bold",
    )
    footer.text(
        0.04,
        0.22,
        "This is inference-time hard pruning on real CIFAR-10 test images.",
        color=MUTED,
        fontsize=11,
    )

    progress_ax = fig.add_axes([0.04, 0.03, 0.92, 0.02])
    progress_ax.set_facecolor("#1B2A43")
    progress_ax.barh([0], [local_progress], height=1.0, color=ACCENT_2)
    progress_ax.set_xlim(0.0, 1.0)
    progress_ax.axis("off")
    return figure_to_frame(fig)


def render_outro_frame(summary_rows: Sequence[Dict], progress: float) -> np.ndarray:
    dense_row = next(row for row in summary_rows if row["name"] == "dense")
    best_pruned = max((row for row in summary_rows if row["name"] != "dense"), key=lambda row: row["accuracy"])
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BACKGROUND)
    ax_bg = fig.add_axes([0, 0, 1, 1])
    ax_bg.set_facecolor(BACKGROUND)
    ax_bg.axis("off")

    card = fig.add_axes([0.08, 0.12, 0.84, 0.76])
    draw_card(card)
    card.text(0.06, 0.82, "Takeaway", color=TEXT, fontsize=26, fontweight="bold")
    card.text(
        0.06,
        0.62,
        f"Dense: {dense_row['accuracy'] * 100:.2f}% accuracy with 64 tokens\n"
        f"{best_pruned['name']}: {best_pruned['accuracy'] * 100:.2f}% accuracy with {int(best_pruned['avg_kept_tokens_per_stage'][-1])} final tokens",
        color=TEXT,
        fontsize=20,
        linespacing=1.6,
    )
    card.text(
        0.06,
        0.34,
        "This is a defensible reproduction of the pruning idea.\n"
        "The strongest evidence is token-budget reduction with limited accuracy loss.\n"
        "Do not oversell GPU speedup on this setup.",
        color=MUTED,
        fontsize=15,
        linespacing=1.6,
    )

    progress_ax = fig.add_axes([0.08, 0.05, 0.84, 0.02])
    progress_ax.set_facecolor("#1B2A43")
    progress_ax.barh([0], [progress], height=1.0, color=ACCENT)
    progress_ax.set_xlim(0.0, 1.0)
    progress_ax.axis("off")
    return figure_to_frame(fig)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create an MP4 demo video for the class-project pruning result.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-video", type=Path, default=None)
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--max-per-class", type=int, default=1)
    parser.add_argument("--selection-batch-size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--intro-seconds", type=float, default=2.5)
    parser.add_argument("--sample-seconds", type=float, default=3.0)
    parser.add_argument("--outro-seconds", type=float, default=2.0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint_model(args.checkpoint, device)
    if checkpoint.get("model_type") != "pruned":
        raise ValueError("Demo video is only useful for a pruned checkpoint.")

    experiment_root = args.experiment_root or args.checkpoint.parent.parent
    output_video = args.output_video or args.checkpoint.parent / "demo_video.mp4"
    output_manifest = args.output_manifest or args.checkpoint.parent / "demo_video_manifest.json"

    plot_image = plt.imread(experiment_root / "plots" / "accuracy_speed_tradeoff.png")
    summary_rows = load_json(experiment_root / "plots" / "summary_table.json")
    dataset = build_dataset(args.data_dir, checkpoint["config"]["image_size"])
    class_names = checkpoint.get("class_names", [])

    selected = select_diverse_examples(
        model=model,
        dataset=dataset,
        device=device,
        num_samples=args.num_samples,
        max_per_class=args.max_per_class,
        batch_size=args.selection_batch_size,
    )
    samples = enrich_examples(model, dataset, selected, class_names, device)

    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        output_video,
        fps=args.fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )

    intro_frames = max(1, int(round(args.intro_seconds * args.fps)))
    for frame_idx in range(intro_frames):
        progress = (frame_idx + 1) / intro_frames
        writer.append_data(render_intro_frame(progress, summary_rows, plot_image))

    sample_frames = max(3, int(round(args.sample_seconds * args.fps)))
    stage_count = len(samples[0]["stages"])
    for clip_index, sample in enumerate(samples):
        for frame_idx in range(sample_frames):
            local_progress = (frame_idx + 1) / sample_frames
            active_stage = min(stage_count - 1, int(math.floor(local_progress * stage_count)))
            writer.append_data(
                render_sample_frame(
                    sample=sample,
                    active_stage=active_stage,
                    local_progress=local_progress,
                    clip_index=clip_index,
                    total_clips=len(samples),
                )
            )

    outro_frames = max(1, int(round(args.outro_seconds * args.fps)))
    for frame_idx in range(outro_frames):
        progress = (frame_idx + 1) / outro_frames
        writer.append_data(render_outro_frame(summary_rows, progress))

    writer.close()

    save_json(
        output_manifest,
        {
            "checkpoint": str(args.checkpoint),
            "output_video": str(output_video),
            "summary_rows": summary_rows,
            "selected_samples": [
                {
                    "dataset_index": sample["dataset_index"],
                    "true_name": sample["true_name"],
                    "pred_name": sample["pred_name"],
                    "confidence": sample["confidence"],
                    "stage_kept_tokens": [stage["kept_count"] for stage in sample["stages"][1:]],
                }
                for sample in samples
            ],
        },
    )
    print(str(output_video))


if __name__ == "__main__":
    main()
