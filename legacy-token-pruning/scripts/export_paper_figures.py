#!/usr/bin/env python3
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import ToyDynamicViTConfig, build_model  # noqa: E402


CIFAR10_MEAN = torch.tensor((0.4914, 0.4822, 0.4465)).view(3, 1, 1)
CIFAR10_STD = torch.tensor((0.2470, 0.2435, 0.2616)).view(3, 1, 1)
BG = "#FFFFFF"
GRID = "#D9E2EC"
TEXT = "#1F2933"
MUTED = "#52606D"
DENSE = "#2C5282"
PRUNED = "#DD6B20"
ACCENT = "#2F855A"
DROP = "#9B2C2C"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.45,
            "grid.linestyle": "-",
            "savefig.bbox": "tight",
        }
    )


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)


def save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


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


def normalize_scores(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    if values.size == 0:
        return values
    vmin = float(values.min())
    vmax = float(values.max())
    if abs(vmax - vmin) < 1e-8:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def build_stage_records(outputs: Dict, batch_index: int) -> List[Dict]:
    num_patches = outputs["num_patches"]
    candidate_ids = np.arange(num_patches, dtype=np.int64)
    stage_records = []
    for stage in outputs["pruning"]:
        local_scores = stage["scores"][batch_index].numpy()
        score_map = np.full(num_patches, np.nan, dtype=np.float32)
        score_map[candidate_ids] = normalize_scores(local_scores)
        kept_indices = stage["selected_patch_indices"][batch_index].numpy().astype(np.int64)
        stage_records.append(
            {
                "layer": int(stage["layer"]),
                "kept_indices": kept_indices,
                "kept_count": int(stage["kept_token_count"]),
                "score_map": score_map,
            }
        )
        candidate_ids = kept_indices
    return stage_records


@torch.no_grad()
def collect_inference_data(
    dense_model,
    pruned_model,
    dataset,
    class_names: Sequence[str],
    device: torch.device,
    batch_size: int,
) -> Dict:
    loader = build_loader(dataset, batch_size)
    num_classes = len(class_names)
    num_patches = pruned_model.num_patches
    num_stages = len(pruned_model.prune_layers)

    dense_preds: List[int] = []
    pruned_preds: List[int] = []
    dense_conf: List[float] = []
    pruned_conf: List[float] = []
    targets_all: List[int] = []
    dense_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    pruned_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    class_totals = np.zeros(num_classes, dtype=np.int64)
    dense_correct = np.zeros(num_classes, dtype=np.int64)
    pruned_correct = np.zeros(num_classes, dtype=np.int64)
    stage_retention = [np.zeros(num_patches, dtype=np.int64) for _ in range(num_stages)]
    stage_score_sum = [np.zeros(num_patches, dtype=np.float64) for _ in range(num_stages)]
    stage_score_count = [np.zeros(num_patches, dtype=np.int64) for _ in range(num_stages)]

    top_success_by_class: Dict[int, Dict] = {}
    top_failure: Dict | None = None
    top_pruned_only: Dict | None = None
    sample_offset = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        dense_logits = dense_model(images)
        pruned_outputs = pruned_model(images, return_info=True)
        pruned_logits = pruned_outputs["logits"]

        dense_probs = torch.softmax(dense_logits, dim=1)
        pruned_probs = torch.softmax(pruned_logits, dim=1)
        dense_batch_conf, dense_batch_preds = dense_probs.max(dim=1)
        pruned_batch_conf, pruned_batch_preds = pruned_probs.max(dim=1)

        for batch_index in range(images.size(0)):
            dataset_index = sample_offset + batch_index
            target = int(targets[batch_index].item())
            dense_pred = int(dense_batch_preds[batch_index].item())
            pruned_pred = int(pruned_batch_preds[batch_index].item())
            dense_prob = float(dense_batch_conf[batch_index].item())
            pruned_prob = float(pruned_batch_conf[batch_index].item())
            stage_records = build_stage_records(pruned_outputs, batch_index)

            class_totals[target] += 1
            dense_correct[target] += int(dense_pred == target)
            pruned_correct[target] += int(pruned_pred == target)
            dense_confusion[target, dense_pred] += 1
            pruned_confusion[target, pruned_pred] += 1

            dense_preds.append(dense_pred)
            pruned_preds.append(pruned_pred)
            dense_conf.append(dense_prob)
            pruned_conf.append(pruned_prob)
            targets_all.append(target)

            for stage_index, stage in enumerate(stage_records):
                stage_retention[stage_index][stage["kept_indices"]] += 1
                valid = ~np.isnan(stage["score_map"])
                stage_score_sum[stage_index][valid] += stage["score_map"][valid]
                stage_score_count[stage_index][valid] += 1

            example_record = {
                "dataset_index": dataset_index,
                "image": images[batch_index].detach().cpu(),
                "target": target,
                "true_name": class_names[target],
                "dense_pred": dense_pred,
                "dense_name": class_names[dense_pred],
                "dense_conf": dense_prob,
                "pruned_pred": pruned_pred,
                "pruned_name": class_names[pruned_pred],
                "pruned_conf": pruned_prob,
                "stage_records": stage_records,
            }

            current_best = top_success_by_class.get(target)
            if pruned_pred == target and (current_best is None or pruned_prob > current_best["pruned_conf"]):
                top_success_by_class[target] = example_record

            if dense_pred == target and pruned_pred != target:
                if top_failure is None or dense_prob > top_failure["dense_conf"]:
                    top_failure = example_record

            if dense_pred != target and pruned_pred == target:
                if top_pruned_only is None or pruned_prob > top_pruned_only["pruned_conf"]:
                    top_pruned_only = example_record

        sample_offset += images.size(0)

    dense_preds_arr = np.asarray(dense_preds, dtype=np.int64)
    pruned_preds_arr = np.asarray(pruned_preds, dtype=np.int64)
    targets_arr = np.asarray(targets_all, dtype=np.int64)
    dense_conf_arr = np.asarray(dense_conf, dtype=np.float32)
    pruned_conf_arr = np.asarray(pruned_conf, dtype=np.float32)

    stage_score_mean = []
    for stage_index in range(num_stages):
        mean = np.full(num_patches, np.nan, dtype=np.float32)
        valid = stage_score_count[stage_index] > 0
        mean[valid] = (stage_score_sum[stage_index][valid] / stage_score_count[stage_index][valid]).astype(np.float32)
        stage_score_mean.append(mean)

    return {
        "class_names": list(class_names),
        "dense_preds": dense_preds_arr,
        "pruned_preds": pruned_preds_arr,
        "targets": targets_arr,
        "dense_conf": dense_conf_arr,
        "pruned_conf": pruned_conf_arr,
        "dense_confusion": dense_confusion,
        "pruned_confusion": pruned_confusion,
        "class_totals": class_totals,
        "dense_correct": dense_correct,
        "pruned_correct": pruned_correct,
        "stage_retention": stage_retention,
        "stage_score_mean": stage_score_mean,
        "top_success_by_class": top_success_by_class,
        "top_failure": top_failure,
        "top_pruned_only": top_pruned_only,
    }


def plot_accuracy_vs_tokens(summary_rows: Sequence[Dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.1))
    rows = sorted(
        summary_rows,
        key=lambda row: row["avg_kept_tokens_per_stage"][-1] if row["avg_kept_tokens_per_stage"] else 64,
        reverse=True,
    )
    x = [row["avg_kept_tokens_per_stage"][-1] if row["avg_kept_tokens_per_stage"] else 64 for row in rows]
    y = [row["accuracy"] for row in rows]
    colors = [DENSE if row["name"] == "dense" else PRUNED for row in rows]
    ax.plot(x, y, color="#829AB1", linewidth=1.4, zorder=1)
    ax.scatter(x, y, s=110, color=colors, zorder=2)
    for row, x_val, y_val in zip(rows, x, y):
        ax.annotate(row["name"], (x_val, y_val), xytext=(8, 6), textcoords="offset points")
    ax.set_title("Accuracy vs Final Kept Tokens")
    ax.set_xlabel("Final kept tokens")
    ax.set_ylabel("Test accuracy")
    ax.set_xlim(0, 68)
    ax.set_ylim(min(y) - 0.01, max(y) + 0.01)
    save_fig(fig, path)


def plot_latency_vs_accuracy(summary_rows: Sequence[Dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.1))
    x = [row["images_per_second"] for row in summary_rows]
    y = [row["accuracy"] for row in summary_rows]
    colors = [DENSE if row["name"] == "dense" else PRUNED for row in summary_rows]
    ax.scatter(x, y, s=120, color=colors)
    for row, x_val, y_val in zip(summary_rows, x, y):
        ax.annotate(row["name"], (x_val, y_val), xytext=(8, 6), textcoords="offset points")
    ax.set_title("Inference Accuracy vs Throughput")
    ax.set_xlabel("Images / second")
    ax.set_ylabel("Test accuracy")
    save_fig(fig, path)


def plot_classwise_accuracy(data: Dict, pruned_label: str, path: Path) -> None:
    class_names = data["class_names"]
    dense_acc = data["dense_correct"] / np.maximum(1, data["class_totals"])
    pruned_acc = data["pruned_correct"] / np.maximum(1, data["class_totals"])
    x = np.arange(len(class_names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.bar(x - width / 2, dense_acc, width=width, label="dense", color=DENSE)
    ax.bar(x + width / 2, pruned_acc, width=width, label=pruned_label, color=PRUNED)
    ax.set_title("Per-Class Accuracy on CIFAR-10 Test Set")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=25, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False)
    save_fig(fig, path)


def plot_prediction_agreement(data: Dict, pruned_label: str, path: Path) -> None:
    targets = data["targets"]
    dense_preds = data["dense_preds"]
    pruned_preds = data["pruned_preds"]
    dense_ok = dense_preds == targets
    pruned_ok = pruned_preds == targets
    labels = ["Both Correct", "Dense Only Correct", "Pruned Only Correct", "Both Wrong"]
    values = np.array(
        [
            np.sum(dense_ok & pruned_ok),
            np.sum(dense_ok & ~pruned_ok),
            np.sum(~dense_ok & pruned_ok),
            np.sum(~dense_ok & ~pruned_ok),
        ],
        dtype=np.int64,
    )
    colors = [ACCENT, DENSE, PRUNED, MUTED]

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title(f"Dense vs {pruned_label} Prediction Agreement")
    ax.set_ylabel("Number of test images")
    ax.bar_label(bars, labels=[f"{value}\n({value / values.sum() * 100:.1f}%)" for value in values], padding=3, fontsize=10)
    save_fig(fig, path)


def plot_confusion_matrix(matrix: np.ndarray, class_names: Sequence[str], title: str, path: Path) -> None:
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = matrix / np.maximum(1, row_sums)

    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    im = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_yticklabels(class_names)
    threshold = 0.55
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            text_color = "white" if normalized[row, col] > threshold else TEXT
            ax.text(
                col,
                row,
                f"{matrix[row, col]}\n{normalized[row, col] * 100:.1f}%",
                ha="center",
                va="center",
                fontsize=7.5,
                color=text_color,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_fig(fig, path)


def stage_heatmap(values: np.ndarray, title: str, path: Path, value_label: str, cmap_name: str, use_mask: bool) -> None:
    grid_size = int(np.sqrt(values.size))
    grid = values.reshape(grid_size, grid_size)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    if use_mask:
        masked = np.ma.masked_invalid(grid)
        cmap = matplotlib.colormaps.get_cmap(cmap_name).copy()
        cmap.set_bad(color="#EEF2F7")
        im = ax.imshow(masked, cmap=cmap)
    else:
        im = ax.imshow(grid, cmap=cmap_name, vmin=0.0, vmax=np.nanmax(grid))
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for row in range(grid_size):
        for col in range(grid_size):
            if np.isnan(grid[row, col]):
                continue
            ax.text(col, row, f"{grid[row, col]:.2f}", ha="center", va="center", fontsize=7, color=TEXT)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=value_label)
    save_fig(fig, path)


def build_patch_mask(kept_indices: Sequence[int], grid_size: int) -> np.ndarray:
    mask = np.zeros(grid_size * grid_size, dtype=np.float32)
    mask[np.asarray(list(kept_indices), dtype=np.int64)] = 1.0
    return mask.reshape(grid_size, grid_size)


def make_focus_image(image: torch.Tensor, kept_indices: Sequence[int], grid_size: int) -> np.ndarray:
    np_image = denormalize(image).copy()
    mask = build_patch_mask(kept_indices, grid_size)
    patch_size = np_image.shape[0] // grid_size
    drop_tint = np.array([0.78, 0.12, 0.10], dtype=np.float32)
    for row in range(grid_size):
        for col in range(grid_size):
            y0 = row * patch_size
            y1 = (row + 1) * patch_size
            x0 = col * patch_size
            x1 = (col + 1) * patch_size
            region = np_image[y0:y1, x0:x1]
            if mask[row, col] < 0.5:
                gray = region.mean(axis=2, keepdims=True)
                np_image[y0:y1, x0:x1] = np.clip(0.18 * region + 0.25 * gray + 0.57 * drop_tint, 0.0, 1.0)
            else:
                np_image[y0:y1, x0:x1] = np.clip(region * 1.05, 0.0, 1.0)
    return np_image


def draw_patch_grid(ax, kept_indices: Sequence[int], grid_size: int, patch_size: int) -> None:
    kept_set = set(int(idx) for idx in kept_indices)
    for patch_idx in range(grid_size * grid_size):
        row = patch_idx // grid_size
        col = patch_idx % grid_size
        x0 = col * patch_size - 0.5
        y0 = row * patch_size - 0.5
        if patch_idx in kept_set:
            rect = Rectangle((x0, y0), patch_size, patch_size, linewidth=1.6, edgecolor=ACCENT, facecolor="none")
        else:
            rect = Rectangle((x0, y0), patch_size, patch_size, linewidth=0.8, edgecolor=DROP, facecolor="none")
        ax.add_patch(rect)


def draw_original(ax, image: torch.Tensor, grid_size: int, patch_size: int, title: str) -> None:
    np_image = denormalize(image)
    ax.imshow(np_image, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    draw_patch_grid(ax, range(grid_size * grid_size), grid_size, patch_size)


def draw_focus_view(ax, image: torch.Tensor, kept_indices: Sequence[int], grid_size: int, patch_size: int, title: str) -> None:
    ax.imshow(make_focus_image(image, kept_indices, grid_size), interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    draw_patch_grid(ax, kept_indices, grid_size, patch_size)


def draw_binary_mask(ax, kept_indices: Sequence[int], grid_size: int, title: str) -> None:
    mask = build_patch_mask(kept_indices, grid_size)
    cmap = LinearSegmentedColormap.from_list("keep_drop", ["#7F1D1D", "#DCFCE7"])
    im = ax.imshow(mask, cmap=cmap, interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color="#FFFFFF", linewidth=1.1)
    for row in range(grid_size):
        for col in range(grid_size):
            ax.text(
                col,
                row,
                "K" if mask[row, col] > 0.5 else "D",
                ha="center",
                va="center",
                fontsize=9,
                color=TEXT if mask[row, col] > 0.5 else "white",
                fontweight="bold",
            )
    return im


def draw_score_map(ax, score_map: np.ndarray, kept_indices: Sequence[int], title: str) -> None:
    grid_size = int(np.sqrt(score_map.size))
    grid = score_map.reshape(grid_size, grid_size)
    masked = np.ma.masked_invalid(grid)
    cmap = LinearSegmentedColormap.from_list("paper_heat", ["#F7FAFC", "#FDBA74", "#C05621"])
    cmap.set_bad(color="#E2E8F0")
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    kept_set = set(int(idx) for idx in kept_indices)
    for patch_idx in range(grid_size * grid_size):
        row = patch_idx // grid_size
        col = patch_idx % grid_size
        if np.isnan(grid[row, col]):
            continue
        rect = Rectangle(
            (col - 0.5, row - 0.5),
            1.0,
            1.0,
            linewidth=2.2 if patch_idx in kept_set else 0.7,
            edgecolor=ACCENT if patch_idx in kept_set else "#FFFFFF",
            facecolor="none",
        )
        ax.add_patch(rect)
    return im


def draw_explanation_panel(ax, record: Dict, grid_size: int) -> None:
    ax.axis("off")
    ax.set_title("How To Read This Figure", loc="left")

    ax.add_patch(Rectangle((0.02, 0.80), 0.045, 0.07, transform=ax.transAxes, facecolor="#DCFCE7", edgecolor=ACCENT, linewidth=2))
    ax.text(0.09, 0.835, "Green patch = retained token position", transform=ax.transAxes, fontsize=11, va="center")
    ax.add_patch(Rectangle((0.02, 0.68), 0.045, 0.07, transform=ax.transAxes, facecolor="#7F1D1D", edgecolor=DROP, linewidth=2))
    ax.text(0.09, 0.715, "Red patch = dropped token position", transform=ax.transAxes, fontsize=11, va="center")
    ax.add_patch(Rectangle((0.02, 0.56), 0.045, 0.07, transform=ax.transAxes, facecolor="#FDBA74", edgecolor="#C05621", linewidth=2))
    ax.text(0.09, 0.595, "Orange map = score before top-k selection", transform=ax.transAxes, fontsize=11, va="center")

    note = (
        "Important:\n"
        "These are contextual token positions after self-attention.\n"
        "They are not detected object boxes."
    )
    ax.text(0.02, 0.42, note, transform=ax.transAxes, fontsize=11.5, color=MUTED, va="top", linespacing=1.5, fontweight="bold")

    summary = (
        f"true = {record['true_name']}\n"
        f"dense = {record['dense_name']} ({record['dense_conf']:.3f})\n"
        f"pruned = {record['pruned_name']} ({record['pruned_conf']:.3f})\n"
        f"grid = {grid_size}x{grid_size} patches\n"
        f"index = {record['dataset_index']}"
    )
    ax.text(0.02, 0.18, summary, transform=ax.transAxes, fontsize=11.2, va="top", linespacing=1.55)


def plot_example(record: Dict, path: Path, title_prefix: str) -> None:
    num_stages = len(record["stage_records"])
    grid_size = int(np.sqrt(record["stage_records"][0]["score_map"].size))
    patch_size = record["image"].shape[-1] // grid_size
    if num_stages != 2:
        raise ValueError("plot_example currently expects exactly 2 pruning stages")

    fig, axes = plt.subplots(2, 4, figsize=(18.8, 9.2))
    draw_original(axes[0, 0], record["image"], grid_size, patch_size, "Original Image")
    draw_explanation_panel(axes[1, 0], record, grid_size)

    color_ref = None
    for stage_index, stage in enumerate(record["stage_records"]):
        row = 0 if stage_index == 0 else 1
        draw_focus_view(
            axes[row, 1],
            record["image"],
            stage["kept_indices"],
            grid_size,
            patch_size,
            f"Stage {stage_index + 1} Retained Token Positions ({stage['kept_count']})",
        )
        draw_binary_mask(
            axes[row, 2],
            stage["kept_indices"],
            grid_size,
            f"Stage {stage_index + 1} Keep / Drop Grid",
        )
        color_ref = draw_score_map(
            axes[row, 3],
            stage["score_map"],
            stage["kept_indices"],
            f"Stage {stage_index + 1} Token Score Map",
        )
    fig.suptitle(
        f"{title_prefix} | true={record['true_name']} | dense={record['dense_name']} ({record['dense_conf']:.3f}) | "
        f"pruned={record['pruned_name']} ({record['pruned_conf']:.3f}) | idx={record['dataset_index']}",
        fontsize=13,
        fontweight="bold",
        y=0.97,
    )
    if color_ref is not None:
        fig.colorbar(color_ref, ax=[axes[0, 3], axes[1, 3]], fraction=0.018, pad=0.02, label="Normalized score")
    fig.subplots_adjust(left=0.04, right=0.97, top=0.88, bottom=0.07, wspace=0.16, hspace=0.12)
    save_fig(fig, path)


def build_figure_index(entries: Sequence[Dict]) -> str:
    lines = ["# Figure Index", "", "All figures below are produced from saved checkpoint inference on the CIFAR-10 test set.", ""]
    for entry in entries:
        lines.append(f"- `{entry['id']}`: {entry['caption']}")
        lines.append(f"  Path: `{entry['path']}`")
    return "\n".join(lines) + "\n"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export paper-style figures from class-project inference outputs.")
    parser.add_argument("--experiment-root", type=Path, default=ROOT / "outputs" / "final_experiments")
    parser.add_argument("--dense-checkpoint", type=Path, default=None)
    parser.add_argument("--pruned-checkpoint", type=Path, default=None)
    parser.add_argument("--qualitative-checkpoint", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-success-examples", type=int, default=4)
    return parser


def main() -> None:
    configure_matplotlib()
    args = build_argparser().parse_args()

    dense_checkpoint = args.dense_checkpoint or (args.experiment_root / "dense" / "best.pt")
    pruned_checkpoint = args.pruned_checkpoint or (args.experiment_root / "pruned_k75" / "best.pt")
    default_qualitative = args.experiment_root / "pruned_k25" / "best.pt"
    qualitative_checkpoint = args.qualitative_checkpoint or (default_qualitative if default_qualitative.exists() else pruned_checkpoint)
    output_dir = args.output_dir or (args.experiment_root / "paper_figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dense_model, dense_meta = load_checkpoint_model(dense_checkpoint, device)
    pruned_model, pruned_meta = load_checkpoint_model(pruned_checkpoint, device)
    qualitative_model, qualitative_meta = load_checkpoint_model(qualitative_checkpoint, device)
    class_names = dense_meta.get("class_names", pruned_meta.get("class_names", []))
    dataset = build_dataset(args.data_dir, dense_meta["config"]["image_size"])
    summary_rows = load_json(args.experiment_root / "plots" / "summary_table.json")
    pruned_label = pruned_checkpoint.parent.name
    qualitative_label = qualitative_checkpoint.parent.name

    inference = collect_inference_data(
        dense_model=dense_model,
        pruned_model=pruned_model,
        dataset=dataset,
        class_names=class_names,
        device=device,
        batch_size=args.batch_size,
    )
    qualitative_inference = collect_inference_data(
        dense_model=dense_model,
        pruned_model=qualitative_model,
        dataset=dataset,
        class_names=class_names,
        device=device,
        batch_size=args.batch_size,
    )

    figure_entries = []

    figure_specs = [
        ("fig01_accuracy_vs_tokens.png", "Accuracy vs final kept tokens across dense and pruned variants."),
        ("fig02_latency_vs_accuracy.png", "Inference accuracy vs throughput for dense and pruned variants."),
        ("fig03_classwise_accuracy.png", f"Per-class CIFAR-10 accuracy comparing dense and `{pruned_label}` inference."),
        ("fig04_prediction_agreement.png", f"Prediction agreement breakdown between dense and `{pruned_label}` checkpoints."),
        ("fig05_dense_confusion_matrix.png", "Dense checkpoint confusion matrix on the test set."),
        ("fig06_pruned_confusion_matrix.png", f"`{pruned_label}` confusion matrix on the test set."),
        ("fig07_stage1_retention_heatmap.png", f"How often each patch survives stage 1 pruning for `{qualitative_label}` across the test set."),
        ("fig08_stage2_retention_heatmap.png", f"How often each patch survives stage 2 pruning for `{qualitative_label}` across the test set."),
        ("fig09_stage1_score_heatmap.png", f"Mean normalized stage 1 token scores for `{qualitative_label}` across the test set."),
        ("fig10_stage2_score_heatmap.png", f"Mean normalized stage 2 token scores for `{qualitative_label}` over scored tokens."),
    ]

    plot_accuracy_vs_tokens(summary_rows, output_dir / figure_specs[0][0])
    plot_latency_vs_accuracy(summary_rows, output_dir / figure_specs[1][0])
    plot_classwise_accuracy(inference, pruned_label, output_dir / figure_specs[2][0])
    plot_prediction_agreement(inference, pruned_label, output_dir / figure_specs[3][0])
    plot_confusion_matrix(inference["dense_confusion"], class_names, "Dense Confusion Matrix", output_dir / figure_specs[4][0])
    plot_confusion_matrix(inference["pruned_confusion"], class_names, f"{pruned_label} Confusion Matrix", output_dir / figure_specs[5][0])

    stage1_retention_pct = qualitative_inference["stage_retention"][0] / qualitative_inference["targets"].size
    stage2_retention_pct = qualitative_inference["stage_retention"][1] / qualitative_inference["targets"].size
    stage_heatmap(stage1_retention_pct, f"Stage 1 Retention Frequency ({qualitative_label})", output_dir / figure_specs[6][0], "fraction kept", "Greens", False)
    stage_heatmap(stage2_retention_pct, f"Stage 2 Retention Frequency ({qualitative_label})", output_dir / figure_specs[7][0], "fraction kept", "Greens", False)
    stage_heatmap(qualitative_inference["stage_score_mean"][0], f"Mean Stage 1 Score Map ({qualitative_label})", output_dir / figure_specs[8][0], "normalized score", "Oranges", True)
    stage_heatmap(qualitative_inference["stage_score_mean"][1], f"Mean Stage 2 Score Map ({qualitative_label})", output_dir / figure_specs[9][0], "normalized score", "Oranges", True)

    for fig_id, caption in figure_specs:
        figure_entries.append({"id": fig_id.replace(".png", ""), "path": str(output_dir / fig_id), "caption": caption})

    success_examples = sorted(
        qualitative_inference["top_success_by_class"].values(),
        key=lambda record: record["pruned_conf"],
        reverse=True,
    )[: args.num_success_examples]
    for index, record in enumerate(success_examples, start=11):
        fig_name = f"fig{index:02d}_success_{record['true_name']}.png"
        plot_example(record, output_dir / fig_name, f"Successful inference example ({qualitative_label}): {record['true_name']}")
        figure_entries.append(
            {
                "id": fig_name.replace(".png", ""),
                "path": str(output_dir / fig_name),
                "caption": f"Successful `{qualitative_label}` inference example for class `{record['true_name']}` with retained token positions, binary keep/drop grids, and score maps.",
            }
        )

    extra_examples = []
    if qualitative_inference["top_failure"] is not None:
        extra_examples.append(("failure_dense_right_pruned_wrong", qualitative_inference["top_failure"], f"Failure case ({qualitative_label}): dense correct, pruned wrong"))
    if qualitative_inference["top_pruned_only"] is not None:
        extra_examples.append(("pruned_recovers_error", qualitative_inference["top_pruned_only"], f"Recovery case ({qualitative_label}): pruned correct, dense wrong"))

    next_index = 11 + len(success_examples)
    for suffix, record, title in extra_examples:
        fig_name = f"fig{next_index:02d}_{suffix}.png"
        plot_example(record, output_dir / fig_name, title)
        figure_entries.append(
            {
                "id": fig_name.replace(".png", ""),
                "path": str(output_dir / fig_name),
                "caption": f"{title}. This figure is taken from actual test-set inference.",
            }
        )
        next_index += 1

    manifest = {
        "dense_checkpoint": str(dense_checkpoint),
        "pruned_checkpoint": str(pruned_checkpoint),
        "qualitative_checkpoint": str(qualitative_checkpoint),
        "num_figures": len(figure_entries),
        "figures": figure_entries,
        "class_names": list(class_names),
    }
    save_json(output_dir / "figure_manifest.json", manifest)
    write_text(output_dir / "FIGURE_INDEX.md", build_figure_index(figure_entries))
    print(str(output_dir))


if __name__ == "__main__":
    main()
