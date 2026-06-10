#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import ToyDynamicViTConfig, build_model  # noqa: E402


CIFAR10_MEAN = torch.tensor((0.4914, 0.4822, 0.4465)).view(3, 1, 1)
CIFAR10_STD = torch.tensor((0.2470, 0.2435, 0.2616)).view(3, 1, 1)


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


def denormalize(image: torch.Tensor) -> torch.Tensor:
    image = image.cpu() * CIFAR10_STD + CIFAR10_MEAN
    return image.clamp(0.0, 1.0)


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def draw_overlay(ax, image: torch.Tensor, kept_indices, grid_size: int, patch_size: int, title: str) -> None:
    np_image = denormalize(image).permute(1, 2, 0).numpy()
    ax.imshow(np_image)
    ax.set_title(title)
    ax.axis("off")

    kept_set = set(int(idx) for idx in kept_indices)
    for patch_idx in range(grid_size * grid_size):
        row = patch_idx // grid_size
        col = patch_idx % grid_size
        x0 = col * patch_size
        y0 = row * patch_size

        if patch_idx in kept_set:
            rect = Rectangle((x0, y0), patch_size, patch_size, linewidth=1.5, edgecolor="lime", facecolor="none")
            ax.add_patch(rect)
        else:
            rect = Rectangle((x0, y0), patch_size, patch_size, linewidth=0.4, edgecolor="black", facecolor="black", alpha=0.25)
            ax.add_patch(rect)


@torch.no_grad()
def select_examples(model, dataset, device: torch.device, selection: str) -> List[Dict]:
    candidates: List[Dict] = []
    for dataset_index in range(len(dataset)):
        image, target = dataset[dataset_index]
        batch = image.unsqueeze(0).to(device)
        outputs = model(batch, return_info=True)
        probs = torch.softmax(outputs["logits"], dim=1)
        confidence, pred = probs.max(dim=1)
        pred_idx = int(pred.item())
        record = {
            "dataset_index": dataset_index,
            "target": int(target),
            "prediction": pred_idx,
            "confidence": float(confidence.item()),
            "final_patch_indices": outputs["final_patch_indices"][0].tolist(),
            "stage_kept_tokens": [stage["kept_token_count"] for stage in outputs["pruning"]],
        }
        if selection == "best_correct" and pred_idx != int(target):
            continue
        candidates.append(record)

    if not candidates:
        raise ValueError("No candidate images matched the requested selection strategy.")

    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    return candidates


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize kept tokens for the class-project pruned ViT.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "token_viz")
    parser.add_argument("--num-images", type=int, default=6)
    parser.add_argument("--selection", choices=["best_correct", "first_n"], default="best_correct")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint_model(args.checkpoint, device)
    if checkpoint.get("model_type") != "pruned":
        raise ValueError("Visualization is only useful for a pruned checkpoint.")

    image_size = checkpoint["config"]["image_size"]
    patch_size = checkpoint["config"]["patch_size"]
    class_names = checkpoint.get("class_names", [])
    dataset = build_dataset(args.data_dir, image_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = select_examples(model, dataset, device, args.selection)
    manifest = []

    for saved, candidate in enumerate(candidates[: args.num_images]):
        image, target = dataset[candidate["dataset_index"]]
        final_indices = candidate["final_patch_indices"]
        pred = candidate["prediction"]
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        draw_overlay(
            axes[0],
            image,
            kept_indices=range(model.num_patches),
            grid_size=model.grid_size,
            patch_size=patch_size,
            title="Original Patch Grid",
        )
        draw_overlay(
            axes[1],
            image,
            kept_indices=final_indices,
            grid_size=model.grid_size,
            patch_size=patch_size,
            title="Kept Tokens",
        )

        true_name = class_names[target] if class_names else str(target)
        pred_name = class_names[pred] if class_names else str(pred)
        fig.suptitle(
            f"true={true_name} | pred={pred_name} | conf={candidate['confidence']:.3f} | kept={len(final_indices)}"
        )
        fig.tight_layout()
        output_path = args.output_dir / f"sample_{saved:02d}.png"
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        manifest.append(
            {
                **candidate,
                "true_name": true_name,
                "pred_name": pred_name,
                "output_path": str(output_path),
            }
        )

    save_json(
        args.output_dir / "manifest.json",
        {
            "selection": args.selection,
            "checkpoint": str(args.checkpoint),
            "samples": manifest,
        },
    )


if __name__ == "__main__":
    main()
