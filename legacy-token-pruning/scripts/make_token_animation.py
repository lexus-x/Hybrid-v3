#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import imageio.v2 as imageio
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


def draw_overlay(ax, image: torch.Tensor, kept_indices: Iterable[int], grid_size: int, patch_size: int, title: str) -> None:
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
            rect = Rectangle((x0, y0), patch_size, patch_size, linewidth=1.6, edgecolor="#2CA02C", facecolor="none")
        else:
            rect = Rectangle((x0, y0), patch_size, patch_size, linewidth=0.35, edgecolor="black", facecolor="black", alpha=0.28)
        ax.add_patch(rect)


@torch.no_grad()
def choose_sample_index(model, dataset, device: torch.device, sample_index: int, strategy: str) -> int:
    if strategy == "index":
        return sample_index

    best_index = None
    best_confidence = float("-inf")
    for dataset_index in range(len(dataset)):
        image, target = dataset[dataset_index]
        outputs = model(image.unsqueeze(0).to(device), return_info=True)
        probs = torch.softmax(outputs["logits"], dim=1)
        confidence, pred = probs.max(dim=1)
        if int(pred.item()) != int(target):
            continue
        if float(confidence.item()) > best_confidence:
            best_confidence = float(confidence.item())
            best_index = dataset_index

    if best_index is None:
        raise ValueError("Could not find a correct prediction for GIF export.")
    return best_index


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Create a GIF showing token pruning stages for one sample.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "animations")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sample-strategy", choices=["index", "best_correct"], default="index")
    parser.add_argument("--fps", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint_model(args.checkpoint, device)
    if checkpoint.get("model_type") != "pruned":
        raise ValueError("Animation is only useful for a pruned checkpoint.")

    dataset = build_dataset(args.data_dir, checkpoint["config"]["image_size"])
    selected_index = choose_sample_index(
        model=model,
        dataset=dataset,
        device=device,
        sample_index=args.sample_index,
        strategy=args.sample_strategy,
    )
    image, target = dataset[selected_index]
    class_names = checkpoint.get("class_names", [])
    true_name = class_names[target] if class_names else str(target)

    outputs = model(image.unsqueeze(0).to(device), return_info=True)
    pred = int(outputs["logits"].argmax(dim=1).item())
    pred_name = class_names[pred] if class_names else str(pred)

    kept_sequences: List[List[int]] = [list(range(outputs["num_patches"]))]
    titles = ["All Tokens"]
    for stage in outputs["pruning"]:
        kept_sequences.append(stage["selected_patch_indices"][0].tolist())
        titles.append(f"Layer {stage['layer']} keep={stage['kept_token_count']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    for idx, (kept_indices, title) in enumerate(zip(kept_sequences, titles)):
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        draw_overlay(
            ax,
            image,
            kept_indices=kept_indices,
            grid_size=outputs["grid_size"],
            patch_size=checkpoint["config"]["patch_size"],
            title=title,
        )
        fig.suptitle(f"true={true_name} | pred={pred_name}")
        frame_path = args.output_dir / f"sample_{selected_index:03d}_frame_{idx:02d}.png"
        fig.tight_layout()
        fig.savefig(frame_path, dpi=160)
        plt.close(fig)
        frame_paths.append(frame_path)

    gif_path = args.output_dir / f"sample_{selected_index:03d}_pruning.gif"
    frames = [imageio.imread(path) for path in frame_paths]
    imageio.mimsave(gif_path, frames, fps=args.fps, loop=0)
    print(str(gif_path))


if __name__ == "__main__":
    main()
