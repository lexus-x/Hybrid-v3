#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import ToyDynamicViTConfig, build_model  # noqa: E402


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def build_loader(data_dir: Path, image_size: int, batch_size: int, num_workers: int) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    dataset = datasets.CIFAR10(root=data_dir, train=False, transform=transform, download=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def load_model(checkpoint_path: Path, device: torch.device):
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


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device, max_batches: int = 0) -> Dict:
    total_correct = 0
    total_examples = 0
    stage_kept = None

    for batch_idx, (images, targets) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images, return_info=True)
        logits = outputs["logits"]
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += images.size(0)

        if outputs["pruning"]:
            if stage_kept is None:
                stage_kept = [0.0 for _ in outputs["pruning"]]
            for idx, stage in enumerate(outputs["pruning"]):
                stage_kept[idx] += stage["kept_token_count"] * images.size(0)

    metrics = {
        "accuracy": total_correct / max(1, total_examples),
        "num_examples": total_examples,
    }
    if stage_kept is not None:
        metrics["avg_kept_tokens_per_stage"] = [value / total_examples for value in stage_kept]
    return metrics


@torch.no_grad()
def benchmark(model, loader: DataLoader, device: torch.device, timing_batches: int, warmup_batches: int) -> Dict[str, float]:
    iterator = iter(loader)

    for _ in range(warmup_batches):
        try:
            images, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, _ = next(iterator)
        images = images.to(device, non_blocking=True)
        _ = model(images)

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_time = 0.0
    total_images = 0

    for _ in range(timing_batches):
        try:
            images, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, _ = next(iterator)

        images = images.to(device, non_blocking=True)
        start = time.perf_counter()
        _ = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.perf_counter() - start
        total_images += images.size(0)

    return {
        "timing_batches": timing_batches,
        "total_images": total_images,
        "avg_batch_seconds": total_time / max(1, timing_batches),
        "avg_image_milliseconds": (total_time / max(1, total_images)) * 1000.0,
        "images_per_second": total_images / max(total_time, 1e-8),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a class-project toy DynamicViT model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "eval_metrics.json")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--timing-batches", type=int, default=20)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, checkpoint = load_model(args.checkpoint, device)
    image_size = checkpoint["config"]["image_size"]
    loader = build_loader(args.data_dir, image_size, args.batch_size, args.num_workers)

    metrics = evaluate(model, loader, device, max_batches=args.max_batches)
    timing = benchmark(model, loader, device, args.timing_batches, args.warmup_batches)

    payload = {
        "checkpoint": str(args.checkpoint),
        "model_type": checkpoint.get("model_type", "unknown"),
        "metrics": metrics,
        "timing": timing,
    }
    save_json(args.output_json, payload)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
