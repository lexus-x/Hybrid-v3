#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_toy_dynamic_pruner import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    set_seed,
    save_json,
)
from src import build_model


class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]


def build_simclr_transforms(image_size: int) -> ContrastiveTransformations:
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    base_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return ContrastiveTransformations(base_transform, n_views=2)


class SimCLRWrapper(nn.Module):
    def __init__(self, base_encoder: nn.Module, embed_dim: int, proj_dim: int = 128):
        super().__init__()
        self.backbone = base_encoder
        # Extract features instead of predicting logits
        self.backbone.head = nn.Identity()

        # Projection head: 2-layer MLP
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns representations before head
        h = self.backbone(x)
        z = self.projection_head(h)
        return z


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0) # [2N, D]
    z = F.normalize(z, dim=1)

    # Compute similarity matrix
    sim_matrix = torch.matmul(z, z.T) / temperature

    # Mask out self-similarity by setting diagonal to a very negative number
    sim_matrix.fill_diagonal_(float('-inf'))

    # Targets: z1_i should match z2_i, and z2_i should match z1_i
    # For a sample at index i, its positive is at i + batch_size
    # For a sample at i + batch_size, its positive is at i
    labels = torch.cat([
        torch.arange(batch_size) + batch_size,
        torch.arange(batch_size)
    ], dim=0).to(z.device)

    return F.cross_entropy(sim_matrix, labels)


def train_simclr_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float,
    max_batches: int = 0,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    total_examples = 0

    for batch_idx, (images, _) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break

        images1 = images[0].to(device, non_blocking=True)
        images2 = images[1].to(device, non_blocking=True)
        batch_size = images1.size(0)

        optimizer.zero_grad(set_to_none=True)

        z1 = model(images1)
        z2 = model(images2)

        loss = nt_xent_loss(z1, z2, temperature)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        total_examples += batch_size

    return {
        "loss": running_loss / max(1, total_examples),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain a class-project toy ViT using SimCLR.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "simclr_run")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_encoder = build_model(
        model_type="dense",
        num_classes=10,
        image_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
    )
    
    model = SimCLRWrapper(base_encoder, embed_dim=args.embed_dim, proj_dim=128).to(device)

    train_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        transform=build_simclr_transforms(args.image_size),
        download=True,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, float]] = []
    best_loss = float("inf")

    print(f"Starting SimCLR Pretraining for {args.epochs} epochs on {device}...")

    for epoch in range(1, args.epochs + 1):
        metrics = train_simclr_epoch(
            model,
            train_loader,
            optimizer,
            device,
            temperature=args.temperature,
            max_batches=args.max_train_batches,
        )

        row = {
            "epoch": epoch,
            "train_loss": metrics["loss"],
        }
        history.append(row)

        print(f"epoch={epoch} train_loss={row['train_loss']:.4f}")

        # Save the actual base encoder we care about
        # Exclude projection head which is discarded for classification
        backbone_state_dict = model.backbone.state_dict()
        
        checkpoint = {
            "state_dict": backbone_state_dict,
            "config": base_encoder.config.to_dict(),
            "model_type": "dense",
            "history": history,
        }
        torch.save(checkpoint, output_dir / "last.pt")

        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            torch.save(checkpoint, output_dir / "best.pt")

    save_json(
        output_dir / "metrics.json",
        {
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "best_train_loss": best_loss,
            "history": history,
        },
    )

if __name__ == "__main__":
    main()
