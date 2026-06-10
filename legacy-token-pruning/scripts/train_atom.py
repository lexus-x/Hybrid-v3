#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.atom_vit import build_atom_model

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

def parse_int_list(value: str) -> Tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(item.strip()) for item in value.split(","))

def parse_float_list(value: str) -> Tuple[float, ...]:
    if not value:
        return ()
    return tuple(float(item.strip()) for item in value.split(","))

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return train_transform, eval_transform

def build_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    train_transform, eval_transform = build_transforms()
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=train_transform,
        download=True,
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        transform=eval_transform,
        download=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader, train_dataset.classes

def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def set_model_keep_ratios(model: nn.Module, keep_ratios: Sequence[float]) -> None:
    if hasattr(model, "set_keep_ratios"):
        model.set_keep_ratios(keep_ratios)

def compute_epoch_keep_ratios(
    target_keep_ratios: Sequence[float],
    epoch: int,
    warmup_epochs: int,
    ramp_epochs: int,
) -> Tuple[float, ...]:
    if not target_keep_ratios:
        return ()
    if epoch <= warmup_epochs:
        return tuple(1.0 for _ in target_keep_ratios)
    if ramp_epochs <= 0:
        return tuple(target_keep_ratios)

    progress = min(1.0, (epoch - warmup_epochs) / ramp_epochs)
    return tuple(1.0 - progress * (1.0 - ratio) for ratio in target_keep_ratios)

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 0,
) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch_idx, (images, targets) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images, return_info=True)
        logits = outputs["logits"]
        loss = criterion(logits, targets)

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += images.size(0)

    metrics = {
        "loss": total_loss / max(1, total_examples),
        "accuracy": total_correct / max(1, total_examples),
    }
    return metrics

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    teacher_model: Optional[nn.Module] = None,
    distill_alpha: float = 0.0,
    distill_temperature: float = 2.0,
    max_batches: int = 0,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_ce_loss = 0.0
    running_kd_loss = 0.0
    running_acc = 0.0
    total_examples = 0

    for batch_idx, (images, targets) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        
        # AToM doesn't have a differentiable masking loss (like DynamicViT)
        # It natively absorbs tokens.
        logits = model(images)
        ce_loss = criterion(logits, targets)
        
        kd_loss = torch.tensor(0.0, device=device)
        loss = ce_loss
        
        if teacher_model is not None and distill_alpha > 0.0:
            with torch.no_grad():
                teacher_logits = teacher_model(images)
            temperature = distill_temperature
            kd_loss = F.kl_div(
                F.log_softmax(logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1),
                reduction="batchmean",
            ) * (temperature ** 2)
            loss = (1.0 - distill_alpha) * ce_loss + distill_alpha * kd_loss
            
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_ce_loss += ce_loss.item() * batch_size
        running_kd_loss += kd_loss.item() * batch_size
        running_acc += accuracy_from_logits(logits, targets) * batch_size
        total_examples += batch_size

    return {
        "loss": running_loss / max(1, total_examples),
        "ce_loss": running_ce_loss / max(1, total_examples),
        "kd_loss": running_kd_loss / max(1, total_examples),
        "accuracy": running_acc / max(1, total_examples),
    }

def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train AToM ViT model.")
    parser.add_argument("--model", choices=["dense", "pruned"], default="pruned")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "atom_train_run")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prune-layers", type=str, default="1,3,5")
    parser.add_argument("--keep-ratios", type=str, default="0.75,0.5,0.25")
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--distill-alpha", type=float, default=0.5)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--pruning-warmup-epochs", type=int, default=2)
    parser.add_argument("--pruning-ramp-epochs", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--pretrained-checkpoint", type=Path, default=None)
    return parser

def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prune_layers = parse_int_list(args.prune_layers)
    keep_ratios = parse_float_list(args.keep_ratios)

    model = build_atom_model(
        model_type=args.model,
        prune_layers=prune_layers,
        keep_ratios=keep_ratios,
    ).to(device)
    
    if args.pretrained_checkpoint is not None:
        checkpoint = torch.load(args.pretrained_checkpoint, map_location=device, weights_only=True)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained checkpoint from {args.pretrained_checkpoint}")

    train_loader, test_loader, class_names = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    teacher_model = None
    if args.teacher_checkpoint is not None:
        teacher_model = build_atom_model(model_type="dense").to(device)
        checkpoint = torch.load(args.teacher_checkpoint, map_location=device, weights_only=True)
        teacher_model.load_state_dict(checkpoint["state_dict"])
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        print(f"Loaded teacher checkpoint from {args.teacher_checkpoint}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, float]] = []
    best_acc = -1.0
    target_keep_ratios = keep_ratios if args.model == "pruned" else ()

    for epoch in range(1, args.epochs + 1):
        epoch_keep_ratios = compute_epoch_keep_ratios(
            target_keep_ratios=target_keep_ratios,
            epoch=epoch,
            warmup_epochs=args.pruning_warmup_epochs,
            ramp_epochs=args.pruning_ramp_epochs,
        )
        if epoch_keep_ratios:
            set_model_keep_ratios(model, epoch_keep_ratios)

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            teacher_model=teacher_model,
            distill_alpha=args.distill_alpha if teacher_model is not None else 0.0,
            distill_temperature=args.distill_temperature,
            max_batches=args.max_train_batches,
        )
        if target_keep_ratios:
            set_model_keep_ratios(model, target_keep_ratios)
        eval_metrics = evaluate(
            model,
            test_loader,
            device,
            max_batches=args.max_eval_batches,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_ce_loss": train_metrics["ce_loss"],
            "train_kd_loss": train_metrics["kd_loss"],
            "train_accuracy": train_metrics["accuracy"],
            "test_loss": eval_metrics["loss"],
            "test_accuracy": eval_metrics["accuracy"],
        }
        if epoch_keep_ratios:
            row["train_keep_ratios"] = list(epoch_keep_ratios)
        history.append(row)

        print(
            f"epoch={epoch} "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.4f} "
            f"test_loss={row['test_loss']:.4f} test_acc={row['test_accuracy']:.4f}"
        )

        checkpoint = {
            "state_dict": model.state_dict(),
            "model_type": args.model,
            "history": history,
        }
        torch.save(checkpoint, output_dir / "last.pt")

        if row["test_accuracy"] > best_acc:
            best_acc = row["test_accuracy"]
            torch.save(checkpoint, output_dir / "best.pt")

    save_json(
        output_dir / "metrics.json",
        {
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "best_test_accuracy": best_acc,
            "history": history,
        },
    )

if __name__ == "__main__":
    main()
