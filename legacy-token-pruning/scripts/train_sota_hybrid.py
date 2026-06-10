#!/usr/bin/env python3
import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm.data import Mixup
from timm.data.mixup import mixup_target
from timm.loss import SoftTargetCrossEntropy

class KDMixup(Mixup):
    def __call__(self, x, target):
        assert len(x) % 2 == 0, 'Batch size should be even when using this'
        if self.mode == 'elem':
            lam = self._mix_elem(x)
        elif self.mode == 'pair':
            lam = self._mix_pair(x)
        else:
            lam = self._mix_batch(x)
        target = mixup_target(target, self.num_classes, lam, self.label_smoothing)
        return x, target, lam

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sota_hybrid_vit import build_sota_model

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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

def build_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform

class IndexedCIFAR10(datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index

def build_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    img_size: int = 224,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    train_transform, eval_transform = build_transforms(img_size)
    train_dataset = IndexedCIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=eval_transform
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
    if targets.dim() == 2:
        targets = targets.argmax(dim=1)
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

    pbar = tqdm(loader, desc="Evaluating", leave=False)
    for batch_idx, (images, targets) in enumerate(pbar):
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
    cached_logits: Optional[torch.Tensor] = None,
    distill_alpha: float = 0.0,
    distill_temperature: float = 2.0,
    max_batches: int = 0,
    mixup_fn: Optional[KDMixup] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_ce_loss = 0.0
    running_kd_loss = 0.0
    running_acc = 0.0
    total_examples = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch_idx, batch_data in enumerate(pbar):
        if max_batches and batch_idx >= max_batches:
            break
            
        if len(batch_data) == 3:
            images, targets, indices = batch_data
        else:
            images, targets = batch_data
            indices = None

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        lam = 1.0
        if mixup_fn is not None:
            images, targets, lam_tensor = mixup_fn(images, targets)
            if isinstance(lam_tensor, float):
                lam = lam_tensor
            else:
                lam = lam_tensor.view(-1, 1).to(device)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(images)
                ce_loss = criterion(logits, targets)
                loss = ce_loss
                kd_loss = torch.tensor(0.0, device=device)
                
                if cached_logits is not None and distill_alpha > 0.0 and indices is not None:
                    teacher_logits_raw = cached_logits[indices].to(device, non_blocking=True)
                    if mixup_fn is not None:
                        teacher_logits = teacher_logits_raw * lam + teacher_logits_raw.flip(0) * (1.0 - lam)
                    else:
                        teacher_logits = teacher_logits_raw
                        
                    temperature = distill_temperature
                    kd_loss = F.kl_div(
                        F.log_softmax(logits / temperature, dim=1),
                        F.softmax(teacher_logits / temperature, dim=1),
                        reduction="batchmean",
                    ) * (temperature ** 2)
                    loss = (1.0 - distill_alpha) * ce_loss + distill_alpha * kd_loss
                    
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            ce_loss = criterion(logits, targets)
            kd_loss = torch.tensor(0.0, device=device)
            loss = ce_loss
            
            if cached_logits is not None and distill_alpha > 0.0 and indices is not None:
                teacher_logits_raw = cached_logits[indices].to(device, non_blocking=True)
                if mixup_fn is not None:
                    teacher_logits = teacher_logits_raw * lam + teacher_logits_raw.flip(0) * (1.0 - lam)
                else:
                    teacher_logits = teacher_logits_raw
                    
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
        pbar.set_postfix({"loss": f"{running_loss / total_examples:.4f}", "acc": f"{running_acc / total_examples:.4f}"})

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
    parser = argparse.ArgumentParser(description="Train SOTA Hybrid ViT model.")
    parser.add_argument("--model", choices=["dense", "pruned"], default="pruned")
    parser.add_argument("--model-name", type=str, default="vit_small_patch16_224")
    parser.add_argument("--img-size", type=int, default=224, help="Image resolution for training")
    parser.add_argument("--prune-mode", choices=["lite", "absorption"], default="lite")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "sota_train_run")
    parser.add_argument("--teacher-model-name", type=str, default="vit_large_patch16_384", help="Name of the teacher model backbone")
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5) # Very gentle LR for fine-tuning massive models
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prune-layers", type=str, default="3,6,9")
    parser.add_argument("--keep-ratios", type=str, default="0.75,0.5,0.25")
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--cached-logits", type=Path, default=None, help="Path to cached teacher logits")
    parser.add_argument("--distill-alpha", type=float, default=0.5)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--pruning-warmup-epochs", type=int, default=2)
    parser.add_argument("--pruning-ramp-epochs", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    return parser

def main() -> None:
    args = build_argparser().parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(args.output_dir / "training.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Starting training with args: {args}")
    
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prune_layers = parse_int_list(args.prune_layers)
    keep_ratios = parse_float_list(args.keep_ratios)

    model = build_sota_model(
        model_type=args.model,
        prune_layers=prune_layers,
        keep_ratios=keep_ratios,
        prune_mode=args.prune_mode,
        model_name=args.model_name,
    ).to(device)

    train_loader, test_loader, class_names = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )

    # A100 Ultimate Speedup: torch.compile
    # torch.compile takes hours for this massive model and hangs, bypassing.
    logging.info("Skipping torch.compile to avoid long hangs.")
    # try:
    #     model = torch.compile(model)
    #     logging.info("Successfully applied torch.compile!")
    # except Exception as e:
    #     logging.info(f"torch.compile failed, continuing without it: {e}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    warmup_epochs = min(5, args.epochs // 5)
    main_epochs = args.epochs - warmup_epochs
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=max(1, warmup_epochs)
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, main_epochs), eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )

    mixup_fn = KDMixup(
        mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, mode='batch',
        label_smoothing=args.label_smoothing, num_classes=10
    )
    criterion = SoftTargetCrossEntropy()
    scaler = torch.amp.GradScaler('cuda')
    
    cached_logits = None
    if args.cached_logits is not None:
        logging.info(f"Loading cached teacher logits from {args.cached_logits}")
        cached_logits = torch.load(args.cached_logits, map_location="cpu", weights_only=True)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, float]] = []
    best_acc = -1.0
    target_keep_ratios = keep_ratios if args.model == "pruned" else ()
    start_epoch = 1

    if args.resume is not None and args.resume.exists():
        logging.info(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        # if "scheduler" in checkpoint:
        #     scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        if "history" in checkpoint:
            history = checkpoint["history"]
            start_epoch = history[-1]["epoch"] + 1
            for _ in range(start_epoch - 1):
                scheduler.step()
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        else:
            if history:
                best_acc = max([row.get("test_accuracy", -1.0) for row in history])

    for epoch in range(start_epoch, args.epochs + 1):
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
            cached_logits=cached_logits,
            distill_alpha=args.distill_alpha if cached_logits is not None else 0.0,
            distill_temperature=args.distill_temperature,
            max_batches=args.max_train_batches,
            mixup_fn=mixup_fn,
            scaler=scaler,
        )
        if target_keep_ratios:
            set_model_keep_ratios(model, target_keep_ratios)
        eval_metrics = evaluate(
            model,
            test_loader,
            device,
            max_batches=args.max_eval_batches,
        )
        
        scheduler.step()

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

        logging.info(
            f"epoch={epoch} "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.4f} "
            f"test_loss={row['test_loss']:.4f} test_acc={row['test_accuracy']:.4f}"
        )

        csv_path = args.output_dir / "metrics.csv"
        with open(csv_path, "a") as f:
            if epoch == 1:
                f.write("Epoch,Train_Loss,Train_Acc,Test_Loss,Test_Acc\n")
            f.write(f"{epoch},{row['train_loss']:.4f},{row['train_accuracy']:.4f},{row['test_loss']:.4f},{row['test_accuracy']:.4f}\n")

        checkpoint = {
            "state_dict": model.state_dict(),
            "model_type": args.model,
            "history": history,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_acc": best_acc,
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
