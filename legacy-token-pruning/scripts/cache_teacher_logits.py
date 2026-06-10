#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sota_hybrid_vit import build_sota_model

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="vit_large_patch16_384")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "teacher_logits.pt")
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build eval transform (NO augmentation for caching base logits)
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Ensure dataset is not shuffled so that indices strictly align 1-to-1 with CIFAR-10 training set
    dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False, # MUST BE FALSE!
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Loading teacher model: {args.model_name}")
    model = build_sota_model(model_type="dense", model_name=args.model_name).to(device)
    
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    all_logits = []

    print("Caching logits...")
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for images, targets in tqdm(loader):
                images = images.to(device, non_blocking=True)
                logits = model(images)
                all_logits.append(logits.cpu())
    
    # Shape: [50000, 10]
    all_logits_tensor = torch.cat(all_logits, dim=0)
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_logits_tensor, args.output)
    
    print(f"Successfully cached logits of shape {all_logits_tensor.shape} to {args.output}")

if __name__ == "__main__":
    main()
