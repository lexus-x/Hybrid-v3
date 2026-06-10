import torch
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.sota_hybrid_vit import build_sota_model
from scripts.train_sota_hybrid import build_dataloaders, evaluate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = ROOT / "outputs" / "sota_dense_final" / "last.pt"

print("Loading checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

dense_model = build_sota_model("dense").to(device)
dense_model.load_state_dict(checkpoint["state_dict"])

pruned_model = build_sota_model("pruned").to(device)
pruned_model.load_state_dict(checkpoint["state_dict"])

print("Building dataloaders...")
_, test_loader, _ = build_dataloaders(ROOT / "data", batch_size=128, num_workers=4)

print("Evaluating Dense Model...")
dense_metrics = evaluate(dense_model, test_loader, device)
print(f"Dense Accuracy: {dense_metrics['accuracy']:.4f}")

print("Evaluating Pruned Model...")
pruned_metrics = evaluate(pruned_model, test_loader, device)
print(f"Pruned Accuracy: {pruned_metrics['accuracy']:.4f}")
print(f"Retention: {pruned_metrics['accuracy'] / dense_metrics['accuracy'] * 100:.2f}%")
