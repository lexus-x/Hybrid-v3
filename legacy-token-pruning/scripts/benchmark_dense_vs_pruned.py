#!/usr/bin/env python3
"""
Fair, same-architecture Dense-vs-Pruned benchmark for the class project.

Loads the actually-trained pruned ViT-Small/384 checkpoint and reports, for the
identical backbone:

  * test accuracy with pruning ON  (keep ratios as trained)
  * test accuracy with pruning OFF (keep=1.0 -> dense behaviour, same weights)
  * kept-token counts per pruned layer + total tokens processed
  * GFLOPs (fvcore) for a true dense model vs the pruned model
  * throughput (img/s), per-image latency, peak VRAM at several batch sizes

The accuracy comparison is the ablation "same weights, pruning toggled". The
speed/FLOPs comparison uses a clean dense backbone (no pruner modules, no
disabled flash-attn) so the dense numbers are not penalised. Results are written
to outputs/benchmark_results.json and outputs/benchmark_table.md.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from fvcore.nn import FlopCountAnalysis

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sota_hybrid_vit import build_sota_model
from scripts.train_sota_hybrid import build_dataloaders


def parse_floats(value):
    return tuple(float(v.strip()) for v in value.split(","))


def parse_ints(value):
    return tuple(int(v.strip()) for v in value.split(","))


@torch.no_grad()
def eval_with_tokens(model, loader, device, max_batches=0):
    """Full-test-set accuracy plus averaged kept-token bookkeeping."""
    model.eval()
    correct = 0
    total = 0
    token_sums = None          # per pruned-layer kept-token totals
    layer_info = None          # static layer / keep_ratio labels
    num_patches = None
    for batch_idx, (images, targets) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        out = model(images, return_info=True)
        logits = out["logits"]
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += images.size(0)
        num_patches = out["num_patches"]
        info = out["pruning"]
        if token_sums is None:
            token_sums = [0 for _ in info]
            layer_info = [(d["layer"], d["keep_ratio"]) for d in info]
        for i, d in enumerate(info):
            token_sums[i] += d["kept_token_count"] * images.size(0)

    per_layer = []
    if layer_info is not None:
        for (layer, keep_ratio), tok in zip(layer_info, token_sums):
            per_layer.append(
                {
                    "layer": layer,
                    "keep_ratio": keep_ratio,
                    "avg_kept_tokens": tok / max(1, total),
                }
            )
    return {
        "accuracy": correct / max(1, total),
        "num_patches": num_patches,
        "per_layer": per_layer,
    }


@torch.no_grad()
def measure_throughput(model, device, img_size, batch_sizes, iters=30, warmup=8):
    model.eval()
    results = {}
    for bs in batch_sizes:
        x = torch.randn(bs, 3, img_size, img_size, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _ = model(x)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        time_per_batch = (ms / 1000.0) / iters
        results[bs] = {
            "throughput_img_s": bs / time_per_batch,
            "latency_ms_per_img": (time_per_batch / bs) * 1000.0,
            "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        }
    return results


def gflops(model, device, img_size):
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    fca = FlopCountAnalysis(model, x)
    fca.unsupported_ops_warnings(False)
    fca.uncalled_modules_warnings(False)
    return fca.total() / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=ROOT / "outputs" / "sota_pruned_small_atp_final" / "best.pt")
    ap.add_argument("--model-name", type=str, default="vit_small_patch16_384")
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--prune-layers", type=str, default="3,6,9")
    ap.add_argument("--keep-ratios", type=str, default="0.75,0.5,0.4")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--bench-batch-sizes", type=str, default="1,32,128")
    ap.add_argument("--max-eval-batches", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prune_layers = parse_ints(args.prune_layers)
    keep_ratios = parse_floats(args.keep_ratios)
    bench_bs = parse_ints(args.bench_batch_sizes)

    print(f"Building pruned model {args.model_name} keep={keep_ratios} ...")
    pruned = build_sota_model(
        "pruned", prune_layers=prune_layers, keep_ratios=keep_ratios,
        prune_mode="lite", model_name=args.model_name,
    ).to(device)
    print(f"Loading checkpoint {args.checkpoint} ...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    pruned.load_state_dict(ckpt["state_dict"])
    best_acc = ckpt.get("best_acc", None)

    # Clean dense model (no pruner modules, flash-attn intact) for fair speed/FLOPs.
    print(f"Building clean dense model {args.model_name} ...")
    dense = build_sota_model("dense", model_name=args.model_name).to(device)

    n_params_dense = sum(p.numel() for p in dense.parameters())
    n_params_pruned = sum(p.numel() for p in pruned.parameters())

    print("Building dataloaders ...")
    _, test_loader, _ = build_dataloaders(
        args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, img_size=args.img_size,
    )

    # --- Accuracy: pruning ON vs OFF on the SAME trained weights ---
    print("Evaluating pruned (pruning ON) ...")
    pruned.set_keep_ratios(keep_ratios)
    acc_on = eval_with_tokens(pruned, test_loader, device, args.max_eval_batches)
    print(f"  pruning ON  accuracy = {acc_on['accuracy']:.4f}")

    print("Evaluating pruned (pruning OFF, keep=1.0) ...")
    pruned.set_keep_ratios(tuple(1.0 for _ in keep_ratios))
    acc_off = eval_with_tokens(pruned, test_loader, device, args.max_eval_batches)
    print(f"  pruning OFF accuracy = {acc_off['accuracy']:.4f}")
    pruned.set_keep_ratios(keep_ratios)  # restore for benchmarking

    # --- FLOPs ---
    print("Counting FLOPs ...")
    dense_gflops = gflops(dense, device, args.img_size)
    pruned_gflops = gflops(pruned, device, args.img_size)

    # --- Throughput ---
    print("Benchmarking dense throughput ...")
    dense_thr = measure_throughput(dense, device, args.img_size, bench_bs)
    print("Benchmarking pruned throughput ...")
    pruned_thr = measure_throughput(pruned, device, args.img_size, bench_bs)

    results = {
        "config": {
            "model_name": args.model_name,
            "img_size": args.img_size,
            "prune_layers": list(prune_layers),
            "keep_ratios": list(keep_ratios),
            "checkpoint": str(args.checkpoint),
            "best_acc_in_ckpt": best_acc,
        },
        "params": {"dense": n_params_dense, "pruned": n_params_pruned},
        "accuracy": {
            "pruned_on": acc_on["accuracy"],
            "pruned_off": acc_off["accuracy"],
            "num_patches": acc_on["num_patches"],
            "kept_tokens_per_layer": acc_on["per_layer"],
        },
        "gflops": {"dense": dense_gflops, "pruned": pruned_gflops},
        "throughput": {"dense": dense_thr, "pruned": pruned_thr},
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "benchmark_results.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)

    # --- Markdown table ---
    lines = []
    lines.append("# Dense vs Pruned ViT-Small/384 — CIFAR-10 Benchmark\n")
    lines.append(f"Backbone: `{args.model_name}` @ {args.img_size}px | "
                 f"prune layers {list(prune_layers)} | keep ratios {list(keep_ratios)}\n")
    lines.append("## Accuracy (same trained weights, pruning toggled)\n")
    lines.append("| Mode | Test accuracy |")
    lines.append("|---|---|")
    lines.append(f"| Pruning OFF (keep=1.0, dense behaviour) | {acc_off['accuracy']*100:.2f}% |")
    lines.append(f"| Pruning ON (keep {','.join(str(k) for k in keep_ratios)}) | {acc_on['accuracy']*100:.2f}% |")
    drop = (acc_off["accuracy"] - acc_on["accuracy"]) * 100
    lines.append(f"\n**Accuracy cost of pruning: {drop:+.2f} pt**\n")

    lines.append("## Token budget\n")
    lines.append(f"Patches per image (dense): **{acc_on['num_patches']}** (+1 CLS)\n")
    lines.append("| Pruned at layer | keep ratio | avg kept tokens |")
    lines.append("|---|---|---|")
    for d in acc_on["per_layer"]:
        lines.append(f"| {d['layer']} | {d['keep_ratio']} | {d['avg_kept_tokens']:.1f} |")
    if acc_on["per_layer"]:
        final_tok = acc_on["per_layer"][-1]["avg_kept_tokens"]
        lines.append(f"\nFinal blocks operate on **{final_tok:.1f} / {acc_on['num_patches']} "
                     f"tokens ({final_tok/acc_on['num_patches']*100:.0f}%)**.\n")

    lines.append("## Compute & speed\n")
    lines.append("| Metric | Dense | Pruned | Change |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Parameters | {n_params_dense:,} | {n_params_pruned:,} | "
                 f"{(n_params_pruned/n_params_dense-1)*100:+.1f}% |")
    lines.append(f"| GFLOPs (bs=1) | {dense_gflops:.2f} | {pruned_gflops:.2f} | "
                 f"{(1-pruned_gflops/dense_gflops)*100:.1f}% reduction |")
    for bs in bench_bs:
        d = dense_thr[bs]
        p = pruned_thr[bs]
        speedup = p["throughput_img_s"] / d["throughput_img_s"]
        speed_note = f"{speedup:.2f}x" + ("" if speedup >= 1 else " (pruned slower)")
        lat_pct = (p["latency_ms_per_img"] / d["latency_ms_per_img"] - 1) * 100
        lat_note = (f"{-lat_pct:.1f}% faster" if lat_pct < 0 else f"{lat_pct:.1f}% slower")
        vram_pct = (p["peak_vram_mb"] / d["peak_vram_mb"] - 1) * 100
        vram_note = (f"{vram_pct:.1f}% more" if vram_pct > 0 else f"{-vram_pct:.1f}% less")
        lines.append(f"| Throughput bs={bs} (img/s) | {d['throughput_img_s']:.1f} | "
                     f"{p['throughput_img_s']:.1f} | {speed_note} |")
        lines.append(f"| Latency bs={bs} (ms/img) | {d['latency_ms_per_img']:.3f} | "
                     f"{p['latency_ms_per_img']:.3f} | {lat_note} |")
        lines.append(f"| Peak VRAM bs={bs} (MB) | {d['peak_vram_mb']:.0f} | "
                     f"{p['peak_vram_mb']:.0f} | {vram_note} |")

    md_path = args.output_dir / "benchmark_table.md"
    md_path.write_text("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
