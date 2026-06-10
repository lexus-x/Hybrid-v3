#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_toy_dynamic_pruner.py"
EVAL_SCRIPT = ROOT / "scripts" / "eval_toy_dynamic_pruner.py"
PLOT_SCRIPT = ROOT / "scripts" / "plot_experiment_results.py"
VIZ_SCRIPT = ROOT / "scripts" / "visualize_tokens.py"
GIF_SCRIPT = ROOT / "scripts" / "make_token_animation.py"


def parse_keep_ratios(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_command(cmd: List[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run dense/pruned class-project experiment variants.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "experiments")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--prune-layers", type=str, default="1,3")
    parser.add_argument("--keep-ratios", type=str, default="0.75,0.5,0.25")
    parser.add_argument("--scorer-hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--teacher-distill-alpha", type=float, default=0.5)
    parser.add_argument("--teacher-distill-temperature", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--pruning-warmup-epochs", type=int, default=2)
    parser.add_argument("--pruning-ramp-epochs", type=int, default=4)
    parser.add_argument("--pretrained-checkpoint", type=Path, default=None, help="Path to pre-trained SimCLR backbone")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--timing-batches", type=int, default=20)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--artifact-images", type=int, default=8)
    parser.add_argument("--skip-artifacts", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    runs = []
    common_train_args = [
        "--data-dir",
        str(args.data_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--image-size",
        str(args.image_size),
        "--patch-size",
        str(args.patch_size),
        "--embed-dim",
        str(args.embed_dim),
        "--depth",
        str(args.depth),
        "--num-heads",
        str(args.num_heads),
        "--mlp-ratio",
        str(args.mlp_ratio),
        "--dropout",
        str(args.dropout),
        "--scorer-hidden-dim",
        str(args.scorer_hidden_dim),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--label-smoothing",
        str(args.label_smoothing),
        "--pruning-warmup-epochs",
        str(args.pruning_warmup_epochs),
        "--pruning-ramp-epochs",
        str(args.pruning_ramp_epochs),
    ]
    if args.pretrained_checkpoint is not None:
        common_train_args += ["--pretrained-checkpoint", str(args.pretrained_checkpoint)]
    if args.max_train_batches:
        common_train_args += ["--max-train-batches", str(args.max_train_batches)]
    if args.max_eval_batches:
        common_train_args += ["--max-eval-batches", str(args.max_eval_batches)]

    dense_dir = output_root / "dense"
    run_command(
        [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--model",
            "dense",
            "--output-dir",
            str(dense_dir),
            *common_train_args,
        ]
    )
    dense_eval = dense_dir / "eval.json"
    run_command(
        [
            sys.executable,
            str(EVAL_SCRIPT),
            "--checkpoint",
            str(dense_dir / "best.pt"),
            "--data-dir",
            str(args.data_dir),
            "--output-json",
            str(dense_eval),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--timing-batches",
            str(args.timing_batches),
            "--warmup-batches",
            str(args.warmup_batches),
            *(["--max-batches", str(args.max_eval_batches)] if args.max_eval_batches else []),
        ]
    )
    runs.append({"name": "dense", "model": "dense", "run_dir": str(dense_dir)})

    for keep_ratio in parse_keep_ratios(args.keep_ratios):
        label = f"pruned_k{int(round(keep_ratio * 100)):02d}"
        run_dir = output_root / label
        train_cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--model",
            "pruned",
            "--output-dir",
            str(run_dir),
            "--prune-layers",
            args.prune_layers,
            "--keep-ratios",
            f"{keep_ratio},{keep_ratio}",
            *common_train_args,
        ]
        if args.teacher_distill_alpha > 0.0:
            train_cmd.extend(
                [
                    "--teacher-checkpoint",
                    str(dense_dir / "best.pt"),
                    "--distill-alpha",
                    str(args.teacher_distill_alpha),
                    "--distill-temperature",
                    str(args.teacher_distill_temperature),
                ]
            )
        run_command(
            train_cmd
        )
        eval_json = run_dir / "eval.json"
        run_command(
            [
                sys.executable,
                str(EVAL_SCRIPT),
                "--checkpoint",
                str(run_dir / "best.pt"),
                "--data-dir",
                str(args.data_dir),
                "--output-json",
                str(eval_json),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--timing-batches",
                str(args.timing_batches),
                "--warmup-batches",
                str(args.warmup_batches),
                *(["--max-batches", str(args.max_eval_batches)] if args.max_eval_batches else []),
            ]
        )
        runs.append(
            {
                "name": label,
                "model": "pruned",
                "keep_ratio": keep_ratio,
                "run_dir": str(run_dir),
            }
        )

    index_path = output_root / "experiment_index.json"
    save_json(
        index_path,
        {
            "runs": runs,
            "notes": "Each run directory should contain best.pt, last.pt, metrics.json, and eval.json.",
        },
    )

    if args.skip_artifacts:
        return

    run_command(
        [
            sys.executable,
            str(PLOT_SCRIPT),
            "--experiment-root",
            str(output_root),
        ]
    )

    best_pruned_run = None
    best_pruned_acc = float("-inf")
    for run in runs:
        if run["model"] != "pruned":
            continue
        eval_payload = load_json(Path(run["run_dir"]) / "eval.json")
        accuracy = eval_payload["metrics"]["accuracy"]
        if accuracy > best_pruned_acc:
            best_pruned_acc = accuracy
            best_pruned_run = Path(run["run_dir"])

    if best_pruned_run is None:
        return

    run_command(
        [
            sys.executable,
            str(VIZ_SCRIPT),
            "--checkpoint",
            str(best_pruned_run / "best.pt"),
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(best_pruned_run / "token_viz"),
            "--num-images",
            str(args.artifact_images),
        ]
    )
    run_command(
        [
            sys.executable,
            str(GIF_SCRIPT),
            "--checkpoint",
            str(best_pruned_run / "best.pt"),
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(best_pruned_run / "animations"),
            "--sample-strategy",
            "best_correct",
        ]
    )


if __name__ == "__main__":
    main()
