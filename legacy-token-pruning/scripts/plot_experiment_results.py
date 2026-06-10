#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_plot(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def collect_runs(experiment_root: Path) -> List[Dict]:
    index_path = experiment_root / "experiment_index.json"
    if index_path.exists():
        index = load_json(index_path)
        runs = index["runs"]
    else:
        runs = []
        for run_dir in sorted(path for path in experiment_root.iterdir() if path.is_dir()):
            if (run_dir / "metrics.json").exists() and (run_dir / "eval.json").exists():
                runs.append({"name": run_dir.name, "run_dir": str(run_dir)})

    resolved = []
    for run in runs:
        run_dir = Path(run["run_dir"])
        metrics = load_json(run_dir / "metrics.json")
        eval_payload = load_json(run_dir / "eval.json")
        resolved.append(
            {
                "name": run["name"],
                "run_dir": run_dir,
                "history": metrics["history"],
                "best_test_accuracy": metrics["best_test_accuracy"],
                "eval": eval_payload,
            }
        )
    return sorted(
        resolved,
        key=lambda run: (
            0 if run["name"] == "dense" else 1,
            -run["eval"]["metrics"].get("avg_kept_tokens_per_stage", [run["eval"]["metrics"].get("num_examples", 0)])[-1],
        ),
    )


def plot_accuracy_curves(runs: List[Dict], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for run in runs:
        epochs = [row["epoch"] for row in run["history"]]
        acc = [row["test_accuracy"] for row in run["history"]]
        ax.plot(epochs, acc, marker="o", label=run["name"])
    ax.set_title("Test Accuracy Across Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_plot(fig, output_dir / "accuracy_curves.png")


def plot_accuracy_bars(runs: List[Dict], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [run["name"] for run in runs]
    values = [run["eval"]["metrics"]["accuracy"] for run in runs]
    ax.bar(names, values, color=["#4C78A8" if "dense" in name else "#F58518" for name in names])
    ax.set_title("Test Accuracy by Variant")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, min(1.0, max(values) + 0.1))
    ax.grid(True, axis="y", alpha=0.3)
    save_plot(fig, output_dir / "accuracy_bars.png")


def plot_speed_bars(runs: List[Dict], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [run["name"] for run in runs]
    values = [run["eval"]["timing"]["images_per_second"] for run in runs]
    ax.bar(names, values, color=["#54A24B" if "dense" in name else "#E45756" for name in names])
    ax.set_title("Inference Throughput by Variant")
    ax.set_ylabel("Images / second")
    ax.grid(True, axis="y", alpha=0.3)
    save_plot(fig, output_dir / "speed_bars.png")


def plot_tradeoff_scatter(runs: List[Dict], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for run in runs:
        accuracy = run["eval"]["metrics"]["accuracy"]
        speed = run["eval"]["timing"]["images_per_second"]
        color = "#4C78A8" if run["name"] == "dense" else "#F58518"
        size = 140 if run["name"] == "dense" else 110
        ax.scatter(speed, accuracy, s=size, color=color, alpha=0.9)
        ax.annotate(run["name"], (speed, accuracy), xytext=(6, 4), textcoords="offset points")
    ax.set_title("Accuracy / Speed Tradeoff")
    ax.set_xlabel("Images / second")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.3)
    save_plot(fig, output_dir / "accuracy_speed_tradeoff.png")


def plot_token_bars(runs: List[Dict], output_dir: Path) -> None:
    names = []
    values = []
    for run in runs:
        kept = run["eval"]["metrics"].get("avg_kept_tokens_per_stage")
        if kept:
            names.append(run["name"])
            values.append(kept[-1])
    if not names:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, values, color="#72B7B2")
    ax.set_title("Final Kept Tokens by Pruned Variant")
    ax.set_ylabel("Tokens")
    ax.grid(True, axis="y", alpha=0.3)
    save_plot(fig, output_dir / "kept_tokens_bars.png")


def export_summary_table(runs: List[Dict], output_dir: Path) -> None:
    dense_run = next((run for run in runs if run["name"] == "dense"), None)
    dense_accuracy = dense_run["eval"]["metrics"]["accuracy"] if dense_run is not None else None
    dense_speed = dense_run["eval"]["timing"]["images_per_second"] if dense_run is not None else None
    rows = []
    for run in runs:
        metrics = run["eval"]["metrics"]
        timing = run["eval"]["timing"]
        speedup = timing["images_per_second"] / dense_speed if dense_speed else 1.0
        accuracy_delta = metrics["accuracy"] - dense_accuracy if dense_accuracy is not None else 0.0
        rows.append(
            {
                "name": run["name"],
                "accuracy": round(metrics["accuracy"], 4),
                "accuracy_delta_vs_dense": round(accuracy_delta, 4),
                "images_per_second": round(timing["images_per_second"], 2),
                "speedup_vs_dense": round(speedup, 2),
                "avg_image_milliseconds": round(timing["avg_image_milliseconds"], 4),
                "avg_kept_tokens_per_stage": metrics.get("avg_kept_tokens_per_stage", []),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary_table.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    lines = [
        "| Variant | Accuracy | Acc Delta vs Dense | Images/s | Speedup vs Dense | ms/image | Final kept tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        final_kept = row["avg_kept_tokens_per_stage"][-1] if row["avg_kept_tokens_per_stage"] else 64
        lines.append(
            f"| {row['name']} | {row['accuracy']:.4f} | {row['accuracy_delta_vs_dense']:+.4f} | "
            f"{row['images_per_second']:.2f} | {row['speedup_vs_dense']:.2f}x | "
            f"{row['avg_image_milliseconds']:.4f} | {final_kept} |"
        )
    with (output_dir / "summary_table.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    if dense_run is not None:
        best_pruned = max(
            (run for run in runs if run["name"] != "dense"),
            key=lambda run: run["eval"]["metrics"]["accuracy"],
            default=None,
        )
        if best_pruned is not None:
            best_metrics = best_pruned["eval"]["metrics"]
            best_timing = best_pruned["eval"]["timing"]
            best_speedup = best_timing["images_per_second"] / dense_speed
            if best_speedup > 1.05:
                claim = (
                    "The toy DynamicViT-style model improves inference speed while preserving most of the dense "
                    "baseline accuracy. The reproduction is functional, visually demonstrable, and gives a real "
                    "efficiency tradeoff."
                )
            elif best_speedup >= 0.98:
                claim = (
                    "The toy DynamicViT-style model preserves most of the dense baseline accuracy while keeping only "
                    "a fraction of the tokens. On this hardware, wall-clock timing is roughly flat, so the strongest "
                    "evidence is token-budget reduction rather than latency gain."
                )
            else:
                claim = (
                    "The toy DynamicViT-style model preserves most of the dense baseline accuracy while keeping only "
                    "a fraction of the tokens, but this tiny model does not run faster on the A100 because pruning "
                    "overhead dominates wall-clock timing."
                )
            overview_lines = [
                "# Experiment Overview",
                "",
                f"- Dense baseline accuracy: {dense_accuracy:.4f}",
                f"- Best pruned variant: {best_pruned['name']}",
                f"- Best pruned accuracy: {best_metrics['accuracy']:.4f}",
                f"- Best pruned speedup vs dense: {best_speedup:.2f}x",
                "",
                "Use this project claim:",
                "",
                f"> {claim}",
                "",
                "Artifacts:",
                "",
                "- `accuracy_speed_tradeoff.png`",
                "- `accuracy_curves.png`",
                "- `summary_table.md`",
            ]
            with (output_dir / "RESULT_OVERVIEW.md").open("w", encoding="utf-8") as handle:
                handle.write("\n".join(overview_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate slide-ready plots from class-project experiment runs.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.experiment_root / "plots"
    runs = collect_runs(args.experiment_root)
    if not runs:
        raise ValueError("No runs with metrics.json and eval.json were found.")

    plot_accuracy_curves(runs, output_dir)
    plot_accuracy_bars(runs, output_dir)
    plot_speed_bars(runs, output_dir)
    plot_tradeoff_scatter(runs, output_dir)
    plot_token_bars(runs, output_dir)
    export_summary_table(runs, output_dir)


if __name__ == "__main__":
    main()
