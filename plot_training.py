"""Create train-vs-validation accuracy and loss plots from a CA-SAM run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_metric(epochs: list[dict[str, float]], task: str, metric: str, output: Path) -> None:
    x = range(1, len(epochs) + 1)
    train = [epoch[f"train_{metric}"] for epoch in epochs]
    validation = [epoch[f"val_{metric}"] for epoch in epochs]

    plt.figure(figsize=(7, 4.5))
    plt.plot(x, train, marker="o", linewidth=2, label="Train")
    plt.plot(x, validation, marker="o", linewidth=2, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Pixel accuracy" if metric == "accuracy" else "BCE + Dice loss")
    plt.title(f"{task}: train vs validation {metric}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True, help="Directory containing status.json")
    args = parser.parse_args()

    status = json.loads((args.run / "status.json").read_text())
    plot_dir = args.run / "plots"
    plot_dir.mkdir(exist_ok=True)

    for task_entry in status["history"]:
        task = task_entry["task"]
        epochs = task_entry["epochs"]
        if not epochs:
            continue
        plot_metric(epochs, task, "accuracy", plot_dir / f"{task}_accuracy.png")
        plot_metric(epochs, task, "loss", plot_dir / f"{task}_loss.png")
        print(f"Saved plots for {task}")


if __name__ == "__main__":
    main()
