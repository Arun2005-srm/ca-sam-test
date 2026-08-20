"""
Create deterministic train/validation manifests from a task's training manifest.

The original test manifest is never read or modified.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "manifest",
        type=Path,
        help="Training manifest, e.g. ebhi_seg/train_all.jsonl",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    args = parser.parse_args()

    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")

    lines = [
        line
        for line in args.manifest.read_text().splitlines()
        if line.strip()
    ]

    if len(lines) < 2:
        raise ValueError("Need at least two records to split.")

    indices = list(range(len(lines)))
    random.Random(args.seed).shuffle(indices)

    val_count = max(1, round(len(lines) * args.val_fraction))
    val_indices = set(indices[:val_count])

    train_lines = [
        line
        for index, line in enumerate(lines)
        if index not in val_indices
    ]

    val_lines = [
        line
        for index, line in enumerate(lines)
        if index in val_indices
    ]

    args.manifest.with_name("train.jsonl").write_text(
        "\n".join(train_lines) + "\n"
    )
    args.manifest.with_name("val.jsonl").write_text(
        "\n".join(val_lines) + "\n"
    )

    print(
        f"Wrote {len(train_lines)} train and "
        f"{len(val_lines)} validation records. "
        "Test data was not accessed."
    )


if __name__ == "__main__":
    main()
