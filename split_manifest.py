"""Create a deterministic validation manifest from a task's official training manifest.

The original test manifest is never read or modified. Run once per task before training.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, help="Official training manifest, e.g. acdc/train_all.jsonl")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")
    lines = [line for line in args.manifest.read_text().splitlines() if line]
    if len(lines) < 2:
        raise ValueError("Need at least two samples to split")
    indices = list(range(len(lines)))
    random.Random(args.seed).shuffle(indices)
    val_count = max(1, round(len(lines) * args.val_fraction))
    val_indices = set(indices[:val_count])
    train = [line for index, line in enumerate(lines) if index not in val_indices]
    validation = [line for index, line in enumerate(lines) if index in val_indices]
    args.manifest.with_name("train.jsonl").write_text("\n".join(train) + "\n")
    args.manifest.with_name("val.jsonl").write_text("\n".join(validation) + "\n")
    print(f"Wrote {len(train)} train and {len(validation)} validation records. Test data was not accessed.")


if __name__ == "__main__":
    main()
