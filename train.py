from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from ca_sam.data import ManifestSegmentationDataset, mask_boxes
from ca_sam.model import (
    CASAM,
    attention_pool,
    dice_bce_loss,
    segmentation_metrics,
)

# Native 2D continual-learning task order.
TASKS = ("ebhi_seg", "kvasir_seg", "ddti", "sts_2d")


def loader(
    manifest: Path,
    batch_size: int,
    workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        ManifestSegmentationDataset(manifest),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
    )


@torch.no_grad()
def validate_adapter(
    model: CASAM,
    task: str,
    data: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Validation only: never accesses the held-out test manifest."""
    model.eval()

    totals = {
        "loss": 0.0,
        "accuracy": 0.0,
        "iou": 0.0,
        "dice": 0.0,
    }

    for batch in data:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        # Deterministic ground-truth box prompts for validation.
        boxes = mask_boxes(masks, jitter=0)
        embeddings = model.encode(images)
        logits = model.masks_from_embeddings(embeddings, boxes, task)

        batch_size = masks.shape[0]
        totals["loss"] += dice_bce_loss(logits, masks).item() * batch_size

        for name, value in segmentation_metrics(logits, masks).items():
            totals[name] += value * batch_size

    return {
        name: value / len(data.dataset)
        for name, value in totals.items()
    }


def train_adapter(
    model: CASAM,
    task: str,
    data: DataLoader,
    validation_data: DataLoader,
    epochs: int,
    accumulation: int,
    device: torch.device,
) -> list[dict[str, float]]:
    """Train only the current task's Alignment Layer."""

    # SAM remains frozen and in evaluation mode throughout.
    model.sam.eval()

    adapter = model.adapters[task]
    adapter.train()

    optimizer = Adam(adapter.parameters(), lr=1e-4)
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        adapter.train()
        optimizer.zero_grad(set_to_none=True)

        totals = {
            "loss": 0.0,
            "accuracy": 0.0,
            "iou": 0.0,
            "dice": 0.0,
        }

        progress = tqdm(
            data,
            desc=f"{task}: epoch {epoch + 1}/{epochs}",
            leave=False,
        )

        for step, batch in enumerate(progress, start=1):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            # Randomly jittered target box prompts during training.
            boxes = mask_boxes(masks)

            # Encoder is frozen; hybrid FP16/FP32 behavior is defined in model.py.
            embeddings = model.encode(images)

            # Adapter, mask-decoder gradient path, and loss run in FP32.
            logits = model.masks_from_embeddings(embeddings, boxes, task)
            loss = dice_bce_loss(logits, masks) / accumulation
            full_loss = loss.detach().item() * accumulation

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss: task={task}, "
                    f"epoch={epoch + 1}, step={step}"
                )

            loss.backward()

            if step % accumulation == 0 or step == len(data):
                torch.nn.utils.clip_grad_norm_(
                    adapter.parameters(),
                    max_norm=1.0,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Correctly accumulate train statistics for the epoch report.
            batch_size = masks.shape[0]
            totals["loss"] += full_loss * batch_size

            for name, value in segmentation_metrics(
                logits.detach(),
                masks,
            ).items():
                totals[name] += value * batch_size

            progress.set_postfix(loss=f"{full_loss:.3f}")

        train_stats = {
            name: value / len(data.dataset)
            for name, value in totals.items()
        }

        validation_stats = validate_adapter(
            model,
            task,
            validation_data,
            device,
        )

        record = (
            {f"train_{name}": value for name, value in train_stats.items()}
            | {f"val_{name}": value for name, value in validation_stats.items()}
        )
        history.append(record)

        print(
            f"{task} | epoch [{epoch + 1}/{epochs}] : "
            f"train acc {train_stats['accuracy']:.4f}/"
            f"val acc {validation_stats['accuracy']:.4f} | "
            f"train loss {train_stats['loss']:.4f}/"
            f"val loss {validation_stats['loss']:.4f} | "
            f"train IoU {train_stats['iou']:.4f}/"
            f"val IoU {validation_stats['iou']:.4f} | "
            f"train Dice {train_stats['dice']:.4f}/"
            f"val Dice {validation_stats['dice']:.4f}"
        )

    return history


def train_vae(
    model: CASAM,
    task: str,
    data: DataLoader,
    epochs: int,
    device: torch.device,
) -> None:
    """Train the current task's VAE router after its adapter is trained."""

    model.sam.eval()

    vae = model.vaes[task]
    vae.train()

    optimizer = Adam(vae.parameters(), lr=5e-4)

    for epoch in range(epochs):
        for batch in tqdm(
            data,
            desc=f"{task}: router {epoch + 1}/{epochs}",
            leave=False,
        ):
            images = batch["image"].to(device, non_blocking=True)

            embeddings = model.encode(images)
            features = attention_pool(embeddings)
            loss = vae.elbo(features).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    # p97 threshold calibration, using only the task training split.
    vae.eval()
    scores = []

    with torch.no_grad():
        for batch in data:
            images = batch["image"].to(device, non_blocking=True)
            embeddings = model.encode(images)
            features = attention_pool(embeddings)
            scores.extend(vae.elbo(features).cpu().tolist())

    model.thresholds[task] = float(np.percentile(scores, 97))


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Directory containing one folder per task.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to sam_vit_b_01ec64.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/four_task"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--vae-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=6)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    from segment_anything import sam_model_registry

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    sam = sam_model_registry["vit_b"](
        checkpoint=str(args.checkpoint)
    ).to(device)

    model = CASAM(sam, TASKS).to(device)
    model.sam.eval()

    if not all(not parameter.requires_grad for parameter in model.sam.parameters()):
        raise RuntimeError("SAM is not fully frozen.")

    print(
        "SAM encoder, prompt encoder, and mask decoder are frozen. "
        "Only the active task Alignment Layer and its VAE are trained."
    )

    args.output.mkdir(parents=True, exist_ok=True)
    state_file = args.output / "ca_sam_four_task.pt"

    completed: list[str] = []
    history: list[dict[str, object]] = []

    if args.resume and state_file.exists():
        state = torch.load(state_file, map_location=device)
        model.load_state_dict(state["model"])
        model.thresholds = state["thresholds"]
        completed = state["completed"]
        history = state.get("history", [])

        print(f"Resuming after: {completed}")

    for task in TASKS:
        required = [
            args.data / task / split
            for split in ("train.jsonl", "val.jsonl", "test.jsonl")
        ]

        missing = [str(path) for path in required if not path.exists()]

        if missing:
            raise FileNotFoundError(
                f"{task} requires train/val/test manifests. "
                f"Missing: {', '.join(missing)}"
            )

        if task in completed:
            continue

        print(f"\n{'=' * 70}\nTraining task: {task}\n{'=' * 70}")

        train_data = loader(
            args.data / task / "train.jsonl",
            args.batch_size,
            args.workers,
            shuffle=True,
        )
        val_data = loader(
            args.data / task / "val.jsonl",
            args.batch_size,
            args.workers,
            shuffle=False,
        )

        epoch_history = train_adapter(
            model,
            task,
            train_data,
            val_data,
            args.epochs,
            args.accumulation,
            device,
        )

        train_vae(
            model,
            task,
            train_data,
            args.vae_epochs,
            device,
        )

        completed.append(task)
        history.append({"task": task, "epochs": epoch_history})

        # Individual weights for later testing/inference.
        torch.save(
            model.adapters[task].state_dict(),
            args.output / f"adapter_{task}.pt",
        )
        torch.save(
            model.vaes[task].state_dict(),
            args.output / f"vae_{task}.pt",
        )

        # Complete resumable state after every finished task.
        payload = {
            "model": model.state_dict(),
            "thresholds": model.thresholds,
            "completed": completed,
            "tasks": TASKS,
            "history": history,
        }

        torch.save(payload, state_file)

        (args.output / "status.json").write_text(
            json.dumps(
                {
                    "completed": completed,
                    "thresholds": model.thresholds,
                    "history": history,
                },
                indent=2,
            )
        )

        print(f"Saved completed task '{task}' to {state_file}")


if __name__ == "__main__":
    main()
