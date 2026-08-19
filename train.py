from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from ca_sam.data import ManifestSegmentationDataset, mask_boxes
from ca_sam.model import CASAM, attention_pool, dice_bce_loss, segmentation_metrics

TASKS = ("acdc", "ebhi_seg", "kvasir_seg", "msd_prostate")


def loader(manifest: Path, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(ManifestSegmentationDataset(manifest), batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=True)


@torch.no_grad()
def validate_adapter(model: CASAM, task: str, data: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "accuracy": 0.0, "iou": 0.0, "dice": 0.0}
    for batch in data:
        images, masks = batch["image"].to(device), batch["mask"].to(device)
        logits = model.masks_from_embeddings(model.encode(images), mask_boxes(masks, jitter=0), task)
        totals["loss"] += dice_bce_loss(logits, masks).item() * masks.shape[0]
        for name, value in segmentation_metrics(logits, masks).items():
            totals[name] += value * masks.shape[0]
    return {name: value / len(data.dataset) for name, value in totals.items()}


def train_adapter(model: CASAM, task: str, data: DataLoader, validation_data: DataLoader, epochs: int, accumulation: int, device: torch.device) -> list[dict[str, float]]:
    adapter = model.adapters[task].train()
    optimizer = Adam(adapter.parameters(), lr=1e-4)
    scaler = GradScaler(enabled=device.type == "cuda")
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        totals = {"loss": 0.0, "accuracy": 0.0, "iou": 0.0, "dice": 0.0}
        progress = tqdm(data, desc=f"{task}: epoch {epoch + 1}/{epochs}", leave=False)
        for step, batch in enumerate(progress, 1):
            images, masks = batch["image"].to(device), batch["mask"].to(device)
            boxes = mask_boxes(masks)
            embeddings = model.encode(images)
            with autocast(enabled=device.type == "cuda"):
                logits = model.masks_from_embeddings(embeddings, boxes, task)
                loss = dice_bce_loss(logits, masks) / accumulation
            scaler.scale(loss).backward()
            if step % accumulation == 0 or step == len(data):
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            full_loss = loss.item() * accumulation
            totals["loss"] += full_loss * masks.shape[0]
            for name, value in segmentation_metrics(logits.detach(), masks).items():
                totals[name] += value * masks.shape[0]
            progress.set_postfix(loss=f"{full_loss:.3f}")
        train_stats = {name: value / len(data.dataset) for name, value in totals.items()}
        validation_stats = validate_adapter(model, task, validation_data, device)
        record = {f"train_{name}": value for name, value in train_stats.items()} | {f"val_{name}": value for name, value in validation_stats.items()}
        history.append(record)
        print(
            f"{task} | epoch [{epoch + 1}/{epochs}] : "
            f"train acc {train_stats['accuracy']:.4f}/val acc {validation_stats['accuracy']:.4f} | "
            f"train loss {train_stats['loss']:.4f}/val loss {validation_stats['loss']:.4f} | "
            f"train IoU {train_stats['iou']:.4f}/val IoU {validation_stats['iou']:.4f} | "
            f"train Dice {train_stats['dice']:.4f}/val Dice {validation_stats['dice']:.4f}"
        )
        adapter.train()
    return history


def train_vae(model: CASAM, task: str, data: DataLoader, epochs: int, device: torch.device) -> None:
    vae = model.vaes[task].train()
    optimizer = Adam(vae.parameters(), lr=5e-4)
    for epoch in range(epochs):
        for batch in tqdm(data, desc=f"{task}: router {epoch + 1}/{epochs}"):
            embeddings = model.encode(batch["image"].to(device))
            loss = vae.elbo(attention_pool(embeddings)).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    vae.eval()
    scores = []
    with torch.no_grad():
        for batch in data:
            embeddings = model.encode(batch["image"].to(device))
            scores.extend(vae.elbo(attention_pool(embeddings)).cpu().tolist())
    model.thresholds[task] = float(np.percentile(scores, 97))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Directory containing <task>/train.jsonl")
    parser.add_argument("--checkpoint", type=Path, required=True, help="sam_vit_b_01ec64.pth")
    parser.add_argument("--output", type=Path, default=Path("runs/four_task"))
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--vae-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=6)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    from segment_anything import sam_model_registry
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry["vit_b"](checkpoint=str(args.checkpoint)).to(device)
    model = CASAM(sam, TASKS).to(device)
    args.output.mkdir(parents=True, exist_ok=True)
    state_file = args.output / "ca_sam_four_task.pt"
    completed: list[str] = []
    history: list[dict[str, object]] = []
    if args.resume and state_file.exists():
        state = torch.load(state_file, map_location=device)
        model.load_state_dict(state["model"]); model.thresholds = state["thresholds"]; completed = state["completed"]
        history = state.get("history", [])
    for task in TASKS:
        required_manifests = [args.data / task / name for name in ("train.jsonl", "val.jsonl", "test.jsonl")]
        missing = [str(path) for path in required_manifests if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{task} must provide separate train/val/test manifests. Missing: {', '.join(missing)}")
        if task in completed:
            continue
        task_loader = loader(args.data / task / "train.jsonl", args.batch_size, args.workers, True)
        validation_loader = loader(args.data / task / "val.jsonl", args.batch_size, args.workers, False)
        epoch_history = train_adapter(model, task, task_loader, validation_loader, args.epochs, args.accumulation, device)
        train_vae(model, task, task_loader, args.vae_epochs, device)
        completed.append(task)
        history.append({"task": task, "epochs": epoch_history})
        torch.save(model.adapters[task].state_dict(), args.output / f"adapter_{task}.pt")
        torch.save(model.vaes[task].state_dict(), args.output / f"vae_{task}.pt")
        payload = {"model": model.state_dict(), "thresholds": model.thresholds, "completed": completed, "tasks": TASKS, "history": history}
        torch.save(payload, state_file)
        (args.output / "status.json").write_text(json.dumps({"completed": completed, "thresholds": model.thresholds, "history": history}, indent=2))
        print(f"Saved completed task {task} to {state_file}")


if __name__ == "__main__":
    main()
