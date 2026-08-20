"""Sequential held-out test evaluation and qualitative visualizations for CA-SAM."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ca_sam.data import ManifestSegmentationDataset, mask_boxes
from ca_sam.model import CASAM


def metrics(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float, float]:
    """Return pixel accuracy, image IoU, Dice, and global intersection/union components."""
    prediction = prediction.bool()
    target = target.bool()
    intersection = (prediction & target).sum().item()
    union = (prediction | target).sum().item()
    predicted_sum = prediction.sum().item()
    target_sum = target.sum().item()
    accuracy = (prediction == target).float().mean().item()
    iou = intersection / max(union, 1)
    dice = 2 * intersection / max(predicted_sum + target_sum, 1)
    return accuracy, iou, dice, intersection, union


def save_panel(image: torch.Tensor, prediction: torch.Tensor, target: torch.Tensor, output: Path, task: str, index: int) -> None:
    image_np = image.permute(1, 2, 0).cpu().numpy().clip(0, 255).astype("uint8")
    prediction_np = prediction.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()
    _, iou, dice, _, _ = metrics(prediction, target)

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_np)
    axes[0].set_title("Original image")
    axes[1].imshow(prediction_np, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Prediction\nIoU={iou:.3f}, Dice={dice:.3f}")
    axes[2].imshow(target_np, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Ground truth")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(f"{task} test sample {index + 1}")
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


@torch.no_grad()
def evaluate_task(model: CASAM, task: str, manifest: Path, device: torch.device, output: Path, samples: int) -> dict[str, float]:
    data = DataLoader(ManifestSegmentationDataset(manifest), batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    task_output = output / task
    task_output.mkdir(parents=True, exist_ok=True)

    total_accuracy = total_iou = total_dice = 0.0
    global_intersection = global_union = 0.0
    routed_correctly = fallback_count = 0

    model.eval()
    for index, batch in enumerate(tqdm(data, desc=f"Testing {task}")):
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        embeddings = model.encode(image)
        route = model.route(embeddings)
        logits = model.masks_from_embeddings(embeddings, mask_boxes(target, jitter=0), route.task)
        prediction = F.interpolate(logits.sigmoid(), size=target.shape[-2:], mode="bilinear", align_corners=False) > 0.5

        accuracy, iou, dice, intersection, union = metrics(prediction, target)
        total_accuracy += accuracy
        total_iou += iou
        total_dice += dice
        global_intersection += intersection
        global_union += union
        routed_correctly += int(route.task == task)
        fallback_count += int(route.task is None)

        if index < samples:
            save_panel(image[0], prediction[0], target[0], task_output / f"sample_{index + 1}.png", task, index)

    count = len(data.dataset)
    return {
        "task": task,
        "samples": count,
        "pixel_accuracy": total_accuracy / count,
        "mIoU": total_iou / count,
        "Dice": total_dice / count,
        "global_IoU": global_intersection / max(global_union, 1),
        "router_accuracy": routed_correctly / count,
        "fallback_rate": fallback_count / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Dataset directory containing task folders")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Pretrained SAM ViT-B weight file")
    parser.add_argument("--run", type=Path, required=True, help="Training output directory containing ca_sam_four_task.pt")
    parser.add_argument("--output", type=Path, default=None, help="Optional test result directory")
    parser.add_argument("--samples", type=int, default=5, help="Visualization count per task")
    args = parser.parse_args()

    state = torch.load(args.run / "ca_sam_four_task.pt", map_location="cpu", weights_only=False)
    tasks = tuple(state["tasks"])
    if state["completed"] != list(tasks):
        raise RuntimeError(f"Training is incomplete. Completed tasks: {state['completed']}")

    from segment_anything import sam_model_registry
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry["vit_b"](checkpoint=str(args.checkpoint)).to(device)
    model = CASAM(sam, tasks).to(device)
    model.load_state_dict(state["model"])
    model.thresholds = state["thresholds"]
    model.eval()

    output = args.output or args.run / "test_results"
    output.mkdir(parents=True, exist_ok=True)

    results = []
    for task in tasks:
        manifest = args.data / task / "test.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(f"Test manifest not found: {manifest}")
        results.append(evaluate_task(model, task, manifest, device, output, args.samples))

    with (output / "metrics.json").open("w") as file:
        json.dump(results, file, indent=2)
    with (output / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print("\nSequential held-out test results")
    for result in results:
        print(
            f"{result['task']}: mIoU={result['mIoU']:.4f}, "
            f"global IoU={result['global_IoU']:.4f}, Dice={result['Dice']:.4f}, "
            f"accuracy={result['pixel_accuracy']:.4f}, "
            f"router accuracy={result['router_accuracy']:.4f}"
        )
    print(f"\nSaved metrics and sample panels to: {output}")


if __name__ == "__main__":
    main()
