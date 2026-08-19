from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ManifestSegmentationDataset(Dataset):
    """Reads JSONL records: {"image": "...", "mask": "..."}, paths relative to manifest."""
    def __init__(self, manifest: str | Path, image_size: int = 1024) -> None:
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        self.records: list[dict[str, Any]] = [json.loads(line) for line in self.manifest.read_text().splitlines() if line]
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        image = Image.open(self.root / record["image"]).convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = Image.open(self.root / record["mask"]).convert("L").resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        image_tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy((np.asarray(mask).copy() > 0).astype(np.float32))[None]
        return {"image": image_tensor, "mask": mask_tensor}


def mask_boxes(masks: torch.Tensor, jitter: int = 10) -> torch.Tensor:
    """Create one valid xyxy box per foreground mask in 1024-pixel coordinates."""
    boxes = []
    height, width = masks.shape[-2:]
    for mask in masks[:, 0]:
        ys, xs = torch.where(mask > 0)
        if len(xs) == 0:
            boxes.append(torch.tensor([0, 0, width - 1, height - 1], device=mask.device))
            continue
        box = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()], device=mask.device)
        noise = torch.randint(-jitter, jitter + 1, (4,), device=mask.device)
        box = box + noise
        box[[0, 2]] = box[[0, 2]].clamp(0, width - 1)
        box[[1, 3]] = box[[1, 3]].clamp(0, height - 1)
        box[2] = torch.maximum(box[2], box[0] + 1)
        box[3] = torch.maximum(box[3], box[1] + 1)
        boxes.append(box)
    return torch.stack(boxes).float()[:, None, :]
