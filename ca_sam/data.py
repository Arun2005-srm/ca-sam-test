from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ManifestSegmentationDataset(Dataset):
    """
    Reads JSONL records such as:
    {"image": "images/example.png", "mask": "masks/example.png"}

    Paths are relative to the JSONL manifest file.
    """

    def __init__(
        self,
        manifest: str | Path,
        image_size: int = 1024,
    ) -> None:
        self.manifest = Path(manifest)
        self.root = self.manifest.parent

        self.records: list[dict[str, Any]] = [
            json.loads(line)
            for line in self.manifest.read_text().splitlines()
            if line.strip()
        ]

        if not self.records:
            raise ValueError(f"Manifest contains no records: {self.manifest}")

        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]

        image_path = self.root / record["image"]
        mask_path = self.root / record["mask"]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR,
        )
        mask = mask.resize(
            (self.image_size, self.image_size),
            Image.Resampling.NEAREST,
        )

        image_tensor = torch.from_numpy(
            np.asarray(image).copy()
        ).permute(2, 0, 1).float()

        # Binary foreground mask: 0 = background, 1 = foreground.
        mask_tensor = torch.from_numpy(
            (np.asarray(mask).copy() > 0).astype(np.float32)
        ).unsqueeze(0)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
        }


def mask_boxes(
    masks: torch.Tensor,
    jitter: int = 10,
) -> torch.Tensor:
    """
    Create one valid XYXY prompt box for every binary foreground mask.

    Input:  [B, 1, H, W]
    Output: [B, 1, 4]
    """
    boxes = []
    height, width = masks.shape[-2:]

    for mask in masks[:, 0]:
        ys, xs = torch.where(mask > 0)

        # Fallback for an empty mask. Your prepared EBHI data excludes empty
        # masks, but this keeps the data loader safe for other datasets.
        if xs.numel() == 0:
            boxes.append(
                torch.tensor(
                    [0, 0, width - 1, height - 1],
                    device=mask.device,
                    dtype=torch.float32,
                )
            )
            continue

        x0 = xs.min()
        y0 = ys.min()
        x1 = xs.max()
        y1 = ys.max()

        if jitter > 0:
            noise = torch.randint(
                low=-jitter,
                high=jitter + 1,
                size=(4,),
                device=mask.device,
            )

            x0 = x0 + noise[0]
            y0 = y0 + noise[1]
            x1 = x1 + noise[2]
            y1 = y1 + noise[3]

        # Clamp while guaranteeing x1 > x0 and y1 > y0.
        x0 = x0.clamp(0, width - 2)
        y0 = y0.clamp(0, height - 2)

        x1 = x1.clamp(x0 + 1, width - 1)
        y1 = y1.clamp(y0 + 1, height - 1)

        boxes.append(
            torch.stack([x0, y0, x1, y1]).float()
        )

    return torch.stack(boxes).unsqueeze(1)
