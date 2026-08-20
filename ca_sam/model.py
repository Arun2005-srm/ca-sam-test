from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)

        return (
            normalized * self.weight[:, None, None]
            + self.bias[:, None, None]
        )


class CAResBlock(nn.Module):
    """Residual convolution block with efficient channel attention."""

    def __init__(self, channels: int = 256, eca_kernel: int = 3) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        # Efficient Channel Attention (ECA).
        self.eca = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=eca_kernel,
            padding=eca_kernel // 2,
            bias=False,
        )

        self.norm = LayerNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        x = F.relu(self.conv1(x), inplace=True)
        x = self.conv2(x)

        channel_weights = F.adaptive_avg_pool2d(x, 1)
        channel_weights = channel_weights.squeeze(-1).transpose(1, 2)
        channel_weights = self.eca(channel_weights)
        channel_weights = torch.sigmoid(channel_weights)
        channel_weights = channel_weights.transpose(1, 2).unsqueeze(-1)

        return self.norm(residual + x * channel_weights)


class AlignmentLayer(nn.Module):
    """Task-specific stack of CAResBlocks placed after SAM's image encoder."""

    def __init__(self, channels: int = 256, blocks: int = 3) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            *(CAResBlock(channels) for _ in range(blocks))
        )

    def forward(self, embeddings: Tensor) -> Tensor:
        return self.blocks(embeddings)


class FeatureVAE(nn.Module):
    """Small task-specific VAE used for ELBO-based task routing."""

    def __init__(
        self,
        feature_dim: int = 256,
        latent_dim: int = 64,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
        )

        self.mu = nn.Linear(feature_dim, latent_dim)
        self.logvar = nn.Linear(feature_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden = self.encoder(features)

        mu = self.mu(hidden)
        logvar = self.logvar(hidden).clamp(-12, 12)

        # Use stochastic sampling only while fitting the VAE.
        if self.training:
            noise = torch.randn_like(mu)
            z = mu + noise * torch.exp(0.5 * logvar)
        else:
            z = mu

        reconstruction = self.decoder(z)

        return reconstruction, mu, logvar

    def elbo(self, features: Tensor, beta: float = 16.5) -> Tensor:
        reconstruction, mu, logvar = self(features)

        reconstruction_loss = (features - reconstruction).pow(2).mean(dim=1)

        kl_divergence = 0.5 * (
            mu.pow(2) + logvar.exp() - 1 - logvar
        ).mean(dim=1)

        return reconstruction_loss + beta * kl_divergence


def attention_pool(
    embeddings: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    """
    Parameter-free attention pooling.

    Input:  [B, C, H, W]
    Output: [B, C]
    """
    _, channels, _, _ = embeddings.shape

    scores = embeddings.norm(p=2, dim=1, keepdim=True)
    scores = scores / (channels * temperature)

    weights = torch.softmax(scores.flatten(2), dim=-1)
    weights = weights.view_as(scores)

    return (embeddings * weights).sum(dim=(2, 3))


def dice_bce_loss(logits: Tensor, masks: Tensor) -> Tensor:
    """Binary BCE + soft Dice loss."""

    masks = F.interpolate(
        masks.float(),
        size=logits.shape[-2:],
        mode="nearest",
    )

    bce = F.binary_cross_entropy_with_logits(logits, masks)

    probabilities = logits.sigmoid()
    intersection = (probabilities * masks).sum(dim=(1, 2, 3))

    dice_loss = 1 - (
        (2 * intersection + 1)
        / (
            probabilities.sum(dim=(1, 2, 3))
            + masks.sum(dim=(1, 2, 3))
            + 1
        )
    )

    return bce + dice_loss.mean()


@torch.no_grad()
def segmentation_metrics(
    logits: Tensor,
    masks: Tensor,
) -> dict[str, float]:
    """Binary pixel accuracy, IoU, and Dice."""

    masks = F.interpolate(
        masks.float(),
        size=logits.shape[-2:],
        mode="nearest",
    ).bool()

    predictions = logits.sigmoid() > 0.5

    intersection = (predictions & masks).sum().item()
    union = (predictions | masks).sum().item()

    predicted_pixels = predictions.sum().item()
    target_pixels = masks.sum().item()

    return {
        "accuracy": (predictions == masks).float().mean().item(),
        "iou": intersection / max(union, 1),
        "dice": 2 * intersection / max(predicted_pixels + target_pixels, 1),
    }


@dataclass
class Route:
    task: str | None
    accepted: Tensor
    scores: dict[str, Tensor]


class CASAM(nn.Module):
    """Frozen SAM with task-specific Alignment Layers and VAE task routing."""

    def __init__(
        self,
        sam: nn.Module,
        task_names: Iterable[str],
    ) -> None:
        super().__init__()

        self.sam = sam

        # SAM remains completely frozen.
        for parameter in self.sam.parameters():
            parameter.requires_grad_(False)

        self.adapters = nn.ModuleDict(
            {name: AlignmentLayer() for name in task_names}
        )

        self.vaes = nn.ModuleDict(
            {name: FeatureVAE() for name in task_names}
        )

        self.thresholds: dict[str, float] = {}

    @torch.no_grad()
    def encode(self, images: Tensor) -> Tensor:
        """
        Frozen SAM encoder in FP16 for T4 efficiency.

        Its output is converted to FP32 before the trainable adapter and
        decoder-gradient path, preventing the earlier FP16 NaN issue.
        """
        with torch.autocast(
            device_type=images.device.type,
            dtype=torch.float16,
            enabled=images.device.type == "cuda",
        ):
            embeddings = self.sam.image_encoder(
                self.sam.preprocess(images)
            )

        return embeddings.float()

    def masks_from_embeddings(
        self,
        embeddings: Tensor,
        boxes: Tensor,
        task: str | None,
    ) -> Tensor:
        """
        Apply selected Alignment Layer, then decode masks.

        SAM's original mask decoder does not natively support a batch of
        independent images with independent box prompts. Decode one
        image-box pair at a time, then concatenate the predictions.
        """
        if task is not None:
            embeddings = self.adapters[task](embeddings)

        predictions = []

        for image_embedding, image_boxes in zip(
            embeddings.split(1, dim=0),
            boxes.split(1, dim=0),
        ):
            sparse_prompts, dense_prompts = self.sam.prompt_encoder(
                points=None,
                boxes=image_boxes,
                masks=None,
            )

            low_resolution_masks, _ = self.sam.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_prompts,
                dense_prompt_embeddings=dense_prompts,
                multimask_output=False,
            )

            predictions.append(low_resolution_masks)

        return torch.cat(predictions, dim=0)

    @torch.no_grad()
    def route(self, embeddings: Tensor) -> Route:
        """
        Select the VAE with the lowest ELBO.

        Return `task=None` when the selected score exceeds its p97
        threshold. In that case inference should use the identity adapter,
        i.e. frozen original SAM.
        """
        features = attention_pool(embeddings)

        scores = {
            name: vae.elbo(features)
            for name, vae in self.vaes.items()
        }

        best_task = min(
            scores,
            key=lambda name: scores[name].mean().item(),
        )

        accepted = scores[best_task] <= self.thresholds.get(
            best_task,
            float("inf"),
        )

        selected_task = best_task if bool(accepted.all()) else None

        return Route(
            task=selected_task,
            accepted=accepted,
            scores=scores,
        )
