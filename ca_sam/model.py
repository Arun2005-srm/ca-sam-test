from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class CAResBlock(nn.Module):
    """Paper-style residual block with efficient channel attention."""
    def __init__(self, channels: int = 256, eca_kernel: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.eca = nn.Conv1d(1, 1, eca_kernel, padding=eca_kernel // 2, bias=False)
        self.norm = LayerNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = F.relu(self.conv1(x), inplace=True)
        x = self.conv2(x)
        channel_weights = F.adaptive_avg_pool2d(x, 1).squeeze(-1).transpose(1, 2)
        channel_weights = torch.sigmoid(self.eca(channel_weights)).transpose(1, 2).unsqueeze(-1)
        return self.norm(residual + x * channel_weights)


class AlignmentLayer(nn.Module):
    def __init__(self, channels: int = 256, blocks: int = 3) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*(CAResBlock(channels) for _ in range(blocks)))

    def forward(self, embeddings: Tensor) -> Tensor:
        return self.blocks(embeddings)


class FeatureVAE(nn.Module):
    def __init__(self, feature_dim: int = 256, latent_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU())
        self.mu = nn.Linear(feature_dim, latent_dim)
        self.logvar = nn.Linear(feature_dim, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, feature_dim), nn.ReLU(), nn.Linear(feature_dim, feature_dim))

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden = self.encoder(features)
        mu, logvar = self.mu(hidden), self.logvar(hidden).clamp(-12, 12)
        # Use the posterior mean at evaluation time so routing and thresholds are stable.
        z = mu if not self.training else mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decoder(z), mu, logvar

    def elbo(self, features: Tensor, beta: float = 16.5) -> Tensor:
        reconstruction, mu, logvar = self(features)
        recon = (features - reconstruction).pow(2).mean(dim=1)
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).mean(dim=1)
        return recon + beta * kl


def attention_pool(embeddings: Tensor, temperature: float = 1.0) -> Tensor:
    """Parameter-free attention pooling from Equation 5 of the paper."""
    _, channels, _, _ = embeddings.shape
    scores = embeddings.norm(p=2, dim=1, keepdim=True) / (channels * temperature)
    weights = torch.softmax(scores.flatten(2), dim=-1).view_as(scores)
    return (embeddings * weights).sum(dim=(2, 3))


def dice_bce_loss(logits: Tensor, masks: Tensor) -> Tensor:
    masks = F.interpolate(masks.float(), size=logits.shape[-2:], mode="nearest")
    bce = F.binary_cross_entropy_with_logits(logits, masks)
    probs = logits.sigmoid()
    intersection = (probs * masks).sum(dim=(1, 2, 3))
    dice = 1 - (2 * intersection + 1) / (probs.sum((1, 2, 3)) + masks.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


@torch.no_grad()
def segmentation_metrics(logits: Tensor, masks: Tensor) -> dict[str, float]:
    """Binary pixel accuracy, IoU, and Dice for reporting; logits can be low-resolution."""
    masks = F.interpolate(masks.float(), size=logits.shape[-2:], mode="nearest").bool()
    predicted = logits.sigmoid() > 0.5
    intersection = (predicted & masks).sum().item()
    union = (predicted | masks).sum().item()
    predicted_sum, target_sum = predicted.sum().item(), masks.sum().item()
    return {
        "accuracy": (predicted == masks).float().mean().item(),
        "iou": intersection / max(union, 1),
        "dice": 2 * intersection / max(predicted_sum + target_sum, 1),
    }


@dataclass
class Route:
    task: str | None
    accepted: Tensor
    scores: dict[str, Tensor]


class CASAM(nn.Module):
    """Frozen SAM with task adapters and exemplar-free VAE routing."""
    def __init__(self, sam: nn.Module, task_names: Iterable[str]) -> None:
        super().__init__()
        self.sam = sam
        for parameter in self.sam.parameters():
            parameter.requires_grad_(False)
        self.adapters = nn.ModuleDict({name: AlignmentLayer() for name in task_names})
        self.vaes = nn.ModuleDict({name: FeatureVAE() for name in task_names})
        self.thresholds: dict[str, float] = {}

    @torch.no_grad()
    def encode(self, images: Tensor) -> Tensor:
        return self.sam.image_encoder(self.sam.preprocess(images))

    def masks_from_embeddings(self, embeddings: Tensor, boxes: Tensor, task: str | None) -> Tensor:
        if task is not None:
            embeddings = self.adapters[task](embeddings)
        sparse, dense = self.sam.prompt_encoder(points=None, boxes=boxes, masks=None)
        low_res, _ = self.sam.mask_decoder(
            image_embeddings=embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        return low_res

    @torch.no_grad()
    def route(self, embeddings: Tensor) -> Route:
        features = attention_pool(embeddings)
        scores = {name: vae.elbo(features) for name, vae in self.vaes.items()}
        best_name = min(scores, key=lambda name: scores[name].mean().item())
        accepted = scores[best_name] <= self.thresholds.get(best_name, float("inf"))
        return Route(best_name if bool(accepted.all()) else None, accepted, scores)
