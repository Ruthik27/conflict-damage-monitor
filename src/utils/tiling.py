"""
tiling.py — Sliding-window inference for 1024×1024 images.

sliding_window_inference(model, image, tile_size=512, overlap=128)
    image : (1, C, H, W) float32 tensor on the same device as model
    returns (num_classes, H, W) float32 logit tensor

Overlapping tile regions are blended with a 2-D Gaussian weight kernel so
tile-boundary seams are suppressed without hard cuts.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian_kernel(size: int, sigma: float | None = None) -> torch.Tensor:
    """Return a (size, size) 2-D Gaussian weight kernel normalised to [0, 1]."""
    if sigma is None:
        sigma = size / 4.0
    ax = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    kernel_1d = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d / kernel_2d.max()


@torch.no_grad()
def sliding_window_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    tile_size: int = 512,
    overlap: int = 128,
    num_classes: int = 5,
) -> torch.Tensor:
    """Run model on a full image via overlapping tiles, Gaussian-blend results.

    Args:
        model      : callable (B, C, tile, tile) → (B, num_classes, tile, tile)
        image      : (1, C, H, W) float32, already on correct device
        tile_size  : square tile side length in pixels
        overlap    : overlap in pixels between adjacent tiles
        num_classes: number of output channels

    Returns:
        (num_classes, H, W) float32 logit tensor
    """
    _, C, H, W = image.shape
    device = image.device
    stride = tile_size - overlap

    weight_kernel = _gaussian_kernel(tile_size).to(device)          # (tile, tile)
    logit_sum  = torch.zeros(num_classes, H, W, device=device)
    weight_sum = torch.zeros(1, H, W, device=device)

    y_starts = list(range(0, H - tile_size + 1, stride))
    x_starts = list(range(0, W - tile_size + 1, stride))

    # Ensure the last column/row is always covered
    if y_starts[-1] + tile_size < H:
        y_starts.append(H - tile_size)
    if x_starts[-1] + tile_size < W:
        x_starts.append(W - tile_size)

    for y in y_starts:
        for x in x_starts:
            tile = image[:, :, y : y + tile_size, x : x + tile_size]  # (1,C,ts,ts)
            logits = model(tile)                                        # (1,nc,ts,ts)
            logits = logits.squeeze(0)                                  # (nc,ts,ts)

            logit_sum[:, y : y + tile_size, x : x + tile_size]  += logits * weight_kernel
            weight_sum[0, y : y + tile_size, x : x + tile_size] += weight_kernel

    # Avoid divide-by-zero (shouldn't happen with correct stride, but be safe)
    weight_sum = weight_sum.clamp(min=1e-6)
    return logit_sum / weight_sum
