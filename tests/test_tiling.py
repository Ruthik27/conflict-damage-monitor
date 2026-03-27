"""Unit tests for src/utils/tiling.py."""

from __future__ import annotations

import torch
import pytest
from src.utils.tiling import sliding_window_inference, _gaussian_kernel


class MockModel(torch.nn.Module):
    """Returns constant logits (class 2 always wins)."""
    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        out = torch.zeros(B, self.num_classes, H, W)
        out[:, 2, :, :] = 1.0  # class 2 logit = 1, rest = 0
        return out


def test_output_shape_1024():
    model = MockModel()
    image = torch.randn(1, 6, 1024, 1024)
    out = sliding_window_inference(model, image, tile_size=512, overlap=128)
    assert out.shape == (5, 1024, 1024)


def test_no_nan_in_output():
    model = MockModel()
    image = torch.randn(1, 6, 1024, 1024)
    out = sliding_window_inference(model, image, tile_size=512, overlap=128)
    assert not out.isnan().any()


def test_full_pixel_coverage():
    """Every pixel must have a non-zero weight (no unvisited pixels)."""
    model = MockModel()
    image = torch.ones(1, 6, 1024, 1024)
    out = sliding_window_inference(model, image, tile_size=512, overlap=128)
    # With constant model, output should be non-zero everywhere for class 2
    assert (out[2] > 0).all(), "Some pixels were never covered by any tile"


def test_constant_model_argmax():
    """With MockModel, argmax should always be class 2."""
    model = MockModel()
    image = torch.randn(1, 6, 1024, 1024)
    out = sliding_window_inference(model, image)
    preds = out.argmax(dim=0)
    assert (preds == 2).all()


def test_gaussian_kernel_shape_and_range():
    k = _gaussian_kernel(512)
    assert k.shape == (512, 512)
    assert float(k.max()) == pytest.approx(1.0)
    assert float(k.min()) > 0.0


def test_zero_overlap_no_nan():
    model = MockModel()
    image = torch.randn(1, 6, 1024, 1024)
    out = sliding_window_inference(model, image, tile_size=512, overlap=0)
    assert not out.isnan().any()
    assert out.shape == (5, 1024, 1024)
