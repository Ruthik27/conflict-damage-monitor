"""Unit tests for src/models/change_detector.py."""

import torch
import yaml
import pytest
from src.models.change_detector import SiameseChangeDetector


@pytest.fixture()
def cfg(tmp_path):
    return {
        "model": {"encoder": "efficientnet-b4", "encoder_weights": None,
                  "in_channels": 3, "num_classes": 5, "decoder_attention": "scse"},
        "class_weights": [0.1, 0.6, 1.4, 1.8, 2.2],
        "loss": {"dice_weight": 0.5, "ce_weight": 0.5},
        "optimizer": {"lr": 5e-5, "weight_decay": 1e-4, "betas": [0.9, 0.999]},
        "scheduler": {"warmup_epochs": 3, "min_lr": 1e-6},
        "trainer": {"max_epochs": 60},
        "data": {"val_tile_size": 512, "val_tile_overlap": 128},
        "checkpoint": {}, "wandb": {},
    }


def test_output_shape_train(cfg):
    m = SiameseChangeDetector(cfg)
    x = torch.randn(2, 6, 512, 512)
    out = m(x)
    assert out.shape == (2, 5, 512, 512)


def test_output_dtype(cfg):
    m = SiameseChangeDetector(cfg)
    x = torch.randn(1, 6, 512, 512)
    out = m(x)
    assert out.dtype == torch.float32


def test_pre_post_split(cfg):
    """Verify that swapping pre/post channels changes the output (model is not symmetric)."""
    m = SiameseChangeDetector(cfg)
    m.eval()
    x = torch.randn(1, 6, 64, 64)
    x_swapped = torch.cat([x[:, 3:], x[:, :3]], dim=1)
    with torch.no_grad():
        out1 = m(x)
        out2 = m(x_swapped)
    assert not torch.allclose(out1, out2), "Model output should differ when pre/post are swapped"


def test_training_step_returns_loss(cfg):
    m = SiameseChangeDetector(cfg)
    x = torch.randn(2, 6, 512, 512)
    y = torch.randint(0, 5, (2, 512, 512))
    loss = m.training_step((x, y), batch_idx=0)
    assert loss.ndim == 0
    assert float(loss) > 0
