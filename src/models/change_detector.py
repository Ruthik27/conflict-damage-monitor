"""
SiameseChangeDetector — Explicit change detection via shared Siamese encoder.

Architecture:
  - Shared EfficientNet-B4 encoder processes pre and post images independently
  - At each encoder stage: difference features (|post - pre|) are concatenated
    with post features before being passed to the UNet++ decoder
  - Decoder produces 5-class per-pixel damage map

Input : pre  (B, 3, 512, 512) float32
        post (B, 3, 512, 512) float32
Output: (B, 5, 512, 512) logits — 0=bg, 1-4=damage classes

Why Siamese over 6-channel concatenation:
  - Encoder sees each image with correct ImageNet-pretrained receptive fields
  - Explicit difference signal forces the model to focus on CHANGE, not just
    the absolute post-disaster appearance
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torchmetrics import F1Score, JaccardIndex

from src.utils.tiling import sliding_window_inference


class _DiffDecoder(nn.Module):
    """Wrap smp UnetPlusPlus decoder to accept diff-augmented encoder features."""

    def __init__(self, base_model: smp.UnetPlusPlus) -> None:
        super().__init__()
        self.encoder  = base_model.encoder
        self.decoder  = base_model.decoder
        self.head     = base_model.segmentation_head

        # 1×1 convs to project (post_feat + diff_feat) → same channel dim
        # Applied at each encoder stage so decoder sees standard channel counts
        enc_channels = self.encoder.out_channels  # e.g. (3, 32, 48, 80, 192, 320)
        self.diff_projs = nn.ModuleList([
            nn.Conv2d(c * 2, c, kernel_size=1, bias=False)
            for c in enc_channels
        ])

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        pre_feats  = self.encoder(pre)   # list of feature maps per stage
        post_feats = self.encoder(post)

        # Fuse: project(concat(post_feat, |post_feat - pre_feat|)) → post channels
        fused = [
            proj(torch.cat([pf, (pf - rf).abs()], dim=1))
            for proj, pf, rf in zip(self.diff_projs, post_feats, pre_feats)
        ]

        decoder_out = self.decoder(*fused)
        return self.head(decoder_out)


class SiameseChangeDetector(pl.LightningModule):
    """Siamese UNet++ change detector for building damage segmentation.

    Args:
        cfg: Parsed train_change.yaml dict.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.save_hyperparameters(cfg)
        mcfg = cfg["model"]

        base = smp.UnetPlusPlus(
            encoder_name=mcfg["encoder"],
            encoder_weights=mcfg["encoder_weights"],
            in_channels=3,                          # single image per branch
            classes=mcfg["num_classes"],
            decoder_attention_type=mcfg.get("decoder_attention", "scse"),
        )
        self.net = _DiffDecoder(base)

        num_classes = mcfg["num_classes"]
        weights = cfg.get("class_weights")
        weight_tensor = torch.tensor(weights, dtype=torch.float32) if weights else None
        self.ce_loss   = nn.CrossEntropyLoss(weight=weight_tensor)
        self.dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        lcfg = cfg.get("loss", {})
        self.dice_w = lcfg.get("dice_weight", 0.5)
        self.ce_w   = lcfg.get("ce_weight", 0.5)

        self.val_iou = JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=0, average="macro")
        self.val_f1  = F1Score(task="multiclass",     num_classes=num_classes, ignore_index=0, average="macro")

        self._tile_size = cfg.get("data", {}).get("val_tile_size", 512)
        self._overlap   = cfg.get("data", {}).get("val_tile_overlap", 128)
        self._num_cls   = num_classes

    # ── forward (accepts pre+post separately) ────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Accept 6-channel (pre+post) tensor for API compatibility with tiling.py."""
        pre, post = x[:, :3], x[:, 3:]
        return self.net(pre, post)

    # ── steps ─────────────────────────────────────────────────────────────────

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch                    # x: (B,6,512,512)  y: (B,512,512)
        logits = self(x)
        loss = self.dice_w * self.dice_loss(logits, y) + self.ce_w * self.ce_loss(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        x, y = batch                    # x: (1,6,1024,1024)
        logits = sliding_window_inference(
            self, x, tile_size=self._tile_size,
            overlap=self._overlap, num_classes=self._num_cls,
        ).unsqueeze(0)

        loss  = self.dice_w * self.dice_loss(logits, y) + self.ce_w * self.ce_loss(logits, y)
        preds = logits.argmax(dim=1)
        self.val_iou(preds, y)
        self.val_f1(preds, y)
        self.log("val_loss",      loss,          on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log("val_iou_macro", self.val_iou,  on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log("val_f1_macro",  self.val_f1,   on_epoch=True, prog_bar=False, sync_dist=True)

    # ── optimiser ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        ocfg = self.hparams["optimizer"]
        scfg = self.hparams["scheduler"]
        tcfg = self.hparams["trainer"]

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=ocfg["lr"],
            weight_decay=ocfg["weight_decay"], betas=tuple(ocfg["betas"]),
        )
        total, warmup = tcfg["max_epochs"], scfg["warmup_epochs"]
        min_ratio = scfg["min_lr"] / ocfg["lr"]

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return (epoch + 1) / warmup
            p = (epoch - warmup) / max(1, total - warmup)
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))

        sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
