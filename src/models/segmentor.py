"""
DamageSegmentor — UNet++ segmentation LightningModule.

Input : (B, 6, 512, 512) float32   — pre+post concatenated tile
Output: (B, 5, 512, 512) logits    — 0=bg, 1-4=damage classes
Val   : full 1024×1024 via sliding_window_inference
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torchmetrics import F1Score, JaccardIndex

from src.utils.tiling import sliding_window_inference


class DamageSegmentor(pl.LightningModule):
    """UNet++ with EfficientNet-B4 encoder for pixel-level damage segmentation.

    Args:
        cfg: Parsed train_seg.yaml dict.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.save_hyperparameters(cfg)
        mcfg = cfg["model"]

        self.model = smp.UnetPlusPlus(
            encoder_name=mcfg["encoder"],
            encoder_weights=mcfg["encoder_weights"],
            in_channels=mcfg["in_channels"],
            classes=mcfg["num_classes"],
            decoder_attention_type=mcfg.get("decoder_attention", "scse"),
        )

        num_classes = mcfg["num_classes"]
        weights = cfg.get("class_weights")
        weight_tensor = torch.tensor(weights, dtype=torch.float32) if weights else None

        self.ce_loss   = nn.CrossEntropyLoss(weight=weight_tensor)
        self.dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        lcfg = cfg.get("loss", {})
        self.dice_w = lcfg.get("dice_weight", 0.5)
        self.ce_w   = lcfg.get("ce_weight", 0.5)

        # Background (class 0) excluded from metrics — matches xBD evaluation standard
        self.val_iou = JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=0, average="macro")
        self.val_f1  = F1Score(task="multiclass", num_classes=num_classes, ignore_index=0, average="macro")

        self._tile_size = cfg.get("data", {}).get("val_tile_size", 512)
        self._overlap   = cfg.get("data", {}).get("val_tile_overlap", 128)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ── steps ─────────────────────────────────────────────────────────────────

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch                        # x: (B,6,512,512)  y: (B,512,512)
        logits = self(x)                    # (B,5,512,512)
        loss = self.dice_w * self.dice_loss(logits, y) + self.ce_w * self.ce_loss(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        x, y = batch                        # x: (1,6,1024,1024)  y: (1,1024,1024)
        logits = sliding_window_inference(
            self, x,
            tile_size=self._tile_size,
            overlap=self._overlap,
            num_classes=self.hparams["model"]["num_classes"],
        )                                   # (5,1024,1024)
        logits = logits.unsqueeze(0)        # (1,5,1024,1024) for loss + metrics

        loss = self.dice_w * self.dice_loss(logits, y) + self.ce_w * self.ce_loss(logits, y)
        preds = logits.argmax(dim=1)        # (1,1024,1024)

        self.val_iou(preds, y)
        self.val_f1(preds, y)
        self.log("val_loss",     loss,         on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log("val_iou_macro", self.val_iou, on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log("val_f1_macro",  self.val_f1,  on_epoch=True, prog_bar=False, sync_dist=True)

    # ── optimiser + scheduler ─────────────────────────────────────────────────

    def configure_optimizers(self):
        import math
        ocfg = self.hparams["optimizer"]
        scfg = self.hparams["scheduler"]
        tcfg = self.hparams["trainer"]

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=ocfg["lr"],
            weight_decay=ocfg["weight_decay"],
            betas=tuple(ocfg["betas"]),
        )

        total_steps  = tcfg["max_epochs"]
        warmup_steps = scfg["warmup_epochs"]
        min_lr_ratio = scfg["min_lr"] / ocfg["lr"]

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_steps:
                return float(epoch + 1) / float(warmup_steps)
            progress = (epoch - warmup_steps) / max(1, total_steps - warmup_steps)
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
