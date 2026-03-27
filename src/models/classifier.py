"""
EfficientNet-B4 damage classifier — PyTorch Lightning module.

Input : (B, 6, 512, 512) float32  — channels 0-2 pre, 3-5 post
Output: (B, 4) logits             — 0 no-damage … 3 destroyed
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score


class DamageClassifier(pl.LightningModule):
    """EfficientNet-B4 fine-tuned for 4-class building damage classification.

    Args:
        cfg: Parsed train.yaml dict (top-level keys: model, optimizer,
             scheduler, trainer, class_weights).
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.save_hyperparameters(cfg)
        mcfg = cfg["model"]

        self.model = timm.create_model(
            mcfg["name"],
            pretrained=mcfg["pretrained"],
            in_chans=mcfg["in_channels"],
            num_classes=mcfg["num_classes"],
            drop_rate=mcfg.get("drop_rate", 0.3),
        )

        weights = cfg.get("class_weights")
        weight_tensor = torch.tensor(weights, dtype=torch.float32) if weights else None
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        num_classes = mcfg["num_classes"]
        task = "multiclass"
        self.train_acc = Accuracy(task=task, num_classes=num_classes)
        self.val_acc   = Accuracy(task=task, num_classes=num_classes)
        self.val_f1    = F1Score(task=task, num_classes=num_classes, average="macro")

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ── steps ─────────────────────────────────────────────────────────────────

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        self.train_acc(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_acc",  self.train_acc, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc",  self.val_acc,  on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_f1_macro", self.val_f1, on_epoch=True, prog_bar=True, sync_dist=True)

    # ── optimiser + scheduler ─────────────────────────────────────────────────

    def configure_optimizers(self):
        ocfg = self.hparams["optimizer"]
        scfg = self.hparams["scheduler"]
        tcfg = self.hparams["trainer"]

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=ocfg["lr"],
            weight_decay=ocfg["weight_decay"],
            betas=tuple(ocfg["betas"]),
        )

        total_steps = tcfg["max_epochs"]  # steps = epochs when interval="epoch"
        warmup_steps = scfg["warmup_epochs"]

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_steps:
                return float(epoch + 1) / float(warmup_steps)
            progress = (epoch - warmup_steps) / max(1, total_steps - warmup_steps)
            import math
            min_lr_ratio = scfg["min_lr"] / ocfg["lr"]
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
