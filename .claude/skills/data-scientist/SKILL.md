---
name: data-scientist
description: >
  ML and data science patterns for the conflict-damage-monitor project. Use this skill
  whenever writing or reviewing model code, training loops, dataset classes, loss functions,
  metrics, or evaluation scripts. Trigger on requests like "write a LightningModule for X",
  "add a loss function", "implement the training loop", "set up wandb logging", "build the
  dataset class", "handle class imbalance", "write the validation step", or any task
  touching PyTorch, PyTorch Lightning, torchmetrics, or model checkpointing. Also trigger
  when the user asks about metric choices, class weighting, or how to structure experiments.
  These patterns apply to both the EfficientNet-B4 classifier and the UNet++ segmentation
  model.
---

# Data Scientist Patterns — conflict-damage-monitor

This project classifies building damage into 4 ordinal classes using satellite imagery.
The class distribution is highly skewed — `destroyed` is rare in xBD — and this shapes
every modelling decision from loss function to evaluation.

## Domain constants

```python
# Always use these — don't redefine them locally
DAMAGE_CLASSES = {0: "no-damage", 1: "minor-damage", 2: "major-damage", 3: "destroyed"}
NUM_CLASSES = 4

# xBD approximate class weights (inverse frequency, normalized)
# destroyed is ~3% of buildings — weight it ~8x no-damage
XBD_CLASS_WEIGHTS = [1.0, 2.5, 4.0, 8.0]  # no / minor / major / destroyed
```

---

## Framework: PyTorch Lightning

Every model is a `LightningModule`; every dataset is wrapped in a `LightningDataModule`.
This keeps training/validation/test logic unified and makes HPC job scripts trivial.

### LightningModule skeleton

```python
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score, JaccardIndex
from pathlib import Path

class DamageClassifier(pl.LightningModule):
    """EfficientNet-B4 building damage classifier."""

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # Build model (segmentation_models_pytorch or timm)
        self.model = ...

        # Class-weighted loss — handles the destroyed-class imbalance
        weights = torch.tensor(XBD_CLASS_WEIGHTS)
        self.loss_fn = nn.CrossEntropyLoss(weight=weights)
        # Alternative: FocalLoss for harder examples
        # self.loss_fn = FocalLoss(gamma=2.0, weight=weights)

        # torchmetrics — always, never manual numpy
        self._init_metrics()

    def _init_metrics(self):
        kwargs = dict(num_classes=NUM_CLASSES, average="macro", task="multiclass")
        for split in ("train", "val", "test"):
            setattr(self, f"{split}_acc",  Accuracy(task="multiclass", num_classes=NUM_CLASSES))
            setattr(self, f"{split}_f1",   F1Score(**kwargs))
            setattr(self, f"{split}_miou", JaccardIndex(**kwargs))

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, split: str):
        imgs, labels = batch
        logits = self(imgs)
        loss = self.loss_fn(logits, labels)
        preds = logits.argmax(dim=1)

        getattr(self, f"{split}_acc")(preds, labels)
        getattr(self, f"{split}_f1")(preds, labels)
        getattr(self, f"{split}_miou")(preds, labels)

        self.log(f"{split}/loss", loss, on_step=(split=="train"), on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val")

    def on_train_epoch_end(self):
        self._log_epoch_metrics("train")

    def _log_epoch_metrics(self, split: str):
        acc  = getattr(self, f"{split}_acc").compute()
        f1   = getattr(self, f"{split}_f1").compute()
        miou = getattr(self, f"{split}_miou").compute()

        self.log(f"{split}/accuracy", acc,  prog_bar=True)
        self.log(f"{split}/f1_macro", f1,   prog_bar=True)
        self.log(f"{split}/mIoU",     miou, prog_bar=True)

        # Log per-class F1 separately so wandb shows individual curves
        f1_per_class = F1Score(
            task="multiclass", num_classes=NUM_CLASSES, average="none"
        ).to(self.device)
        # (recompute from stored preds if needed, or log in validation_step)
        for cls_idx, cls_name in DAMAGE_CLASSES.items():
            self.log(f"{split}/f1_{cls_name}", f1_per_class_scores[cls_idx])

        # Reset for next epoch
        for m in [f"{split}_acc", f"{split}_f1", f"{split}_miou"]:
            getattr(self, m).reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.cfg.training.max_epochs
        )
        return [opt], [scheduler]
```

---

## Metrics: always torchmetrics

Never compute metrics manually with numpy — torchmetrics handles distributed reduction
correctly across GPUs, which matters when running 2×A100.

| Metric | torchmetrics class | Log name |
|---|---|---|
| Overall accuracy | `Accuracy(task="multiclass")` | `val/accuracy` |
| Macro F1 | `F1Score(average="macro")` | `val/f1_macro` |
| Per-class F1 | `F1Score(average="none")` | `val/f1_no-damage` etc. |
| mIoU | `JaccardIndex(average="macro")` | `val/mIoU` |

Log per-class F1 individually so you can track whether `destroyed` F1 is improving
independently of the macro average — this is the number that matters most for the
conflict monitoring use case.

---

## Class imbalance: destroyed is rare

In xBD, `destroyed` is ~3% of buildings. Without intervention, models learn to ignore it.
Three complementary strategies:

**1. Weighted cross-entropy** (default, always on):
```python
weights = torch.tensor(XBD_CLASS_WEIGHTS).to(device)
loss_fn = nn.CrossEntropyLoss(weight=weights)
```

**2. Focal loss** (add when destroyed F1 stalls below 0.4):
```python
# pip install focal-loss-torch or implement:
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits, targets):
        ce = self.ce(logits, targets)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()
```

**3. Oversampling** (in the DataModule):
```python
from torch.utils.data import WeightedRandomSampler

sample_weights = [XBD_CLASS_WEIGHTS[label] for label in all_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights))
# Pass sampler= to DataLoader, drop shuffle=True
```

Start with (1). Add (2) if destroyed F1 < 0.4 after 10 epochs. Add (3) only if still stuck.

---

## Checkpointing

```python
from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_cb = ModelCheckpoint(
    dirpath="/blue/smin.fgcu/rkale.fgcu/cdm/checkpoints",
    filename="{epoch:03d}-{val/f1_macro:.4f}",
    monitor="val/f1_macro",
    mode="max",
    save_top_k=3,
    every_n_epochs=5,   # save every 5 epochs regardless of monitor
    save_last=True,
)
```

Pass `checkpoint_cb` to `Trainer(callbacks=[checkpoint_cb, ...])`.

---

## Trainer setup

```python
import wandb
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(
    project="conflict-damage-monitor",
    log_model=False,   # checkpoints go to /blue, not wandb artifacts
)

trainer = pl.Trainer(
    max_epochs=cfg.training.max_epochs,
    accelerator="gpu",
    devices=2,                    # 2×A100 on HiperGator
    strategy="ddp",               # DistributedDataParallel
    precision="16-mixed",         # mixed precision, always
    logger=wandb_logger,
    callbacks=[checkpoint_cb, lr_monitor],
    log_every_n_steps=50,
    val_check_interval=1.0,       # validate every epoch
)
```

---

## Model choices

| Task | Model | Library |
|---|---|---|
| Classification | EfficientNet-B4 | `timm.create_model("efficientnet_b4", pretrained=True, num_classes=4)` |
| Segmentation | UNet++ + EfficientNet-B4 encoder | `smp.UnetPlusPlus(encoder_name="efficientnet-b4", encoder_weights="imagenet", classes=4)` |

For segmentation, `classes=4` outputs a 4-channel mask; use `argmax` for predictions and
`JaccardIndex` from torchmetrics for mIoU.

---

## wandb logging checklist

Before starting a run, confirm these appear in the wandb run:

- [ ] `val/accuracy` — overall pixel/building accuracy
- [ ] `val/f1_macro` — macro F1 across all 4 classes
- [ ] `val/mIoU` — mean IoU across all 4 classes
- [ ] `val/f1_no-damage`, `val/f1_minor-damage`, `val/f1_major-damage`, `val/f1_destroyed`
- [ ] `val/loss` and `train/loss`
- [ ] Learning rate (add `LearningRateMonitor(logging_interval="epoch")` callback)
- [ ] Hyperparameters logged via `self.save_hyperparameters()` in `__init__`
