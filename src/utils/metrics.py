"""
metrics.py — Per-class F1, IoU, and confusion matrix for damage evaluation.

CLASS_NAMES maps the 5 segmentation classes (0-4).
Background (class 0) is excluded from macro averages to match xBD standard.

Usage:
    from src.utils.metrics import DamageMetrics
    m = DamageMetrics()
    m.update(preds, targets)   # torch tensors, any shape, values 0-4
    print(m.summary())
    m.reset()
"""

from __future__ import annotations

import torch
from torchmetrics import ConfusionMatrix, F1Score, JaccardIndex

CLASS_NAMES = ["background", "no-damage", "minor-damage", "major-damage", "destroyed"]
NUM_CLASSES = 5
DAMAGE_CLASSES = list(range(1, NUM_CLASSES))  # exclude background for macro metrics


class DamageMetrics:
    """Accumulates predictions and computes per-class + macro metrics.

    Args:
        device: torch device for metric tensors.
    """

    def __init__(self, device: str | torch.device = "cpu") -> None:
        kw = dict(task="multiclass", num_classes=NUM_CLASSES)
        self._f1_per  = F1Score(**kw, average="none").to(device)
        self._iou_per = JaccardIndex(**kw, average="none").to(device)
        self._cm      = ConfusionMatrix(**kw).to(device)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate a batch. preds and targets are flat or any-shape int tensors."""
        p = preds.reshape(-1)
        t = targets.reshape(-1)
        self._f1_per.update(p, t)
        self._iou_per.update(p, t)
        self._cm.update(p, t)

    def compute(self) -> dict:
        """Return dict with per-class and macro-averaged metrics."""
        f1_all  = self._f1_per.compute().cpu()   # (5,)
        iou_all = self._iou_per.compute().cpu()  # (5,)
        cm      = self._cm.compute().cpu()       # (5,5)

        per_class = {
            CLASS_NAMES[i]: {"f1": float(f1_all[i]), "iou": float(iou_all[i])}
            for i in range(NUM_CLASSES)
        }
        # Macro over damage classes only (exclude background)
        macro_f1  = float(f1_all[DAMAGE_CLASSES].mean())
        macro_iou = float(iou_all[DAMAGE_CLASSES].mean())

        return {
            "per_class": per_class,
            "macro_f1":  macro_f1,
            "macro_iou": macro_iou,
            "confusion_matrix": cm,
        }

    def reset(self) -> None:
        self._f1_per.reset()
        self._iou_per.reset()
        self._cm.reset()

    def summary(self) -> str:
        """Return a human-readable evaluation table string."""
        results = self.compute()
        lines = [
            "",
            f"{'Class':<16} {'F1':>8} {'IoU':>8}",
            "-" * 34,
        ]
        for cls_name, vals in results["per_class"].items():
            marker = " *" if cls_name == "background" else ""
            lines.append(f"{cls_name:<16} {vals['f1']:>8.4f} {vals['iou']:>8.4f}{marker}")
        lines += [
            "-" * 34,
            f"{'Macro (no bg)':<16} {results['macro_f1']:>8.4f} {results['macro_iou']:>8.4f}",
            "",
            "* background excluded from macro averages",
        ]
        return "\n".join(lines)
