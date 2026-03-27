"""Unit tests for src/utils/metrics.py."""

import torch
import pytest
from src.utils.metrics import DamageMetrics, CLASS_NAMES, NUM_CLASSES


def test_perfect_predictions_f1_one():
    m = DamageMetrics()
    preds   = torch.tensor([0, 1, 2, 3, 4])
    targets = torch.tensor([0, 1, 2, 3, 4])
    m.update(preds, targets)
    r = m.compute()
    assert r["macro_f1"] == pytest.approx(1.0, abs=1e-4)
    assert r["macro_iou"] == pytest.approx(1.0, abs=1e-4)


def test_all_wrong_f1_zero():
    m = DamageMetrics()
    # predict class 1 for everything but ground truth is class 2
    preds   = torch.ones(100, dtype=torch.long)
    targets = torch.full((100,), 2, dtype=torch.long)
    m.update(preds, targets)
    r = m.compute()
    # class 1 f1 = 0 (no true positives), class 2 f1 = 0
    assert r["per_class"]["no-damage"]["f1"] == pytest.approx(0.0, abs=1e-4)
    assert r["per_class"]["minor-damage"]["f1"] == pytest.approx(0.0, abs=1e-4)


def test_background_excluded_from_macro():
    """Macro metrics must not include background class (index 0)."""
    m = DamageMetrics()
    # Only background pixels — damage classes unseen
    preds   = torch.zeros(100, dtype=torch.long)
    targets = torch.zeros(100, dtype=torch.long)
    m.update(preds, targets)
    r = m.compute()
    # background F1 = 1.0 but macro should NOT be 1.0
    assert r["per_class"]["background"]["f1"] == pytest.approx(1.0, abs=1e-4)
    # macro is over unseen damage classes → 0
    assert r["macro_f1"] == pytest.approx(0.0, abs=1e-4)


def test_reset_clears_state():
    m = DamageMetrics()
    preds = targets = torch.tensor([1, 2, 3])
    m.update(preds, targets)
    m.reset()
    m.update(torch.tensor([1]), torch.tensor([1]))
    r = m.compute()
    # Only one sample after reset — should not accumulate previous
    assert r["per_class"]["no-damage"]["f1"] == pytest.approx(1.0, abs=1e-4)


def test_summary_contains_all_classes():
    m = DamageMetrics()
    m.update(torch.tensor([0, 1, 2, 3, 4]), torch.tensor([0, 1, 2, 3, 4]))
    summary = m.summary()
    for name in CLASS_NAMES:
        assert name in summary


def test_confusion_matrix_shape():
    m = DamageMetrics()
    m.update(torch.randint(0, 5, (50,)), torch.randint(0, 5, (50,)))
    r = m.compute()
    assert r["confusion_matrix"].shape == (NUM_CLASSES, NUM_CLASSES)
