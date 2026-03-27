---
name: testing-quality
description: >
  Test-writing discipline for the conflict-damage-monitor project. Use this skill
  immediately after writing any new Python script, module, dataset class, model, or
  pipeline component. Trigger on: "write tests for X", "add tests", "I just wrote Y",
  finishing a new src/ file, or any time new code is created without a corresponding
  test. Also trigger when the user says "sanity check this", "make sure this works",
  or "verify the shapes are right". Tests are written alongside code, not after —
  treat this skill as the mandatory second step of every coding task. If a new file
  was created in src/ and no test exists in tests/, this skill should run.
---

# Testing Quality — conflict-damage-monitor

The reason tests are written immediately (not "later") is that bugs in data pipelines
are silent — wrong tile shapes, label values outside 0–3, or split overlap won't raise
exceptions, they'll just produce bad metrics. A 5-minute sanity check catches these
before a 48-hour training run produces garbage results.

## File placement: mirror src/ in tests/

```
src/data/xbd_dataset.py      → tests/data/test_xbd_dataset.py
src/data/datamodule.py       → tests/data/test_datamodule.py
src/training/train.py        → tests/training/test_train.py
src/models/classifier.py     → tests/models/test_classifier.py
src/preprocessing/tiling.py  → tests/preprocessing/test_tiling.py
```

Create `tests/__init__.py` and `tests/<subdir>/__init__.py` as needed. Never put
test files in `src/` or at the project root.

---

## Shared fixtures (tests/conftest.py)

Put reusable fixtures here so every test file can import them without duplication.
All fixtures use synthetic data — no dependency on `/blue` at all.

```python
# tests/conftest.py
import pytest
import torch
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path

NUM_CLASSES = 4
TILE_SIZE = 512

@pytest.fixture
def cfg():
    """Minimal config object that mirrors configs/train.yaml structure."""
    return OmegaConf.create({
        "data": {
            "root": "/tmp/cdm_test_data",  # never /blue in tests
            "tile_size": TILE_SIZE,
            "num_classes": NUM_CLASSES,
            "batch_size": 2,
        },
        "training": {
            "max_epochs": 2,
            "lr": 1e-4,
            "weight_decay": 1e-5,
        },
        "model": {
            "encoder": "efficientnet-b4",
        },
    })

@pytest.fixture
def tiny_image_batch():
    """Two RGB tiles at the expected spatial resolution."""
    return torch.rand(2, 3, TILE_SIZE, TILE_SIZE)   # (B, C, H, W)

@pytest.fixture
def tiny_label_batch():
    """Integer labels in valid range [0, NUM_CLASSES-1]."""
    return torch.randint(0, NUM_CLASSES, (2,))       # classification
    # For segmentation: torch.randint(0, NUM_CLASSES, (2, TILE_SIZE, TILE_SIZE))

@pytest.fixture
def tiny_mask_batch():
    """Per-pixel labels for segmentation tasks."""
    return torch.randint(0, NUM_CLASSES, (2, TILE_SIZE, TILE_SIZE))
```

---

## Data script tests

Every dataset class and preprocessing script needs these checks.

```python
# tests/data/test_xbd_dataset.py
import pytest
import torch
import numpy as np

def test_tile_shape(tiny_image_batch):
    """Tiles must be exactly 512x512 — downstream models are fixed to this size."""
    img = tiny_image_batch[0]
    assert img.shape == (3, 512, 512), f"Expected (3,512,512), got {img.shape}"

def test_label_range(tiny_label_batch):
    """Labels must be integers in [0, 3] — 4 damage classes only."""
    assert tiny_label_batch.dtype in (torch.int64, torch.long)
    assert tiny_label_batch.min() >= 0
    assert tiny_label_batch.max() <= 3

def test_dataset_length(tmp_path):
    """Dataset __len__ returns a positive integer."""
    from src.data.xbd_dataset import XBDDataset
    # Build synthetic tile dir
    _make_synthetic_tiles(tmp_path, n=10)
    ds = XBDDataset(root=tmp_path)
    assert len(ds) > 0

def test_dataset_getitem(tmp_path):
    """__getitem__ returns (image_tensor, label_tensor) with correct types."""
    from src.data.xbd_dataset import XBDDataset
    _make_synthetic_tiles(tmp_path, n=4)
    ds = XBDDataset(root=tmp_path)
    img, label = ds[0]
    assert isinstance(img, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert img.shape[-2:] == (512, 512)

def test_split_no_overlap(tmp_path):
    """Train, val, and test splits must not share any scene IDs."""
    from src.data.xbd_dataset import XBDDataset
    _make_synthetic_tiles(tmp_path, n=20)
    train_ds = XBDDataset(root=tmp_path, split="train")
    val_ds   = XBDDataset(root=tmp_path, split="val")
    test_ds  = XBDDataset(root=tmp_path, split="test")
    train_ids = set(train_ds.scene_ids)
    val_ids   = set(val_ds.scene_ids)
    test_ids  = set(test_ds.scene_ids)
    assert train_ids.isdisjoint(val_ids),  "Train/val overlap!"
    assert train_ids.isdisjoint(test_ids), "Train/test overlap!"
    assert val_ids.isdisjoint(test_ids),   "Val/test overlap!"

def test_split_sizes_nonzero(tmp_path):
    """All three splits should be non-empty."""
    from src.data.xbd_dataset import XBDDataset
    _make_synthetic_tiles(tmp_path, n=20)
    for split in ("train", "val", "test"):
        ds = XBDDataset(root=tmp_path, split=split)
        assert len(ds) > 0, f"{split} split is empty"

def _make_synthetic_tiles(root, n: int):
    """Helper — writes n fake PNG tiles so dataset tests don't need /blue."""
    import numpy as np
    from PIL import Image
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        img.save(root / "images" / f"tile_{i:04d}.png")
        lbl = np.random.randint(0, 4, (512, 512), dtype=np.uint8)
        Image.fromarray(lbl).save(root / "labels" / f"tile_{i:04d}.png")
```

---

## Model script tests

Smoke tests for LightningModules — CPU-only, 2 steps, tiny batch.

```python
# tests/models/test_classifier.py
import pytest
import torch
import pytorch_lightning as pl

def test_model_forward_shape(cfg, tiny_image_batch):
    """Forward pass returns logits of shape (batch, num_classes)."""
    from src.models.classifier import DamageClassifier
    model = DamageClassifier(cfg)
    model.eval()
    with torch.no_grad():
        logits = model(tiny_image_batch)
    assert logits.shape == (2, 4), f"Expected (2,4), got {logits.shape}"

def test_model_output_finite(cfg, tiny_image_batch):
    """Logits must be finite — NaN/Inf means a broken init or loss."""
    from src.models.classifier import DamageClassifier
    model = DamageClassifier(cfg)
    logits = model(tiny_image_batch)
    assert torch.isfinite(logits).all(), "Model output contains NaN or Inf"

def test_training_smoke(cfg, tiny_image_batch, tiny_label_batch):
    """2-step fast_dev_run completes without error on CPU."""
    from src.models.classifier import DamageClassifier
    from src.data.datamodule import DamageDataModule

    model = DamageClassifier(cfg)
    datamodule = DamageDataModule(cfg)

    trainer = pl.Trainer(
        fast_dev_run=2,      # 2 train + 2 val batches, then stop
        accelerator="cpu",   # never require GPU in tests
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, datamodule=datamodule)
    # If we got here without exception, the training loop is wired correctly

def test_segmentation_output_shape(cfg, tiny_image_batch):
    """UNet++ output mask must match spatial dims of input."""
    from src.models.segmentation import DamageSegmentor
    model = DamageSegmentor(cfg)
    model.eval()
    with torch.no_grad():
        masks = model(tiny_image_batch)
    B, C, H, W = masks.shape
    assert C == 4,   f"Expected 4 class channels, got {C}"
    assert H == 512, f"Expected H=512, got {H}"
    assert W == 512, f"Expected W=512, got {W}"
```

---

## Running tests

```bash
# All tests (login node is fine — no GPU needed)
cd ~/local_project/conflict-damage-monitor
conda activate cdm
pytest tests/ -v

# Specific module
pytest tests/data/ -v

# Fast check (stop on first failure)
pytest tests/ -x -q

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

Add `pytest.ini` or `pyproject.toml` section to set the default test root:
```ini
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

---

## Checklist: after writing any new src/ file

- [ ] Created `tests/<matching_subdir>/test_<filename>.py`
- [ ] Added fixtures to `tests/conftest.py` if new shared setup is needed
- [ ] Data tests: shape, label range, split no-overlap, len > 0
- [ ] Model tests: forward shape, output finite, `fast_dev_run=2` smoke test
- [ ] All tests pass on CPU with `pytest tests/ -x -q`
- [ ] No test touches `/blue` — synthetic data only
