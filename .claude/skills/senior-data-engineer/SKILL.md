---
name: senior-data-engineer
description: >
  Senior-level Python data engineering standards for the conflict-damage-monitor project.
  Use this skill whenever writing any Python code that touches files, data pipelines,
  dataset classes, preprocessing scripts, loaders, or I/O utilities. Trigger on requests
  like "write a data loader", "create a preprocessing script", "build a pipeline for X",
  "add a dataset class", "write a utility to read/write files", or any time new Python
  source files are being created. Also trigger when reviewing or refactoring existing code
  for quality issues — hardcoded paths, print statements, missing type hints, os.path usage.
  These standards apply to every .py file in this project, not just data code.
---

# Senior Data Engineer Standards — conflict-damage-monitor

These conventions exist for a concrete reason: this project runs on HPC with multiple
people, shared `/blue` storage, and jobs that run overnight without supervision. Sloppy
I/O paths break other users' jobs; print statements disappear in SLURM logs; hardcoded
values make configs useless. The patterns below are the minimum bar for production-quality
research code.

---

## Paths: pathlib only

Use `pathlib.Path` everywhere. It's safer, cross-platform, and composes cleanly.

```python
# Good
from pathlib import Path
data_root = Path(os.environ["DATA_ROOT"])          # from env var
output_dir = data_root / "processed" / "xbd"      # / operator, never os.path.join
tile_path = output_dir / f"{scene_id}_pre.tif"

# Bad — never do these
import os
path = "/blue/smin.fgcu/rkale.fgcu/cdm/data"      # hardcoded
path = os.path.join(data_root, "processed")        # os.path
path = str(data_root) + "/processed"               # string concat
```

Always call `output_dir.mkdir(parents=True, exist_ok=True)` before writing to a new dir.

---

## Data root: env var, never hardcoded

The `/blue` path is an environment detail, not a code detail. Load it once at the top of
each module that needs it.

```python
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/blue/smin.fgcu/rkale.fgcu/cdm/data"))
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "/blue/smin.fgcu/rkale.fgcu/cdm/checkpoints"))
LOG_DIR = Path(os.environ.get("LOG_DIR", "/blue/smin.fgcu/rkale.fgcu/cdm/logs"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/blue/smin.fgcu/rkale.fgcu/cdm/outputs"))
```

In `.sbatch` scripts, export these before calling python:
```bash
export DATA_ROOT=/blue/smin.fgcu/rkale.fgcu/cdm/data
export CHECKPOINT_DIR=/blue/smin.fgcu/rkale.fgcu/cdm/checkpoints
```

For scripts driven by `configs/train.yaml`, load paths from the config object instead:
```python
from omegaconf import DictConfig
data_root = Path(cfg.data.root)
```

---

## Logging: stdlib logging, never print

Every module gets its own logger. This makes SLURM log files parseable and lets you filter
by severity.

```python
import logging

logger = logging.getLogger(__name__)  # one line per module, at the top

# In CLI scripts, configure the root logger in main():
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("Starting preprocessing job")
```

Use levels meaningfully: `DEBUG` for per-item progress, `INFO` for stage transitions,
`WARNING` for skipped/missing data, `ERROR` for recoverable failures, `CRITICAL` + raise
for unrecoverable ones. Never use `print()` — it bypasses log routing and gets lost in
SLURM output.

---

## Config: OmegaConf / PyYAML, no magic numbers

Load `configs/train.yaml` at the entry point and pass the config object down. Don't read
YAML in a dozen places.

```python
# CLI entry point
import argparse
from omegaconf import OmegaConf

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train.yaml")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    # pass cfg to everything downstream
```

No magic numbers in source. `tile_size = 512` in a function body is a code smell —
it belongs in `configs/train.yaml` under `data.tile_size`.

---

## Code style

**Type hints on all functions** — parameters and return type:
```python
def load_tile(path: Path, bands: list[int]) -> np.ndarray:
    ...
```

**Docstrings on public classes and functions** — one-line summary is enough for simple
functions; add Args/Returns for anything non-obvious:
```python
def tile_image(src_path: Path, tile_size: int, output_dir: Path) -> list[Path]:
    """Slice a GeoTIFF into non-overlapping square tiles.

    Args:
        src_path: Path to the source raster.
        tile_size: Side length in pixels.
        output_dir: Directory to write tiles into (created if absent).

    Returns:
        List of paths to the written tile files.
    """
```

**argparse for all CLI scripts** — no `sys.argv` indexing, no bare `if __name__ == "__main__"` without argument parsing.

---

## Data pipeline patterns

**Context managers for file I/O** — always, so handles close even on exception:
```python
with rasterio.open(tile_path) as src:
    data = src.read()
```

**Generators for large datasets** — avoid loading everything into RAM:
```python
def iter_scenes(data_root: Path) -> Generator[Path, None, None]:
    """Yield pre/post scene pairs without loading them."""
    for pre in sorted((data_root / "pre").glob("*.tif")):
        post = data_root / "post" / pre.name
        if post.exists():
            yield pre, post
        else:
            logger.warning("Missing post image for %s, skipping", pre.name)
```

**Chunked processing for memory efficiency** — don't read 850k rows at once:
```python
import pandas as pd

for chunk in pd.read_csv(label_file, chunksize=10_000):
    process(chunk)
```

**Fail loudly on bad data** — raise with context rather than silently skipping:
```python
if not tile_path.exists():
    raise FileNotFoundError(f"Expected tile at {tile_path}; check DATA_ROOT={DATA_ROOT}")
```

---

## Quick checklist

Before submitting any Python file for review, confirm:

- [ ] No `import os.path` or `os.path.join` — only `pathlib.Path`
- [ ] No hardcoded `/blue/...` strings — env vars or config only
- [ ] No `print()` — `logger.info/debug/warning/error` instead
- [ ] All functions have type hints and public ones have docstrings
- [ ] CLI script uses `argparse` with `--config` pointing to `configs/train.yaml`
- [ ] File I/O uses `with` blocks
- [ ] Large iterations use generators or chunked reads
