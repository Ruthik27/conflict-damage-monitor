"""
make_hold_split.py — Carve a hold (val) split from the xBD train directory.

Groups train scenes by disaster event name, holds out a configurable fraction
of events, and creates hold/ with symlinks so no data is duplicated.

Usage:
    python src/data/make_hold_split.py --paths-cfg configs/paths.yaml
    python src/data/make_hold_split.py --paths-cfg configs/paths.yaml --hold-frac 0.2 --seed 42 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _resolve_xbd_root(paths_cfg: Path) -> Path:
    with paths_cfg.open() as fh:
        cfg = yaml.safe_load(fh)
    return Path(cfg["data"]["raw"]) / "xbd"


def _disaster_name(filename: str) -> str:
    """Extract disaster event name from an image filename.

    e.g. 'hurricane-michael_00000042_pre_disaster.png' → 'hurricane-michael'
    """
    return filename.split("_")[0]


def _make_symlink(src: Path, dst: Path, dry_run: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if not dry_run:
        dst.symlink_to(src)
    logger.debug("  link %s → %s", dst, src)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create xBD hold split via symlinks.")
    parser.add_argument("--paths-cfg", default="configs/paths.yaml", type=Path)
    parser.add_argument("--hold-frac", default=0.2, type=float,
                        help="Fraction of disaster events to hold out (default 0.2)")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without creating symlinks")
    args = parser.parse_args(argv)

    xbd_root = _resolve_xbd_root(args.paths_cfg)
    train_images = xbd_root / "train" / "images"
    train_labels = xbd_root / "train" / "labels"
    hold_images = xbd_root / "hold" / "images"
    hold_labels = xbd_root / "hold" / "labels"

    if not train_images.exists():
        logger.error("train/images not found: %s", train_images)
        return 1

    # Group image stems by disaster event
    event_stems: dict[str, list[str]] = defaultdict(list)
    for f in train_images.iterdir():
        if "_pre_disaster" in f.name:
            event_stems[_disaster_name(f.name)].append(
                f.name.replace("_pre_disaster" + f.suffix, "")
            )

    events = sorted(event_stems)
    n_hold = max(1, round(len(events) * args.hold_frac))

    rng = random.Random(args.seed)
    hold_events = set(rng.sample(events, n_hold))
    train_events = set(events) - hold_events

    logger.info("Total events : %d", len(events))
    logger.info("Hold events  : %s", sorted(hold_events))
    logger.info("Train events : %s", sorted(train_events))

    if args.dry_run:
        logger.info("[dry-run] No symlinks created.")
        return 0

    # Check hold/ doesn't already exist with content
    if hold_images.exists() and any(hold_images.iterdir()):
        logger.error(
            "hold/images already exists and is non-empty. "
            "Delete it first if you want to re-split: %s",
            hold_images,
        )
        return 1

    ext_map: dict[str, str] = {}
    for f in train_images.iterdir():
        if "_pre_disaster" in f.name:
            stem = f.name.replace("_pre_disaster" + f.suffix, "")
            ext_map[stem] = f.suffix  # .png or .tif

    linked_images = linked_labels = 0
    for event in hold_events:
        for stem in event_stems[event]:
            ext = ext_map.get(stem, ".png")
            for suffix in (f"_pre_disaster{ext}", f"_post_disaster{ext}"):
                src = (train_images / f"{stem}{suffix}").resolve()
                dst = hold_images / f"{stem}{suffix}"
                if src.exists():
                    _make_symlink(src, dst, dry_run=False)
                    linked_images += 1

            label_src = (train_labels / f"{stem}_post_disaster.json").resolve()
            label_dst = hold_labels / f"{stem}_post_disaster.json"
            if label_src.exists():
                _make_symlink(label_src, label_dst, dry_run=False)
                linked_labels += 1

    logger.info("Created %d image symlinks and %d label symlinks in hold/", linked_images, linked_labels)
    logger.info("Done. hold/ is ready at: %s", xbd_root / "hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
