"""
rasterize_xbd_labels.py — Convert xBD polygon JSON labels → uint8 PNG target masks.

Reads  : {split_dir}/labels/{stem}_post_disaster.json
Writes : {split_dir}/targets/{stem}_post_disaster_target.png

Class mapping (matches XBDSegDataset):
    0  background / un-classified
    1  no-damage
    2  minor-damage
    3  major-damage
    4  destroyed

Usage:
    python src/data/rasterize_xbd_labels.py --split hold
    python src/data/rasterize_xbd_labels.py --split hold test --paths-cfg configs/paths.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw
from shapely import wkt as shapely_wkt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SUBTYPE_TO_CLASS: dict[str, int] = {
    "no-damage":     1,
    "minor-damage":  2,
    "major-damage":  3,
    "destroyed":     4,
    "un-classified": 0,
}


def _rasterize_one(json_path: Path, targets_dir: Path, img_size: int = 1024) -> None:
    """Rasterize a single post-disaster JSON label file into a target PNG."""
    stem = json_path.stem.replace("_post_disaster", "")
    out_path = targets_dir / f"{stem}_post_disaster_target.png"
    if out_path.exists():
        return  # already done

    with json_path.open() as fh:
        data = json.load(fh)

    # Pull image size from metadata if present
    meta = data.get("metadata", {})
    h = int(meta.get("height", img_size))
    w = int(meta.get("width",  img_size))

    mask = np.zeros((h, w), dtype=np.uint8)

    for feat in data.get("features", {}).get("xy", []):
        subtype = feat["properties"].get("subtype", "un-classified")
        cls = SUBTYPE_TO_CLASS.get(subtype, 0)
        if cls == 0:
            continue  # skip background polygons

        geom = shapely_wkt.loads(feat["wkt"])
        # shapely exterior coords → list of (x, y) pixel tuples
        coords = list(geom.exterior.coords)

        # Draw filled polygon onto a temporary single-channel image
        poly_img = Image.new("L", (w, h), 0)
        ImageDraw.Draw(poly_img).polygon(coords, fill=cls)
        poly_arr = np.array(poly_img)

        # Later (higher-damage) classes overwrite earlier ones
        mask = np.where(poly_arr > 0, poly_arr, mask)

    Image.fromarray(mask, mode="L").save(out_path)


def rasterize_split(split_dir: Path) -> None:
    labels_dir  = split_dir / "labels"
    targets_dir = split_dir / "targets"

    if not labels_dir.exists():
        logger.error("labels/ not found: %s", labels_dir)
        return

    targets_dir.mkdir(exist_ok=True)

    post_jsons = sorted(labels_dir.glob("*_post_disaster.json"))
    logger.info("Split %s: %d post-disaster JSON files", split_dir.name, len(post_jsons))

    done = 0
    for json_path in post_jsons:
        try:
            _rasterize_one(json_path, targets_dir)
            done += 1
        except Exception as exc:
            logger.warning("Failed %s: %s", json_path.name, exc)

    logger.info("Done: %d / %d masks written to %s", done, len(post_jsons), targets_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rasterize xBD JSON labels to PNG masks.")
    parser.add_argument("--split",     nargs="+", default=["hold"],
                        help="Subdirectory names under xbd_root (default: hold)")
    parser.add_argument("--paths-cfg", default="configs/paths.yaml", type=Path)
    args = parser.parse_args()

    with args.paths_cfg.open() as fh:
        cfg = yaml.safe_load(fh)
    xbd_root = Path(cfg["data"]["raw"]) / "xbd"

    for split_name in args.split:
        split_dir = xbd_root / split_name
        if not split_dir.exists():
            logger.warning("Split dir not found, skipping: %s", split_dir)
            continue
        rasterize_split(split_dir)


if __name__ == "__main__":
    main()
