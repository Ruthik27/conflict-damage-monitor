"""
Download the xBD building damage dataset from xview2.org.

Usage:
    python src/data/download_xbd.py --token <YOUR_TOKEN> [--splits train tier3 hold test]

The script:
  1. Downloads each requested split archive from the xView2 CDN.
  2. Streams to disk in chunks (large files — ~50 GB total).
  3. Extracts each archive in-place under the raw xBD data directory.
  4. Verifies the expected top-level sub-directories are present post-extraction.

Registration and token: https://xview2.org/dataset

Output directory layout (from configs/paths.yaml → data.raw/xbd/):
    xbd/
        train/
            images/
            labels/
        tier3/
            images/
            labels/
        hold/
            images/
            labels/
        test/
            images/
            labels/
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tarfile
from pathlib import Path

import requests
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── xView2 download endpoints ─────────────────────────────────────────────────
# Archive filenames follow the pattern: {split}_images_labels_targets.tar.gz
# The token is appended as a query parameter.
XBD_BASE_URL = "https://download.xview2.org"

SPLIT_ARCHIVES: dict[str, str] = {
    "train": "train_images_labels_targets.tar.gz",
    "tier3": "tier3_images_labels_targets.tar.gz",
    "hold":  "hold_images_labels_targets.tar.gz",
    "test":  "test_images_labels_targets.tar.gz",
}

CHUNK_SIZE = 1024 * 1024 * 8  # 8 MiB streaming chunks
MIN_FREE_GB = 80.0             # warn if less than this is free before download


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_xbd_root(paths_cfg: Path) -> Path:
    with paths_cfg.open() as fh:
        cfg = yaml.safe_load(fh)
    return Path(cfg["data"]["raw"]) / "xbd"


def _check_free_space(path: Path, min_gb: float) -> None:
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < min_gb:
        logger.warning(
            "Low disk space on %s — %.1f GB free, %.1f GB recommended",
            path, free_gb, min_gb,
        )
    else:
        logger.info("Disk space OK: %.1f GB free on %s", free_gb, path)


def _archive_already_extracted(xbd_root: Path, split: str) -> bool:
    """Return True if the split directory already looks fully extracted."""
    split_dir = xbd_root / split
    return (split_dir / "images").exists() and (split_dir / "labels").exists()


def _download_archive(url: str, dest: Path) -> None:
    """Stream a remote archive to *dest*, printing a simple progress counter."""
    logger.info("Downloading  %s", url)
    logger.info("         →   %s", dest)

    response = requests.get(url, stream=True, timeout=60)
    if response.status_code == 401:
        logger.error("Authentication failed — check your --token value.")
        sys.exit(1)
    if response.status_code == 404:
        logger.error("Archive not found at %s — the URL or token may be wrong.", url)
        sys.exit(1)
    response.raise_for_status()

    total_bytes = int(response.headers.get("Content-Length", 0))
    downloaded = 0

    with dest.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if not chunk:
                continue
            fh.write(chunk)
            downloaded += len(chunk)
            if total_bytes:
                pct = downloaded / total_bytes * 100
                logger.info("  %.1f %%  (%d / %d MB)", pct, downloaded >> 20, total_bytes >> 20)

    logger.info("Download complete: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)


def _extract_archive(archive: Path, dest_dir: Path) -> None:
    """Extract a .tar.gz archive into *dest_dir*."""
    logger.info("Extracting  %s  →  %s", archive.name, dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(path=dest_dir)
    logger.info("Extraction complete: %s", archive.name)


def _verify_split(xbd_root: Path, split: str) -> bool:
    """Check that the expected sub-directories exist after extraction."""
    split_dir = xbd_root / split
    ok = (split_dir / "images").exists() and (split_dir / "labels").exists()
    if ok:
        n_images = len(list((split_dir / "images").glob("*.tif")))
        logger.info("Verified %s: %d .tif files in images/", split, n_images)
    else:
        logger.error(
            "Verification FAILED for %s — expected images/ and labels/ under %s",
            split, split_dir,
        )
    return ok


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the xBD dataset from xview2.org.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--token",
        required=True,
        help="xView2 download token (register at https://xview2.org/dataset).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLIT_ARCHIVES.keys()),
        choices=list(SPLIT_ARCHIVES.keys()),
        help="Which dataset splits to download.",
    )
    parser.add_argument(
        "--paths-cfg",
        default="configs/paths.yaml",
        help="Path to the project paths config.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        default=False,
        help="Keep .tar.gz files on disk after extraction (default: delete).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip download + extraction if the split directory already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    paths_cfg = Path(args.paths_cfg)
    if not paths_cfg.exists():
        logger.error("paths config not found: %s", paths_cfg)
        sys.exit(1)

    xbd_root = _load_xbd_root(paths_cfg)
    xbd_root.mkdir(parents=True, exist_ok=True)
    logger.info("xBD root: %s", xbd_root)

    _check_free_space(xbd_root, MIN_FREE_GB)

    failed: list[str] = []

    for split in args.splits:
        logger.info("=" * 60)
        logger.info("Processing split: %s", split)

        if args.skip_existing and _archive_already_extracted(xbd_root, split):
            logger.info("Split %s already extracted — skipping.", split)
            continue

        archive_name = SPLIT_ARCHIVES[split]
        url = f"{XBD_BASE_URL}/{archive_name}?token={args.token}"
        archive_path = xbd_root / archive_name

        try:
            if not archive_path.exists():
                _download_archive(url, archive_path)
            else:
                logger.info("Archive already on disk, skipping download: %s", archive_path)

            _extract_archive(archive_path, xbd_root)

            if not args.keep_archives:
                archive_path.unlink()
                logger.info("Removed archive: %s", archive_path)

            if not _verify_split(xbd_root, split):
                failed.append(split)

        except requests.RequestException as exc:
            logger.error("Network error for split %s: %s", split, exc)
            failed.append(split)
        except tarfile.TarError as exc:
            logger.error("Extraction error for split %s: %s", split, exc)
            failed.append(split)

    logger.info("=" * 60)
    if failed:
        logger.error("Failed splits: %s", failed)
        sys.exit(1)
    else:
        logger.info("All splits downloaded and verified successfully.")


if __name__ == "__main__":
    main()
