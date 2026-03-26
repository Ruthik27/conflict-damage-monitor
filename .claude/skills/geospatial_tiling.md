
Skill: Tiling Satellite Images with Rasterio
Standard 512x512 tiling pattern:

python
import rasterio
from rasterio.windows import Window
import numpy as np
from pathlib import Path

TILE_SIZE = 512
OVERLAP = 64  # pixels overlap between tiles

def tile_image(src_path: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with rasterio.open(src_path) as src:
        width, height = src.width, src.height
        profile = src.profile.copy()
        profile.update(width=TILE_SIZE, height=TILE_SIZE)
        
        stride = TILE_SIZE - OVERLAP
        for row in range(0, height - TILE_SIZE + 1, stride):
            for col in range(0, width - TILE_SIZE + 1, stride):
                window = Window(col, row, TILE_SIZE, TILE_SIZE)
                tile = src.read(window=window)
                
                # Skip mostly empty tiles
                if tile.mean() < 1.0:
                    continue
                
                tile_name = f"tile_r{row:05d}_c{col:05d}.tif"
                transform = src.window_transform(window)
                profile.update(transform=transform)
                
                with rasterio.open(output_dir / tile_name, "w", **profile) as dst:
                    dst.write(tile)

Rules:

Always use 512x512 tiles with 64px overlap

Skip near-empty tiles (mean < 1.0)

Preserve geospatial metadata (transform, CRS) in output tiles

Process pre and post images together to keep spatial alignment

Output to /blue/.../cdm/data/processed/tiles/
