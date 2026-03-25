# inference/

Pipeline to run trained models over target regions and produce damage maps.

| File | Purpose |
|------|---------|
| `run_inference.py` | Load model checkpoint, tile AOI, run inference, save GeoTIFF + GeoJSON |

## Output files

- `outputs/<region>/<date>/damage_map.tif` - per-pixel damage probabilities (4 channels)
- `outputs/<region>/<date>/damage_overlay.geojson` - simplified vector layer for web map

## Target regions (v1)

| Region | Coordinates (approx.) | Time window |
|--------|----------------------|-------------|
| Gaza Strip | 31.3°N, 34.3°E | Oct 2023 – present |
| Southern Lebanon | 33.0°N, 35.4°E | Sep 2024 – present |
