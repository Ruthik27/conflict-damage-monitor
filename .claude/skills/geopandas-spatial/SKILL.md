---
name: geopandas-spatial
description: >
  Geospatial data handling for the conflict-damage-monitor project. Use this skill
  whenever working with GeoTIFFs, building footprint polygons, coordinate reference
  systems, spatial joins, raster clipping, or any code that touches rasterio, GeoPandas,
  GDAL, or shapely. Trigger on: "read the satellite image", "load building polygons",
  "clip to AOI", "reproject to", "spatial join", "extract pixels for buildings",
  "convert coordinates", "tile the raster", "mask raster with polygons", or any
  geospatial I/O task. Also trigger when the user mentions xBD or BRIGHT data loading,
  since both require specific CRS handling. CRS errors and silent reprojection bugs
  are the #1 source of bad training data in this project — always verify CRS first.
---

# Geopandas-Spatial — conflict-damage-monitor

The most common silent bug in geospatial ML pipelines is a CRS mismatch: rasters and
vectors in different coordinate systems that spatially overlap on a map but misalign
when you try to extract pixels per polygon. Always verify CRS before any spatial
operation, and reproject to a common CRS before joining or clipping.

## Default CRS

- **Storage CRS**: EPSG:4326 (WGS84 geographic, degrees) — xBD and BRIGHT polygons are in this
- **Processing CRS for pixel extraction**: match the raster's native CRS (often UTM)
- **Rule**: reproject vectors to match the raster, not the other way around — raster reprojection introduces interpolation artifacts that corrupt pixel values

---

## CRS: always set and verify

```python
import geopandas as gpd
import rasterio
from pyproj import CRS

# Loading vector data — always check CRS immediately
gdf = gpd.read_file(polygon_path)
assert gdf.crs is not None, f"No CRS set on {polygon_path}"
print(f"Vector CRS: {gdf.crs.to_epsg()}")  # expect 4326 for xBD/BRIGHT

# If CRS is missing (older xBD versions), set it explicitly
if gdf.crs is None:
    gdf = gdf.set_crs("EPSG:4326")

# Loading raster — log CRS and transform for debugging
with rasterio.open(tif_path) as src:
    raster_crs = src.crs
    print(f"Raster CRS: {raster_crs.to_epsg()}, shape: {src.shape}, res: {src.res}")

# Reproject vector to match raster before any pixel extraction
if gdf.crs != raster_crs:
    gdf = gdf.to_crs(raster_crs)
```

---

## Library responsibilities

| Task | Library | Notes |
|---|---|---|
| Read/write GeoTIFF | `rasterio` | Use context manager, read bands explicitly |
| Vector polygons (GeoJSON, shapefile, GeoPackage) | `geopandas` | Check CRS on load |
| Geometry operations (union, buffer, intersection) | `shapely` via geopandas | |
| Raster clipping to AOI | `rasterio.mask` | Clip before tiling |
| Pixel extraction per polygon | `rasterio.features` + `numpy` | Match CRS first |
| Reprojection | `gdf.to_crs()` / `rasterio.warp` | Vectors to raster CRS |
| Spatial joins | `gpd.sjoin()` | Both must share CRS |

Never mix rasterio and geopandas on data with different CRS — always align first.

---

## AOI clipping: clip before processing

Clip large rasters to the area of interest before tiling or extracting features. Running
operations on full Sentinel scenes is slow and wastes I/O on pixels that will be discarded.

```python
import rasterio
from rasterio.mask import mask
from shapely.geometry import box, mapping
import geopandas as gpd
from pathlib import Path

def clip_raster_to_aoi(
    src_path: Path,
    aoi_gdf: gpd.GeoDataFrame,
    dst_path: Path,
) -> None:
    """Clip a GeoTIFF to the bounding box of an AOI polygon GeoDataFrame.

    Reprojects AOI to raster CRS before clipping. Writes a new GeoTIFF
    with updated transform and identical metadata otherwise.
    """
    with rasterio.open(src_path) as src:
        # Reproject AOI to raster CRS
        aoi_reproj = aoi_gdf.to_crs(src.crs)
        # Use bounding box of AOI, not exact geometry (faster, avoids edge artifacts)
        bbox = box(*aoi_reproj.total_bounds)
        out_image, out_transform = mask(src, [mapping(bbox)], crop=True)
        out_meta = src.meta.copy()

    out_meta.update({
        "height": out_image.shape[1],
        "width":  out_image.shape[2],
        "transform": out_transform,
    })
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **out_meta) as dst:
        dst.write(out_image)
```

---

## xBD and BRIGHT dataset specifics

### xBD
- Building polygons: GeoJSON files, one per disaster event, **EPSG:4326**
- Imagery: pre/post GeoTIFFs, typically **UTM** (varies by event location)
- Label field: `"subtype"` → map to int: `{"no-damage":0, "minor-damage":1, "major-damage":2, "destroyed":3, "un-classified":-1}`
- Pattern: load polygons → reproject to UTM → extract 512×512 chips centered on each building

```python
DAMAGE_LABEL_MAP = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
    "un-classified": -1,  # exclude from training
}

def load_xbd_polygons(geojson_path: Path) -> gpd.GeoDataFrame:
    """Load xBD building footprints with integer damage labels."""
    gdf = gpd.read_file(geojson_path)
    assert gdf.crs.to_epsg() == 4326, f"Unexpected CRS: {gdf.crs}"
    gdf["label"] = gdf["subtype"].map(DAMAGE_LABEL_MAP)
    # Drop unclassified buildings
    gdf = gdf[gdf["label"] >= 0].reset_index(drop=True)
    return gdf
```

### BRIGHT
- Building polygons: GeoJSON/shapefile, **EPSG:4326**
- Imagery: Sentinel-1 (SAR) + Sentinel-2 (optical), **UTM**
- Label field: `"damage_class"` (already integer 0–3 in most releases)
- Key difference from xBD: multi-temporal (more than 2 timestamps), handle by selecting pre/post pair

---

## Tiling: 512×512 chips per building

```python
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import box

TILE_SIZE = 512  # pixels

def extract_chip(
    src: rasterio.DatasetReader,
    centroid_x: float,
    centroid_y: float,
    tile_size: int = TILE_SIZE,
) -> np.ndarray:
    """Extract a square pixel chip centered on a building centroid.

    Args:
        src: Open rasterio dataset (raster CRS, not WGS84).
        centroid_x: X coordinate in raster CRS.
        centroid_y: Y coordinate in raster CRS.
        tile_size: Output chip size in pixels.

    Returns:
        Array of shape (bands, tile_size, tile_size), float32.
    """
    half = tile_size * src.res[0] / 2
    window = from_bounds(
        centroid_x - half, centroid_y - half,
        centroid_x + half, centroid_y + half,
        src.transform,
    )
    chip = src.read(window=window, boundless=True, fill_value=0)
    # Resize to exact tile_size if boundary effects cause off-by-one
    chip = chip[:, :tile_size, :tile_size]
    return chip.astype(np.float32)
```

---

## Common pitfalls

| Problem | Symptom | Fix |
|---|---|---|
| CRS mismatch | Polygons don't overlap raster pixels | `gdf.to_crs(src.crs)` before extraction |
| No CRS on load | `gdf.crs is None` | `gdf.set_crs("EPSG:4326")` for xBD/BRIGHT |
| `boundless=True` missing | Edge buildings raise `WindowError` | Always pass `boundless=True, fill_value=0` |
| Wrong band order | RGB vs BGR confusion | Check `src.descriptions` or `src.colorinterp` |
| Sentinel-2 scale | Values 0–10000, not 0–1 | Normalize: `chip / 10000.0` |
| Clipping without reprojection | Clip returns empty array | Reproject AOI to raster CRS before `rasterio.mask` |

---

## Sanity checks for any spatial pipeline

```python
# After loading any GeoDataFrame
assert gdf.crs is not None
assert len(gdf) > 0
assert gdf.geometry.is_valid.all(), "Invalid geometries — run gdf.geometry.buffer(0)"

# After reprojection
assert gdf.crs == target_crs

# After chip extraction
assert chip.shape == (n_bands, 512, 512)
assert np.isfinite(chip).all(), "Chip contains NaN/Inf"
assert chip.max() <= 1.0 or chip.max() > 1.0  # log actual range for debugging
```
