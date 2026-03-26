# ML Pipeline Architecture

## Data Sources
- xBD: 850k+ buildings, pre/post pairs, 4-class damage labels
- BRIGHT: 350k+ buildings, high-res optical
- Sentinel-1/2: Free ESA SAR + multispectral

## Damage Classes
0: No damage | 1: Minor | 2: Major | 3: Destroyed

## Model
- Backbone: EfficientNet-B4 (pretrained ImageNet)
- Head: UNet++ decoder
- Input: 6-channel (3ch pre + 3ch post concatenated)
- Output: 512x512 segmentation mask (4 classes)
- Library: segmentation-models-pytorch

## Flow
Raw → Tile 512x512 → Augment → Train/Val/Test
→ EfficientNet-B4 + UNet++ → Change Detection → F1 per class
→ FastAPI + PostGIS → React Dashboard
