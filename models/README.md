# models/

Model architecture definitions.

| File | Purpose |
|------|---------|
| `unet_rgb.py` | U-Net for RGB pre/post damage segmentation (xBD, v1) |
| `unet_multimodal.py` | Dual-branch U-Net for optical + SAR inputs (BRIGHT, v2) |

## Architecture overview

- **Encoder**: ResNet-34 or EfficientNet-B4 backbone (pretrained on ImageNet)
- **Decoder**: U-Net++ style skip connections
- **Output**: Per-pixel 4-class softmax (Intact / Minor / Major / Destroyed)
- **Loss**: Weighted cross-entropy + Dice loss
- **Metrics**: mIoU, F1 per class, overall accuracy
