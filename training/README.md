# training/

Training scripts for all model variants.

| File | Purpose |
|------|---------|
| `train_xbd_rgb.py` | Train RGB U-Net on xBD dataset (v1) |
| `train_bright_multimodal.py` | Train multimodal U-Net on BRIGHT dataset (v2) |

## How to run (HiperGator)

```bash
# v1 RGB training
python training/train_xbd_rgb.py \
  --data_dir /path/to/xbd/tiles \
  --epochs 50 \
  --batch_size 16 \
  --gpus 1
```

## Experiment tracking

- Logs saved to `runs/` (TensorBoard or W&B)
- Checkpoints saved to `checkpoints/`
