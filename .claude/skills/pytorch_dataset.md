
Skill: PyTorch Dataset Classes for Geospatial Data
Pattern for xBD/BRIGHT datasets:

python
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
from pathlib import Path

class BuildingDamageDataset(Dataset):
    """Dataset for pre/post disaster image pairs with damage labels."""
    
    def __init__(self, data_dir: str, split: str = "train", transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.samples = self._load_samples()
    
    def _load_samples(self) -> list:
        # Returns list of (pre_path, post_path, label_path) tuples
        ...
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        pre_path, post_path, label_path = self.samples[idx]
        
        with rasterio.open(pre_path) as src:
            pre = src.read().astype(np.float32)  # (C, H, W)
        with rasterio.open(post_path) as src:
            post = src.read().astype(np.float32)
        
        image = np.concatenate([pre, post], axis=0)  # 6-channel
        
        with rasterio.open(label_path) as src:
            label = src.read(1).astype(np.int64)  # (H, W)
        
        sample = {"image": torch.from_numpy(image), "label": torch.from_numpy(label)}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
Rules:

Always concatenate pre+post along channel dim (6-channel input)

Always return dict with "image" and "label" keys

Labels: 0=no damage, 1=minor, 2=major, 3=destroyed

Use rasterio for all geotiff reading, never PIL/cv2 for satellite data
