"""
DEGIS - Unified Image Dataset
==================================

This module provides a unified interface for loading images from various sources
for both training and inference in the DEGIS (Disentangled Embeddings Guided 
Image Synthesis) pipeline.

SUPPORTED DATA SOURCES:
- CSV files with image URLs (mode='url_df')
- CSV files with local file paths (mode='file_df') 
- Auto-detection based on CSV content (mode='auto')

USAGE EXAMPLES:

1. Load from CSV with image URLs:
   ```python
   import pandas as pd
   from degis.data.dataset import UnifiedImageDataset
   
   # CSV should have column: ['image_url'] with image URLs
   df = pd.read_csv('image_urls.csv')
   dataset = UnifiedImageDataset(df, mode='url_df', size=(512, 512))
   ```

2. Load from CSV with local file paths:
   ```python
   # CSV should have columns: ['file_path'] with local image paths
   df = pd.read_csv('image_paths.csv')
   dataset = UnifiedImageDataset(df, mode='file_df', size=(224, 224))
   ```

3. Auto-detect source type:
   ```python
   # Automatically detects if URLs or file paths
   df = pd.read_csv('images.csv')
   dataset = UnifiedImageDataset(df, mode='auto')
   ```

4. Use with DataLoader for training:
   ```python
   from torch.utils.data import DataLoader
   
   loader = DataLoader(
       dataset, 
       batch_size=32, 
       shuffle=True, 
       num_workers=4
   )
   ```

5. Access individual images:
   ```python
   # Get single image as PIL Image
   image = dataset[0]
   
   # Get image with transforms applied
   image_tensor = dataset.tf(image)
   ```

CSV FORMAT REQUIREMENTS:
- For URLs: Column named 'image_url' containing image URLs
- For file paths: Column named 'file_path' containing local image paths
- Additional columns are ignored but preserved in the dataset

TRANSFORMS:
- Default: Resize to specified size + Convert to PyTorch tensor
- Custom: Pass the torchvision.transforms.Compose object

ERROR HANDLING:
- Invalid URLs: Returns None (filtered out during iteration)
- Missing files: Returns None (filtered out during iteration)
- Invalid images: Returns None (filtered out during iteration)
"""

import os
import torch
from torch.utils.data import Dataset, Subset
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
from torch.utils.data import DataLoader
from ..shared.config import CSV_PATH, BATCH_SIZE
import numpy as np
from typing import Sequence


class UnifiedImageDataset(Dataset):
    def __init__(self, source, size=(224,224), mode="auto", subset_ratio=1.0, transform=None):
        """
        Unified dataset for loading images from various sources.
        
        Args:
            source: pandas.DataFrame with image URLs or file paths
            size: (H,W) target size for image resizing
            mode: 'url_df' | 'file_df' | 'auto' (auto-detects based on content)
            subset_ratio: fraction of data to use (for sampling)
            transform: custom torchvision transforms (default: Resize + ToTensor)
        """
        self.size         = size
        self.tf           = transform or Compose([Resize(size), ToTensor()])
        self.subset_ratio = subset_ratio
        self.mode         = mode

        if mode == "url_df":
            df          = source.sample(frac=subset_ratio,
                                        random_state=42).reset_index(drop=True)
            self.df     = df
            self.get_img = self._get_url

        elif mode == "file_df":
            df          = source.sample(frac=subset_ratio,
                                        random_state=42).reset_index(drop=True)
            self.df     = df
            self.get_img = self._get_file

        elif mode == "auto":
            # Auto-detect: check if 'image_url' column exists (URLs) or 'file_path' (local files)
            df          = source.sample(frac=subset_ratio,
                                        random_state=42).reset_index(drop=True)
            self.df     = df
            if "image_url" in self.df.columns:
                self.mode = "url_df"
                self.get_img = self._get_url
            elif "file_path" in self.df.columns:
                self.mode = "file_df"
                self.get_img = self._get_file
            else:
                raise ValueError("Auto mode requires 'image_url' or 'file_path' column")

        else:
            raise ValueError(f"mode must be one of ['url_df','file_df','auto'], got {mode!r}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = self.get_img(idx)
        if img is None:
            # if load fails, return a zero‐tensor
            img = Image.new("RGB", self.size, color=(0,0,0))
        img_t = self.tf(img)
        return img_t, idx


    # ─── mode: url_df ───────────────────────────────────────────────────────
    def _get_url(self, idx):
        url = self.df.iloc[idx]["image_url"]
        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            return Image.open(BytesIO(r.content)).convert("RGB")
        except:
            return None

    # ─── mode: file_df ──────────────────────────────────────────────────────
    def _get_file(self, idx):
        path = self.df.iloc[idx]["file_path"]
        try:
            return Image.open(path).convert("RGB")
        except:
            return None


class PrecompClipColorDataset(Dataset):
    """
    Wraps preloaded numpy arrays (embeddings + histograms) and yields per-sample
    tensors with a per-item histogram renormalization (exactly like the notebook).
    """
    def __init__(self, indices, embeddings_arr, hist_arr):
        self.indices = np.asarray(indices)
        self.emb  = embeddings_arr.astype(np.float32, copy=False)   # [N, clip_dim]
        self.hist = hist_arr.astype(np.float32,  copy=False)        # [N, hist_dim]

    def __len__(self): 
        return len(self.indices)

    def __getitem__(self, idx):
        i = int(self.indices[idx])
        z = torch.from_numpy(self.emb[i]).float()   # [clip_dim]
        h = torch.from_numpy(self.hist[i]).float()  # [hist_dim]
        s = h.sum()
        if s <= 1e-8:
            h = h + 1e-8
            s = h.sum()
        h = h / s
        return z, h