import os
import torch
from torch.utils.data import Dataset, Subset
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
from torchvision.datasets import CocoDetection
import pandas as pd
from torch.utils.data import DataLoader
from config import CSV_PATH, BATCH_SIZE
import numpy as np
from typing import Sequence


class UnifiedImageDataset(Dataset):
    def __init__(self, source, size=(224,224), mode="auto", subset_ratio=1.0, transform=None):
        """
        source      :
            - if mode=='coco'   → a torchvision.datasets.CocoDetection
            - if mode in {'url_df','file_df','auto'} → a pandas.DataFrame
        size        : (H,W) target for Resize
        mode        : 'coco' | 'url_df' | 'file_df' | 'auto'
        subset_ratio: fraction to sample from your DataFrame or COCO set
        """
        self.size         = size
        self.tf           = transform or Compose([Resize(size), ToTensor()])
        self.subset_ratio = subset_ratio
        self.mode         = mode

        if mode == "coco":
            full_ds    = source
            n_keep     = int(len(full_ds) * subset_ratio)
            self.ds    = Subset(full_ds, range(n_keep))
            self.get_img = self._get_coco

        elif mode == "url_df":
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
            # expects your DF to have both 'file_path' and 'value' columns
            df          = source.sample(frac=subset_ratio,
                                        random_state=42).reset_index(drop=True)
            self.df     = df
            self.get_img = self._get_auto

        else:
            raise ValueError(f"mode must be one of "
                             f"['coco','url_df','file_df','auto'], got {mode!r}")

    def __len__(self):
        if self.mode == "coco":
            return len(self.ds)
        else:
            return len(self.df)

    def __getitem__(self, idx):
        img = self.get_img(idx)
        if img is None:
            # if load fails, return a zero‐tensor
            img = Image.new("RGB", self.size, color=(0,0,0))
        img_t = self.tf(img)
        return img_t, idx

    # ─── mode: coco ────────────────────────────────────────────────────────
    def _get_coco(self, idx):
        img, _ = self.ds[idx]
        return img

    # ─── mode: url_df ───────────────────────────────────────────────────────
    def _get_url(self, idx):
        url = self.df.iloc[idx]["value"]
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

    # ─── mode: auto ─────────────────────────────────────────────────────────
    def _get_auto(self, idx):
        row = self.df.iloc[idx]
        # 1) try file_path
        fp = row.get("file_path", None)
        if isinstance(fp, str) and os.path.exists(fp):
            try:
                return Image.open(fp).convert("RGB")
            except:
                pass
        # 2) fallback to URL
        url = row.get("value", None)
        if isinstance(url, str):
            try:
                r = requests.get(url, timeout=5)
                r.raise_for_status()
                return Image.open(BytesIO(r.content)).convert("RGB")
            except:
                pass
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

class PrecompClipEdgeDataset(Dataset):
    """
    Dataset for (precomputed CLIP embedding → edge map) pairs.

    Args:
        indices: subset indices referring into the embeddings/edge arrays
        embeddings: np.ndarray [N, D] or memmap; float32 preferred (we cast anyway)
        edge_maps:  np.ndarray [N, 224*224] or memmap; can be uint8 (0–255) or float
        normalize_edges: if True, convert to float32 in [0,1] (divides by 255 if needed)
    """
    def __init__(
        self,
        indices: Sequence[int],
        embeddings: np.ndarray,
        edge_maps: np.ndarray,
        normalize_edges: bool = True,
    ):
        self.indices = np.asarray(indices)
        self.emb = embeddings
        self.edges = edge_maps
        self.normalize_edges = normalize_edges

        # quick sanity (same N)
        assert self.emb.shape[0] == self.edges.shape[0], \
            "Embeddings and edge_maps must have same number of rows."

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        i = int(self.indices[idx])

        z = self.emb[i]
        if z.dtype != np.float32:
            z = z.astype(np.float32, copy=False)
        z = torch.from_numpy(z).float()

        e = self.edges[i]
        if e.dtype != np.float32:
            e = e.astype(np.float32, copy=False)
        if self.normalize_edges and e.max() > 1.0:
            e = e / 255.0
        e = torch.from_numpy(e).float()

        return z, e
