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

class UnifiedImageDataset(Dataset):
    def __init__(self, source, size=(224,224), mode="auto", subset_ratio=1.0):
        """
        source      :
            - if mode=='coco'   → a torchvision.datasets.CocoDetection
            - if mode in {'url_df','file_df','auto'} → a pandas.DataFrame
        size        : (H,W) target for Resize
        mode        : 'coco' | 'url_df' | 'file_df' | 'auto'
        subset_ratio: fraction to sample from your DataFrame or COCO set
        """
        self.size         = size
        self.tf           = Compose([Resize(size), ToTensor()])
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

df = pd.read_csv(CSV_PATH)
assert "local_path" in df.columns, "CSV must have a local_path column!"

dataset = UnifiedImageDataset(
    df.rename(columns={"local_path": "file_path"}), 
    mode="file_df"
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)