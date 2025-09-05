import pandas as pd
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
import torch

from data.dataset import UnifiedImageDataset
from features.clip_embeddings_xl import (
    generate_embeddings_xl, preprocess_xl  # <- use the XL transform
)
from config import CSV_PATH, BATCH_SIZE, XL_EMBEDDINGS_TARGET_PATH

# ---- system knobs (speed + deterministic enough) ----
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

df = pd.read_csv(CSV_PATH)
assert "local_path" in df.columns, "CSV must have a local_path column!"

# If your dataset uses a 'transform=' kwarg, pass preprocess_xl.
# If not, adapt UnifiedImageDataset to apply it internally.
dataset = UnifiedImageDataset(
    df.rename(columns={"local_path": "file_path"}),
    mode="file_df",
    transform=preprocess_xl,          # <-- IMPORTANT for SDXL/bigG
)

num_cpu = max(1, cpu_count())
nw = min(32, max(8, num_cpu // 8))    # 8–32 workers is usually good

# pin_memory_device requires torch>=2.0; keep optional
loader_kwargs = dict(
    batch_size=BATCH_SIZE,            # use your config
    shuffle=False,
    num_workers=nw,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=6,
)
try:
    loader_kwargs["pin_memory_device"] = "cuda"
except TypeError:
    pass  # older torch doesn't have this arg

loader = DataLoader(dataset, **loader_kwargs)

embeddings = generate_embeddings_xl(
    loader,
    XL_EMBEDDINGS_TARGET_PATH,
    force_recompute=True
)
print("Saved:", XL_EMBEDDINGS_TARGET_PATH, "shape:", embeddings.shape)
