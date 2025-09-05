import pandas as pd
from torch.utils.data import DataLoader

from data.dataset import UnifiedImageDataset
from features.clip_embeddings_xl_hf import generate_embeddings_xl
from config import CSV_PATH, BATCH_SIZE, HF_XL_EMBEDDINGS_TARGET_PATH
from multiprocessing import cpu_count


df = pd.read_csv(CSV_PATH)
print('xxxx')
print(df.head())
print(df.shape)
# assert "local_path" in df.columns, "CSV must have a local_path column!"

def print_system_profile():
    import os, shutil, platform, psutil, torch
    print("=== SYSTEM PROFILE ===")
    print("Python:", platform.python_version())
    print("PyTorch:", torch.__version__)
    print("CPU cores:", psutil.cpu_count(logical=True))
    vm = psutil.virtual_memory()
    print(f"RAM: {vm.total/1e9:.1f} GB, free {vm.available/1e9:.1f} GB")
    du = shutil.disk_usage("/data")
    print(f"/data disk: total {du.total/1e9:.1f} GB, free {du.free/1e9:.1f} GB")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        print("GPU:", torch.cuda.get_device_name(i))
        print(f"VRAM total: {torch.cuda.get_device_properties(i).total_memory/1e9:.1f} GB")
    print("======================")

print_system_profile()

dataset = UnifiedImageDataset(
    df.rename(columns={"local_path": "file_path"}), 
    mode="file_df"
)

num_cpu = cpu_count()

loader = DataLoader(
    dataset,
    batch_size=224,                    # then try 160/192/224
    shuffle=False,
    num_workers=min(32, max(8, num_cpu // 8)),  # 16–32 is a good sweet spot
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=6,                 # 4–8
    pin_memory_device="cuda",
)


embeddings = generate_embeddings_xl(
    loader,
    HF_XL_EMBEDDINGS_TARGET_PATH,
    force_recompute=True
)


# THE ONE I RAN FOR EDGE MAPS
# import config
# from features.color_histograms import generate_color_histograms
# from data.dataset import UnifiedImageDataset
# import time
# import pandas as pd
# from torch.utils.data import DataLoader
# from features.edge_maps import generate_edge_maps

# df = pd.read_csv(config.CSV_PATH)

# dataset = UnifiedImageDataset(
#     df.rename(columns={"local_path": "file_path"}), 
#     mode="file_df"
# )

# loader = DataLoader(
#     dataset,
#     batch_size=265,                    # then try 160/192/224
#     shuffle=False,
#     num_workers=64,  # 16–32 is a good sweet spot
#     pin_memory=True,
#     persistent_workers=True,
#     prefetch_factor=6,                 # 4–8
#     pin_memory_device="cuda",
# )

# edge_maps = generate_edge_maps(loader, config.EDGE_MAPS_PATH, method="canny", force_recompute=False)




# import time
# import numpy as np
# from tqdm import tqdm
# import torch
# import pandas as pd
# from torch.utils.data import DataLoader

# import config
# from data.dataset import UnifiedImageDataset
# from features.color_histograms import (
#     fast_rgb_histogram,
#     fast_lab_histogram,
#     fast_hcl_histogram,
# )

# def batch_histograms(batch, bins=8):
#     B = batch.shape[0]
#     out = {
#         "rgb":     np.zeros((B, bins**3),     dtype=np.float32),
#         "lab_514": np.zeros((B, bins**3 + 2), dtype=np.float32),
#         "hcl_514": np.zeros((B, bins**3 + 2), dtype=np.float32),
#     }
#     batch = batch.cpu()
#     for i in range(B):
#         img = batch[i]
#         out["rgb"][i]     = fast_rgb_histogram(img, bins=bins)
#         out["lab_514"][i] = fast_lab_histogram(img, bins=bins)   # returns 514-dim (includes BW)
#         out["hcl_514"][i] = fast_hcl_histogram(img, bins=bins)   # returns 514-dim (includes BW)
#     return out

# def generate_all_histograms(loader, out_paths, bins=8):
#     num_images = len(loader.dataset)
#     results = {
#         "rgb":     np.zeros((num_images, bins**3),     dtype=np.float32),
#         "lab_514": np.zeros((num_images, bins**3 + 2), dtype=np.float32),
#         "hcl_514": np.zeros((num_images, bins**3 + 2), dtype=np.float32),
#     }
#     idx = 0
#     for batch, _ in tqdm(loader, desc="Computing histograms", total=len(loader), unit="batch"):
#         bh = batch_histograms(batch, bins=bins)
#         B = batch.shape[0]
#         results["rgb"][idx:idx+B]     = bh["rgb"]
#         results["lab_514"][idx:idx+B] = bh["lab_514"]
#         results["hcl_514"][idx:idx+B] = bh["hcl_514"]
#         idx += B

#     for k, v in results.items():
#         np.save(out_paths[k], v)
#         print(f"✔ Saved {k} histograms to {out_paths[k]} (shape={v.shape})")
#     return results

# if __name__ == "__main__":
#     # Build dataset/loader HERE
#     df = pd.read_csv(config.CSV_PATH)
#     dataset = UnifiedImageDataset(
#         df.rename(columns={"local_path": "file_path"}),
#         mode="file_df"
#     )
#     loader = DataLoader(
#         dataset,
#         batch_size=256,
#         shuffle=False,
#         num_workers=64,
#         pin_memory=False,
#         persistent_workers=True,
#         prefetch_factor=6,
#     )

#     OUT_PATHS = {
#         "rgb":     config.COLOR_HIST_PATH_RGB,
#         "lab_514": config.COLOR_HIST_PATH_LAB_514,
#         "hcl_514": config.COLOR_HIST_PATH_HCL_514,
#     }

#     generate_all_histograms(loader, OUT_PATHS, bins=8)  # ← pass the loader, not CSV path
