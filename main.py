# import pandas as pd
# from torch.utils.data import DataLoader

# from data.dataset import UnifiedImageDataset
# from features.clip_embeddings import generate_embeddings_fp16
# from features.color_histograms import generate_color_histograms  # if you use it
# from config import CSV_PATH, BATCH_SIZE, EMBEDDINGS_TARGET_PATH  # etc.
# from multiprocessing import cpu_count


# df = pd.read_csv(CSV_PATH)
# print('xxxx')
# print(df.head())
# print(df.shape)
# assert "local_path" in df.columns, "CSV must have a local_path column!"

# def print_system_profile():
#     import os, shutil, platform, psutil, torch
#     print("=== SYSTEM PROFILE ===")
#     print("Python:", platform.python_version())
#     print("PyTorch:", torch.__version__)
#     print("CPU cores:", psutil.cpu_count(logical=True))
#     vm = psutil.virtual_memory()
#     print(f"RAM: {vm.total/1e9:.1f} GB, free {vm.available/1e9:.1f} GB")
#     du = shutil.disk_usage("/data")
#     print(f"/data disk: total {du.total/1e9:.1f} GB, free {du.free/1e9:.1f} GB")
#     print("CUDA available:", torch.cuda.is_available())
#     if torch.cuda.is_available():
#         i = torch.cuda.current_device()
#         print("GPU:", torch.cuda.get_device_name(i))
#         print(f"VRAM total: {torch.cuda.get_device_properties(i).total_memory/1e9:.1f} GB")
#     print("======================")

# print_system_profile()

# dataset = UnifiedImageDataset(
#     df.rename(columns={"local_path": "file_path"}), 
#     mode="file_df"
# )

# num_cpu = cpu_count()

# loader = DataLoader(
#     dataset,
#     batch_size=224,                    # then try 160/192/224
#     shuffle=False,
#     num_workers=min(32, max(8, num_cpu // 8)),  # 16–32 is a good sweet spot
#     pin_memory=True,
#     persistent_workers=True,
#     prefetch_factor=6,                 # 4–8
#     pin_memory_device="cuda",
# )


# embeddings = generate_embeddings_fp16(
#     loader,
#     EMBEDDINGS_TARGET_PATH,
#     force_recompute=True
# )

# import config
# from features.color_histograms import generate_color_histograms
# from data.dataset import UnifiedImageDataset
# import time
# import pandas as pd
# from torch.utils.data import DataLoader

# df = pd.read_csv(config.CSV_PATH)

# dataset = UnifiedImageDataset(
#     df.rename(columns={"local_path": "file_path"}), 
#     mode="file_df",
#     subset_ratio=0.001
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

# # Extraction plan
# tasks = [
#     ("rgb", config.COLOR_HIST_PATH_RGB, False),
#     ("lab", config.COLOR_HIST_PATH_LAB, False),
#     ("hcl", config.COLOR_HIST_PATH_HCL, False),
#     ("lab", config.COLOR_HIST_PATH_LAB_514, True),
#     ("hcl", config.COLOR_HIST_PATH_HCL_514, True),
# ]

# start = time.time()
# for color_space, save_path, add_bw in tasks:
#     print(f"\n=== Starting {color_space.upper()} histogram "
#           f"({'514' if add_bw else '512'}) ===")

#     start = time.time()
#     generate_color_histograms(
#         loader=loader,
#         hist_path=save_path,
#         force_recompute=True,
#         hist_bins=8,
#         method="fast",
#         color_space=color_space,
#         add_bw_bins=add_bw
#     )
#     print(f"✔ Completed in {(time.time()-start)/3600:.2f}h")

#     print(f"✔ Completed {color_space.upper()} "
#           f"({'514' if add_bw else '512'})\n")

# print(f"✔ Completed ALL histograms in {(time.time()-start)/3600:.2f}h")

# import os
# import numpy as np
# from tqdm import tqdm
# from torchvision.io import read_image
# import torchvision.transforms.functional as TF
# import cv2
# from skimage import color

# import config


# def compute_histograms(img, bins=8):
#     """
#     Compute RGB, LAB, HCL histograms (512 + 514 bins).
#     Returns a dict of numpy arrays.
#     """

#     # Convert tensor [C,H,W] -> [H,W,C], uint8
#     img = img.permute(1, 2, 0).numpy()
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # cv2 expects BGR

#     hists = {}

#     # RGB hist (512 bins = 8^3)
#     hist_rgb = cv2.calcHist([img], [0, 1, 2], None,
#                             [bins, bins, bins],
#                             [0, 256, 0, 256, 0, 256])
#     hists["rgb"] = hist_rgb.flatten()

#     # Convert to LAB
#     lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#     hist_lab_512 = cv2.calcHist([lab], [0, 1, 2], None,
#                                 [bins, bins, bins],
#                                 [0, 256, 0, 256, 0, 256])
#     hists["lab_512"] = hist_lab_512.flatten()

#     # LAB 514 (add white + black bins)
#     h_lab = hist_lab_512.flatten()
#     hists["lab_514"] = np.concatenate([h_lab, np.array([0, 0])])

#     # Convert to HCL (via skimage)
#     rgb_norm = img[..., ::-1] / 255.0  # back to RGB + normalize
#     hcl = color.rgb2hsv(rgb_norm)  # approx HCL via HSV
#     hcl = (hcl * 255).astype(np.uint8)
#     hist_hcl_512 = cv2.calcHist([hcl], [0, 1, 2], None,
#                                 [bins, bins, bins],
#                                 [0, 256, 0, 256, 0, 256])
#     hists["hcl_512"] = hist_hcl_512.flatten()
#     hists["hcl_514"] = np.concatenate([hist_hcl_512.flatten(),
#                                        np.array([0, 0])])

#     return hists


# def generate_all_histograms(csv, out_paths, bins=8):
#     """
#     Compute all histograms (RGB, LAB, HCL) in one pass.
#     Saves to npy files.
#     """
#     import pandas as pd
#     df = pd.read_csv(csv)
#     files = df["local_path"].tolist()

#     results = {k: [] for k in out_paths.keys()}

#     for path in tqdm(files, desc="Computing histograms"):
#         try:
#             img = read_image(path)  # fast decode
#             img = TF.resize(img, config.IMG_SIZE, antialias=True)

#             hists = compute_histograms(img, bins=bins)

#             results["rgb"].append(hists["rgb"])
#             results["lab_512"].append(hists["lab_512"])
#             results["lab_514"].append(hists["lab_514"])
#             results["hcl_512"].append(hists["hcl_512"])
#             results["hcl_514"].append(hists["hcl_514"])

#         except Exception as e:
#             print(f"❌ Error on {path}: {e}")
#             continue

#     # Save all histograms
#     for k, v in results.items():
#         arr = np.stack(v, axis=0)
#         np.save(out_paths[k], arr)
#         print(f"✔ Saved {k} histograms to {out_paths[k]} (shape={arr.shape})")


# if __name__ == "__main__":
#     OUT_PATHS = {
#         "rgb": config.COLOR_HIST_PATH_RGB,
#         "lab_512": config.COLOR_HIST_PATH_LAB,
#         "lab_514": config.COLOR_HIST_PATH_LAB_514,
#         "hcl_512": config.COLOR_HIST_PATH_HCL,
#         "hcl_514": config.COLOR_HIST_PATH_HCL_514,
#     }

#     generate_all_histograms(config.CSV_PATH, OUT_PATHS, bins=8)



# import time
# import numpy as np
# from tqdm import tqdm
# import torch

# import config
# from data.dataset import UnifiedImageDataset
# from features.color_histograms import (
#     fast_rgb_histogram,
#     fast_lab_histogram,
#     fast_hcl_histogram,
# )
# import pandas as pd
# from torch.utils.data import DataLoader


# def batch_histograms(batch, bins=8):
#     """
#     Compute RGB, LAB, HCL histograms (512 + 514 bins) for a whole batch at once.
#     Returns dict of numpy arrays: [B, hist_dim].
#     """
#     B = batch.shape[0]
#     batch_hists = {
#         "rgb": np.zeros((B, bins**3), dtype=np.float32),
#         "lab_512": np.zeros((B, bins**3), dtype=np.float32),
#         "lab_514": np.zeros((B, bins**3 + 2), dtype=np.float32),
#         "hcl_512": np.zeros((B, bins**3), dtype=np.float32),
#         "hcl_514": np.zeros((B, bins**3 + 2), dtype=np.float32),
#     }

#     # Convert to CPU once (faster than calling .cpu() inside the loop)
#     batch = batch.cpu()

#     # Loop in NumPy only, avoid PyTorch overhead
#     for i in range(B):
#         img_tensor = batch[i]

#         # RGB
#         batch_hists["rgb"][i] = fast_rgb_histogram(img_tensor, bins=bins)

#         # Lab
#         batch_hists["lab_512"][i] = fast_lab_histogram(img_tensor, bins=bins, add_bw_bins=False)
#         batch_hists["lab_514"][i] = fast_lab_histogram(img_tensor, bins=bins, add_bw_bins=True)

#         # HCL
#         batch_hists["hcl_512"][i] = fast_hcl_histogram(img_tensor, bins=bins, add_bw_bins=False)
#         batch_hists["hcl_514"][i] = fast_hcl_histogram(img_tensor, bins=bins, add_bw_bins=True)

#     return batch_hists


# def generate_all_histograms(loader, out_paths, bins=8):
#     """
#     Generate RGB, Lab (512 & 514), HCL (512 & 514) histograms for all images in loader.
#     Saves results to .npy arrays.
#     """
#     num_images = len(loader.dataset)

#     results = {
#         "rgb": np.zeros((num_images, bins**3), dtype=np.float32),
#         "lab_512": np.zeros((num_images, bins**3), dtype=np.float32),
#         "lab_514": np.zeros((num_images, bins**3 + 2), dtype=np.float32),
#         "hcl_512": np.zeros((num_images, bins**3), dtype=np.float32),
#         "hcl_514": np.zeros((num_images, bins**3 + 2), dtype=np.float32),
#     }

#     idx = 0
#     for batch, _ in tqdm(loader, desc="Computing histograms"):
#         bh = batch_histograms(batch, bins=bins)

#         B = batch.shape[0]
#         for k in results.keys():
#             results[k][idx:idx+B] = bh[k]

#         idx += B

#     # Save
#     for k, v in results.items():
#         np.save(out_paths[k], v)
#         print(f"✔ Saved {k} histograms to {out_paths[k]} (shape={v.shape})")

#     return results


# if __name__ == "__main__":
#     df = pd.read_csv(config.CSV_PATH)
#     dataset = UnifiedImageDataset(
#         df.rename(columns={"local_path": "file_path"}),
#         mode="file_df",
#         subset_ratio=0.001
#     )

#     loader = DataLoader(
#         dataset,
#         batch_size=256,  # tune up or down depending on RAM
#         shuffle=False,
#         num_workers=64,  # high for good CPU throughput
#         pin_memory=True,
#         persistent_workers=True,
#         prefetch_factor=6,
#         pin_memory_device="cuda",
#     )

#     OUT_PATHS = {
#         "rgb": config.COLOR_HIST_PATH_RGB,
#         "lab_512": config.COLOR_HIST_PATH_LAB,
#         "lab_514": config.COLOR_HIST_PATH_LAB_514,
#         "hcl_512": config.COLOR_HIST_PATH_HCL,
#         "hcl_514": config.COLOR_HIST_PATH_HCL_514,
#     }

#     start = time.time()
#     generate_all_histograms(loader, OUT_PATHS, bins=8)
#     print(f"✔ Completed ALL histograms in {(time.time()-start)/3600:.2f}h")






# import time
# import numpy as np
# from tqdm import tqdm
# import torch
# import cv2
# from skimage import color
# import pandas as pd
# from torch.utils.data import DataLoader

# import config
# from data.dataset import UnifiedImageDataset


# def batch_histograms(batch, bins=8):
#     """
#     Vectorized RGB, Lab, HCL histograms for a whole batch.
#     Returns dict of numpy arrays: [B, hist_dim].
#     """
#     B, C, H, W = batch.shape
#     batch = batch.permute(0, 2, 3, 1).numpy().astype(np.uint8)  # [B,H,W,C], uint8

#     # ---- RGB ----
#     rgb = batch.reshape(B, -1, 3)
#     hist_rgb = np.array([
#         np.histogramdd(img, bins=(bins, bins, bins),
#                        range=((0,256),(0,256),(0,256)))[0].flatten()
#         for img in rgb
#     ])

#     # ---- Lab ----
#     lab = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2LAB) for img in batch])
#     lab_flat = lab.reshape(B, -1, 3)
#     hist_lab_512 = np.array([
#         np.histogramdd(img, bins=(bins,bins,bins),
#                        range=((0,256),(0,256),(0,256)))[0].flatten()
#         for img in lab_flat
#     ])
#     hist_lab_514 = np.concatenate([hist_lab_512, np.zeros((B,2))], axis=1)

#     # ---- HCL (approx via HSV) ----
#     rgb_norm = batch / 255.0
#     hsv = np.array([color.rgb2hsv(img) for img in rgb_norm]) * 255
#     hsv = hsv.astype(np.uint8)
#     hsv_flat = hsv.reshape(B, -1, 3)
#     hist_hcl_512 = np.array([
#         np.histogramdd(img, bins=(bins,bins,bins),
#                        range=((0,256),(0,256),(0,256)))[0].flatten()
#         for img in hsv_flat
#     ])
#     hist_hcl_514 = np.concatenate([hist_hcl_512, np.zeros((B,2))], axis=1)

#     return {
#         "rgb": hist_rgb.astype(np.float32),
#         "lab_512": hist_lab_512.astype(np.float32),
#         "lab_514": hist_lab_514.astype(np.float32),
#         "hcl_512": hist_hcl_512.astype(np.float32),
#         "hcl_514": hist_hcl_514.astype(np.float32),
#     }


# def generate_all_histograms(loader, out_paths, bins=8):
#     """
#     Generate RGB, Lab (512 & 514), HCL (512 & 514) histograms for all images in loader.
#     Saves results to .npy arrays.
#     """
#     num_images = len(loader.dataset)

#     results = {
#         "rgb": np.zeros((num_images, bins**3), dtype=np.float32),
#         "lab_512": np.zeros((num_images, bins**3), dtype=np.float32),
#         "lab_514": np.zeros((num_images, bins**3 + 2), dtype=np.float32),
#         "hcl_512": np.zeros((num_images, bins**3), dtype=np.float32),
#         "hcl_514": np.zeros((num_images, bins**3 + 2), dtype=np.float32),
#     }

#     idx = 0
#     for batch, _ in tqdm(loader, desc="Computing histograms"):
#         bh = batch_histograms(batch, bins=bins)

#         B = batch.shape[0]
#         for k in results.keys():
#             results[k][idx:idx+B] = bh[k]

#         idx += B

#     # Save
#     for k, v in results.items():
#         np.save(out_paths[k], v)
#         print(f"✔ Saved {k} histograms to {out_paths[k]} (shape={v.shape})")

#     return results


# if __name__ == "__main__":
#     df = pd.read_csv(config.CSV_PATH)
#     dataset = UnifiedImageDataset(
#         df.rename(columns={"local_path": "file_path"}),
#         mode="file_df",
#         subset_ratio=0.001   # 🔹 small subset for testing
#     )

#     loader = DataLoader(
#         dataset,
#         batch_size=128,        # adjust up/down depending on RAM
#         shuffle=False,
#         num_workers=64,        # CPU workers
#         pin_memory=True,
#         persistent_workers=True,
#         prefetch_factor=6,
#         pin_memory_device="cuda",
#     )

#     OUT_PATHS = {
#         "rgb": config.COLOR_HIST_PATH_RGB,
#         "lab_512": config.COLOR_HIST_PATH_LAB,
#         "lab_514": config.COLOR_HIST_PATH_LAB_514,
#         "hcl_512": config.COLOR_HIST_PATH_HCL,
#         "hcl_514": config.COLOR_HIST_PATH_HCL_514,
#     }

#     start = time.time()
#     generate_all_histograms(loader, OUT_PATHS, bins=8)
#     print(f"✔ Completed ALL histograms in {(time.time()-start)/3600:.2f}h")


# THE ONE I RAN FOR EDGE MAPS
import config
from features.color_histograms import generate_color_histograms
from data.dataset import UnifiedImageDataset
import time
import pandas as pd
from torch.utils.data import DataLoader
from features.edge_maps import generate_edge_maps

df = pd.read_csv(config.CSV_PATH)

dataset = UnifiedImageDataset(
    df.rename(columns={"local_path": "file_path"}), 
    mode="file_df"
)

loader = DataLoader(
    dataset,
    batch_size=265,                    # then try 160/192/224
    shuffle=False,
    num_workers=64,  # 16–32 is a good sweet spot
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=6,                 # 4–8
    pin_memory_device="cuda",
)

edge_maps = generate_edge_maps(loader, config.EDGE_MAPS_PATH, method="canny", force_recompute=False)




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
