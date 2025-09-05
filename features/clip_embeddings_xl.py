# clip_embeddings_xl.py
"""
OpenCLIP XL embeddings (SDXL-friendly).

Defaults:
  - model: ViT-bigG-14
  - weights: laion2b-39b-b160k
These are commonly used for SDXL IP-Adapter weights.

Usage:
  from clip_embeddings_xl import (
      device, preprocess_xl, compute_clip_embedding_xl, generate_embeddings_xl
  )

  # single image tensor [3,H,W] in [0,1]:
  z = compute_clip_embedding_xl(img_tensor)     # torch.float32, shape [D]

  # batched loader -> .npy on disk:
  arr = generate_embeddings_xl(loader, "/path/emb_sdxl_bigG.npy")
"""

from __future__ import annotations
import os
import numpy as np
import torch
from tqdm import tqdm
import open_clip

# --- config ---
try:
    from config import HF_HUB_CACHE as HF_CACHE
except Exception:
    HF_CACHE = None

# Choose the XL encoder + weights (change if your IP-Adapter expects a different one)
MODEL_NAME = "ViT-bigG-14"             # SDXL-friendly
PRETRAINED = "laion2b_s39b_b160k"       # common bigG weights

# --- device / perf flags ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# --- load model + preprocess ---
_clip_model_xl, _, preprocess_xl = open_clip.create_model_and_transforms(
    model_name=MODEL_NAME,
    pretrained=PRETRAINED,
    cache_dir=HF_CACHE,
)

_clip_model_xl = _clip_model_xl.to(device)
# Prefer fp16 on CUDA for speed; stay fp32 on CPU.
if device.type == "cuda":
    _clip_model_xl = _clip_model_xl.half()
_clip_model_xl.eval()

# Get output dimension (bigG -> 1280)
try:
    XL_EMB_DIM = int(getattr(_clip_model_xl.visual, "output_dim", 0)) or int(
        _clip_model_xl.text_projection.shape[-1]
    )
except Exception:
    XL_EMB_DIM = 1280  # sensible default


@torch.no_grad()
def compute_clip_embedding_xl(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    image_tensor: [3,H,W] in [0,1] (already normalized/resize handled upstream
                   OR pass through preprocess_xl before calling)
    returns: torch.float32 [D] on CPU
    """
    x = image_tensor.unsqueeze(0).to(device)  # [1,3,H,W]
    if device.type == "cuda":
        x = x.half()
        with torch.cuda.amp.autocast(dtype=torch.float16):
            z = _clip_model_xl.encode_image(x)  # [1,D], fp16
    else:
        z = _clip_model_xl.encode_image(x)      # [1,D], fp32

    # Normalize (optional but standard for CLIP embeddings)
    z = torch.nn.functional.normalize(z, dim=-1)

    return z.float().squeeze(0).cpu()


@torch.no_grad()
def generate_embeddings_xl(
    loader,
    save_path: str,
    force_recompute: bool = False,
    normalize: bool = True,
) -> np.ndarray:
    """
    Batched embedding extraction.

    loader: yields (imgs, _) with imgs [B,3,H,W] in [0,1]
            (ideally transformed by preprocess_xl)
    save_path: .npy path
    force_recompute: if False and file exists -> loads and returns
    normalize: L2-normalize embeddings

    returns: numpy array [N,D] float32
    """
    if (not force_recompute) and os.path.exists(save_path):
        arr = np.load(save_path, mmap_mode=None)
        print(f"→ Loaded precomputed from {save_path} (shape={arr.shape})")
        return arr

    all_chunks = []
    for imgs, _ in tqdm(loader, desc="OpenCLIP bigG batched encode"):
        imgs = imgs.to(device)
        if device.type == "cuda":
            imgs = imgs.half()
            with torch.cuda.amp.autocast(dtype=torch.float16):
                z = _clip_model_xl.encode_image(imgs)  # [B,D], fp16
        else:
            z = _clip_model_xl.encode_image(imgs)      # [B,D], fp32

        if normalize:
            z = torch.nn.functional.normalize(z, dim=-1)

        all_chunks.append(z.float().cpu().numpy())

    embeddings = np.concatenate(all_chunks, axis=0)
    np.save(save_path, embeddings)
    print(f"→ Saved embeddings to {save_path} (shape={embeddings.shape})")
    return embeddings
