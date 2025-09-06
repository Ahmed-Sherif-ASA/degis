# clip_embeddings_xl_hf.py
"""
HF CLIPVisionModelWithProjection embeddings for IPAdapterXL.

- Uses a ViT-H/14 vision tower (LAION OpenCLIP weights on HF).
- Returns the GLOBAL projected embedding (projection_dim) that IPAdapterXL's
  ImageProjModel expects.
- Keep MODEL_ID consistent with the vision model you load in IPAdapterXL
  (self.image_encoder_path) to avoid dim mismatches.

Usage:
  from clip_embeddings_xl_hf import (
      device, preprocess_xl, XL_EMB_DIM,
      compute_clip_embedding_xl, generate_embeddings_xl
  )

  z = compute_clip_embedding_xl(img_tensor)          # torch.float32 [D] on CPU
  arr = generate_embeddings_xl(loader, "/path/emb.npy")
"""

from __future__ import annotations
import os
import numpy as np
import torch
from tqdm import tqdm
from typing import Union, List
from PIL import Image
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

# ---- config ----
from ..models_config.models import get_model_config, setup_huggingface_cache, get_clip_vit_bigg14_path

# Use model management system
config = get_model_config()

# IMPORTANT: use the SAME model ID as IPAdapterXL's image_encoder_path
MODEL_ID = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

# ---- device / perf ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    _dtype = torch.float16
else:
    _dtype = torch.float32

# ---- load vision tower + processor ----
# Initialize as None - will be loaded lazily
_vision = None
preprocess_xl = None
XL_EMB_DIM = None

def _ensure_model_loaded():
    """Ensure the model is loaded (lazy loading)."""
    global _vision, preprocess_xl, XL_EMB_DIM
    
    if _vision is None:
        setup_huggingface_cache()
        
        _vision = CLIPVisionModelWithProjection.from_pretrained(MODEL_ID)
        preprocess_xl = CLIPImageProcessor.from_pretrained(MODEL_ID)
        
        _vision = _vision.to(device, dtype=_dtype).eval()
        
        # Embedding dim IPAdapterXL expects from .image_embeds
        XL_EMB_DIM = int(getattr(_vision.config, "projection_dim", 0)) or 1024
        print(f"Model projection_dim: {XL_EMB_DIM}")


def _to_pil_list(x: Union[torch.Tensor, Image.Image, List[Image.Image]]):
    """Accept [3,H,W] tensor, PIL.Image, or list of PIL; return list of PIL."""
    if isinstance(x, list):
        return x
    if isinstance(x, Image.Image):
        return [x]
    if torch.is_tensor(x):
        # x: [3,H,W] or [B,3,H,W]
        from torchvision.transforms.functional import to_pil_image
        if x.ndim == 3:
            return [to_pil_image(x)]
        elif x.ndim == 4:
            return [to_pil_image(img) for img in x]
    raise TypeError("Expected a [3,H,W] tensor, a PIL.Image, or a list of PIL images.")


@torch.no_grad()
def compute_clip_embedding_xl(image: Union[torch.Tensor, Image.Image, List[Image.Image]]) -> torch.Tensor:
    """
    Returns a single/global embedding for one image (float32 CPU) shaped [XL_EMB_DIM].
    If a list is passed, only the first image is encoded (use batched API for many).
    """
    _ensure_model_loaded()
    
    pil_list = _to_pil_list(image)
    px = preprocess_xl(images=pil_list[:1], return_tensors="pt").pixel_values.to(device, dtype=_dtype)

    if device.type == "cuda":
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = _vision(px)
    else:
        out = _vision(px)

    z = out.image_embeds                     # [1, XL_EMB_DIM] - GLOBAL embedding for IPAdapterXL
    z = torch.nn.functional.normalize(z, dim=-1)
    z = z.float().squeeze(0).cpu()
    assert z.shape[-1] == XL_EMB_DIM, f"Got {z.shape[-1]}, expected {XL_EMB_DIM}"
    return z


@torch.no_grad()
def generate_embeddings_xl(
    loader,
    save_path: str,
    force_recompute: bool = False,
    normalize: bool = True,
) -> np.ndarray:
    """
    Batched extraction of GLOBAL embeddings for IPAdapterXL.
    - loader should yield (imgs, _) with imgs as tensors [B,3,H,W] in [0,1] or PIL lists.
    - Saves float32 numpy array of shape [N, XL_EMB_DIM].
    """
    _ensure_model_loaded()
    
    if (not force_recompute) and os.path.exists(save_path):
        arr = np.load(save_path, mmap_mode=None)
        print(f"→ Loaded precomputed from {save_path} (shape={arr.shape})")
        return arr

    all_chunks = []
    for imgs, _ in tqdm(loader, desc=f"HF {MODEL_ID} batched encode (global)"):
        # Normalize input types to PIL list for HF processor
        pil_list = _to_pil_list(imgs)
        px = preprocess_xl(images=pil_list, return_tensors="pt").pixel_values.to(device, dtype=_dtype)

        if device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = _vision(px)
        else:
            out = _vision(px)

        z = out.image_embeds  # [B, XL_EMB_DIM] - GLOBAL embeddings for IPAdapterXL
        if normalize:
            z = torch.nn.functional.normalize(z, dim=-1)

        all_chunks.append(z.float().cpu().numpy())

    embeddings = np.concatenate(all_chunks, axis=0)
    np.save(save_path, embeddings)
    print(f"→ Saved embeddings to {save_path} (shape={embeddings.shape}, dim={XL_EMB_DIM})")
    return embeddings


# Function to check compatibility with your IP-Adapter model
def check_ip_adapter_compatibility(ip_adapter_model):
    """
    Check if the embeddings from this script are compatible with your IP-Adapter model.
    
    ip_adapter_model: Your IPAdapterXL instance
    
    Returns: (is_compatible, expected_dim, actual_dim)
    """
    _ensure_model_loaded()
    
    expected_dim = ip_adapter_model.image_encoder.config.projection_dim
    actual_dim = XL_EMB_DIM
    
    is_compatible = expected_dim == actual_dim
    
    print(f"IP-Adapter expected projection_dim: {expected_dim}")
    print(f"This script produces projection_dim: {actual_dim}")
    print(f"Compatible: {is_compatible}")
    
    if not is_compatible:
        print(f"\nTo fix compatibility, change MODEL_ID to:")
        print(f"'{ip_adapter_model.image_encoder_path}'")
    
    return is_compatible, expected_dim, actual_dim


# Update the model to match your IP-Adapter
def update_model_id(ip_adapter_model):
    """
    Update the model ID to match your IP-Adapter model.
    Call this after loading your IP-Adapter model.
    """
    global MODEL_ID, _vision, preprocess_xl, XL_EMB_DIM
    
    new_id = ip_adapter_model.image_encoder_path
    if new_id != MODEL_ID:
        print(f"Updating model from {MODEL_ID} to {new_id}")
        MODEL_ID = new_id
        
        # Reload the model
        setup_huggingface_cache()
        _vision = CLIPVisionModelWithProjection.from_pretrained(MODEL_ID)
        preprocess_xl = CLIPImageProcessor.from_pretrained(MODEL_ID)
        
        _vision = _vision.to(device, dtype=_dtype).eval()
        XL_EMB_DIM = int(getattr(_vision.config, "projection_dim", 0)) or 1024
        print(f"Updated model projection_dim: {XL_EMB_DIM}")
