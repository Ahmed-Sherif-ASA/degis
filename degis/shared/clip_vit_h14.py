import open_clip
import torch
import numpy as np
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use model management system
from .config import MODEL_CACHE
from .models_config.models import get_clip_vit_h14_path

# Get model info from registry
model_id = get_clip_vit_h14_path()
model_name = "ViT-H-14"
pretrained = "laion2b_s32b_b79k"

# Initialize as None - will be loaded lazily
clip_model = None
preprocess = None

def _ensure_model_loaded():
    """Ensure the model is loaded (lazy loading)."""
    global clip_model, preprocess
    
    if clip_model is None:
        print("Loading CLIP model...")
        try:
            clip_model, _, preprocess = open_clip.create_model_and_transforms(
                model_name=model_name,
                pretrained=pretrained,
                cache_dir=MODEL_CACHE,
            )
            print(f"CLIP model loaded successfully. Preprocess: {preprocess is not None}")

            # half precision only on CUDA
            clip_model = clip_model.to(device)
            if device.type == "cuda":
                clip_model = clip_model.half()
            clip_model.eval()
            print("CLIP model ready for inference")
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            raise
    # CLIP model already loaded


def compute_clip_embedding(image_tensor: torch.Tensor) -> torch.Tensor:
    _ensure_model_loaded()
    
    x = image_tensor.unsqueeze(0).to(device)  # [1,3,H,W]
    with torch.no_grad():
        if device.type == "cuda":
            x = x.half()
            with torch.cuda.amp.autocast():
                z = clip_model.encode_image(x)
        else:
            z = clip_model.encode_image(x)
    return z.float().squeeze(0)  # always return fp32 on CPU-friendly dtype


def generate_embeddings_fp16(loader, save_path, force_recompute=False):
    _ensure_model_loaded()
    
    try:
        if not force_recompute:
            emb = np.load(save_path)
            print(f"→ Loaded precomputed from {save_path}")
            return emb
    except FileNotFoundError:
        print("→ No existing file, recomputing…")

    all_embs = []
    for imgs, _ in tqdm(loader, desc="CLIP batched encode"):
        imgs = imgs.to(device)
        if device.type == "cuda":
            imgs = imgs.half()
            with torch.no_grad(), torch.cuda.amp.autocast():
                z = clip_model.encode_image(imgs)
        else:
            with torch.no_grad():
                z = clip_model.encode_image(imgs)

        all_embs.append(z.float().cpu().numpy())

    embeddings = np.concatenate(all_embs, axis=0)
    np.save(save_path, embeddings)
    print(f"→ Saved embeddings to {save_path}")
    return embeddings
