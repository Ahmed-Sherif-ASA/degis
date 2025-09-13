"""
Model configuration and path management.
"""
import os
from typing import Dict, Optional
from pathlib import Path

# Model registry with HuggingFace IDs and local paths
MODEL_REGISTRY = {
    # Stable Diffusion models
    "sd15": {
        "hf_id": "runwayml/stable-diffusion-v1-5",
        "local_path": None,  # Will be cached by HuggingFace
    },
    "sdxl": {
        "hf_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "local_path": None,
    },
    
    # ControlNet models
    "controlnet_sd15_canny": {
        "hf_id": "lllyasviel/control_v11p_sd15_canny",
        "local_path": None,
    },
    "controlnet_sdxl_canny": {
        "hf_id": "diffusers/controlnet-canny-sdxl-1.0",
        "local_path": None,
    },
    
    # CLIP models
    "clip_vit_h14": {
        "hf_id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "local_path": None,
    },
    "clip_vit_bigg14": {
        "hf_id": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        "local_path": None,
    },
    
    # IP-Adapter checkpoints (these need to be downloaded separately)
    "ip_adapter_sd15": {
        "hf_id": "h94/IP-Adapter",  # HuggingFace model ID
        "checkpoint_name": "ip-adapter_sd15.bin",
        "local_path": None,  # Will be set by download
    },
    "ip_adapter_sdxl": {
        "hf_id": "h94/IP-Adapter",  # HuggingFace model ID
        "checkpoint_name": "ip-adapter_sdxl.bin", 
        "local_path": None,
    },
}



def get_model_path(model_key: str, config: Optional[Dict[str, str]] = None) -> str:
    """
    Get the path for a specific model.
    
    Args:
        model_key: Key from MODEL_REGISTRY
        config: Optional configuration dict
        
    Returns:
        Path to the model (local path if available, otherwise HF ID)
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key: {model_key}")
    
    model_info = MODEL_REGISTRY[model_key]
    
    # If local path is set, use it
    if model_info.get("local_path"):
        return model_info["local_path"]
    
    # Otherwise, return HF ID for automatic downloading
    return model_info["hf_id"]

def set_model_path(model_key: str, local_path: str):
    """Set a local path for a model (e.g., after downloading)."""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key: {model_key}")
    
    MODEL_REGISTRY[model_key]["local_path"] = local_path

def download_model(model_key: str, force_download: bool = False) -> str:
    """
    Download a model and return its local path.
    
    Args:
        model_key: Key from MODEL_REGISTRY
        force_download: Whether to force re-download
        
    Returns:
        Local path to the downloaded model
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key: {model_key}")
    
    model_info = MODEL_REGISTRY[model_key]
    
    # For HuggingFace models, use the library's caching
    if "hf_id" in model_info:
        return model_info["hf_id"]  # Let HuggingFace handle caching
    
    # For custom checkpoints, implement download logic here
    # This would download from the HF model and extract specific files
    raise NotImplementedError(f"Download not implemented for {model_key}")

# Convenience functions for common models
def get_sd15_path() -> str:
    """Get path to SD 1.5 model."""
    return get_model_path("sd15")

def get_sdxl_path() -> str:
    """Get path to SDXL model."""
    return get_model_path("sdxl")

def get_controlnet_sd15_path() -> str:
    """Get path to SD 1.5 ControlNet."""
    return get_model_path("controlnet_sd15_canny")

def get_controlnet_sdxl_path() -> str:
    """Get path to SDXL ControlNet."""
    return get_model_path("controlnet_sdxl_canny")

def get_ip_adapter_sd15_path() -> str:
    """Get path to IP-Adapter SD 1.5 checkpoint."""
    return get_model_path("ip_adapter_sd15")

def get_ip_adapter_sdxl_path() -> str:
    """Get path to IP-Adapter SDXL checkpoint."""
    return get_model_path("ip_adapter_sdxl")

def get_clip_vit_h14_path() -> str:
    """Get path to CLIP ViT-H-14 model."""
    return get_model_path("clip_vit_h14")

def get_clip_vit_bigg14_path() -> str:
    """Get path to CLIP ViT-bigG-14 model."""
    return get_model_path("clip_vit_bigg14")
