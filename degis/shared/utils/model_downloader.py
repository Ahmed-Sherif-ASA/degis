"""
Model downloading utilities.
"""
import os
from typing import Optional

from ..config import MODEL_CACHE


def download_ip_adapter_checkpoint(model_type: str, cache_dir: Optional[str] = None) -> str:
    """
    Download IP-Adapter checkpoint to cache directory.
    
    Args:
        model_type: Type of model ('sd15' or 'sdxl')
        cache_dir: Cache directory (defaults to MODEL_CACHE)
        
    Returns:
        Path to the downloaded checkpoint file
        
    Raises:
        ImportError: If huggingface_hub is not installed
        RuntimeError: If download fails
    """
    if cache_dir is None:
        cache_dir = MODEL_CACHE
    
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("huggingface_hub is required for model downloading. Install with: pip install huggingface_hub")
    
    # Determine the correct subdirectory and filename
    if model_type == 'sd15':
        filename = "models/ip-adapter_sd15.bin"
    elif model_type == 'sdxl':
        filename = "sdxl_models/ip-adapter_sdxl.bin"
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Must be 'sd15' or 'sdxl'")
    
    try:
        downloaded_path = hf_hub_download(
            repo_id="h94/IP-Adapter",
            filename=filename,
            cache_dir=cache_dir
        )
        return downloaded_path
    except Exception as e:
        raise RuntimeError(f"Failed to download IP-Adapter {model_type} checkpoint: {e}")


def download_all_ip_adapter_models(cache_dir: Optional[str] = None) -> dict:
    """
    Download all IP-Adapter models.
    
    Args:
        cache_dir: Cache directory (defaults to MODEL_CACHE)
        
    Returns:
        Dictionary with model types as keys and paths as values
        
    Raises:
        ImportError: If huggingface_hub is not installed
        RuntimeError: If any download fails
    """
    if cache_dir is None:
        cache_dir = MODEL_CACHE
    
    print(f"Downloading IP-Adapter models to: {cache_dir}")
    
    models = {}
    
    # Download SD 1.5
    print("Downloading IP-Adapter SD 1.5...")
    models['sd15'] = download_ip_adapter_checkpoint('sd15', cache_dir)
    print("Downloaded IP-Adapter SD 1.5")
    
    # Download SDXL
    print("Downloading IP-Adapter SDXL...")
    models['sdxl'] = download_ip_adapter_checkpoint('sdxl', cache_dir)
    print("Downloaded IP-Adapter SDXL")
    
    print("\nAll IP-Adapter models downloaded!")
    return models
