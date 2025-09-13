"""
Model configuration and path management.
"""

from .models import *

__all__ = [
    "MODEL_REGISTRY",
    "get_model_path",
    "set_model_path", 
    "download_model",
    "get_sd15_path",
    "get_sdxl_path",
    "get_controlnet_sd15_path",
    "get_controlnet_sdxl_path",
    "get_ip_adapter_sd15_path",
    "get_ip_adapter_sdxl_path",
    "get_clip_vit_h14_path",
    "get_clip_vit_bigg14_path"
]
