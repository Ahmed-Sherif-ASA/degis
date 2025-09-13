"""
Core image generation functionality using IP-Adapter and ControlNet.
"""

import os
import torch
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
from typing import List, Optional, Tuple, Union
from pathlib import Path

try:
    # Import DEGIS IP-Adapter implementation directly (no original IP-Adapter needed)
    from ip_adapter_patch.degis_ip_adapter_patch import IPAdapter, IPAdapterXL
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline
    GENERATION_AVAILABLE = True
except ImportError:
    GENERATION_AVAILABLE = False


def _download_checkpoint_if_needed(checkpoint_path: str, model_type: str) -> str:
    """
    Download IP-Adapter checkpoint if it doesn't exist locally.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model_type: Type of model ('sd15' or 'sdxl')
        
    Returns:
        Path to the checkpoint file (downloaded if needed)
    """
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    
    print(f"IP-Adapter {model_type} checkpoint not found at {checkpoint_path}")
    print(f"Downloading IP-Adapter {model_type} checkpoint...")
    
    from ..shared.utils.model_downloader import download_ip_adapter_checkpoint
    from ..shared.utils.image_utils import create_control_edge_pil
    
    downloaded_path = download_ip_adapter_checkpoint(model_type)
    print(f"✓ Downloaded IP-Adapter {model_type} checkpoint")
    return downloaded_path


class ImageGenerator:
    """Base class for image generation with IP-Adapter."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.pipe = None
        self.ip_adapter = None
        self.controlnet = None
        
    def setup_pipeline(self, **kwargs):
        """Setup the generation pipeline. To be implemented by subclasses."""
        raise NotImplementedError
        
    def generate(
        self,
        control_image: Image.Image,
        prompt: str,
        negative_prompt: str = None,
        num_samples: int = 1,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        controlnet_conditioning_scale: float = 1.0,
        # Support both modes
        color_embedding: torch.Tensor = None,
        pil_image: Image.Image = None,
        **kwargs
    ) -> List[Image.Image]:
        """Generate images using the pipeline."""
        if not GENERATION_AVAILABLE:
            raise ImportError("IP-Adapter and diffusers are required for image generation")
        
        if self.ip_adapter is None:
            raise RuntimeError("Pipeline not set up. Call setup_pipeline() first.")
        
        # Mode detection
        if pil_image is not None and color_embedding is not None:
            raise ValueError("Provide either pil_image OR color_embedding, not both")
        elif pil_image is not None:
            # IP-Adapter mode: pass the image directly
            return self.ip_adapter.generate(
                pil_image=pil_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_samples=num_samples,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                image=control_image,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                **kwargs
            )
        elif color_embedding is not None:
            # Pre-computed embedding mode
            return self.ip_adapter.generate(
                clip_image_embeds=color_embedding,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_samples=num_samples,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                image=control_image,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                **kwargs
            )
        else:
            raise ValueError("Must provide either pil_image or color_embedding")


class IPAdapterGenerator(ImageGenerator):
    """Image generator using IP-Adapter with SD 1.5."""
    
    def setup_pipeline(
        self,
        model_id: str = None,
        controlnet_id: str = None,
        ip_ckpt: str = None,
        image_encoder_path: str = None,
        cache_dir: str = None,
        torch_dtype: torch.dtype = torch.float16,
    ):
        """Setup IP-Adapter with SD 1.5 pipeline."""
        if not GENERATION_AVAILABLE:
            raise ImportError("IP-Adapter and diffusers are required for image generation")
        
        # Import model management
        from ..shared.models_config.models import (
            get_sd15_path, get_controlnet_sd15_path, 
            get_ip_adapter_sd15_path, get_clip_vit_h14_path
        )
        from ..shared.config import MODEL_CACHE
        
        # Use model management system for defaults
        model_id = model_id or get_sd15_path()
        controlnet_id = controlnet_id or get_controlnet_sd15_path()
        image_encoder_path = image_encoder_path or get_clip_vit_h14_path()
        cache_dir = cache_dir or MODEL_CACHE
        
        # Create ControlNet
        self.controlnet = ControlNetModel.from_pretrained(controlnet_id)
        
        # Create pipeline
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_id,
            controlnet=self.controlnet,
            torch_dtype=torch_dtype,
            safety_checker=None,
            feature_extractor=None,
        ).to(self.device)
        
        self.pipe.controlnet = self.pipe.controlnet.to(dtype=torch_dtype)
        
        # Create IP-Adapter
        if ip_ckpt is None:
            ip_ckpt = get_ip_adapter_sd15_path()
        
        # Download checkpoint if needed
        ip_ckpt = _download_checkpoint_if_needed(ip_ckpt, 'sd15')
            
        self.ip_adapter = IPAdapter(
            sd_pipe=self.pipe,
            image_encoder_path=image_encoder_path,
            ip_ckpt=ip_ckpt,
            device=self.device,
            embedding_type='clip'
        )


class IPAdapterXLGenerator(ImageGenerator):
    """Image generator using IP-Adapter XL with SDXL."""
    
    def setup_pipeline(
        self,
        model_id: str = None,
        controlnet_id: str = None,
        ip_ckpt: str = None,
        image_encoder_path: str = None,
        cache_dir: str = None,
        torch_dtype: torch.dtype = torch.float16,
    ):
        """Setup IP-Adapter XL with SDXL pipeline."""
        if not GENERATION_AVAILABLE:
            raise ImportError("IP-Adapter and diffusers are required for image generation")
        
        # Import model management
        from ..shared.models_config.models import (
            get_sdxl_path, get_controlnet_sdxl_path, 
            get_ip_adapter_sdxl_path, get_clip_vit_bigg14_path
        )
        from ..shared.config import MODEL_CACHE
        
        # Use model management system for defaults
        model_id = model_id or get_sdxl_path()
        controlnet_id = controlnet_id or get_controlnet_sdxl_path()
        image_encoder_path = image_encoder_path or get_clip_vit_bigg14_path()
        cache_dir = cache_dir or MODEL_CACHE
        
        # Create ControlNet
        self.controlnet = ControlNetModel.from_pretrained(controlnet_id)
        
        # Create pipeline
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            model_id,
            controlnet=self.controlnet,
            torch_dtype=torch_dtype,
            safety_checker=None,
            feature_extractor=None,
        ).to(self.device)
        
        self.pipe.controlnet = self.pipe.controlnet.to(dtype=torch_dtype)
        
        # Create IP-Adapter XL
        if ip_ckpt is None:
            ip_ckpt = get_ip_adapter_sdxl_path()
        
        # Download checkpoint if needed
        ip_ckpt = _download_checkpoint_if_needed(ip_ckpt, 'sdxl')
            
        self.ip_adapter = IPAdapterXL(
            sd_pipe=self.pipe,
            image_encoder_path=image_encoder_path,
            ip_ckpt=ip_ckpt,
            device=self.device,
            embedding_type='clip'
        )


# Edge control image creation moved to shared/utils/image_utils.py


def generate_from_embeddings(
    color_embedding: torch.Tensor,
    layout_embedding: torch.Tensor,
    edge_data: np.ndarray,
    prompt: str,
    generator: ImageGenerator,
    negative_prompt: str = None,
    num_samples: int = 1,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
    controlnet_conditioning_scale: float = 1.0,
    **kwargs
) -> List[Image.Image]:
    """Generate images from color and layout embeddings."""
    
    # Create control image from edge data
    control_image = create_control_edge_pil(edge_data)
    
    # Generate images
    images = generator.generate(
        color_embedding=color_embedding,
        control_image=control_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_samples=num_samples,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        **kwargs
    )
    
    return images


def load_trained_color_head(
    checkpoint_path: str,
    clip_dim: int,
    hist_dim: int,
    device: str = "cuda"
):
    """Load a trained color head model."""
    from ..training.models.color_heads import ColorHead
    
    color_head = ColorHead(clip_dim=clip_dim, hist_dim=hist_dim).to(device)
    color_head.load_state_dict(torch.load(checkpoint_path, map_location=device))
    color_head.eval()
    
    return color_head


def get_color_embedding(
    color_head,
    clip_embedding: torch.Tensor
) -> torch.Tensor:
    """Get color embedding from CLIP embedding using trained color head."""
    with torch.no_grad():
        logits, probs, color_embedding = color_head(clip_embedding)
    return color_embedding
