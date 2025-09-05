"""
Core image generation functionality using IP-Adapter and ControlNet.
"""

import os
import torch
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
from typing import List, Optional, Tuple, Union

try:
    import ip_adapter
    from ip_adapter import IPAdapter, IPAdapterXL
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline
    GENERATION_AVAILABLE = True
except ImportError:
    GENERATION_AVAILABLE = False


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
        color_embedding: torch.Tensor,
        control_image: Image.Image,
        prompt: str,
        negative_prompt: str = None,
        num_samples: int = 1,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        controlnet_conditioning_scale: float = 1.0,
        **kwargs
    ) -> List[Image.Image]:
        """Generate images using the pipeline."""
        if not GENERATION_AVAILABLE:
            raise ImportError("IP-Adapter and diffusers are required for image generation")
        
        if self.ip_adapter is None:
            raise RuntimeError("Pipeline not set up. Call setup_pipeline() first.")
            
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


class IPAdapterGenerator(ImageGenerator):
    """Image generator using IP-Adapter with SD 1.5."""
    
    def setup_pipeline(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        controlnet_id: str = "lllyasviel/control_v11p_sd15_canny",
        ip_ckpt: str = None,
        image_encoder_path: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        cache_dir: str = None,
        torch_dtype: torch.dtype = torch.float16,
    ):
        """Setup IP-Adapter with SD 1.5 pipeline."""
        if not GENERATION_AVAILABLE:
            raise ImportError("IP-Adapter and diffusers are required for image generation")
            
        # Setup cache directory
        if cache_dir:
            os.environ["HF_HOME"] = cache_dir
            os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(cache_dir, "hub")
            os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")
            os.environ["DIFFUSERS_CACHE"] = os.path.join(cache_dir, "diffusers")
            os.environ["TORCH_HOME"] = os.path.join(cache_dir, "torch")
        
        # Create ControlNet
        self.controlnet = ControlNetModel.from_pretrained(controlnet_id)
        
        # Create pipeline
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_id,
            controlnet=self.controlnet,
            torch_dtype=torch_dtype,
            safety_checker=None,
            feature_extractor=None,
            cache_dir=cache_dir,
        ).to(self.device)
        
        self.pipe.controlnet = self.pipe.controlnet.to(dtype=torch_dtype)
        
        # Create IP-Adapter
        if ip_ckpt is None:
            ip_ckpt = "/data/thesis/models/ip-adapter_sd15.bin"  # Default path
            
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
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet_id: str = "diffusers/controlnet-canny-sdxl-1.0",
        ip_ckpt: str = None,
        image_encoder_path: str = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        cache_dir: str = None,
        torch_dtype: torch.dtype = torch.float16,
    ):
        """Setup IP-Adapter XL with SDXL pipeline."""
        if not GENERATION_AVAILABLE:
            raise ImportError("IP-Adapter and diffusers are required for image generation")
            
        # Setup cache directory
        if cache_dir:
            os.environ["HF_HOME"] = cache_dir
            os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(cache_dir, "hub")
            os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")
            os.environ["DIFFUSERS_CACHE"] = os.path.join(cache_dir, "diffusers")
            os.environ["TORCH_HOME"] = os.path.join(cache_dir, "torch")
        
        # Create ControlNet
        self.controlnet = ControlNetModel.from_pretrained(controlnet_id)
        
        # Create pipeline
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            model_id,
            controlnet=self.controlnet,
            torch_dtype=torch_dtype,
            safety_checker=None,
            feature_extractor=None,
            cache_dir=cache_dir,
        ).to(self.device)
        
        self.pipe.controlnet = self.pipe.controlnet.to(dtype=torch_dtype)
        
        # Create IP-Adapter XL
        if ip_ckpt is None:
            ip_ckpt = "/data/thesis/models/ip-adapter_sdxl.bin"  # Default path
            
        self.ip_adapter = IPAdapterXL(
            sd_pipe=self.pipe,
            image_encoder_path=image_encoder_path,
            ip_ckpt=ip_ckpt,
            device=self.device,
            embedding_type='clip'
        )


def create_edge_control_image(
    edge_data: np.ndarray,
    size: int = 512,
    img_size: Tuple[int, int] = (224, 224)
) -> Image.Image:
    """Create a ControlNet-ready edge image from edge data."""
    H, W = img_size
    if edge_data.ndim == 1 and edge_data.shape[0] != H * W:
        side = int(np.sqrt(edge_data.shape[0]))
        H = W = side
    
    edge = edge_data.reshape(H, W)
    
    # Normalize to 0-255 if needed
    if edge.dtype != np.uint8:
        edge = (edge * 255.0).clip(0, 255).astype(np.uint8)
    
    # Convert to PIL and process
    pil = Image.fromarray(edge)
    pil = ImageOps.autocontrast(pil)
    pil = pil.resize((size, size), Image.BILINEAR).convert("RGB")
    
    return pil


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
    control_image = create_edge_control_image(edge_data)
    
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
    from ..models.color_heads import ColorHead
    
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
        _, _, color_embedding = color_head(clip_embedding)
    return color_embedding
