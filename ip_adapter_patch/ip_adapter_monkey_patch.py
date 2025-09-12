"""
DEGIS IP-Adapter Patch
=====================

This module provides DEGIS-enhanced IP-Adapter functionality by monkey patching
the original IP-Adapter classes with additional features:

- EmbeddingAdapter for multiple embedding types
- DEGIS token mixing with separate scaling controls
- Support for pre-computed embeddings
- Backward compatibility with original API

Usage:
    import ip_adapter_patch  # Apply patches
    from ip_adapter import IPAdapter, IPAdapterXL  # Use DEGIS-enhanced classes
"""

import torch
from typing import List, Optional, Union, Tuple
from PIL import Image

# Import original classes
from ip_adapter.ip_adapter import (
    ImageProjModel, MLPProjModel, IPAdapter, IPAdapterXL, 
    IPAdapterPlus, IPAdapterFull, IPAdapterPlusXL
)


class EmbeddingAdapter(torch.nn.Module):
    """Adapter for different embedding types"""
    
    def __init__(self, cross_attention_dim=1024, embedding_dim=1024, num_tokens=4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.embedding_dim = embedding_dim
        self.num_tokens = num_tokens
        
        # Projection layers for different embedding types
        self.proj_layers = torch.nn.ModuleDict({
            'clip': ImageProjModel(
                cross_attention_dim=cross_attention_dim,
                clip_embeddings_dim=embedding_dim,
                clip_extra_context_tokens=num_tokens
            ),
            'dino': MLPProjModel(
                cross_attention_dim=cross_attention_dim,
                clip_embeddings_dim=embedding_dim
            ),
            'custom': torch.nn.Sequential(
                torch.nn.Linear(embedding_dim, cross_attention_dim * num_tokens),
                torch.nn.LayerNorm(cross_attention_dim * num_tokens)
            )
        })
        
    def forward(self, embeddings, embedding_type='clip'):
        if embedding_type not in self.proj_layers:
            raise ValueError(f"Unsupported embedding type: {embedding_type}. Supported types: {list(self.proj_layers.keys())}")
        
        proj_layer = self.proj_layers[embedding_type]
        if embedding_type == 'custom':
            # Reshape for custom embeddings
            return proj_layer(embeddings).reshape(-1, self.num_tokens, self.cross_attention_dim)
        return proj_layer(embeddings)


def degis_mix_text_ip_tokens(
    self,
    prompt_embeds_text,          # (B, T_text, C)
    negative_prompt_embeds_text, # (B, T_text, C)
    image_prompt_embeds,         # (B, T_ip,   C)
    uncond_image_prompt_embeds,  # (B, T_ip,   C)
    *,
    text_token_scale=1.0,
    ip_token_scale=1.0,
    ip_uncond_scale=None,        # default: same as ip_token_scale
    zero_ip_in_uncond=False,     # True => "pure" unconditional (recommended if you want clean separation)
    pooled_prompt_embeds=None,   # (B, C) - for SDXL
    negative_pooled_prompt_embeds=None,  # (B, C) - for SDXL
):
    """
    Mix text and IP tokens with separate scaling controls.
    
    Args:
        prompt_embeds_text: Text embeddings for positive prompt
        negative_prompt_embeds_text: Text embeddings for negative prompt
        image_prompt_embeds: IP image embeddings for positive prompt
        uncond_image_prompt_embeds: IP image embeddings for negative prompt
        text_token_scale: Scale factor for text tokens (both positive and negative)
        ip_token_scale: Scale factor for IP tokens in positive prompt
        ip_uncond_scale: Scale factor for IP tokens in negative prompt (defaults to ip_token_scale)
        zero_ip_in_uncond: If True, zero out IP tokens in negative prompt for clean separation
        
    Returns:
        Tuple of (prompt_embeds, negative_prompt_embeds) ready for pipeline
    """
    dtype = prompt_embeds_text.dtype
    text_s = torch.as_tensor(text_token_scale, device=prompt_embeds_text.device, dtype=dtype)
    ip_s   = torch.as_tensor(ip_token_scale,   device=prompt_embeds_text.device, dtype=dtype)

    # scale text tokens in both branches (keeps CFG well-behaved)
    prompt_embeds_text  = prompt_embeds_text  * text_s
    negative_prompt_embeds_text = negative_prompt_embeds_text * text_s
    
    # scale pooled text embeddings for SDXL (if provided)
    if pooled_prompt_embeds is not None:
        pooled_prompt_embeds = pooled_prompt_embeds * text_s
    if negative_pooled_prompt_embeds is not None:
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds * text_s

    # scale IP tokens; you can optionally zero them in the unconditional branch
    if zero_ip_in_uncond:
        uncond_image_prompt_embeds = torch.zeros_like(uncond_image_prompt_embeds)
        ip_uncond_s = torch.as_tensor(0.0, device=prompt_embeds_text.device, dtype=dtype)
    else:
        if ip_uncond_scale is None:
            ip_uncond_s = ip_s
        else:
            ip_uncond_s = torch.as_tensor(ip_uncond_scale, device=prompt_embeds_text.device, dtype=dtype)

    image_prompt_embeds        = image_prompt_embeds        * ip_s
    uncond_image_prompt_embeds = uncond_image_prompt_embeds * ip_uncond_s

    # finally concatenate: [text || ip]
    prompt_embeds  = torch.cat([prompt_embeds_text,          image_prompt_embeds],        dim=1)
    negative_embeds = torch.cat([negative_prompt_embeds_text, uncond_image_prompt_embeds], dim=1)
    
    # Return pooled embeddings if provided (for SDXL)
    if pooled_prompt_embeds is not None:
        return prompt_embeds, negative_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
    else:
        return prompt_embeds, negative_embeds


def degis_get_image_embeds(self, pil_image=None, clip_image_embeds=None, embedding_type=None):
    """Enhanced get_image_embeds with support for pre-computed embeddings"""
    if pil_image is not None and self.embedding_type == 'clip':
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
    elif clip_image_embeds is not None:
        clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
    else:
        raise ValueError("Either pil_image (for CLIP) or clip_image_embeds must be provided")
        
    if self.embedding_type == 'clip': # Indicates self.image_proj_model is ImageProjModel
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
    else: # Indicates self.image_proj_model is EmbeddingAdapter
        # Use the embedding_type passed to this function, or default to the adapter's main type
        type_for_adapter_call = embedding_type or self.embedding_type
        image_prompt_embeds = self.image_proj_model(clip_image_embeds, embedding_type=type_for_adapter_call)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds), embedding_type=type_for_adapter_call)
        
    return image_prompt_embeds, uncond_image_prompt_embeds


def degis_generate(self, 
    pil_image=None,
    clip_image_embeds=None,
    prompt=None,
    negative_prompt=None,
    attn_ip_scale=1.0,
    text_token_scale=1.0,
    ip_token_scale=None,
    ip_uncond_scale=None,
    zero_ip_in_uncond=False,
    num_samples=4,
    seed=None,
    guidance_scale=7.5,
    num_inference_steps=30,
    **kwargs,
):
    """
    Enhanced generate method with separate scaling controls.
    
    Args:
        pil_image: Optional PIL image or list of PIL images
        clip_image_embeds: Optional pre-computed CLIP image embeddings
        prompt: Text prompt for generation
        negative_prompt: Negative text prompt
        attn_ip_scale: Per-layer IP attention gate scale (0.0-2.0)
        text_token_scale: Text token magnitude scaling (0.0-2.0)
        ip_token_scale: IP token magnitude scaling (0.0-2.0)
        ip_uncond_scale: IP token scaling in negative prompt (defaults to ip_token_scale)
        zero_ip_in_uncond: Zero out IP tokens in negative prompt for clean separation
        num_samples: Number of samples to generate
        seed: Random seed
        guidance_scale: Guidance scale for classifier-free guidance
        num_inference_steps: Number of denoising steps
        **kwargs: Additional arguments for the pipeline
    """
    if pil_image is not None:
        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
    elif clip_image_embeds is not None:
        num_prompts = clip_image_embeds.size(0)
    else:
        raise ValueError("Either pil_image or clip_image_embeds must be provided")

    if prompt is None:
        prompt = "best quality, high quality"
    if negative_prompt is None:
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

    if not isinstance(prompt, (list, tuple)):
        prompt = [prompt] * num_prompts
    if not isinstance(negative_prompt, (list, tuple)):
        negative_prompt = [negative_prompt] * num_prompts

    image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
        pil_image=pil_image, clip_image_embeds=clip_image_embeds
    )
    bs_embed, seq_len, _ = image_prompt_embeds.shape
    image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
    image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
    uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
    uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

    # Set the per-layer IP attention gate
    self.set_scale(attn_ip_scale)
    
    # Handle token scale parameters
    if ip_token_scale is None:
        ip_token_scale = attn_ip_scale  # Backward-compatible default

    with torch.inference_mode():
        prompt_embeds_text, negative_prompt_embeds_text = self.pipe.encode_prompt(
            prompt,
            device=self.device,
            num_images_per_prompt=num_samples,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )
        
        # Use the DEGIS mixing helper with separate scales
        prompt_embeds, negative_prompt_embeds = self._mix_text_ip_tokens(
            prompt_embeds_text=prompt_embeds_text,
            negative_prompt_embeds_text=negative_prompt_embeds_text,
            image_prompt_embeds=image_prompt_embeds,
            uncond_image_prompt_embeds=uncond_image_prompt_embeds,
            text_token_scale=text_token_scale,
            ip_token_scale=ip_token_scale,
            ip_uncond_scale=ip_uncond_scale,
            zero_ip_in_uncond=zero_ip_in_uncond,
        )

    from ip_adapter.utils import get_generator
    generator = get_generator(seed, self.device)

    images = self.pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        **kwargs,
    ).images

    return images


def degis_generate_from_embeddings(self,
    clip_image_embeds,
    prompt=None,
    negative_prompt=None,
    attn_ip_scale=1.0,
    text_token_scale=1.0,
    ip_token_scale=None,
    ip_uncond_scale=None,
    zero_ip_in_uncond=False,
    num_samples=4,
    seed=None,
    guidance_scale=7.5,
    num_inference_steps=30,
    **kwargs,
):
    """
    Generate images using pre-computed CLIP image embeddings.
    """
    return self.generate(
        clip_image_embeds=clip_image_embeds,
        prompt=prompt,
        negative_prompt=negative_prompt,
        attn_ip_scale=attn_ip_scale,
        text_token_scale=text_token_scale,
        ip_token_scale=ip_token_scale,
        ip_uncond_scale=ip_uncond_scale,
        zero_ip_in_uncond=zero_ip_in_uncond,
        num_samples=num_samples,
        seed=seed,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        **kwargs,
    )


def degis_init_proj(self):
    """Enhanced init_proj with EmbeddingAdapter support"""
    if hasattr(self, 'embedding_type') and self.embedding_type != 'clip':
        # For other embedding types, use the EmbeddingAdapter
        image_proj_model = EmbeddingAdapter(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            embedding_dim=1024,  # Default dimension, can be overridden
            num_tokens=self.num_tokens
        ).to(self.device, dtype=torch.float16)
    else:
        # Original CLIP-based projection
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
    return image_proj_model


def degis_init(self, sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4, embedding_type='clip'):
    """Enhanced __init__ with embedding_type support"""
    self.device = device
    self.image_encoder_path = image_encoder_path
    self.ip_ckpt = ip_ckpt
    self.num_tokens = num_tokens
    self.embedding_type = embedding_type

    self.pipe = sd_pipe.to(self.device)
    self.set_ip_adapter()

    # load image encoder if using CLIP
    if embedding_type == 'clip':
        from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()
    
    # image proj model
    self.image_proj_model = self.init_proj()
    self.load_ip_adapter()


def apply_patches():
    """Apply all monkey patches to IP-Adapter classes"""
    
    # Add EmbeddingAdapter to the module
    import ip_adapter.ip_adapter as ip_module
    ip_module.EmbeddingAdapter = EmbeddingAdapter
    
    # Patch IPAdapter class
    IPAdapter.__init__ = degis_init
    IPAdapter.init_proj = degis_init_proj
    IPAdapter.get_image_embeds = degis_get_image_embeds
    IPAdapter.generate = degis_generate
    IPAdapter.generate_from_embeddings = degis_generate_from_embeddings
    IPAdapter._mix_text_ip_tokens = degis_mix_text_ip_tokens
    
    # Patch IPAdapterXL class
    IPAdapterXL.get_image_embeds = degis_get_image_embeds
    IPAdapterXL.generate = degis_generate
    IPAdapterXL._mix_text_ip_tokens = degis_mix_text_ip_tokens
    
    # Patch IPAdapterPlus class
    IPAdapterPlus.get_image_embeds = degis_get_image_embeds
    
    # Patch IPAdapterFull class
    IPAdapterFull.get_image_embeds = degis_get_image_embeds
    
    # Patch IPAdapterPlusXL class
    IPAdapterPlusXL.get_image_embeds = degis_get_image_embeds
    IPAdapterPlusXL.generate = degis_generate
    IPAdapterPlusXL._mix_text_ip_tokens = degis_mix_text_ip_tokens


# Auto-apply patches when module is imported
apply_patches()
