"""
EMD-based image generation utilities.
"""
import numpy as np
import torch
from PIL import Image
from typing import List, Optional, Tuple
from tqdm import tqdm

try:
    import geomloss
except ImportError:
    raise ImportError("Please install geomloss: pip install geomloss")

from ..core.generation import IPAdapterXLGenerator
from ..features.color_histograms import compute_color_histogram


def calculate_emd_distance(hist1: np.ndarray, hist2: np.ndarray, blur: float = 0.01) -> float:
    """
    Calculate Earth Mover's Distance between two histograms using Sinkhorn algorithm.
    
    Args:
        hist1: First histogram (normalized)
        hist2: Second histogram (normalized)
        blur: Blur parameter for Sinkhorn algorithm
        
    Returns:
        EMD distance value
    """
    # Convert to torch tensors and add batch dimension
    h1 = torch.tensor(hist1, dtype=torch.float32).unsqueeze(0)
    h2 = torch.tensor(hist2, dtype=torch.float32).unsqueeze(0)
    
    # Calculate EMD using Sinkhorn
    loss = geomloss.SamplesLoss("sinkhorn", p=2, blur=blur, backend="tensorized")
    emd = loss(h1, h2).item()
    
    return emd


def generate_with_emd_constraint(
    generator: IPAdapterXLGenerator,
    color_embedding: torch.Tensor,
    control_image: Image.Image,
    original_histogram: np.ndarray,
    prompt: str = "a dog on the hoodie, artistic style, professional photography",
    target_emd_threshold: float = 0.1,
    max_attempts: int = 20,
    guidance_scale: float = 7.5,
    steps: int = 50,
    controlnet_conditioning_scale: float = 0.9,
    attn_ip_scale: float = 0.8,
    text_token_scale: float = 1.0,
    ip_token_scale: Optional[float] = None,
    ip_uncond_scale: float = 0.0,
    zero_ip_in_uncond: bool = True,
    negative_prompt: Optional[str] = None,
    blur: float = 0.01,
    verbose: bool = True
) -> Tuple[List[Image.Image], float, int]:
    """
    Generate images with EMD constraint using the "Dog on hoodie" hyperparameters.
    
    Keeps generating until the EMD between original and generated image histograms
    is below the target threshold.
    
    Args:
        generator: IP-Adapter XL generator instance
        color_embedding: Color embedding tensor
        control_image: Control image for layout
        original_histogram: Target histogram to match
        prompt: Text prompt for generation
        target_emd_threshold: Maximum acceptable EMD value
        max_attempts: Maximum number of generation attempts
        guidance_scale: Guidance scale for generation
        steps: Number of inference steps
        controlnet_conditioning_scale: ControlNet conditioning scale
        attn_ip_scale: Attention IP scale
        text_token_scale: Text token scale
        ip_token_scale: IP token scale
        ip_uncond_scale: IP unconditional scale
        zero_ip_in_uncond: Zero IP in unconditional
        negative_prompt: Negative prompt
        blur: Blur parameter for EMD calculation
        verbose: Whether to print progress
        
    Returns:
        Tuple of (best_images, best_emd, attempts_made)
    """
    if negative_prompt is None:
        negative_prompt = (
            "monochrome, lowres, bad anatomy, worst quality, low quality, blurry, "
            "sketch, cartoon, drawing, anime:1.4, comic, illustration, posterized, "
            "mosaic, stained glass, abstract, surreal, psychedelic, trippy, texture artifact, "
            "embroidery, knitted, painting, oversaturated, unrealistic, bad shading"
        )
    
    best_images = None
    best_emd = float('inf')
    attempts_made = 0
    
    if verbose:
        print(f"Generating with EMD constraint (target: {target_emd_threshold:.3f})")
        print(f"Using prompt: '{prompt}'")
        print(f"Max attempts: {max_attempts}")
        print("-" * 50)
    
    for attempt in tqdm(range(max_attempts), desc="EMD-constrained generation", disable=not verbose):
        # Generate images
        images = generator.generate(
            color_embedding=color_embedding,
            control_image=control_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_samples=1,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            attn_ip_scale=attn_ip_scale,
            text_token_scale=text_token_scale,
            ip_token_scale=ip_token_scale,
            ip_uncond_scale=ip_uncond_scale,
            zero_ip_in_uncond=zero_ip_in_uncond,
        )
        
        # Calculate histogram for generated image
        generated_hist = compute_color_histogram(images[0], bins=8)
        
        # Calculate EMD distance
        emd_distance = calculate_emd_distance(original_histogram, generated_hist, blur=blur)
        
        attempts_made += 1
        
        if verbose:
            print(f"Attempt {attempt + 1:2d}: EMD = {emd_distance:.4f}", end="")
        
        # Check if this is the best result so far
        if emd_distance < best_emd:
            best_emd = emd_distance
            best_images = images
            if verbose:
                print(" ✓ (new best)")
        else:
            if verbose:
                print()
        
        # Check if we've reached the target threshold
        if emd_distance <= target_emd_threshold:
            if verbose:
                print(f"\n✓ Target EMD reached in {attempts_made} attempts!")
            break
    
    if verbose:
        if best_emd > target_emd_threshold:
            print(f"\n⚠️  Target EMD not reached after {attempts_made} attempts")
            print(f"Best EMD achieved: {best_emd:.4f}")
        else:
            print(f"✓ Success! Final EMD: {best_emd:.4f}")
    
    return best_images, best_emd, attempts_made


def generate_from_dataset_id_xl_with_emd(
    generator: IPAdapterXLGenerator,
    color_head: torch.nn.Module,
    embeddings: np.ndarray,
    histograms: np.ndarray,
    edge_maps: np.ndarray,
    colour_dataset,
    colour_index: int,
    layout_index: int,
    prompt: str = "a dog on the hoodie, artistic style, professional photography",
    target_emd_threshold: float = 0.1,
    max_attempts: int = 20,
    device: str = "cuda",
    **generation_kwargs
) -> Tuple[List[Image.Image], float, int]:
    """
    Generate images from dataset indices with EMD constraint.
    
    This is a wrapper around generate_with_emd_constraint that handles
    the dataset loading and setup.
    
    Args:
        generator: IP-Adapter XL generator instance
        color_head: Trained color head model
        embeddings: CLIP embeddings array
        histograms: Color histograms array
        edge_maps: Edge maps array
        colour_dataset: Color dataset instance
        colour_index: Index for color selection
        layout_index: Index for layout selection
        prompt: Text prompt for generation
        target_emd_threshold: Maximum acceptable EMD value
        max_attempts: Maximum number of generation attempts
        device: Device to use
        **generation_kwargs: Additional generation parameters
        
    Returns:
        Tuple of (best_images, best_emd, attempts_made)
    """
    from torchvision import transforms
    
    # Get original image and its histogram
    img_t, _ = colour_dataset[colour_index]
    pil_img = transforms.ToPILImage()(img_t)
    original_histogram = histograms[colour_index]
    
    # Get CLIP embedding and compute color embedding
    z_clip_raw = embeddings[colour_index]
    print(f"Debug: Raw embedding shape: {z_clip_raw.shape}")
    print(f"Debug: Raw embedding dtype: {z_clip_raw.dtype}")
    
    z_clip = torch.as_tensor(z_clip_raw, dtype=torch.float32, device=device).unsqueeze(0)
    print(f"Debug: Processed embedding shape: {z_clip.shape}")
    print(f"Debug: Color head expects: {color_head.fc1.in_features} dimensions")
    
    # Check dimension mismatch
    if z_clip.shape[1] != color_head.fc1.in_features:
        raise ValueError(
            f"Embedding dimension mismatch! "
            f"Got {z_clip.shape[1]}D embedding but color head expects {color_head.fc1.in_features}D. "
            f"This suggests you're using the wrong embedding type or the embeddings array is corrupted."
        )
    
    # Use the degis function to get color embedding (handles the color head properly)
    from ..core.generation import get_color_embedding
    
    print(f"Debug: About to call color head with input shape: {z_clip.shape}")
    
    # Debug the color head outputs
    with torch.no_grad():
        logits, probs, c_emb = color_head(z_clip)
        print(f"Debug: Color head outputs:")
        print(f"  - logits shape: {logits.shape}")
        print(f"  - probs shape: {probs.shape}")
        print(f"  - c_emb shape: {c_emb.shape}")
    
    color_embedding = degis.get_color_embedding(color_head, z_clip)
    print(f"Debug: Final color_embedding shape: {color_embedding.shape}")
    
    # Create control image from edge data
    from ..core.generation import create_edge_control_image
    control_image = create_edge_control_image(edge_maps[layout_index], size=512)
    
    # Generate with EMD constraint
    return generate_with_emd_constraint(
        generator=generator,
        color_embedding=color_embedding,
        control_image=control_image,
        original_histogram=original_histogram,
        prompt=prompt,
        target_emd_threshold=target_emd_threshold,
        max_attempts=max_attempts,
        **generation_kwargs
    )
