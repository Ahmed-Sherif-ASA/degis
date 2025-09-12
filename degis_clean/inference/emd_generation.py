"""
EMD-Constrained Image Generation

This module provides functions for generating images with Earth Mover's Distance (EMD) constraints
on color histograms, supporting multiple color spaces (RGB, HCL, LAB).
"""

import torch
import numpy as np
import geomloss
from typing import List, Tuple, Optional, Union
from PIL import Image
import gc

from .visualization import display_comparison_grid
from ..training.features.color_histograms import compute_lab_histogram, compute_color_histogram, compute_hcl_histogram


def calculate_emd_distance_topk(
    hist1: np.ndarray, 
    hist2: np.ndarray, 
    top_k: int = 20, 
    blur: float = 0.01
) -> float:
    """
    Calculate Earth Mover's Distance between top-k histogram values using Sinkhorn algorithm.
    Supports histograms of variable lengths (e.g., 512 for RGB or 514 for LAB/HCL).
    
    Args:
        hist1: First histogram (1D array)
        hist2: Second histogram (1D array)
        top_k: Number of top values to consider for EMD calculation
        blur: Blur parameter for Sinkhorn algorithm
        
    Returns:
        EMD distance as float
    """
    # Ensure both histograms are same length
    if hist1.shape[0] != hist2.shape[0]:
        raise ValueError(f"Histogram size mismatch: {hist1.shape[0]} vs {hist2.shape[0]}")

    # Get top-k indices for both histograms
    k = min(top_k, hist1.shape[0])  # prevent out-of-bounds
    top_indices_1 = np.argsort(hist1)[-k:]
    top_indices_2 = np.argsort(hist2)[-k:]

    # Get union of indices
    unique_indices = np.union1d(top_indices_1, top_indices_2)

    # Extract top values
    h1_topk = hist1[unique_indices]
    h2_topk = hist2[unique_indices]

    # Normalize to sum to 1
    h1_topk = h1_topk / (h1_topk.sum() + 1e-8)
    h2_topk = h2_topk / (h2_topk.sum() + 1e-8)

    # Convert to tensors
    h1 = torch.tensor(h1_topk, dtype=torch.float32).unsqueeze(0)
    h2 = torch.tensor(h2_topk, dtype=torch.float32).unsqueeze(0)

    # Calculate Sinkhorn EMD
    loss = geomloss.SamplesLoss("sinkhorn", p=2, blur=blur, backend="tensorized")
    emd = loss(h1, h2).item()

    return emd


def detect_color_space(histogram: np.ndarray) -> str:
    """
    Detect color space from histogram length.
    
    Args:
        histogram: Histogram array
        
    Returns:
        Color space name ('rgb', 'hcl', or 'lab')
    """
    length = histogram.shape[0]
    if length == 512:
        return "rgb"
    elif length == 514:
        # For 514-length histograms, we need to determine if it's HCL or LAB
        # This is a heuristic - in practice, you might want to store this info
        # For now, we'll default to LAB as it's more common
        return "lab"
    else:
        raise ValueError(f"Unsupported histogram length: {length}. Expected 512 (RGB) or 514 (LAB/HCL)")


def compute_histogram_for_color_space(
    image: Image.Image, 
    color_space: str, 
    bins: int = 8
) -> np.ndarray:
    """
    Compute histogram for the specified color space.
    
    Args:
        image: PIL Image
        color_space: Color space ('rgb', 'hcl', or 'lab')
        bins: Number of bins per channel
        
    Returns:
        Histogram as numpy array
    """
    if color_space == "rgb":
        return compute_color_histogram(image, bins=bins)
    elif color_space == "hcl":
        return compute_hcl_histogram(image, bins=bins)
    elif color_space == "lab":
        return compute_lab_histogram(image, bins=bins)
    else:
        raise ValueError(f"Unsupported color space: {color_space}")


def generate_from_dataset_id_xl_with_emd(
    colour_index: int,
    layout_index: int,
    prompt: str = "a cat playing with a ball",
    target_emd_threshold: float = 0.1,
    max_attempts: int = 20,
    top_k: int = 20,
    guidance_scale: float = 6.5,
    steps: int = 40,
    controlnet_conditioning_scale: float = 0.8,
    num_samples: int = 1,
    attn_ip_scale: float = 0.8,
    text_token_scale: float = 1.0,
    ip_token_scale: Optional[float] = None,
    ip_uncond_scale: float = 0.0,
    zero_ip_in_uncond: bool = True,
    # Additional parameters for EMD generation
    color_space: Optional[str] = None,  # Auto-detect if None
    blur: float = 0.01,
    verbose: bool = False,
    # Required dependencies (should be passed from calling context)
    generator=None,
    colour_dataset=None,
    embeddings=None,
    histograms=None,
    edge_maps=None,
    color_head=None,
    device=None,
    transforms=None,
) -> Tuple[List[Image.Image], float, int]:
    """
    Generate images using IP-Adapter XL with EMD constraint on color histogram values.
    
    This function automatically detects the color space from the histogram dimension
    and uses the appropriate histogram computation method.
    
    Args:
        colour_index: Index in the color dataset
        layout_index: Index in the layout/edge dataset
        prompt: Text prompt for generation
        target_emd_threshold: Target EMD threshold to achieve
        max_attempts: Maximum number of generation attempts
        top_k: Number of top histogram values to consider for EMD
        guidance_scale: Guidance scale for generation
        steps: Number of inference steps
        controlnet_conditioning_scale: ControlNet conditioning scale
        num_samples: Number of samples to generate per attempt
        attn_ip_scale: IP attention scale
        text_token_scale: Text token scale
        ip_token_scale: IP token scale (defaults to attn_ip_scale)
        ip_uncond_scale: IP unconditional scale
        zero_ip_in_uncond: Whether to zero IP tokens in negative prompt
        color_space: Color space ('rgb', 'hcl', 'lab') - auto-detect if None
        blur: Blur parameter for Sinkhorn EMD calculation
        verbose: Whether to print progress information
        # Required dependencies (passed from calling context)
        generator: DEGIS generator instance
        colour_dataset: Color dataset
        embeddings: CLIP embeddings
        histograms: Color histograms
        edge_maps: Edge maps for layout
        color_head: Color head model
        device: Device to use
        transforms: Image transforms
        
    Returns:
        Tuple of (best_images, best_emd, attempts_made)
    """
    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Validate required dependencies
    if generator is None:
        raise ValueError("generator is required")
    if colour_dataset is None:
        raise ValueError("colour_dataset is required")
    if embeddings is None:
        raise ValueError("embeddings is required")
    if histograms is None:
        raise ValueError("histograms is required")
    if edge_maps is None:
        raise ValueError("edge_maps is required")
    if color_head is None:
        raise ValueError("color_head is required")
    if device is None:
        raise ValueError("device is required")
    if transforms is None:
        raise ValueError("transforms is required")

    # Get original image for display
    img_t, _ = colour_dataset[colour_index]
    pil_img = transforms.ToPILImage()(img_t)

    # Get CLIP embedding and compute color embedding
    z_clip = torch.as_tensor(embeddings[colour_index], dtype=torch.float32, device=device).unsqueeze(0)
    color_embedding = generator.get_color_embedding(color_head, z_clip)

    # Get original histogram for EMD comparison
    original_histogram = histograms[colour_index]

    # Auto-detect color space if not provided
    if color_space is None:
        color_space = detect_color_space(original_histogram)
    
    if verbose:
        print(f"Generating with EMD constraint (target: {target_emd_threshold:.3f})")
        print(f"Prompt: '{prompt}'")
        print(f"Max attempts: {max_attempts}")
        print(f"EMD calculation: top-{top_k} histogram values, color_space={color_space}")
        print("-" * 80)
        print(f"{'Attempt':<8} {'EMD':<10} {'attn_ip_scale':<15} {'ip_token_scale':<15} {'guidance_scale':<15}")
        print("-" * 80)

    # Create control image from edge data
    control_image = generator.create_edge_control_image(edge_maps[layout_index], size=512)

    # EMD-constrained generation
    best_images = None
    best_emd = float("inf")
    attempts_made = 0

    for attempt in range(max_attempts):
        # Generate images with IP-Adapter XL
        images = generator.generate(
            color_embedding=color_embedding,
            control_image=control_image,
            prompt=prompt,
            negative_prompt=(
                "monochrome, lowres, bad anatomy, worst quality, low quality, blurry, "
                "sketch, cartoon, drawing, anime:1.4, comic, illustration, posterized, "
                "mosaic, stained glass, abstract, surreal, psychedelic, trippy, texture artifact, "
                "embroidery, knitted, painting, oversaturated, unrealistic, bad shading"
            ),
            num_samples=num_samples,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            attn_ip_scale=attn_ip_scale,
            text_token_scale=text_token_scale,
            ip_token_scale=ip_token_scale,
            ip_uncond_scale=ip_uncond_scale,
            zero_ip_in_uncond=zero_ip_in_uncond,
        )

        # Calculate histogram for generated image using detected color space
        generated_hist = compute_histogram_for_color_space(images[0], color_space, bins=8)
        emd_distance = calculate_emd_distance_topk(
            original_histogram, generated_hist, top_k=top_k, blur=blur
        )

        attempts_made += 1

        # Structured logging
        if verbose:
            print(f"{attempt+1:<8} {emd_distance:<10.4f} {attn_ip_scale:<15.3f} {ip_token_scale or 0:<15.3f} {guidance_scale:<15.3f}", end="")

        # Check if this is the best result so far
        if emd_distance < best_emd:
            best_emd = emd_distance
            best_images = images
            if verbose:
                print("  ← NEW BEST!")
        else:
            if verbose:
                print()

        # Check if we've reached the target threshold
        if emd_distance <= target_emd_threshold:
            if verbose:
                print(f"\n✓ Target EMD reached! ({emd_distance:.4f} <= {target_emd_threshold:.3f})")
            break

    # Display results
    if best_images and verbose:
        comparison = display_comparison_grid(
            original=pil_img, control=control_image, generated=best_images, cols=3
        )
        # Note: display() is typically used in Jupyter notebooks
        # In a regular Python script, you might want to save the image instead
        try:
            from IPython.display import display
            display(comparison)
        except ImportError:
            # Fallback for non-Jupyter environments
            comparison.save(f"emd_generation_comparison_{colour_index}_{layout_index}.png")
            print(f"Comparison saved as emd_generation_comparison_{colour_index}_{layout_index}.png")

        print(f"\n✓ Generation complete!")
        print(f"Best EMD achieved: {best_emd:.4f}")
        print(f"Attempts made: {attempts_made}")

    return best_images, best_emd, attempts_made


def generate_with_emd_constraint(
    generator,
    color_embedding: torch.Tensor,
    control_image: Image.Image,
    original_histogram: np.ndarray,
    prompt: str = "a beautiful image",
    target_emd_threshold: float = 0.1,
    max_attempts: int = 20,
    top_k: int = 20,
    color_space: Optional[str] = None,
    **generation_kwargs
) -> Tuple[List[Image.Image], float, int]:
    """
    Generate images with EMD constraint using pre-computed embeddings and control image.
    
    This is a more flexible version that works with pre-computed components.
    
    Args:
        generator: DEGIS generator instance
        color_embedding: Pre-computed color embedding
        control_image: Control image for layout
        original_histogram: Original histogram for EMD comparison
        prompt: Text prompt for generation
        target_emd_threshold: Target EMD threshold
        max_attempts: Maximum number of attempts
        top_k: Number of top histogram values for EMD
        color_space: Color space ('rgb', 'hcl', 'lab') - auto-detect if None
        **generation_kwargs: Additional generation parameters
        
    Returns:
        Tuple of (best_images, best_emd, attempts_made)
    """
    # Auto-detect color space if not provided
    if color_space is None:
        color_space = detect_color_space(original_histogram)
    
    # EMD-constrained generation
    best_images = None
    best_emd = float("inf")
    attempts_made = 0

    for attempt in range(max_attempts):
        # Generate images
        images = generator.generate(
            color_embedding=color_embedding,
            control_image=control_image,
            prompt=prompt,
            **generation_kwargs
        )

        # Calculate histogram for generated image
        generated_hist = compute_histogram_for_color_space(images[0], color_space, bins=8)
        emd_distance = calculate_emd_distance_topk(
            original_histogram, generated_hist, top_k=top_k
        )

        attempts_made += 1

        # Check if this is the best result so far
        if emd_distance < best_emd:
            best_emd = emd_distance
            best_images = images

        # Check if we've reached the target threshold
        if emd_distance <= target_emd_threshold:
            break

    return best_images, best_emd, attempts_made


def generate_with_images_and_emd(
    colour_image: Image.Image,
    edge_image: Image.Image,
    prompt: str = "a beautiful image",
    target_emd_threshold: float = 0.1,
    max_attempts: int = 20,
    top_k: int = 20,
    guidance_scale: float = 6.5,
    steps: int = 40,
    controlnet_conditioning_scale: float = 0.8,
    num_samples: int = 1,
    attn_ip_scale: float = 0.8,
    text_token_scale: float = 1.0,
    ip_token_scale: Optional[float] = None,
    ip_uncond_scale: float = 0.0,
    zero_ip_in_uncond: bool = True,
    color_space: Optional[str] = None,
    blur: float = 0.01,
    verbose: bool = True,
    # Only the essential dependencies
    generator=None,
    color_head=None,
    device=None,
) -> Tuple[List[Image.Image], float, int]:
    """
    Generate images with EMD constraint using specific images (much cleaner approach).
    
    This is the recommended function - much cleaner than the dataset-based version.
    
    Args:
        colour_image: PIL Image for color reference
        edge_image: PIL Image for layout/edge control
        prompt: Text prompt for generation
        target_emd_threshold: Target EMD threshold
        max_attempts: Maximum number of attempts
        top_k: Number of top histogram values for EMD
        # ... other generation parameters
        generator: DEGIS generator instance
        color_head: Trained color head model
        device: Device to use
        
    Returns:
        Tuple of (best_images, best_emd, attempts_made)
    """
    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Validate required dependencies
    if generator is None:
        raise ValueError("generator is required")
    if color_head is None:
        raise ValueError("color_head is required")
    if device is None:
        raise ValueError("device is required")

    # Get CLIP embedding from color image
    z_clip = generator.get_clip_embedding(colour_image, device)
    color_embedding = generator.get_color_embedding(color_head, z_clip)

    # Compute original histogram for EMD comparison
    original_histogram = compute_histogram_for_color_space(colour_image, color_space or 'lab', bins=8)
    
    # Auto-detect color space if not provided
    if color_space is None:
        color_space = detect_color_space(original_histogram)
    
    if verbose:
        print(f"Generating with EMD constraint (target: {target_emd_threshold:.3f})")
        print(f"Prompt: '{prompt}'")
        print(f"Max attempts: {max_attempts}")
        print(f"EMD calculation: top-{top_k} histogram values, color_space={color_space}")
        print("-" * 80)
        print(f"{'Attempt':<8} {'EMD':<10} {'attn_ip_scale':<15} {'ip_token_scale':<15} {'guidance_scale':<15}")
        print("-" * 80)

    # EMD-constrained generation
    best_images = None
    best_emd = float("inf")
    attempts_made = 0

    for attempt in range(max_attempts):
        # Generate images with IP-Adapter XL
        images = generator.generate(
            color_embedding=color_embedding,
            control_image=edge_image,
            prompt=prompt,
            negative_prompt=(
                "monochrome, lowres, bad anatomy, worst quality, low quality, blurry, "
                "sketch, cartoon, drawing, anime:1.4, comic, illustration, posterized, "
                "mosaic, stained glass, abstract, surreal, psychedelic, trippy, texture artifact, "
                "embroidery, knitted, painting, oversaturated, unrealistic, bad shading"
            ),
            num_samples=num_samples,
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
        generated_hist = compute_histogram_for_color_space(images[0], color_space, bins=8)
        emd_distance = calculate_emd_distance_topk(
            original_histogram, generated_hist, top_k=top_k, blur=blur
        )

        attempts_made += 1

        # Structured logging
        if verbose:
            print(f"{attempt+1:<8} {emd_distance:<10.4f} {attn_ip_scale:<15.3f} {ip_token_scale or 0:<15.3f} {guidance_scale:<15.3f}", end="")

        # Check if this is the best result so far
        if emd_distance < best_emd:
            best_emd = emd_distance
            best_images = images
            if verbose:
                print("  ← NEW BEST!")
        else:
            if verbose:
                print()

        # Check if we've reached the target threshold
        if emd_distance <= target_emd_threshold:
            if verbose:
                print(f"\n✓ Target EMD reached! ({emd_distance:.4f} <= {target_emd_threshold:.3f})")
            break

    # Display results
    if best_images and verbose:
        try:
            from IPython.display import display
            comparison = display_comparison_grid(
                original=colour_image, control=edge_image, generated=best_images, cols=3
            )
            display(comparison)
        except ImportError:
            # Fallback for non-Jupyter environments
            print("Comparison image would be displayed in Jupyter notebook")

        print(f"\n✓ Generation complete!")
        print(f"Best EMD achieved: {best_emd:.4f}")
        print(f"Attempts made: {attempts_made}")

    return best_images, best_emd, attempts_made
