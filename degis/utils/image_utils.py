"""
Image processing utilities for the degis package.
"""

import numpy as np
from PIL import Image, ImageOps
from typing import Union


def create_control_edge_pil(
    edge_data: np.ndarray, 
    size: int = 512
) -> Image.Image:
    """
    Create a ControlNet-ready edge PIL image from edge data.
    
    Args:
        edge_data: Edge data array, can be 1D (flattened) or 2D
        size: Target size for the output image (default: 512)
        
    Returns:
        PIL Image in RGB format, resized to (size, size)
    """
    # Handle 1D input by inferring dimensions
    if edge_data.ndim == 1:
        side = int(np.sqrt(edge_data.shape[0]))
        H = W = side
        edge = edge_data.reshape(H, W)
    else:
        edge = edge_data
    
    # Normalize to 0-255 if needed
    if edge.dtype != np.uint8:
        edge = (edge * 255.0).clip(0, 255).astype(np.uint8)
    
    # Convert to PIL and process
    pil = Image.fromarray(edge)
    pil = ImageOps.autocontrast(pil)
    pil = pil.resize((size, size), Image.BILINEAR).convert("RGB")
    
    return pil
