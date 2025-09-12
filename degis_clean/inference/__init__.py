"""
Inference module for DEGIS Clean.

Contains inference-specific functionality including generation, features, and CLI.
"""

from .generation import IPAdapterGenerator, IPAdapterXLGenerator, load_trained_color_head, get_color_embedding
from .features import generate_color_histograms, generate_edge_maps
from .visualization import plot_color_palette, display_images_grid

__all__ = [
    "IPAdapterGenerator",
    "IPAdapterXLGenerator", 
    "load_trained_color_head",
    "get_color_embedding",
    "generate_color_histograms",
    "generate_edge_maps",
    "plot_color_palette",
    "display_images_grid"
]
