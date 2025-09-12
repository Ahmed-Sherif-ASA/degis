"""
Inference module for DEGIS Clean.

Contains inference-specific functionality including generation, visualization, and CLI.
"""

from .generation import IPAdapterGenerator, IPAdapterXLGenerator, load_trained_color_head, get_color_embedding
from .visualization import plot_color_palette, display_images_grid, display_comparison_grid
from .emd_generation import generate_from_dataset_id_xl_with_emd, generate_with_emd_constraint

__all__ = [
    "IPAdapterGenerator",
    "IPAdapterXLGenerator", 
    "load_trained_color_head",
    "get_color_embedding",
    "plot_color_palette",
    "display_images_grid",
    "display_comparison_grid",
    "generate_from_dataset_id_xl_with_emd",
    "generate_with_emd_constraint"
]
