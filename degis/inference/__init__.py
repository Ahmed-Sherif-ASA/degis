"""
Inference module for DEGIS.

Contains inference-specific functionality including generation, visualization, and CLI.
"""

from .core_generation import IPAdapterGenerator, IPAdapterXLGenerator, load_trained_color_head, get_color_embedding
from ..shared.utils.visualization import plot_color_palette, display_images_grid, display_comparison_grid, plot_training_curves
from .generation_functions import generate_from_dataset_id_xl_with_sinkhorn, generate_by_style, generate_by_colour_sinkhorn_constrained

__all__ = [
    "IPAdapterGenerator",
    "IPAdapterXLGenerator", 
    "load_trained_color_head",
    "get_color_embedding",
    "plot_color_palette",
    "display_images_grid",
    "display_comparison_grid",
    "plot_training_curves",
    "generate_from_dataset_id_xl_with_sinkhorn",
    "generate_by_style",
    "generate_by_colour_sinkhorn_constrained"
]
