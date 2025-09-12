"""
DEGIS: Disentangled Embeddings for Generative Image Synthesis

A package for training disentangled representations of images using CLIP embeddings
and various visual features (color histograms, edge maps, etc.).
"""

from .core.embeddings import generate_clip_embeddings, generate_xl_embeddings
from .core.features import generate_color_histograms, generate_edge_maps
from .core.training import train_color_model
from .core.generation import IPAdapterGenerator, IPAdapterXLGenerator, generate_from_embeddings, load_trained_color_head, get_color_embedding, create_edge_control_image
from .core.visualization import (
    plot_color_palette, display_images_grid, display_comparison_grid,
    extract_top_palette, visualize_histogram_comparison, plot_training_metrics,
    save_generation_results, create_side_by_side_comparison
)

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "generate_clip_embeddings",
    "generate_xl_embeddings", 
    "generate_color_histograms",
    "generate_edge_maps",
    "train_color_model",
    "IPAdapterGenerator",
    "IPAdapterXLGenerator",
    "generate_from_embeddings",
    "load_trained_color_head",
    "get_color_embedding",
    "plot_color_palette",
    "display_images_grid",
    "display_comparison_grid",
    "extract_top_palette",
    "visualize_histogram_comparison",
    "plot_training_metrics",
    "save_generation_results",
    "create_side_by_side_comparison",
    "create_edge_control_image",
]
