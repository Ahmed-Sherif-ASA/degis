"""
Core functionality for DEGIS package.
"""

from .embeddings import generate_clip_embeddings, generate_xl_embeddings
from .features import generate_color_histograms, generate_edge_maps
from .training import train_color_model, train_edge_model
from .generation import IPAdapterGenerator, IPAdapterXLGenerator, generate_from_embeddings, load_trained_color_head, get_color_embedding
from .visualization import (
    plot_color_palette, display_images_grid, display_comparison_grid,
    extract_top_palette, visualize_histogram_comparison, plot_training_metrics,
    save_generation_results, create_side_by_side_comparison
)

__all__ = [
    "generate_clip_embeddings",
    "generate_xl_embeddings",
    "generate_color_histograms", 
    "generate_edge_maps",
    "train_color_model",
    "train_edge_model",
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
]
