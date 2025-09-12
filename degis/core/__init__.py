"""
Core functionality for DEGIS package.
"""

from .embeddings import generate_clip_embeddings, generate_xl_embeddings
from .features import generate_color_histograms, generate_edge_maps
from .training import train_color_model
from .generation import IPAdapterGenerator, IPAdapterXLGenerator, generate_from_embeddings, load_trained_color_head, get_color_embedding
from .emd_generation import (
    calculate_emd_distance_topk, detect_color_space, compute_histogram_for_color_space,
    generate_from_dataset_id_xl_with_emd, generate_with_emd_constraint, generate_with_images_and_emd
)
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
    "IPAdapterGenerator",
    "IPAdapterXLGenerator",
    "generate_from_embeddings",
    "load_trained_color_head",
    "get_color_embedding",
    "calculate_emd_distance_topk",
    "detect_color_space",
    "compute_histogram_for_color_space",
    "generate_from_dataset_id_xl_with_emd",
    "generate_with_emd_constraint",
    "generate_with_images_and_emd",
    "plot_color_palette",
    "display_images_grid",
    "display_comparison_grid",
    "extract_top_palette",
    "visualize_histogram_comparison",
    "plot_training_metrics",
    "save_generation_results",
    "create_side_by_side_comparison",
]
