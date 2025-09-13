"""
Shared utilities for DEGIS.

Contains utility functions used by both training and inference modules.
"""

from .image_utils import create_control_edge_pil
from .auto_setup import setup_environment, detect_environment, download_required_models, print_setup_summary
from .model_downloader import download_ip_adapter_checkpoint, download_all_ip_adapter_models
from .visualization import plot_color_palette, display_images_grid, display_comparison_grid, plot_training_curves

__all__ = [
    'create_control_edge_pil',
    'setup_environment', 
    'detect_environment',
    'download_required_models',
    'print_setup_summary',
    'download_ip_adapter_checkpoint',
    'download_all_ip_adapter_models',
    'plot_color_palette',
    'display_images_grid',
    'display_comparison_grid',
    'plot_training_curves',
]
