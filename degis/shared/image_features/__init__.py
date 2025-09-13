"""
Image features module for DEGIS.

Contains shared image feature extraction functionality used by both training and inference.
"""

from .color_histograms import *
from .edge_maps import *

__all__ = [
    # Color histograms
    "generate_color_histograms",
    "compute_color_histogram",
    "compute_lab_histogram", 
    "compute_hcl_histogram",
    "fast_rgb_histogram",
    "fast_lab_histogram",
    "fast_hcl_histogram",
    # Edge maps
    "generate_edge_maps",
    "compute_edge_map_canny"
]
