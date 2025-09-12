"""
Training features module for DEGIS Clean.

Contains feature extraction functionality used during training.
"""

from .color_histograms import *
from .edge_maps import *

__all__ = [
    "generate_color_histograms",
    "generate_edge_maps",
    "compute_color_histogram",
    "compute_lab_histogram", 
    "compute_hcl_histogram",
    "fast_rgb_histogram",
    "fast_lab_histogram",
    "fast_hcl_histogram"
]
