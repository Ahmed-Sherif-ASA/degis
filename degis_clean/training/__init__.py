"""
Training module for DEGIS Clean.

Contains training-specific functionality including models, training loops, and CLI.
"""

from .training import train_color_model
from .features import generate_color_histograms, generate_edge_maps
from .models import *

__all__ = [
    "train_color_model",
    "generate_color_histograms", 
    "generate_edge_maps"
]
