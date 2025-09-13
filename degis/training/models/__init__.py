"""
Training models module for DEGIS.

Contains model architectures and training utilities.
"""

from .color_heads import *
from .train_color import *

__all__ = [
    "ColorHead",
    "RestHead",
    "train_color_disentanglement"
]
