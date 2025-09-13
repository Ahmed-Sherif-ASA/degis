"""
Training module for DEGIS.

Contains training-specific functionality including models, training loops, and CLI.
"""

from .training import train_color_model
from .batch_embeddings import generate_clip_embeddings, generate_xl_embeddings
from .models import *

__all__ = [
    "train_color_model",
    "generate_clip_embeddings",
    "generate_xl_embeddings"
]
