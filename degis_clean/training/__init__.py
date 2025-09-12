"""
Training module for DEGIS Clean.

Contains training-specific functionality including models, training loops, and CLI.
"""

from .training import train_color_model
from .models import *

__all__ = ["train_color_model"]
