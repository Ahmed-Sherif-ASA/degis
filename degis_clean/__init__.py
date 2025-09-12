"""
DEGIS Clean - Disentangled Embeddings Guided Image Synthesis

A clean, well-structured package for training and inference of disentangled
image representations using CLIP embeddings and various visual features.

Structure:
- data/          : Shared data handling (used by both training & inference)
- training/      : Training-specific code (models, training loops, CLI)
- inference/     : Inference-specific code (generation, features, CLI)
- shared/        : Shared utilities (embeddings, utils, config)
- ip_adapter_patch/ : Custom IP-Adapter implementation
"""

__version__ = "2.0.0"
__author__ = "DEGIS Team"

# Import main functionality for easy access
from .shared.config import *
from .data.dataset import UnifiedImageDataset
from .shared.embeddings import generate_clip_embeddings, generate_xl_embeddings
# from .inference.generation import IPAdapterGenerator, IPAdapterXLGenerator
# from .inference.features import generate_color_histograms, generate_edge_maps
# from .training.training import train_color_model

__all__ = [
    "UnifiedImageDataset",
    "generate_clip_embeddings", 
    "generate_xl_embeddings",
    # "IPAdapterGenerator",
    # "IPAdapterXLGenerator", 
    # "generate_color_histograms",
    # "generate_edge_maps",
    # "train_color_model"
]
