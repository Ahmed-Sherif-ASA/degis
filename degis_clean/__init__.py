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
from .training import train_color_model, generate_color_histograms, generate_edge_maps
from .inference import IPAdapterGenerator, IPAdapterXLGenerator, load_trained_color_head, get_color_embedding

__all__ = [
    "UnifiedImageDataset",
    "generate_clip_embeddings", 
    "generate_xl_embeddings",
    "train_color_model",
    "generate_color_histograms",
    "generate_edge_maps",
    "IPAdapterGenerator",
    "IPAdapterXLGenerator",
    "load_trained_color_head",
    "get_color_embedding"
]
