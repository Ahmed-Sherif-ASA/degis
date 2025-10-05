"""
DEGIS - Disentangled Embeddings Guided Image Synthesis

A well-structured package for training and inference of disentangled
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

from .shared.config import *
from .data.dataset import UnifiedImageDataset
from .training.batch_embeddings import generate_clip_embeddings, generate_xl_embeddings
from .shared.image_features import generate_color_histograms, generate_edge_maps
from .training import train_color_model
from .inference import IPAdapterGenerator, IPAdapterXLGenerator, load_trained_color_head, get_color_embedding

__all__ = [
    "UnifiedImageDataset",
    "generate_clip_embeddings", 
    "generate_xl_embeddings",
    "generate_color_histograms",
    "generate_edge_maps",
    "train_color_model",
    "IPAdapterGenerator",
    "IPAdapterXLGenerator",
    "load_trained_color_head",
    "get_color_embedding"
]
