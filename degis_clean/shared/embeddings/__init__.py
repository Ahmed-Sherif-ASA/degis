"""
Shared embeddings for DEGIS Clean.

Contains embedding generation functionality used by both training and inference.
"""

from .embeddings import generate_clip_embeddings, generate_xl_embeddings
from .clip_embeddings import *
from .clip_embeddings_xl_hf import *

__all__ = [
    'generate_clip_embeddings',
    'generate_xl_embeddings'
]
