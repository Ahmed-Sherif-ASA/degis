"""
Core embeddings generation functionality.
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from tqdm import tqdm

from ..data.dataset import UnifiedImageDataset
from .clip_embeddings import generate_embeddings_fp16
from .clip_embeddings_xl_hf import generate_embeddings_xl as generate_embeddings_xl_hf, preprocess_xl
from .config import CSV_PATH, BATCH_SIZE, EMBEDDINGS_TARGET_PATH, HF_XL_EMBEDDINGS_TARGET_PATH
from .utils.auto_setup import setup_environment


def generate_clip_embeddings(
    csv_path=None,
    output_path=None,
    batch_size=None,
    num_workers=None,
    model="base",
    force_recompute=False,
    **kwargs
):
    """
    Generate CLIP embeddings from images.
    
    Args:
        csv_path: Path to CSV file with image paths
        output_path: Path to save embeddings
        batch_size: Batch size for processing
        num_workers: Number of worker processes
        model: Model type ("base" or "xl")
        force_recompute: Whether to recompute if output exists
        **kwargs: Additional arguments passed to dataset
    
    Returns:
        numpy.ndarray: Generated embeddings
    """
    # Setup environment (detects server/local, sets cache dirs, etc.)
    setup_environment()
    
    # Use defaults from config if not provided
    csv_path = csv_path or CSV_PATH
    output_path = output_path or EMBEDDINGS_TARGET_PATH
    batch_size = batch_size or BATCH_SIZE
    
    if num_workers is None:
        num_cpu = cpu_count()
        num_workers = min(32, max(8, num_cpu // 8))
    
    # Load dataset
    df = pd.read_csv(csv_path)
    assert "local_path" in df.columns or "file_path" in df.columns, "CSV must have either 'local_path' or 'file_path' column!"
    
    # Create dataset with appropriate transform
    if model == "xl":
        dataset = UnifiedImageDataset(
            df.rename(columns={"local_path": "file_path"}),
            mode="file_df",
            transform=preprocess_xl,
        )
    else:
        dataset = UnifiedImageDataset(
            df.rename(columns={"local_path": "file_path"}),
            mode="file_df",
            **kwargs
        )
    
    # Create loader
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=6,
    )
    
    try:
        loader_kwargs["pin_memory_device"] = "cuda"
    except TypeError:
        pass  # older torch doesn't have this arg
    
    loader = DataLoader(dataset, **loader_kwargs)
    
    # Generate embeddings
    if model == "xl":
        embeddings = generate_embeddings_xl_hf(
            loader,
            output_path,
            force_recompute=force_recompute
        )
    else:
        embeddings = generate_embeddings_fp16(
            loader,
            output_path,
            force_recompute=force_recompute
        )
    
    print(f"Saved: {output_path}, shape: {embeddings.shape}")
    return embeddings


def generate_xl_embeddings(
    csv_path=None,
    output_path=None,
    batch_size=None,
    num_workers=None,
    force_recompute=False,
    **kwargs
):
    """
    Generate XL CLIP embeddings from images.
    
    Args:
        csv_path: Path to CSV file with image paths
        output_path: Path to save embeddings
        batch_size: Batch size for processing
        num_workers: Number of worker processes
        force_recompute: Whether to recompute if output exists
        **kwargs: Additional arguments passed to dataset
    
    Returns:
        numpy.ndarray: Generated embeddings
    """
    # Setup environment (detects server/local, sets cache dirs, etc.)
    setup_environment()
    
    return generate_clip_embeddings(
        csv_path=csv_path,
        output_path=output_path,
        batch_size=batch_size,
        num_workers=num_workers,
        model="xl",
        force_recompute=force_recompute,
        **kwargs
    )
