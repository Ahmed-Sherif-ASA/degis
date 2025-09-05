"""
Core training functionality.
"""

import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from ..data.dataset import PrecompClipColorDataset, PrecompClipEdgeDataset
from ..models.color_heads import ColorHead, RestHead
from ..models.edge_heads import EdgeHead
from ..models.train_color import train_color_disentanglement
from ..models.train_edge import train_edge_decoder
from ..utils.logger import MetricsLogger, short_git_hash


def train_color_model(
    embeddings_path,
    histograms_path,
    output_dir=None,
    hist_kind="hcl514",
    epochs=200,
    batch_size=128,
    val_batch_size=256,
    lr=1e-3,
    weight_decay=1e-2,
    blur=0.05,
    lambda_ortho=0.1,
    top_k=None,
    weighting=False,
    device=None,
    **kwargs
):
    """
    Train a color disentanglement model.
    
    Args:
        embeddings_path: Path to CLIP embeddings
        histograms_path: Path to color histograms
        output_dir: Directory to save model and logs
        hist_kind: Type of histograms ("rgb512", "lab514", "hcl514")
        epochs: Number of training epochs
        batch_size: Training batch size
        val_batch_size: Validation batch size
        lr: Learning rate
        weight_decay: Weight decay
        blur: Sinkhorn blur parameter
        lambda_ortho: Orthogonality loss weight
        top_k: Top-k filtering for histograms
        weighting: Whether to use rarity weighting
        device: Device to use for training
        **kwargs: Additional arguments
    
    Returns:
        dict: Training results and metadata
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    if output_dir is None:
        import time
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_name = f"color_{hist_kind}_tk{top_k or 'all'}_b{batch_size}"
        output_dir = os.path.join("runs", f"{run_name}-{stamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    print("Run dir:", output_dir)
    
    # Load data
    emb = np.load(embeddings_path).astype(np.float32, copy=False)
    hist = np.load(histograms_path).astype(np.float32, copy=False)
    assert emb.shape[0] == hist.shape[0], "Embeddings and histograms must share N"
    
    N, clip_dim = emb.shape
    hist_dim = hist.shape[1]
    print(f"Loaded → emb: {emb.shape} | hist: {hist.shape} | kind={hist_kind}")
    
    # Split data
    idx_train, idx_val = train_test_split(np.arange(N), test_size=0.2, random_state=42)
    
    # Create datasets and loaders
    train_ds = PrecompClipColorDataset(idx_train, emb, hist)
    val_ds = PrecompClipColorDataset(idx_val, emb, hist)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=16, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=val_batch_size, shuffle=False,
        num_workers=16, pin_memory=True
    )
    
    # Optional rarity weighting
    weights_vec = None
    if weighting:
        counts = hist.sum(0).astype(np.float64) + 1e-6
        alpha = 0.5
        w = (1.0 / (counts ** alpha))
        w = (w / w.mean()).astype(np.float32)
        weights_vec = torch.tensor(w, device=device)
    
    # Create models
    color_head = ColorHead(clip_dim=clip_dim, hist_dim=hist_dim).to(device)
    rest_head = RestHead(clip_dim=clip_dim).to(device)
    
    # Setup logging
    logger = MetricsLogger(outdir=output_dir)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    logger.set_meta(
        run_name=f"color_{hist_kind}",
        hist_kind=hist_kind,
        top_k=top_k, blur=blur,
        lambda_ortho=lambda_ortho, lambda_leak=0.25,
        epochs=epochs, batch_size=batch_size, val_batch_size=val_batch_size,
        optimizer="AdamW", lr=lr, weight_decay=weight_decay,
        device=str(device), gpu_name=gpu_name, seed=42,
        param_count_color=MetricsLogger.param_count(color_head),
        param_count_rest=MetricsLogger.param_count(rest_head),
        git_commit=short_git_hash(),
    )
    
    # Train model
    train_color_disentanglement(
        train_loader=train_loader,
        val_loader=val_loader,
        color_head=color_head,
        rest_head=rest_head,
        device=device,
        num_epochs=epochs,
        lambda_ortho=lambda_ortho,
        top_k=top_k,
        use_weighting=bool(weighting),
        weights_vec=weights_vec,
        T_min=1.0,
        blur=blur,
        lr=lr, wd=weight_decay,
        save_prefix=os.path.join(output_dir, "best_color_head_tmp"),
        logger=logger,
    )
    
    return {
        "output_dir": output_dir,
        "color_head": color_head,
        "rest_head": rest_head,
        "logger": logger,
    }


def train_edge_model(
    embeddings_path,
    edge_maps_path,
    output_dir=None,
    epochs=200,
    batch_size=512,
    val_batch_size=1024,
    lr=1e-4,
    weight_decay=0.0,
    lambda_ortho=0.1,
    patience=10,
    device=None,
    **kwargs
):
    """
    Train an edge decoder model.
    
    Args:
        embeddings_path: Path to CLIP embeddings
        edge_maps_path: Path to edge maps
        output_dir: Directory to save model and logs
        epochs: Number of training epochs
        batch_size: Training batch size
        val_batch_size: Validation batch size
        lr: Learning rate
        weight_decay: Weight decay
        lambda_ortho: Orthogonality loss weight
        patience: Early stopping patience
        device: Device to use for training
        **kwargs: Additional arguments
    
    Returns:
        dict: Training results and metadata
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    if output_dir is None:
        import time
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_name = f"edge_b{batch_size}"
        output_dir = os.path.join("runs", f"{run_name}-{stamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    print("Run dir:", output_dir)
    
    # Load data
    emb = np.load(embeddings_path, mmap_mode="r")
    edges = np.load(edge_maps_path, mmap_mode="r")
    
    N, clip_dim = emb.shape
    edge_dim = edges.shape[1]
    print(f"Loaded → emb: {emb.shape} | edges: {edges.shape}")
    
    # Split data
    idx_train, idx_val = train_test_split(np.arange(N), test_size=0.2, random_state=42)
    
    # Create datasets and loaders
    train_ds = PrecompClipEdgeDataset(idx_train, emb, edges, normalize_edges=True)
    val_ds = PrecompClipEdgeDataset(idx_val, emb, edges, normalize_edges=True)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True,
        pin_memory_device="cuda", persistent_workers=True, prefetch_factor=6
    )
    val_loader = DataLoader(
        val_ds, batch_size=val_batch_size, shuffle=False,
        num_workers=8, pin_memory=True,
        pin_memory_device="cuda", persistent_workers=True, prefetch_factor=6
    )
    
    # Create models
    edge_head = EdgeHead(clip_dim=clip_dim, edge_dim=edge_dim).to(device)
    rest_head = RestHead(clip_dim=clip_dim).to(device)
    
    # Train model
    train_edge_decoder(
        train_loader=train_loader,
        val_loader=val_loader,
        edge_head=edge_head,
        rest_head=rest_head,
        device=device,
        num_epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        lambda_ortho=lambda_ortho,
        patience=patience,
        ckpt_path=os.path.join(output_dir, "best_edge_head.pth"),
    )
    
    return {
        "output_dir": output_dir,
        "edge_head": edge_head,
        "rest_head": rest_head,
    }
