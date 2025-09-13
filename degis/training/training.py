"""
Core training functionality.
"""

import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from ..data.dataset import PrecompClipColorDataset, PrecompClipEdgeDataset
from .models.color_heads import ColorHead, RestHead
from .models.train_color import train_color_disentanglement
from ..shared.utils.logger import MetricsLogger, short_git_hash


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
    lambda_consistency=0.1,  # New: consistency loss weight
    top_k=None,
    weighting=False,
    device=None,
    csv_name=None,  # New: dataset name for run naming
    emb_kind=None,  # New: embedding kind for run naming
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
        lambda_consistency: Consistency loss weight (color + rest ≈ original)
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
        
        # Use new naming convention if csv_name and emb_kind are provided
        if csv_name and emb_kind:
            run_name = f"{csv_name}_{emb_kind}_{hist_kind}_tk{top_k or 'all'}_b{batch_size}"
            # Use evaluation_runs directory for new naming convention
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            output_dir = os.path.join(project_root, "evaluation_runs", f"{run_name}-{stamp}")
        else:
            # Fallback to old naming convention
            run_name = f"color_{hist_kind}_tk{top_k or 'all'}_b{batch_size}"
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            output_dir = os.path.join(project_root, "runs", f"{run_name}-{stamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Also create the rest directory (needed for saving rest_head)
    rest_output_dir = output_dir.replace("color_", "rest_")
    os.makedirs(rest_output_dir, exist_ok=True)
    
    print("Run dir:", output_dir)
    print("Rest dir:", rest_output_dir)
    
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
        lambda_ortho=lambda_ortho, lambda_consistency=lambda_consistency, lambda_leak=0.25,
        epochs=epochs, batch_size=batch_size, val_batch_size=val_batch_size,
        optimizer="AdamW", lr=lr, weight_decay=weight_decay,
        device=str(device), gpu_name=gpu_name, seed=42,
        param_count_color=MetricsLogger.param_count(color_head),
        param_count_rest=MetricsLogger.param_count(rest_head),
        git_commit=short_git_hash(),
    )
    
    # Create hparams.yaml
    try:
        import yaml
        # Extract dataset name and embedding kind from paths
        dataset_name = "unknown"
        if "laion_5m" in embeddings_path:
            dataset_name = "laion_5m"
        elif "coco" in embeddings_path:
            dataset_name = "coco"
        
        # Determine encoder ID based on embedding kind or path
        if 'emb_kind' in locals() and emb_kind == "xl":
            encoder_id = "ViT-bigG/14"
        elif 'emb_kind' in locals() and emb_kind == "base":
            encoder_id = "ViT-H/14"
        elif "hf_xl" in embeddings_path:
            encoder_id = "ViT-bigG/14"
        else:
            encoder_id = "ViT-H/14"
        hparams = {
            "encoder_id": encoder_id,
            "dataset": dataset_name,
            "hist_kind": hist_kind,
            "epochs": epochs,
            "batch_size": batch_size,
            "val_batch_size": val_batch_size,
            "optimizer": "AdamW",
            "betas": [0.9, 0.999],  # AdamW default
            "weight_decay": weight_decay,
            "scheduler": None,
            "scheduler_params": {},
            "top_k": top_k,
            "regularizers": {
                "blur": blur,
                "lambda_ortho": lambda_ortho,
                "lambda_consistency": lambda_consistency,
                "lambda_leak": 0.25,
            },
            "precision": "fp16" if str(device) == "cuda" else "fp32",
            "seed": 42,
            "device": str(device),
        }
        with open(os.path.join(output_dir, "hparams.yaml"), "w") as f:
            yaml.safe_dump(hparams, f, sort_keys=False)
    except ImportError:
        print("Warning: PyYAML not available, skipping hparams.yaml creation")
    except Exception as e:
        print(f"Warning: Failed to create hparams.yaml: {e}")
    
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
        lambda_consistency=lambda_consistency,
        lr=lr, wd=weight_decay,
        save_prefix=os.path.join(output_dir, "best_color_head_tmp"),
        logger=logger,
    )
    
    # Generate training curves
    from ..shared.utils.visualization import plot_training_curves
    plot_training_curves(
        metrics_csv_path=os.path.join(output_dir, "metrics.csv"),
        output_dir=output_dir,
        dataset_name=dataset_name,
        hist_kind=hist_kind
    )
    
    return {
        "output_dir": output_dir,
        "color_head": color_head,
        "rest_head": rest_head,
        "logger": logger,
    }