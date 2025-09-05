"""
Core features generation functionality.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..features.color_histograms import (
    generate_color_histograms,
    fast_rgb_histogram,
    fast_lab_histogram,
    fast_hcl_histogram,
)
from ..features.edge_maps import generate_edge_maps


def generate_color_histograms_batch(
    loader,
    output_paths,
    bins=8,
    color_spaces=None,
    force_recompute=False
):
    """
    Generate color histograms for multiple color spaces in batch.
    
    Args:
        loader: DataLoader with images
        output_paths: Dict mapping color space to output path
        bins: Number of histogram bins per dimension
        color_spaces: List of color spaces to generate ("rgb", "lab", "hcl")
        force_recompute: Whether to recompute if output exists
    
    Returns:
        dict: Generated histograms for each color space
    """
    if color_spaces is None:
        color_spaces = ["rgb", "lab", "hcl"]
    
    num_images = len(loader.dataset)
    results = {}
    
    # Initialize result arrays
    for space in color_spaces:
        if space == "rgb":
            dim = bins ** 3
        else:  # lab, hcl
            dim = bins ** 3 + 2  # includes black/white bins
        results[space] = np.zeros((num_images, dim), dtype=np.float32)
    
    idx = 0
    for batch, _ in tqdm(loader, desc="Computing histograms", total=len(loader), unit="batch"):
        B = batch.shape[0]
        
        # Process each image in the batch
        for i in range(B):
            img = batch[i]
            
            if "rgb" in color_spaces:
                results["rgb"][idx + i] = fast_rgb_histogram(img, bins=bins)
            
            if "lab" in color_spaces:
                results["lab"][idx + i] = fast_lab_histogram(img, bins=bins)
            
            if "hcl" in color_spaces:
                results["hcl"][idx + i] = fast_hcl_histogram(img, bins=bins)
        
        idx += B
    
    # Save results
    for space, histograms in results.items():
        if space in output_paths:
            np.save(output_paths[space], histograms)
            print(f"✔ Saved {space} histograms to {output_paths[space]} (shape={histograms.shape})")
    
    return results


def generate_all_features(
    loader,
    output_paths,
    features=None,
    bins=8,
    force_recompute=False,
    **kwargs
):
    """
    Generate all supported features (histograms, edge maps) in batch.
    
    Args:
        loader: DataLoader with images
        output_paths: Dict mapping feature type to output path
        features: List of features to generate ("histograms", "edges")
        bins: Number of histogram bins per dimension
        force_recompute: Whether to recompute if output exists
        **kwargs: Additional arguments for feature generation
    
    Returns:
        dict: Generated features for each type
    """
    if features is None:
        features = ["histograms", "edges"]
    
    results = {}
    
    if "histograms" in features:
        hist_paths = output_paths.get("histograms", {})
        if hist_paths:
            results["histograms"] = generate_color_histograms_batch(
                loader, hist_paths, bins=bins, force_recompute=force_recompute
            )
    
    if "edges" in features:
        edge_path = output_paths.get("edges")
        if edge_path:
            results["edges"] = generate_edge_maps(
                loader, edge_path, force_recompute=force_recompute, **kwargs
            )
    
    return results


# Convenience functions for individual feature types
def generate_rgb_histograms(loader, output_path, bins=8, force_recompute=False):
    """Generate RGB histograms."""
    return generate_color_histograms_batch(
        loader, {"rgb": output_path}, bins=bins, 
        color_spaces=["rgb"], force_recompute=force_recompute
    )["rgb"]


def generate_lab_histograms(loader, output_path, bins=8, force_recompute=False):
    """Generate LAB histograms."""
    return generate_color_histograms_batch(
        loader, {"lab": output_path}, bins=bins,
        color_spaces=["lab"], force_recompute=force_recompute
    )["lab"]


def generate_hcl_histograms(loader, output_path, bins=8, force_recompute=False):
    """Generate HCL histograms."""
    return generate_color_histograms_batch(
        loader, {"hcl": output_path}, bins=bins,
        color_spaces=["hcl"], force_recompute=force_recompute
    )["hcl"]
