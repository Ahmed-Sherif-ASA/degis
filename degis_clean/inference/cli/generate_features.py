#!/usr/bin/env python3
"""
CLI tool for generating visual features (histograms, edge maps).
"""

import argparse
import pandas as pd
from torch.utils.data import DataLoader
from multiprocessing import cpu_count

from ...training.features import (
    generate_color_histograms_batch,
    generate_edge_maps,
    generate_rgb_histograms,
    generate_lab_histograms,
    generate_hcl_histograms,
)
from ...data.dataset import UnifiedImageDataset
from ...shared.config import CSV_PATH, BATCH_SIZE, COLOR_HIST_PATH_RGB, COLOR_HIST_PATH_LAB_514, COLOR_HIST_PATH_HCL_514, EDGE_MAPS_PATH


def main():
    parser = argparse.ArgumentParser(description="Generate visual features from images")
    parser.add_argument("--csv-path", default=CSV_PATH, help="Path to CSV file with image paths")
    parser.add_argument("--type", choices=["histograms", "edges", "all"], default="all", help="Type of features to generate")
    parser.add_argument("--color-space", choices=["rgb", "lab", "hcl", "all"], default="all", help="Color space for histograms")
    parser.add_argument("--bins", type=int, default=8, help="Number of histogram bins per dimension")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for processing")
    parser.add_argument("--num-workers", type=int, help="Number of worker processes")
    parser.add_argument("--force-recompute", action="store_true", help="Force recomputation even if output exists")
    
    # Output paths
    parser.add_argument("--rgb-path", default=COLOR_HIST_PATH_RGB, help="Path for RGB histograms")
    parser.add_argument("--lab-path", default=COLOR_HIST_PATH_LAB_514, help="Path for LAB histograms")
    parser.add_argument("--hcl-path", default=COLOR_HIST_PATH_HCL_514, help="Path for HCL histograms")
    parser.add_argument("--edge-path", default=EDGE_MAPS_PATH, help="Path for edge maps")
    
    args = parser.parse_args()
    
    # Set default num_workers
    if args.num_workers is None:
        num_cpu = cpu_count()
        args.num_workers = min(32, max(8, num_cpu // 8))
    
    print(f"Generating features: {args.type}")
    print(f"CSV: {args.csv_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Workers: {args.num_workers}")
    
    # Load dataset
    df = pd.read_csv(args.csv_path)
    dataset = UnifiedImageDataset(
        df.rename(columns={"local_path": "file_path"}),
        mode="file_df"
    )
    
    # Create loader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=6,
    )
    
    # Generate features
    if args.type in ["histograms", "all"]:
        print("\n=== Generating Histograms ===")
        
        # Determine color spaces to generate
        if args.color_space == "all":
            color_spaces = ["rgb", "lab", "hcl"]
        else:
            color_spaces = [args.color_space]
        
        # Prepare output paths
        hist_paths = {}
        if "rgb" in color_spaces:
            hist_paths["rgb"] = args.rgb_path
        if "lab" in color_spaces:
            hist_paths["lab"] = args.lab_path
        if "hcl" in color_spaces:
            hist_paths["hcl"] = args.hcl_path
        
        # Generate histograms
        generate_color_histograms_batch(
            loader=loader,
            output_paths=hist_paths,
            bins=args.bins,
            color_spaces=color_spaces,
            force_recompute=args.force_recompute
        )
    
    if args.type in ["edges", "all"]:
        print("\n=== Generating Edge Maps ===")
        
        generate_edge_maps(
            loader=loader,
            output_path=args.edge_path,
            force_recompute=args.force_recompute
        )
    
    print("\n✓ Feature generation complete!")


if __name__ == "__main__":
    main()
