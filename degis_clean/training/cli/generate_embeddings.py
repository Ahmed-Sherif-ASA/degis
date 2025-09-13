#!/usr/bin/env python3
"""
CLI tool for generating CLIP embeddings.
"""

import argparse
import pandas as pd
from torch.utils.data import DataLoader
from multiprocessing import cpu_count

from ...training.batch_embeddings import generate_clip_embeddings, generate_xl_embeddings
from ...data.dataset import UnifiedImageDataset
from ...shared.config import CSV_PATH, BATCH_SIZE, EMBEDDINGS_TARGET_PATH, HF_XL_EMBEDDINGS_TARGET_PATH


def main():
    parser = argparse.ArgumentParser(description="Generate CLIP embeddings from images")
    parser.add_argument("--csv-path", default=CSV_PATH, help="Path to CSV file with image paths")
    parser.add_argument("--output-path", help="Path to save embeddings")
    parser.add_argument("--model", choices=["base", "xl"], default="xl", help="CLIP model to use")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for processing")
    parser.add_argument("--num-workers", type=int, help="Number of worker processes")
    parser.add_argument("--force-recompute", action="store_true", help="Force recomputation even if output exists")
    
    args = parser.parse_args()
    
    # Set default output path based on model
    if args.output_path is None:
        if args.model == "xl":
            args.output_path = HF_XL_EMBEDDINGS_TARGET_PATH  # Use HF XL by default
        else:
            args.output_path = EMBEDDINGS_TARGET_PATH
    
    # Set default num_workers
    if args.num_workers is None:
        num_cpu = cpu_count()
        args.num_workers = min(32, max(8, num_cpu // 8))
    
    print(f"Generating {args.model.upper()} embeddings...")
    print(f"CSV: {args.csv_path}")
    print(f"Output: {args.output_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Workers: {args.num_workers}")
    
    # Generate embeddings
    if args.model == "xl":
        embeddings = generate_xl_embeddings(
            csv_path=args.csv_path,
            output_path=args.output_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            force_recompute=args.force_recompute
        )
    else:
        embeddings = generate_clip_embeddings(
            csv_path=args.csv_path,
            output_path=args.output_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            force_recompute=args.force_recompute
        )
    
    print(f"✓ Generated embeddings with shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
