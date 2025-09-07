#!/usr/bin/env python3
"""
CLI tool for training color disentanglement models.
"""

import argparse
import os

from ..core.training import train_color_model
from ..config import EMBEDDINGS_TARGET_PATH, HF_XL_EMBEDDINGS_TARGET_PATH, COLOR_HIST_PATH_HCL_514, COLOR_HIST_PATH_RGB_512, COLOR_HIST_PATH_LAB_514


def main():
    parser = argparse.ArgumentParser(description="Train color disentanglement model")
    
    # Data paths
    parser.add_argument("--embeddings-path", help="Path to CLIP embeddings")
    parser.add_argument("--histograms-path", help="Path to color histograms (auto-determined from --hist-kind if not provided)")
    parser.add_argument("--output-dir", help="Directory to save model and logs")
    
    # Model configuration
    parser.add_argument("--emb-kind", choices=["base", "xl"], help="Type of embeddings (base=EMBEDDINGS_TARGET_PATH, xl=HF_XL_EMBEDDINGS_TARGET_PATH)")
    parser.add_argument("--hist-kind", choices=["rgb512", "lab514", "hcl514"], default="hcl514", help="Type of histograms")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--val-batch-size", type=int, default=256, help="Validation batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--blur", type=float, default=0.05, help="Sinkhorn blur parameter")
    parser.add_argument("--lambda-ortho", type=float, default=0.1, help="Orthogonality loss weight")
    parser.add_argument("--lambda-consistency", type=float, default=0.1, help="Consistency loss weight (color + rest ≈ original)")
    parser.add_argument("--top-k", type=int, help="Top-k filtering for histograms")
    parser.add_argument("--weighting", action="store_true", help="Use rarity weighting")
    
    # System
    parser.add_argument("--device", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Determine embeddings path based on emb_kind if not provided
    if args.embeddings_path is None:
        if args.emb_kind == "xl":
            args.embeddings_path = HF_XL_EMBEDDINGS_TARGET_PATH
        else:
            args.embeddings_path = EMBEDDINGS_TARGET_PATH
    
    # Determine histograms path based on hist_kind if not provided
    if args.histograms_path is None:
        if args.hist_kind == "rgb512":
            args.histograms_path = COLOR_HIST_PATH_RGB_512
        elif args.hist_kind == "lab514":
            args.histograms_path = COLOR_HIST_PATH_LAB_514
        else:  # hcl514
            args.histograms_path = COLOR_HIST_PATH_HCL_514
    
    print("Training color disentanglement model...")
    print(f"Embeddings: {args.embeddings_path}")
    print(f"Histograms: {args.histograms_path}")
    print(f"Hist kind: {args.hist_kind}")
    print(f"Emb kind: {args.emb_kind or 'default (base)'}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    
    # Train model
    results = train_color_model(
        embeddings_path=args.embeddings_path,
        histograms_path=args.histograms_path,
        output_dir=args.output_dir,
        hist_kind=args.hist_kind,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        blur=args.blur,
        lambda_ortho=args.lambda_ortho,
        lambda_consistency=args.lambda_consistency,
        top_k=args.top_k,
        weighting=args.weighting,
        device=args.device,
    )
    
    print(f"\n✓ Training complete!")
    print(f"Output directory: {results['output_dir']}")


if __name__ == "__main__":
    main()
