#!/usr/bin/env python3
"""
CLI tool for training edge decoder models.
"""

import argparse

from ..core.training import train_edge_model
from ..config import HF_XL_EMBEDDINGS_TARGET_PATH, EDGE_MAPS_PATH


def main():
    parser = argparse.ArgumentParser(description="Train edge decoder model")
    
    # Data paths
    parser.add_argument("--embeddings-path", default=HF_XL_EMBEDDINGS_TARGET_PATH, help="Path to CLIP embeddings")
    parser.add_argument("--edge-maps-path", default=EDGE_MAPS_PATH, help="Path to edge maps")
    parser.add_argument("--output-dir", help="Directory to save model and logs")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Training batch size")
    parser.add_argument("--val-batch-size", type=int, default=1024, help="Validation batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--lambda-ortho", type=float, default=0.1, help="Orthogonality loss weight")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    
    # System
    parser.add_argument("--device", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    print("Training edge decoder model...")
    print(f"Embeddings: {args.embeddings_path}")
    print(f"Edge maps: {args.edge_maps_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    
    # Train model
    results = train_edge_model(
        embeddings_path=args.embeddings_path,
        edge_maps_path=args.edge_maps_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_ortho=args.lambda_ortho,
        patience=args.patience,
        device=args.device,
    )
    
    print(f"\n✓ Training complete!")
    print(f"Output directory: {results['output_dir']}")


if __name__ == "__main__":
    main()
