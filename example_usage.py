#!/usr/bin/env python3
"""
Example usage of the DEGIS package.

This demonstrates how to use the package both as a Python library
and via CLI commands.
"""

import pandas as pd
from torch.utils.data import DataLoader
from multiprocessing import cpu_count

# Import the package
import degis
from degis.data.dataset import UnifiedImageDataset
from degis.config import CSV_PATH, BATCH_SIZE


def example_python_api():
    """Example of using DEGIS as a Python library."""
    print("=== Python API Example ===")
    
    # Load dataset
    df = pd.read_csv(CSV_PATH)
    dataset = UnifiedImageDataset(
        df.rename(columns={"local_path": "file_path"}),
        mode="file_df"
    )
    
    # Create loader
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=min(32, max(8, cpu_count() // 8)),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=6,
    )
    
    print(f"Dataset loaded: {len(dataset)} images")
    
    # Example 1: Generate embeddings
    print("\n1. Generating CLIP embeddings...")
    embeddings = degis.generate_xl_embeddings(
        csv_path=CSV_PATH,
        output_path="embeddings_example.npy",
        batch_size=BATCH_SIZE,
        force_recompute=True
    )
    print(f"✓ Generated embeddings: {embeddings.shape}")
    
    # Example 2: Generate features
    print("\n2. Generating color histograms...")
    histograms = degis.generate_hcl_histograms(
        loader=loader,
        output_path="histograms_example.npy",
        bins=8,
        force_recompute=True
    )
    print(f"✓ Generated histograms: {histograms.shape}")
    
    # Example 3: Train a model
    print("\n3. Training color model...")
    results = degis.train_color_model(
        embeddings_path="embeddings_example.npy",
        histograms_path="histograms_example.npy",
        output_dir="example_run",
        hist_kind="hcl514",
        epochs=5,  # Short run for example
        batch_size=64,
    )
    print(f"✓ Training complete: {results['output_dir']}")


def example_cli_commands():
    """Example CLI commands that users can run."""
    print("\n=== CLI Commands Example ===")
    
    commands = [
        "# Generate XL embeddings",
        "degis-embeddings --model xl --batch-size 256 --force-recompute",
        "",
        "# Generate all features",
        "degis-features --type all --color-space all --bins 8 --force-recompute",
        "",
        "# Train color model",
        "degis-train-color --hist-kind hcl514 --epochs 200 --batch-size 128",
        "",
        "# Train edge model", 
        "degis-train-edge --epochs 200 --batch-size 512",
    ]
    
    for cmd in commands:
        print(cmd)


if __name__ == "__main__":
    print("DEGIS Package Example")
    print("=" * 50)
    
    # Show CLI examples
    example_cli_commands()
    
    # Uncomment to run Python API example
    # example_python_api()
    
    print("\n✓ Example complete!")
    print("\nTo install the package:")
    print("pip install -e .")
    print("\nTo use CLI commands:")
    print("degis-embeddings --help")
    print("degis-features --help")
    print("degis-train-color --help")
    print("degis-train-edge --help")
