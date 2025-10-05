#!/usr/bin/env python3
"""
CLI tool for running comprehensive final training experiments.
Runs all combinations of embedding sizes, datasets, and color histograms.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def get_dataset_paths(dataset_name):
    """Get dataset-specific paths for the given dataset name."""
    DATA_DIR = "/data/thesis/data"
    MODELS_DIR = "/data/thesis/models"
    
    return {
        'edge_maps_path': f"{DATA_DIR}/{dataset_name}_edge_maps.npy",
        'csv_path': f"/data/thesis/{dataset_name}_manifest.csv",
        'embeddings_path': f"{DATA_DIR}/{dataset_name}_clip_embeddings.npy",
        'embeddings_target_path': f"{MODELS_DIR}/{dataset_name}_embeddings.npy",
        'hf_xl_embeddings_target_path': f"{MODELS_DIR}/hf_xl_{dataset_name}_embeddings.npy",
        'color_hist_path_rgb': f"{DATA_DIR}/{dataset_name}_color_histograms_rgb_512.npy",
        'color_hist_path_lab_514': f"{DATA_DIR}/{dataset_name}_color_histograms_lab_514.npy",
        'color_hist_path_hcl_514': f"{DATA_DIR}/{dataset_name}_color_histograms_hcl_514.npy",
    }


def run_training_experiment(emb_kind, csv_name, hist_kind, base_params, verbose=True):
    """Run a single training experiment with the given parameters."""
    
    # Create the run name in the format: {csv_name}_{emb_kind}_{hist_kind}_tk{top_k}_b{batch_size}
    run_name = f"{csv_name}_{emb_kind}_{hist_kind}_tk{base_params['top_k']}_b{base_params['batch_size']}"
    
    # Get dataset-specific paths
    dataset_paths = get_dataset_paths(csv_name)
    
    # Determine embeddings path based on emb_kind
    if emb_kind == "xl":
        embeddings_path = dataset_paths['hf_xl_embeddings_target_path']
    else:
        embeddings_path = dataset_paths['embeddings_target_path']
    
    # Determine histograms path based on hist_kind
    if hist_kind == "rgb512":
        histograms_path = dataset_paths['color_hist_path_rgb']
    elif hist_kind == "lab514":
        histograms_path = dataset_paths['color_hist_path_lab_514']
    else:  # hcl514
        histograms_path = dataset_paths['color_hist_path_hcl_514']
    
    # Build the command
    cmd = [
        "python3", "-m", "degis.cli.train_color",
        "--embeddings-path", embeddings_path,
        "--histograms-path", histograms_path,
        "--emb-kind", emb_kind,
        "--hist-kind", hist_kind,
        "--epochs", str(base_params['epochs']),
        "--batch-size", str(base_params['batch_size']),
        "--val-batch-size", str(base_params['val_batch_size']),
        "--lr", str(base_params['lr']),
        "--weight-decay", str(base_params['weight_decay']),
        "--blur", str(base_params['blur']),
        "--lambda-ortho", str(base_params['lambda_ortho']),
        "--lambda-consistency", str(base_params['lambda_consistency']),
        "--top-k", str(base_params['top_k']),
        "--device", base_params['device']
    ]
    
    if base_params['weighting']:
        cmd.append("--weighting")
    
    # Add output directory to save to evaluation_runs
    output_dir = f"evaluation_runs/{run_name}"
    cmd.extend(["--output-dir", output_dir])
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Starting experiment: {run_name}")
        print(f"Embeddings: {embeddings_path}")
        print(f"Histograms: {histograms_path}")
        print(f"Output dir: {output_dir}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*80}")
    
    # Run the command
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        end_time = time.time()
        duration = end_time - start_time
        
        if verbose:
            print(f"\nExperiment {run_name} completed successfully in {duration:.2f} seconds")
        return True, duration
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        if verbose:
            print(f"\nExperiment {run_name} failed after {duration:.2f} seconds")
            print(f"Error: {e}")
        return False, duration


def main():
    """Run all training experiments."""
    
    parser = argparse.ArgumentParser(description="Run comprehensive final training experiments")
    
    # Experiment selection
    parser.add_argument("--emb-kinds", nargs="+", default=["xl", "base"], 
                       help="Embedding kinds to test (default: xl base)")
    parser.add_argument("--csv-names", nargs="+", default=["laion_5m", "coco"], 
                       help="Dataset names to test (default: laion_5m coco)")
    parser.add_argument("--hist-kinds", nargs="+", default=["rgb512", "hcl514", "lab514"], 
                       help="Color histogram kinds to test (default: rgb512 hcl514 lab514)")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4096, help="Training batch size")
    parser.add_argument("--val-batch-size", type=int, default=8192, help="Validation batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay")
    parser.add_argument("--blur", type=float, default=0.01, help="Sinkhorn blur parameter")
    parser.add_argument("--lambda-ortho", type=float, default=0.2, help="Orthogonality loss weight")
    parser.add_argument("--lambda-consistency", type=float, default=0.1, help="Consistency loss weight")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k filtering for histograms")
    parser.add_argument("--weighting", action="store_true", help="Use rarity weighting")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    
    # Control options
    parser.add_argument("--test", action="store_true", help="Run test mode with minimal parameters")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--delay", type=float, default=5.0, help="Delay between experiments in seconds")
    
    args = parser.parse_args()
    
    # Test mode adjustments
    if args.test:
        args.epochs = 2
        args.batch_size = 4096
        args.val_batch_size = 8192
        args.delay = 1.0
        print("Running in TEST MODE with minimal parameters")
    
    # Base parameters
    base_params = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'val_batch_size': args.val_batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'blur': args.blur,
        'lambda_ortho': args.lambda_ortho,
        'lambda_consistency': args.lambda_consistency,
        'top_k': args.top_k,
        'weighting': args.weighting,
        'device': args.device
    }
    
    # Generate all combinations
    experiments = []
    for emb_kind in args.emb_kinds:
        for csv_name in args.csv_names:
            for hist_kind in args.hist_kinds:
                experiments.append((emb_kind, csv_name, hist_kind))
    
    verbose = not args.quiet
    
    if verbose:
        print(f"Starting final training with {len(experiments)} experiments")
        print(f"Embedding kinds: {args.emb_kinds}")
        print(f"Datasets: {args.csv_names}")
        print(f"Color histograms: {args.hist_kinds}")
        print(f"Training parameters:")
        for key, value in base_params.items():
            print(f"  {key}: {value}")
        print(f"\nExperiment combinations:")
        for i, (emb, csv, hist) in enumerate(experiments, 1):
            print(f"  {i:2d}. {csv}_{emb}_{hist}_tk{base_params['top_k']}_b{base_params['batch_size']}")
    
    # Create evaluation_runs directory
    os.makedirs("evaluation_runs", exist_ok=True)
    
    # Track results
    results = []
    total_start_time = time.time()
    
    # Run each experiment
    for i, (emb_kind, csv_name, hist_kind) in enumerate(experiments, 1):
        if verbose:
            print(f"\n{'='*100}")
            print(f"EXPERIMENT {i}/{len(experiments)}")
            print(f"{'='*100}")
        
        success, duration = run_training_experiment(
            emb_kind, csv_name, hist_kind, base_params, verbose=verbose
        )
        
        results.append({
            'experiment': f"{csv_name}_{emb_kind}_{hist_kind}_tk{base_params['top_k']}_b{base_params['batch_size']}",
            'emb_kind': emb_kind,
            'csv_name': csv_name,
            'hist_kind': hist_kind,
            'success': success,
            'duration': duration
        })
        
        # Brief pause between experiments
        if i < len(experiments) and args.delay > 0:
            if verbose:
                print(f"\nWaiting {args.delay} seconds before next experiment...")
            time.sleep(args.delay)
    
    # Final summary
    total_duration = time.time() - total_start_time
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"\n{'='*100}")
    print(f"FINAL SUMMARY")
    print(f"{'='*100}")
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_duration:.2f} seconds ({total_duration/3600:.2f} hours)")
    print(f"Average time per experiment: {total_duration/len(results):.2f} seconds")
    
    if verbose:
        print(f"\nDetailed results:")
        for r in results:
            status = "PASS" if r['success'] else "FAIL"
            print(f"  {status} {r['experiment']} ({r['duration']:.2f}s)")
    
    if failed > 0:
        print(f"\nFailed experiments:")
        for r in results:
            if not r['success']:
                print(f"  - {r['experiment']}")
    
    print(f"\nAll results saved in: evaluation_runs/")
    print(f"Each experiment has its own directory with:")
    print(f"  - best_color_head_tmp.pth")
    print(f"  - best_rest_head_tmp.pth") 
    print(f"  - best_summary.json")
    print(f"  - metrics.csv")
    print(f"  - run_meta.json")
    
    # Exit with error code if any experiments failed
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()