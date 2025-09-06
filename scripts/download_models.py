#!/usr/bin/env python3
"""
Script to download and set up external model checkpoints.
"""
import os
import sys
import argparse
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from degis.models_config.models import MODEL_REGISTRY, set_model_path
from degis.config import MODEL_CACHE
from degis.utils.model_manager import get_model_manager

def download_ip_adapter_models():
    """Download IP-Adapter model checkpoints."""
    print("Downloading IP-Adapter models...")
    
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Error: huggingface_hub not installed. Install with: pip install huggingface_hub")
        return False
    
    models_dir = Path(os.getenv("DEGIS_MODELS_DIR", "./models"))
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Download IP-Adapter SD 1.5
    try:
        sd15_path = hf_hub_download(
            repo_id="h94/IP-Adapter",
            filename="ip-adapter_sd15.bin",
            cache_dir=config["cache_dir"],
            local_dir=models_dir
        )
        set_model_path("ip_adapter_sd15", str(sd15_path))
        print(f"✓ Downloaded IP-Adapter SD 1.5: {sd15_path}")
    except Exception as e:
        print(f"✗ Failed to download IP-Adapter SD 1.5: {e}")
        return False
    
    # Download IP-Adapter SDXL
    try:
        sdxl_path = hf_hub_download(
            repo_id="h94/IP-Adapter",
            filename="ip-adapter_sdxl.bin",
            cache_dir=config["cache_dir"],
            local_dir=models_dir
        )
        set_model_path("ip_adapter_sdxl", str(sdxl_path))
        print(f"✓ Downloaded IP-Adapter SDXL: {sdxl_path}")
    except Exception as e:
        print(f"✗ Failed to download IP-Adapter SDXL: {e}")
        return False
    
    return True

def download_all_models():
    """Download all external models."""
    print("Setting up model environment...")
    
    # Set up model manager
    manager = get_model_manager(
        cache_dir=MODEL_CACHE,
        models_dir=os.getenv("DEGIS_MODELS_DIR", "./models")
    )
    
    print(f"Cache directory: {MODEL_CACHE} (local to project)")
    print(f"Models directory: {os.getenv('DEGIS_MODELS_DIR', './models')}")
    print(f"You can see the cache structure in: {MODEL_CACHE}/")
    
    # Download IP-Adapter models
    if not download_ip_adapter_models():
        print("Failed to download IP-Adapter models")
        return False
    
    print("\n✓ All models downloaded successfully!")
    print("\nNext steps:")
    print("1. Set environment variables if needed:")
    print("   export DEGIS_CACHE_DIR=/path/to/cache")
    print("   export DEGIS_MODELS_DIR=/path/to/models")
    print("2. Use the models in your code:")
    print("   from degis.config.models import get_ip_adapter_sd15_path")
    print("   ip_ckpt = get_ip_adapter_sd15_path()")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Download external model checkpoints")
    parser.add_argument("--models-dir", help="Directory to save models")
    parser.add_argument("--cache-dir", help="Directory for HuggingFace cache")
    parser.add_argument("--ip-adapter-only", action="store_true", 
                       help="Download only IP-Adapter models")
    
    args = parser.parse_args()
    
    # Set environment variables if provided
    if args.models_dir:
        os.environ["DEGIS_MODELS_DIR"] = args.models_dir
    if args.cache_dir:
        os.environ["DEGIS_CACHE_DIR"] = args.cache_dir
    
    if args.ip_adapter_only:
        success = download_ip_adapter_models()
    else:
        success = download_all_models()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
