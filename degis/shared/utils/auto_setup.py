"""
Automatic setup utilities for notebooks and demos.
"""
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def detect_environment() -> Dict[str, str]:
    """
    Detect the current environment and suggest appropriate paths.
    
    Returns:
        Dictionary with detected environment info and suggested paths
    """
    env_info = {
        "is_server": False,
        "has_data_dir": False,
        "has_gpu": False,
        "suggested_data_dir": None,
        "suggested_models_dir": None,
        "suggested_cache_dir": None,
    }
    
    # Check if running on a server with /data directory
    if os.path.exists("/data"):
        env_info["is_server"] = True
        env_info["has_data_dir"] = True
        env_info["suggested_data_dir"] = "/data/thesis/data"
        env_info["suggested_models_dir"] = "/data/thesis/models"
        env_info["suggested_cache_dir"] = "/data/model-cache"
    else:
        # Local development environment
        env_info["suggested_data_dir"] = "./data"
        env_info["suggested_models_dir"] = "./models"
        env_info["suggested_cache_dir"] = "./model-cache"
    
    # Check for GPU
    try:
        import torch
        env_info["has_gpu"] = torch.cuda.is_available()
    except ImportError:
        pass
    
    return env_info

def setup_environment(
    data_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    auto_download: bool = True
) -> Dict[str, str]:
    """
    Set up the environment with appropriate paths and download models if needed.
    
    Args:
        data_dir: Custom data directory path
        models_dir: Custom models directory path
        cache_dir: Custom cache directory path
        auto_download: Whether to automatically download required models
        
    Returns:
        Dictionary with configured paths
    """
    # Detect environment
    env_info = detect_environment()
    
    # Use provided paths or defaults
    paths = {
        "data_dir": data_dir or env_info["suggested_data_dir"],
        "models_dir": models_dir or env_info["suggested_models_dir"],
        "cache_dir": cache_dir or env_info["suggested_cache_dir"],
    }
    
    # Create directories if they don't exist
    for path_name, path_value in paths.items():
        os.makedirs(path_value, exist_ok=True)
        logger.info(f"Created {path_name}: {path_value}")
    
    # Set environment variables
    os.environ["DEGIS_DATA_DIR"] = paths["data_dir"]
    os.environ["DEGIS_MODELS_DIR"] = paths["models_dir"]
    os.environ["DEGIS_CACHE_DIR"] = paths["cache_dir"]
    
    # Download models if requested
    if auto_download:
        try:
            from .model_manager import get_model_manager
            
            manager = get_model_manager(
                cache_dir=paths["cache_dir"],
                models_dir=paths["models_dir"]
            )
            # Model downloading would be handled by the manager
            logger.info("Model manager initialized")
        except ImportError:
            logger.warning("Model manager not available")
    
    return paths

def create_notebook_config(
    notebook_name: str,
    data_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> str:
    """
    Create a configuration cell for a notebook with appropriate paths.
    
    Args:
        notebook_name: Name of the notebook (e.g., "01_data_extraction", "02_generation")
        data_dir: Custom data directory
        models_dir: Custom models directory
        cache_dir: Custom cache directory
        
    Returns:
        Python code string for the configuration cell
    """
    # Set up environment
    paths = setup_environment(data_dir, models_dir, cache_dir, auto_download=False)
    
    config_code = f'''# Auto-setup configuration for {notebook_name}
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# This file contains the setup_environment function

print("Environment configured:")
print(f"  Data directory: {paths['data_dir']}")
print(f"  Models directory: {paths['models_dir']}")
print(f"  Cache directory: {paths['cache_dir']}")

# Set up paths for this notebook
DATA_DIR = paths['data_dir']
MODELS_DIR = paths['models_dir']
CACHE_DIR = paths['cache_dir']

# Create dataset-specific paths
DATASET_NAME = "adimagenet"  # Change this for different datasets
CSV_PATH = os.path.join(DATA_DIR, f"{DATASET_NAME}_manifest.csv")
EMBEDDINGS_PATH = os.path.join(MODELS_DIR, f"{DATASET_NAME}_embeddings.npy")
HISTOGRAMS_PATH = os.path.join(DATA_DIR, f"{DATASET_NAME}_color_histograms_hcl_514.npy")
EDGE_MAPS_PATH = os.path.join(DATA_DIR, f"{DATASET_NAME}_edge_maps.npy")

print("\\nDataset paths configured:")
print(f"  CSV: {CSV_PATH}")
print(f"  Embeddings: {EMBEDDINGS_PATH}")
print(f"  Histograms: {HISTOGRAMS_PATH}")
print(f"  Edge maps: {EDGE_MAPS_PATH}")

# Check if files exist
missing_files = []
for name, path in [("CSV", CSV_PATH), ("Embeddings", EMBEDDINGS_PATH), 
                   ("Histograms", HISTOGRAMS_PATH), ("Edge maps", EDGE_MAPS_PATH)]:
    if not os.path.exists(path):
        missing_files.append(f"{name}: {path}")

if missing_files:
    print("\\n Missing files (will be generated if needed):")
    for file in missing_files:
        print(f"  - {file}")
else:
    print("\\nAll dataset files found!")

print("\\nReady to run the notebook!")
'''
    
    return config_code

def download_required_models(force_download: bool = False) -> bool:
    """
    Download all required models for the notebooks.
    
    Args:
        force_download: Whether to force re-download
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from .model_manager import get_model_manager
        from ..config import MODEL_CACHE
        
        manager = get_model_manager(
            cache_dir=MODEL_CACHE,
            models_dir=os.getenv("DEGIS_MODELS_DIR", "./models")
        )
        
        print("Downloading required models...")
        
        # Download IP-Adapter models
        try:
            from huggingface_hub import hf_hub_download
            
            # Download IP-Adapter SD 1.5
            sd15_path = hf_hub_download(
                repo_id="h94/IP-Adapter",
                filename="ip-adapter_sd15.bin",
                cache_dir=MODEL_CACHE,
                local_dir=os.getenv("DEGIS_MODELS_DIR", "./models")
            )
            print(f"Downloaded IP-Adapter SD 1.5: {sd15_path}")
            
            # Download IP-Adapter SDXL
            sdxl_path = hf_hub_download(
                repo_id="h94/IP-Adapter", 
                filename="ip-adapter_sdxl.bin",
                cache_dir=MODEL_CACHE,
                local_dir=os.getenv("DEGIS_MODELS_DIR", "./models")
            )
            print(f"Downloaded IP-Adapter SDXL: {sdxl_path}")
            
        except ImportError:
            print(" huggingface_hub not available. Install with: pip install huggingface_hub")
            return False
        except Exception as e:
            print(f"Failed to download models: {e}")
            return False
        
        print("All required models downloaded!")
        return True
        
    except Exception as e:
        print(f"Error setting up models: {e}")
        return False

def print_setup_summary():
    """Print a summary of the current setup."""
    env_info = detect_environment()
    paths = {
        "data_dir": os.getenv("DEGIS_DATA_DIR", "./data"),
        "models_dir": os.getenv("DEGIS_MODELS_DIR", "./models"),
        "cache_dir": os.getenv("DEGIS_CACHE_DIR", "./model-cache"),
    }
    
    print("DEGIS Setup Summary")
    print("=" * 50)
    print(f"Environment: {'Server' if env_info['is_server'] else 'Local'}")
    print(f"GPU Available: {'Yes' if env_info['has_gpu'] else 'No'}")
    print(f"Data Directory: {paths['data_dir']}")
    print(f"Models Directory: {paths['models_dir']}")
    print(f"Cache Directory: {paths['cache_dir']}")
    print("=" * 50)