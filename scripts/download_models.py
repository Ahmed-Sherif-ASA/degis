#!/usr/bin/env python3
"""
Script to download models to MODEL_CACHE.
"""
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from degis.utils.model_downloader import download_all_ip_adapter_models

def main():
    """Download all IP-Adapter models."""
    try:
        download_all_ip_adapter_models()
        print("✓ All models downloaded successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"✗ Failed to download models: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
