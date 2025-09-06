#!/usr/bin/env python3
"""
Script to set up notebooks for easy running.
"""
import os
import sys
import argparse
from pathlib import Path
import json

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from degis.utils.auto_setup import create_notebook_config, setup_environment, download_required_models

def setup_notebook(notebook_path: str, data_dir: str = None, models_dir: str = None, cache_dir: str = None):
    """Set up a notebook with auto-configuration."""
    notebook_path = Path(notebook_path)
    
    if not notebook_path.exists():
        print(f"❌ Notebook not found: {notebook_path}")
        return False
    
    print(f"🔧 Setting up notebook: {notebook_path.name}")
    
    # Read the notebook
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    # Create configuration cell
    config_code = create_notebook_config(
        notebook_path.stem,
        data_dir=data_dir,
        models_dir=models_dir,
        cache_dir=cache_dir
    )
    
    # Create the configuration cell
    config_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": config_code.split('\n')
    }
    
    # Insert the config cell after the first markdown cell
    insert_index = 0
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'markdown' and 'setup' in cell['source'][0].lower():
            insert_index = i + 1
            break
    
    notebook['cells'].insert(insert_index, config_cell)
    
    # Create backup
    backup_path = notebook_path.with_suffix('.ipynb.backup')
    with open(backup_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    print(f"📄 Created backup: {backup_path}")
    
    # Write updated notebook
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"✅ Notebook updated: {notebook_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Set up notebooks for easy running")
    parser.add_argument("--notebooks", nargs="+", 
                       default=["01_data_extraction_and_training.ipynb", "02_image_generation_ipadapter.ipynb"],
                       help="Notebooks to set up")
    parser.add_argument("--data-dir", help="Data directory path")
    parser.add_argument("--models-dir", help="Models directory path")
    parser.add_argument("--cache-dir", help="Cache directory path")
    parser.add_argument("--download-models", action="store_true", 
                       help="Download required models")
    parser.add_argument("--no-backup", action="store_true",
                       help="Don't create backup files")
    
    args = parser.parse_args()
    
    # Set up environment
    print("🔧 Setting up environment...")
    paths = setup_environment(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        cache_dir=args.cache_dir,
        auto_download=False
    )
    
    # Download models if requested
    if args.download_models:
        print("\n📥 Downloading required models...")
        success = download_required_models()
        if not success:
            print("❌ Failed to download some models. You may need to download them manually.")
    
    # Set up notebooks
    print(f"\n📓 Setting up {len(args.notebooks)} notebooks...")
    for notebook in args.notebooks:
        if not setup_notebook(notebook, args.data_dir, args.models_dir, args.cache_dir):
            print(f"❌ Failed to set up {notebook}")
        else:
            print(f"✅ Successfully set up {notebook}")
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Open the notebooks in Jupyter")
    print("2. Run the first cell to auto-configure everything")
    print("3. The notebooks will automatically detect your environment and set up paths")
    print("4. If you need models, run: python scripts/download_models.py")

if __name__ == "__main__":
    main()
