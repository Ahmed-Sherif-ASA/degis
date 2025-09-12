#!/usr/bin/env python3
"""
Environment setup utilities for DEGIS notebooks.
This module provides common setup functions for Jupyter notebooks.

Note: This script is designed to be run from within Jupyter notebooks
or virtual environments. It may not work directly from system Python
due to externally-managed environment restrictions.
"""

import sys
import subprocess
import importlib
import os
from pathlib import Path
from typing import List, Optional


def install_and_import(package: str, import_name: Optional[str] = None, pip_name: Optional[str] = None) -> None:
    """
    Install package if not available and import it.
    
    Args:
        package: Package name to install
        import_name: Name to import (defaults to package)
        pip_name: Name for pip install (defaults to package)
    """
    if import_name is None:
        import_name = package
    if pip_name is None:
        pip_name = package
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {import_name} already available")
    except ImportError:
        print(f"📦 Installing {pip_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
        importlib.import_module(import_name)
        print(f"✅ {import_name} installed and imported")


def setup_degis_environment(
    include_generation: bool = False,
    include_sdxl: bool = False,
    additional_deps: Optional[List[str]] = None
) -> bool:
    """
    Set up the DEGIS environment for notebooks.
    
    Args:
        include_generation: Whether to install image generation dependencies
        include_sdxl: Whether to install SDXL-specific dependencies
        additional_deps: Additional packages to install
        
    Returns:
        True if setup successful, False otherwise
    """
    print("🚀 Setting up DEGIS environment...")
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Please run this notebook from the DEGIS project root directory")
        print("   The directory should contain pyproject.toml")
        return False
    
    # Install the package in development mode
    try:
        print("📦 Installing DEGIS package in development mode...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
        print("✅ DEGIS package installed")
    except Exception as e:
        print(f"⚠️  Warning: Could not install DEGIS package: {e}")
        print("   You may need to run: pip install -e .")
    
    # Install Jupyter dependencies
    jupyter_deps = ["jupyter", "ipykernel", "notebook"]
    for dep in jupyter_deps:
        install_and_import(dep)
    
    # Install generation dependencies if requested
    if include_generation:
        print("🤖 Installing IP-Adapter and generation dependencies...")
        try:
            # Install IP-Adapter from your fork
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "git+https://github.com/Ahmed-Sherif-ASA/IP-Adapter@main"
            ])
            print("✅ IP-Adapter installed")
        except Exception as e:
            print(f"⚠️  Warning: Could not install IP-Adapter: {e}")
        
        # Install other generation dependencies
        generation_deps = ["diffusers", "transformers", "accelerate", "controlnet-aux"]
        
        # Add SDXL-specific dependencies
        if include_sdxl:
            generation_deps.append("safetensors")
            print("🎨 Including SDXL-specific dependencies...")
        
        for dep in generation_deps:
            install_and_import(dep)
    
    # Install additional dependencies if provided
    if additional_deps:
        print(f"📦 Installing additional dependencies: {additional_deps}")
        for dep in additional_deps:
            install_and_import(dep)
    
    print("🎉 Setup complete! You can now run the rest of the notebook.")
    return True


def setup_training_environment() -> bool:
    """Set up environment for training notebooks."""
    return setup_degis_environment(
        include_generation=False,
        additional_deps=["matplotlib", "seaborn", "tqdm"]
    )


def setup_generation_environment() -> bool:
    """Set up environment for SD 1.5 generation notebooks."""
    return setup_degis_environment(
        include_generation=True,
        include_sdxl=False
    )


def setup_sdxl_environment() -> bool:
    """Set up environment for SDXL generation notebooks."""
    return setup_degis_environment(
        include_generation=True,
        include_sdxl=True
    )


# Convenience function for quick setup
def quick_setup(notebook_type: str = "training") -> bool:
    """
    Quick setup based on notebook type.
    
    Args:
        notebook_type: Type of notebook ("training", "generation", "sdxl")
        
    Returns:
        True if setup successful, False otherwise
    """
    if notebook_type == "training":
        return setup_training_environment()
    elif notebook_type == "generation":
        return setup_generation_environment()
    elif notebook_type == "sdxl":
        return setup_sdxl_environment()
    else:
        print(f"❌ Unknown notebook type: {notebook_type}")
        print("   Valid types: 'training', 'generation', 'sdxl'")
        return False


if __name__ == "__main__":
    # If run directly, set up for training by default
    setup_training_environment()
