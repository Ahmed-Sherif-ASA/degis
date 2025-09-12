#!/usr/bin/env python3
"""
Setup script to configure Jupyter kernels for DEGIS notebooks.
Run this after setting up your environment to ensure notebooks work seamlessly.
"""

import subprocess
import sys
import json
import os
from pathlib import Path

def setup_degis_kernel():
    """Set up a dedicated DEGIS kernel for Jupyter notebooks."""
    print("🔧 Setting up DEGIS Jupyter kernel...")
    
    try:
        # Install ipykernel if not available
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ipykernel"])
        
        # Create a dedicated kernel
        subprocess.check_call([
            sys.executable, "-m", "ipykernel", "install", 
            "--user", 
            "--name=degis", 
            "--display-name=DEGIS Environment"
        ])
        
        print("✅ DEGIS kernel installed successfully!")
        print("📝 Kernel name: 'degis'")
        print("📝 Display name: 'DEGIS Environment'")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to install kernel: {e}")
        return False

def update_notebook_kernels():
    """Update notebook kernel metadata to use the DEGIS kernel."""
    print("📝 Updating notebook kernel metadata...")
    
    notebooks = [
        "01_data_extraction_and_training.ipynb",
        "02_image_generation_ipadapter.ipynb", 
        "02b_image_generation_ipadapter_xl.ipynb"
    ]
    
    for notebook_path in notebooks:
        if not Path(notebook_path).exists():
            print(f"⚠️  Notebook not found: {notebook_path}")
            continue
            
        try:
            # Read notebook
            with open(notebook_path, 'r') as f:
                notebook = json.load(f)
            
            # Update kernel metadata
            if 'kernelspec' in notebook['metadata']:
                notebook['metadata']['kernelspec'] = {
                    "display_name": "DEGIS Environment",
                    "language": "python",
                    "name": "degis"
                }
                
                # Write back
                with open(notebook_path, 'w') as f:
                    json.dump(notebook, f, indent=2)
                
                print(f"✅ Updated {notebook_path}")
            else:
                print(f"⚠️  No kernelspec found in {notebook_path}")
                
        except Exception as e:
            print(f"❌ Failed to update {notebook_path}: {e}")

def main():
    """Main setup function."""
    print("🚀 Setting up DEGIS Jupyter environment...")
    print("=" * 50)
    
    # Set up kernel
    if setup_degis_kernel():
        # Update notebook metadata
        update_notebook_kernels()
        
        print("\n🎉 Setup complete!")
        print("=" * 50)
        print("Your notebooks should now:")
        print("1. Automatically use the 'DEGIS Environment' kernel")
        print("2. Not prompt for kernel selection")
        print("3. Work seamlessly for your professor")
        print("\nTo test: Open any notebook and run the first cell")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
