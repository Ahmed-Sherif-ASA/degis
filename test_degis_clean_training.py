#!/usr/bin/env python3
"""
Test script to validate degis_clean training module works correctly.
"""

import os

def test_training_structure():
    """Test that the training module structure is correct."""
    print("Testing degis_clean training module structure...")
    print("=" * 60)
    
    # Check training directories exist
    required_dirs = [
        "degis_clean/training",
        "degis_clean/training/features",
        "degis_clean/training/models", 
        "degis_clean/training/cli"
    ]
    
    success = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path} exists")
        else:
            print(f"❌ {dir_path} missing")
            success = False
    
    # Check training files exist
    required_files = [
        "degis_clean/training/__init__.py",
        "degis_clean/training/training.py",
        "degis_clean/training/features.py",
        "degis_clean/training/features/__init__.py",
        "degis_clean/training/features/color_histograms.py",
        "degis_clean/training/features/edge_maps.py",
        "degis_clean/training/models/__init__.py",
        "degis_clean/training/models/color_heads.py",
        "degis_clean/training/cli/__init__.py",
        "degis_clean/training/cli/train_color.py"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            success = False
    
    return success

def test_import_syntax():
    """Test that the training module imports have correct syntax."""
    print("\nTesting training module import syntax...")
    print("=" * 60)
    
    try:
        # Test training __init__.py
        with open("degis_clean/training/__init__.py", "r") as f:
            content = f.read()
            compile(content, "degis_clean/training/__init__.py", "exec")
        print("✅ degis_clean/training/__init__.py syntax OK")
        
        # Test training/features __init__.py
        with open("degis_clean/training/features/__init__.py", "r") as f:
            content = f.read()
            compile(content, "degis_clean/training/features/__init__.py", "exec")
        print("✅ degis_clean/training/features/__init__.py syntax OK")
        
        # Test training/models __init__.py
        with open("degis_clean/training/models/__init__.py", "r") as f:
            content = f.read()
            compile(content, "degis_clean/training/models/__init__.py", "exec")
        print("✅ degis_clean/training/models/__init__.py syntax OK")
        
        # Test main __init__.py
        with open("degis_clean/__init__.py", "r") as f:
            content = f.read()
            compile(content, "degis_clean/__init__.py", "exec")
        print("✅ degis_clean/__init__.py syntax OK")
        
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in __init__.py files: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading __init__.py files: {e}")
        return False

def test_notebook_compatibility():
    """Test that training module works with notebook-style imports."""
    print("\nTesting notebook compatibility...")
    print("=" * 60)
    
    try:
        # Test the imports that would be used in notebooks
        from degis_clean.training import train_color_model, generate_color_histograms, generate_edge_maps
        from degis_clean.shared.embeddings import generate_clip_embeddings, generate_xl_embeddings
        from degis_clean.data.dataset import UnifiedImageDataset
        from degis_clean.shared.config import CSV_PATH, BATCH_SIZE, EMBEDDINGS_TARGET_PATH
        
        print("✅ All training notebook-style imports work")
        print(f"Available functions: train_color_model, generate_color_histograms, generate_edge_maps")
        
        return True
        
    except Exception as e:
        print(f"❌ Notebook compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    success = True
    success &= test_training_structure()
    success &= test_import_syntax()
    success &= test_notebook_compatibility()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ degis_clean training module migration successful!")
        print("Ready for notebook testing on server.")
    else:
        print("❌ degis_clean training module has issues.")
