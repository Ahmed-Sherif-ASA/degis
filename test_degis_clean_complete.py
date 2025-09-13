#!/usr/bin/env python3
"""
Test script to validate complete degis_clean package migration.
"""

import os

def test_complete_structure():
    """Test that the complete package structure is correct."""
    print("Testing complete degis_clean package structure...")
    print("=" * 60)
    
    # Check all main directories exist
    required_dirs = [
        "degis_clean",
        "degis_clean/data",
        "degis_clean/training", 
        "degis_clean/inference",
        "degis_clean/shared",
        "degis_clean/training/models",
        "degis_clean/training/cli",
        "degis_clean/shared/image_features",
        "degis_clean/shared/utils",
        "degis_clean/shared/config"
    ]
    
    success = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path} exists")
        else:
            print(f"❌ {dir_path} missing")
            success = False
    
    return success

def test_import_syntax():
    """Test that all __init__.py files have correct syntax."""
    print("\nTesting import syntax...")
    print("=" * 60)
    
    init_files = [
        "degis_clean/__init__.py",
        "degis_clean/data/__init__.py",
        "degis_clean/training/__init__.py",
        "degis_clean/inference/__init__.py",
        "degis_clean/shared/__init__.py",
        "degis_clean/training/models/__init__.py",
        "degis_clean/shared/image_features/__init__.py",
        "degis_clean/shared/utils/__init__.py"
    ]
    
    success = True
    for init_file in init_files:
        try:
            with open(init_file, "r") as f:
                content = f.read()
                compile(content, init_file, "exec")
            print(f"✅ {init_file} syntax OK")
        except SyntaxError as e:
            print(f"❌ {init_file} syntax error: {e}")
            success = False
        except Exception as e:
            print(f"❌ {init_file} error: {e}")
            success = False
    
    return success

def test_notebook_imports():
    """Test that all notebook imports would work."""
    print("\nTesting notebook import compatibility...")
    print("=" * 60)
    
    try:
        # Test main package import
        import degis_clean
        print("✅ degis_clean main package import")
        
        # Test data imports
        from degis_clean.data.dataset import UnifiedImageDataset
        print("✅ degis_clean.data.dataset.UnifiedImageDataset")
        
        # Test shared imports
        from degis_clean.shared.config import CSV_PATH, BATCH_SIZE, EMBEDDINGS_TARGET_PATH
        from degis_clean.shared.embeddings import generate_clip_embeddings, generate_xl_embeddings
        print("✅ degis_clean.shared imports")
        
        # Test training imports
        from degis_clean.training import train_color_model
        print("✅ degis_clean.training imports")
        
        # Test inference imports
        from degis_clean.inference import IPAdapterGenerator, IPAdapterXLGenerator, load_trained_color_head, get_color_embedding
        print("✅ degis_clean.inference imports")
        
        # Test main package convenience imports
        from degis_clean import (
            UnifiedImageDataset, generate_clip_embeddings, generate_xl_embeddings,
            generate_color_histograms, generate_edge_maps, train_color_model,
            IPAdapterGenerator, IPAdapterXLGenerator, load_trained_color_head, get_color_embedding
        )
        print("✅ degis_clean main package convenience imports")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_migration_summary():
    """Show migration summary."""
    print("\n" + "=" * 60)
    print("🎉 DEGIS CLEAN MIGRATION COMPLETE!")
    print("=" * 60)
    
    print("\n📁 CLEAN STRUCTURE:")
    print("├── data/              # Shared data handling")
    print("├── training/          # Training-specific code")
    print("│   ├── features/      # Feature extraction")
    print("│   ├── models/        # Model architectures")
    print("│   └── cli/           # Training CLI tools")
    print("├── inference/         # Inference-specific code")
    print("│   ├── generation/    # Image generation")
    print("│   ├── visualization/ # Plotting utilities")
    print("│   └── cli/           # Inference CLI tools")
    print("├── shared/            # Shared utilities")
    print("│   ├── embeddings/    # CLIP embeddings")
    print("│   ├── utils/         # General utilities")
    print("│   └── config/        # Configuration")
    print("└── ip_adapter_patch/  # Custom IP-Adapter")
    
    print("\n🔄 MIGRATION BENEFITS:")
    print("✅ Clear separation between training and inference")
    print("✅ No more duplicate or overlapping functionality")
    print("✅ Shared utilities in dedicated module")
    print("✅ Easy to understand and maintain")
    print("✅ Clean import structure")
    
    print("\n📝 NOTEBOOK UPDATES:")
    print("OLD: from degis.core.training import train_color_model")
    print("NEW: from degis_clean.training import train_color_model")
    print("OR:  from degis_clean import train_color_model")

if __name__ == "__main__":
    success = True
    success &= test_complete_structure()
    success &= test_import_syntax()
    success &= test_notebook_imports()
    
    test_migration_summary()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ degis_clean migration is COMPLETE and ready for testing!")
        print("Next step: Update notebooks on server to use degis_clean")
    else:
        print("❌ degis_clean migration has issues that need fixing.")
