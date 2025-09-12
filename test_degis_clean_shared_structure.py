#!/usr/bin/env python3
"""
Test script to validate degis_clean shared utilities structure.
"""

import os

def test_shared_structure():
    """Test that the shared utilities structure is correct."""
    print("Testing degis_clean shared utilities structure...")
    print("=" * 60)
    
    # Check shared directories exist
    required_dirs = [
        "degis_clean/shared",
        "degis_clean/shared/embeddings",
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
    
    # Check shared files exist
    required_files = [
        "degis_clean/shared/__init__.py",
        "degis_clean/shared/config.py",
        "degis_clean/shared/embeddings.py",
        "degis_clean/shared/clip_embeddings.py",
        "degis_clean/shared/clip_embeddings_xl_hf.py",
        "degis_clean/shared/utils/__init__.py",
        "degis_clean/shared/utils/file_utils.py",
        "degis_clean/shared/utils/image_utils.py",
        "degis_clean/shared/utils/logger.py",
        "degis_clean/shared/utils/model_downloader.py",
        "degis_clean/shared/utils/model_manager.py",
        "degis_clean/shared/utils/auto_setup.py",
        "degis_clean/shared/utils/emd_generation.py"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            success = False
    
    return success

def test_import_syntax():
    """Test that the __init__.py files have correct syntax."""
    print("\nTesting import syntax...")
    print("=" * 60)
    
    try:
        # Test main __init__.py
        with open("degis_clean/__init__.py", "r") as f:
            content = f.read()
            compile(content, "degis_clean/__init__.py", "exec")
        print("✅ degis_clean/__init__.py syntax OK")
        
        # Test shared __init__.py
        with open("degis_clean/shared/__init__.py", "r") as f:
            content = f.read()
            compile(content, "degis_clean/shared/__init__.py", "exec")
        print("✅ degis_clean/shared/__init__.py syntax OK")
        
        # Test shared/utils __init__.py
        with open("degis_clean/shared/utils/__init__.py", "r") as f:
            content = f.read()
            compile(content, "degis_clean/shared/utils/__init__.py", "exec")
        print("✅ degis_clean/shared/utils/__init__.py syntax OK")
        
        # Test shared/embeddings __init__.py
        with open("degis_clean/shared/embeddings/__init__.py", "r") as f:
            content = f.read()
            compile(content, "degis_clean/shared/embeddings/__init__.py", "exec")
        print("✅ degis_clean/shared/embeddings/__init__.py syntax OK")
        
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in __init__.py files: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading __init__.py files: {e}")
        return False

if __name__ == "__main__":
    success = True
    success &= test_shared_structure()
    success &= test_import_syntax()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ degis_clean shared utilities structure is correct!")
        print("Ready for testing in full environment or notebook migration.")
    else:
        print("❌ degis_clean shared utilities structure has issues.")
