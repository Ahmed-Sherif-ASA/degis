#!/usr/bin/env python3
"""
Test script to validate degis_clean package structure without dependencies.
"""

import os
import sys

def test_package_structure():
    """Test that the package structure is correct."""
    print("Testing degis_clean package structure...")
    print("=" * 50)
    
    # Check main directories exist
    required_dirs = [
        "degis_clean",
        "degis_clean/data", 
        "degis_clean/training",
        "degis_clean/inference",
        "degis_clean/shared",
        "degis_clean/training/cli",
        "degis_clean/inference/cli",
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
    
    # Check __init__.py files exist
    init_files = [
        "degis_clean/__init__.py",
        "degis_clean/data/__init__.py",
        "degis_clean/training/__init__.py", 
        "degis_clean/inference/__init__.py",
        "degis_clean/shared/__init__.py"
    ]
    
    for init_file in init_files:
        if os.path.exists(init_file):
            print(f"✅ {init_file} exists")
        else:
            print(f"❌ {init_file} missing")
            success = False
    
    # Check data files were copied
    data_files = [
        "degis_clean/data/dataset.py",
        "degis_clean/shared/config.py"
    ]
    
    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"✅ {data_file} copied successfully")
        else:
            print(f"❌ {data_file} missing")
            success = False
    
    return success

def test_import_structure():
    """Test that import structure is syntactically correct."""
    print("\nTesting import structure...")
    print("=" * 50)
    
    try:
        # Test that we can at least parse the __init__.py files
        with open("degis_clean/__init__.py", "r") as f:
            content = f.read()
            compile(content, "degis_clean/__init__.py", "exec")
        print("✅ degis_clean/__init__.py syntax OK")
        
        with open("degis_clean/data/__init__.py", "r") as f:
            content = f.read()
            compile(content, "degis_clean/data/__init__.py", "exec")
        print("✅ degis_clean/data/__init__.py syntax OK")
        
        with open("degis_clean/shared/__init__.py", "r") as f:
            content = f.read()
            compile(content, "degis_clean/shared/__init__.py", "exec")
        print("✅ degis_clean/shared/__init__.py syntax OK")
        
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in __init__.py files: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading __init__.py files: {e}")
        return False

if __name__ == "__main__":
    success = True
    success &= test_package_structure()
    success &= test_import_structure()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Package structure is correct! Ready for next phase.")
    else:
        print("❌ Package structure has issues. Fix before proceeding.")
