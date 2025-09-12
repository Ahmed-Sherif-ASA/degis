#!/usr/bin/env python3
"""
Test script to validate degis_clean data module works correctly.
"""

def test_data_imports():
    """Test that data module imports work correctly."""
    try:
        from degis_clean.data import UnifiedImageDataset
        print("✅ degis_clean.data.UnifiedImageDataset import successful")
        return True
    except ImportError as e:
        print(f"❌ degis_clean.data import failed: {e}")
        return False

def test_config_imports():
    """Test that config imports work correctly."""
    try:
        from degis_clean.shared.config import CSV_PATH, BATCH_SIZE, EMBEDDINGS_TARGET_PATH
        print("✅ degis_clean.shared.config imports successful")
        return True
    except ImportError as e:
        print(f"❌ degis_clean.shared.config import failed: {e}")
        return False

def test_main_imports():
    """Test that main degis_clean imports work."""
    try:
        import degis_clean
        from degis_clean import UnifiedImageDataset
        print("✅ degis_clean main imports successful")
        return True
    except ImportError as e:
        print(f"❌ degis_clean main import failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing degis_clean data module...")
    print("=" * 50)
    
    success = True
    success &= test_data_imports()
    success &= test_config_imports() 
    success &= test_main_imports()
    
    print("=" * 50)
    if success:
        print("✅ All tests passed! Data module migration successful.")
    else:
        print("❌ Some tests failed. Check the errors above.")
