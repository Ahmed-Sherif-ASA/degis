#!/usr/bin/env python3
"""
Test script that simulates notebook imports to validate degis_clean works.
This simulates the imports from 01_data_extraction_and_training.ipynb
"""

def test_notebook_imports():
    """Test imports that would be used in the training notebook."""
    print("Testing notebook-style imports with degis_clean...")
    print("=" * 60)
    
    try:
        # Simulate the imports from 01_data_extraction_and_training.ipynb
        print("Testing data imports...")
        from degis_clean.data.dataset import UnifiedImageDataset
        print("✅ degis_clean.data.dataset.UnifiedImageDataset")
        
        print("Testing config imports...")
        from degis_clean.shared.config import CSV_PATH, BATCH_SIZE, EMBEDDINGS_TARGET_PATH
        print("✅ degis_clean.shared.config (CSV_PATH, BATCH_SIZE, EMBEDDINGS_TARGET_PATH)")
        
        print("Testing main package imports...")
        import degis_clean
        from degis_clean import UnifiedImageDataset as UID
        print("✅ degis_clean main package and UnifiedImageDataset")
        
        print("\nTesting that we can access the variables...")
        print(f"CSV_PATH: {CSV_PATH}")
        print(f"BATCH_SIZE: {BATCH_SIZE}")
        print(f"EMBEDDINGS_TARGET_PATH: {EMBEDDINGS_TARGET_PATH}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

def test_import_compatibility():
    """Test that the new imports are compatible with expected usage."""
    print("\nTesting import compatibility...")
    print("=" * 60)
    
    try:
        # Test that we can create the same import patterns as the notebook
        from degis_clean.data.dataset import UnifiedImageDataset
        from degis_clean.shared.config import CSV_PATH, BATCH_SIZE, EMBEDDINGS_TARGET_PATH
        
        # Test that the classes/functions exist and are callable
        print(f"UnifiedImageDataset type: {type(UnifiedImageDataset)}")
        print(f"CSV_PATH type: {type(CSV_PATH)}")
        print(f"BATCH_SIZE type: {type(BATCH_SIZE)}")
        print(f"EMBEDDINGS_TARGET_PATH type: {type(EMBEDDINGS_TARGET_PATH)}")
        
        print("✅ All imports are compatible with expected usage")
        return True
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    success = True
    success &= test_notebook_imports()
    success &= test_import_compatibility()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ degis_clean data module is ready for notebook migration!")
        print("Next step: Update 01_data_extraction_and_training.ipynb to use degis_clean")
    else:
        print("❌ degis_clean data module has issues. Fix before proceeding.")
