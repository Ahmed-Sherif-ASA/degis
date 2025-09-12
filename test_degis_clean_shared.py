#!/usr/bin/env python3
"""
Test script to validate degis_clean shared utilities work correctly.
"""

def test_shared_imports():
    """Test that shared utilities import correctly."""
    print("Testing degis_clean shared utilities...")
    print("=" * 60)
    
    try:
        # Test config imports
        print("Testing config imports...")
        from degis_clean.shared.config import CSV_PATH, BATCH_SIZE, EMBEDDINGS_TARGET_PATH
        print("✅ degis_clean.shared.config imports successful")
        
        # Test embeddings imports
        print("Testing embeddings imports...")
        from degis_clean.shared.embeddings import generate_clip_embeddings, generate_xl_embeddings
        print("✅ degis_clean.shared.embeddings imports successful")
        
        # Test utils imports
        print("Testing utils imports...")
        from degis_clean.shared.utils import create_control_edge_pil
        print("✅ degis_clean.shared.utils imports successful")
        
        # Test main package imports
        print("Testing main package imports...")
        import degis_clean
        from degis_clean import generate_clip_embeddings, generate_xl_embeddings
        print("✅ degis_clean main package imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

def test_notebook_compatibility():
    """Test that the shared utilities work with notebook-style imports."""
    print("\nTesting notebook compatibility...")
    print("=" * 60)
    
    try:
        # Test the imports that would be used in notebooks
        from degis_clean.shared.embeddings import generate_clip_embeddings, generate_xl_embeddings
        from degis_clean.shared.config import CSV_PATH, BATCH_SIZE, EMBEDDINGS_TARGET_PATH
        from degis_clean.data.dataset import UnifiedImageDataset
        
        print("✅ All notebook-style imports work")
        print(f"CSV_PATH: {CSV_PATH}")
        print(f"BATCH_SIZE: {BATCH_SIZE}")
        print(f"EMBEDDINGS_TARGET_PATH: {EMBEDDINGS_TARGET_PATH}")
        
        return True
        
    except Exception as e:
        print(f"❌ Notebook compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    success = True
    success &= test_shared_imports()
    success &= test_notebook_compatibility()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ degis_clean shared utilities migration successful!")
        print("Next step: Update notebooks to use shared utilities")
    else:
        print("❌ degis_clean shared utilities have issues. Check the errors above.")
