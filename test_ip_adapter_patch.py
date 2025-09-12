#!/usr/bin/env python3
"""
Test script for DEGIS IP-Adapter patch
=====================================

This script tests that the DEGIS patch is working correctly.
"""

def test_ip_adapter_patch():
    """Test that DEGIS IP-Adapter patch is working"""
    print("🧪 Testing DEGIS IP-Adapter patch...")
    
    try:
        # Import the DEGIS patches (this applies the patches)
        import ip_adapter_patch
        
        # Import IP-Adapter classes
        from ip_adapter import IPAdapter, IPAdapterXL
        from ip_adapter.ip_adapter import EmbeddingAdapter
        
        print("✅ Successfully imported patched IP-Adapter classes")
        
        # Test that new methods exist
        assert hasattr(IPAdapter, '_mix_text_ip_tokens'), "Missing _mix_text_ip_tokens method"
        assert hasattr(IPAdapter, 'generate_from_embeddings'), "Missing generate_from_embeddings method"
        assert hasattr(IPAdapter, 'embedding_type'), "Missing embedding_type attribute"
        
        print("✅ All enhanced methods are available")
        
        # Test EmbeddingAdapter
        adapter = EmbeddingAdapter(cross_attention_dim=1024, embedding_dim=1024, num_tokens=4)
        print("✅ EmbeddingAdapter created successfully")
        
        print("🎉 All tests passed! DEGIS patch is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ip_adapter_patch()
    exit(0 if success else 1)
