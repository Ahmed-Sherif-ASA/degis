# DEGIS IP-Adapter Implementation

This directory contains a complete IP-Adapter implementation with DEGIS enhancements.

## What's Included

### Complete Implementation:
- **Full IP-Adapter Replacement**: Complete replacement of all IP-Adapter classes
- **All Classes**: `IPAdapter`, `IPAdapterXL`, `IPAdapterPlus`, `IPAdapterFull`, `IPAdapterPlusXL`
- **Core Models**: `ImageProjModel`, `MLPProjModel`, `EmbeddingAdapter`
- **Advanced Token Mixing**: Separate scaling controls for text and IP tokens
- **Multiple Embedding Types**: Support for CLIP, DINO, and custom embeddings
- **SDXL Support**: Full SDXL compatibility with pooled embeddings
- **No Dependencies**: No original IP-Adapter library needed

### Files:
- `degis_ip_adapter_patch.py`: Complete IP-Adapter implementation with DEGIS enhancements
- `__init__.py`: Exports all classes directly

## Usage

### In Notebooks:
```python
# Import DEGIS IP-Adapter implementation directly
from ip_adapter_patch import IPAdapter, IPAdapterXL
# or
from ip_adapter_patch.degis_ip_adapter_patch import IPAdapter, IPAdapterXL

# Enhanced features are now available
adapter = IPAdapter(
    sd_pipe=pipe,
    image_encoder_path="path/to/encoder",
    ip_ckpt="path/to/checkpoint",
    device=device,
    embedding_type='clip'  # New parameter
)

# Use enhanced generation with separate scaling
images = adapter.generate(
    pil_image=image,
    prompt="a beautiful landscape",
    attn_ip_scale=1.0,      # Per-layer IP attention scale
    text_token_scale=1.0,    # Text token magnitude
    ip_token_scale=1.0,      # IP token magnitude
    zero_ip_in_uncond=False  # Clean separation option
)
```

### In Python Scripts:
```python
# Same as notebooks
from ip_adapter_patch import IPAdapter, IPAdapterXL
```

## What Changed

### Original IP-Adapter:
- Basic CLIP-based image projection
- Simple token concatenation
- Limited scaling options
- Separate classes for different variants

### DEGIS-Enhanced IP-Adapter:
- Complete IP-Adapter implementation
- Multiple embedding type support (CLIP, DINO, custom)
- Advanced token mixing with separate controls
- Pre-computed embedding support
- Full SDXL compatibility
- Unified API across all variants
- Backward compatible with original API

## Installation

The DEGIS IP-Adapter implementation is available when you:
1. Run the setup script: `./setup.sh`
2. Import directly: `from ip_adapter_patch import IPAdapter`

No original IP-Adapter library needed!

## Testing

To test that the implementation is working:
```bash
# Activate environment first
source degis-env/bin/activate

# Run test
python test_ip_adapter_patch.py
```

## License

This DEGIS implementation is based on the original IP-Adapter library (Apache-2.0 licensed) with additional functionality while maintaining full backward compatibility.
