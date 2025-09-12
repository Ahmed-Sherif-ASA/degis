# DEGIS IP-Adapter Patch

This directory contains DEGIS enhancements for the IP-Adapter library via monkey patching.

## What's Included

### Enhanced Features:
- **EmbeddingAdapter**: Universal adapter for different embedding types (CLIP, DINO, custom)
- **Advanced Token Mixing**: Separate scaling controls for text and IP tokens
- **Pre-computed Embeddings**: Support for using pre-computed CLIP embeddings
- **Backward Compatibility**: All original functionality preserved

### Files:
- `ip_adapter_monkey_patch.py`: Main DEGIS patch file with all enhancements
- `__init__.py`: Auto-applies DEGIS patches when imported

## Usage

### In Notebooks:
```python
# Import DEGIS patches (this applies the monkey patches automatically)
import ip_adapter_patch

# Now use IP-Adapter as normal - all DEGIS enhancements are available
from ip_adapter import IPAdapter, IPAdapterXL

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
import ip_adapter_patch
from ip_adapter import IPAdapter
```

## What Changed

### Original IP-Adapter:
- Basic CLIP-based image projection
- Simple token concatenation
- Limited scaling options

### DEGIS-Enhanced IP-Adapter:
- Multiple embedding type support
- DEGIS token mixing with separate controls
- Pre-computed embedding support
- Backward compatible API

## Installation

The DEGIS patches are automatically applied when you:
1. Run the setup script: `./setup.sh`
2. Import the patches: `import ip_adapter_patch`

No additional installation steps required!

## Testing

To test that patches are working:
```bash
# Activate environment first
source degis-env/bin/activate

# Run test
python test_ip_adapter_patch.py
```

## License

This DEGIS monkey patch enhances the original IP-Adapter library (Apache-2.0 licensed) with additional functionality while maintaining full backward compatibility.
