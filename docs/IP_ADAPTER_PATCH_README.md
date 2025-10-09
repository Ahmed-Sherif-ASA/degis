# DEGIS IP-Adapter Implementation

This directory contains a complete IP-Adapter implementation with DEGIS (Disentangled Embeddings Guided Image Synthesis) enhancements, providing advanced multi-embedding support, fine-grained control, and robust error handling.

## Key Features

### Complete Implementation:
- **Full IP-Adapter Replacement**: Complete replacement of all IP-Adapter classes
- **All Classes**: `IPAdapter`, `IPAdapterXL`, `IPAdapterPlus`, `IPAdapterFull`, `IPAdapterPlusXL`
- **Core Models**: `ImageProjModel`, `MLPProjModel`, `EmbeddingAdapter`
- **Advanced Token Mixing**: Separate scaling controls for text and IP tokens
- **Multiple Embedding Types**: Support for CLIP, DINO, and custom embeddings
- **SDXL Support**: Full SDXL compatibility with pooled embeddings
- **Robust Imports**: Fallback system for missing dependencies
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
    embedding_type='clip'
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
from ip_adapter_patch import IPAdapter, IPAdapterXL
```

## Implementation Details

### **Import Structure**
- **Global imports with fallbacks** - handles missing dependencies gracefully

### **Embedding Support**
- **Multi-modal support** - CLIP, DINO, and custom embeddings
- **EmbeddingAdapter class** - unified interface for different embedding types
- **Pre-computed embeddings** - can reuse pre-computed embeddings for performance

### **Token Mixing System**
- **Advanced scaling system** - separate scaling for text vs image tokens
- **Granular parameters** - 6+ parameters for fine-tuning generation quality
- **SDXL support** - advanced pooled embedding mixing for better quality

### **Generation Control**
- **Granular parameters** - control every aspect of generation
- **Multiple methods** - `generate()` + `generate_from_embeddings()` for different use cases
- **Comprehensive validation** - better error messages and validation


## New Features in DEGIS

### 1. **EmbeddingAdapter Class**
```python
class EmbeddingAdapter(torch.nn.Module):
    """Adapter for different embedding types"""
    
    def __init__(self, cross_attention_dim=1024, embedding_dim=1024, num_tokens=4):
        # Projection layers for different embedding types
        self.proj_layers = torch.nn.ModuleDict({
            'clip': ImageProjModel(...),
            'dino': MLPProjModel(...),
            'custom': torch.nn.Sequential(...)
        })
```
**Impact:** **Multi-embedding support** - use CLIP, DINO, or custom embeddings seamlessly.

### 2. **Advanced Token Mixing**
```python
def _mix_text_ip_tokens(
    self,
    prompt_embeds_text,          # (B, T_text, C)
    negative_prompt_embeds_text, # (B, T_text, C)
    image_prompt_embeds,         # (B, T_ip,   C)
    uncond_image_prompt_embeds,  # (B, T_ip,   C)
    *,
    text_token_scale=1.0,        # Scale text tokens
    ip_token_scale=1.0,          # Scale IP tokens
    ip_uncond_scale=None,        # Scale IP in negative prompt
    zero_ip_in_uncond=False,     # Clean separation option
    pooled_prompt_embeds=None,   # SDXL support
    negative_pooled_prompt_embeds=None,
):
```
**Impact:** **Fine-grained control** - separate scaling for different token types.

### 3. **Enhanced Generation Methods**
```python
def generate(
    self,
    pil_image=None,
    clip_image_embeds=None,
    prompt=None,
    negative_prompt=None,
    attn_ip_scale=1.0,        # Per-layer IP attention scale
    text_token_scale=1.0,     # Text token magnitude
    ip_token_scale=None,      # IP token magnitude
    ip_uncond_scale=None,     # IP token in negative prompt
    zero_ip_in_uncond=False,  # Clean separation option
    num_samples=4,
    seed=None,
    guidance_scale=7.5,
    num_inference_steps=30,
    **kwargs,
):
```
**Impact:** **Granular control** - fine-tune every aspect of generation.

### 4. **Pre-computed Embedding Support**
```python
def generate_from_embeddings(
    self,
    clip_image_embeds,  # Pre-computed embeddings
    prompt=None,
    negative_prompt=None,
    # ... all generation parameters
):
    """Generate images using pre-computed CLIP image embeddings."""
    return self.generate(
        clip_image_embeds=clip_image_embeds,
        # ... pass through all parameters
    )
```
**Impact:** **Performance optimization** - reuse pre-computed embeddings for faster generation.

## **Backward Compatibility**

The DEGIS implementation maintains **full backward compatibility** with the original IP-Adapter API:

```python
# Original API still works
adapter = IPAdapter(sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4)
images = adapter.generate(pil_image=image, prompt="a beautiful landscape", scale=1.0)

# New DEGIS features are optional
adapter = IPAdapter(sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4, embedding_type='clip')
images = adapter.generate(
    pil_image=image, 
    prompt="a beautiful landscape", 
    attn_ip_scale=1.0,
    text_token_scale=1.0,
    ip_token_scale=1.0
)
```

## Installation

The DEGIS IP-Adapter implementation is available when you:
1. Run the setup script: `./setup.sh`
2. Import directly: `from ip_adapter_patch import IPAdapter`


## Key Benefits

1. **Multi-Embedding Support**: Use CLIP, DINO, or custom embeddings seamlessly
2. **Fine-Grained Control**: Separate scaling for text vs image tokens
3. **Performance Optimization**: Pre-computed embedding support
4. **Backward Compatible**: Original API still works perfectly

## License

This DEGIS implementation is based on the original IP-Adapter library (Apache-2.0 licensed) with additional functionality while maintaining full backward compatibility.
