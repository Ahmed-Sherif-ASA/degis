# Model Cache Directory

This directory contains cached machine learning models and will be populated when you first download models.

## What Gets Cached Here

- **Stable Diffusion models** (SD 1.5, SDXL)
- **ControlNet models** (Canny, etc.)
- **CLIP models** (ViT-H-14, ViT-bigG-14)
- **IP-Adapter models** (when downloaded via script)

## Usage

Models are automatically cached here when you:
1. Run `python scripts/download_models.py`
2. Use any HuggingFace model in your code
3. Load pre-trained models with `from_pretrained()`

## Cache Management

- **Size**: Can grow to several GB as models are downloaded
- **Location**: Local to this project (not system-wide)
- **Cleanup**: Safe to delete - models will be re-downloaded as needed
- **Git**: Contents are ignored, but directory and README are tracked

## Environment Variables

You can override the cache location by setting:
```bash
export DEGIS_CACHE_DIR="/path/to/custom/cache"
```

## First Time Setup

Run this to download all required models:
```bash
python scripts/download_models.py
```

This will populate the cache with all necessary models for the project.
