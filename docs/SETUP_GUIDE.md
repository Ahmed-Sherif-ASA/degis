# DEGIS Setup Guide

This guide provides multiple setup options for the DEGIS project, ensuring easy installation regardless of the system configuration.

## Quick Start (Recommended)

### Automated Setup Script

**Linux/Mac/Server:**
```bash
chmod +x setup.sh
./setup.sh
```

This single script works on all platforms and handles:
- Poetry/pip installation
- Virtual environment setup
- Jupyter kernel configuration
- Disk space management
- GPU/CUDA support

### Option 2: Manual Setup with Poetry

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install the project
poetry install
poetry install --with dev

# Models will be downloaded automatically when first used

# Start Jupyter
poetry run jupyter lab
```

### Option 3: Manual Setup with pip (Fallback)

```bash
# Create virtual environment
python3 -m venv degis-env
source degis-env/bin/activate  # Linux/Mac
# or
degis-env\Scripts\activate     # Windows

# Install the package
pip install -e .

# Install Jupyter
pip install jupyter jupyterlab ipykernel

# Models will be downloaded automatically when first used

# Start Jupyter
jupyter lab
```

## Running the Notebooks

1. **Start Jupyter:**
   ```bash
   # With Poetry
   poetry run jupyter lab
   
   # With pip
   jupyter lab
   ```

2. **Open the notebooks:**
   - `01_data_extraction_and_training.ipynb` - Data extraction and model training
   - `02_image_generation_ipadapter.ipynb` - Image generation with IP-Adapter (SD 1.5)
   - `02b_image_generation_ipadapter_xl.ipynb` - Image generation with IP-Adapter XL

3. **Select the correct kernel:**
   - If using Poetry: Select "DEGIS (Python 3)" kernel
   - If using pip: Select "Python 3" kernel

## Troubleshooting

### Poetry Issues
If Poetry doesn't work, use the pip approach (Option 3 above).

### Model Download Issues
If models fail to download, you can manually download them:
```bash
# The models will be cached in model-cache/ directory
# You can also set a custom cache directory:
export DEGIS_CACHE_DIR="/path/to/cache"
```

### Jupyter Kernel Issues
If notebooks don't work, try:
```bash
# Install kernel manually
python -m ipykernel install --user --name=degis --display-name="DEGIS (Python 3)"
```

## Package Information

- **Python Version:** 3.10+
- **Main Dependencies:** PyTorch, Transformers, Diffusers, OpenCV
- **Development Dependencies:** Jupyter, JupyterLab, pytest
- **Package Size:** ~2GB (including models)

## Getting Help

If you encounter any issues:
1. Check the main README_PACKAGE.md
2. Try the pip approach if Poetry fails
3. Contact the author: A.Sherif Akl

## What This Project Does

DEGIS (Disentangled Embeddings Guided Synthesis) is a Python package for:
- Training disentangled representations of images using CLIP embeddings
- Generating color histograms and edge maps
- Training color disentanglement models
- Generating images using IP-Adapter with color and layout control

The project includes both CLI tools and Jupyter notebooks for interactive exploration.
