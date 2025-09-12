# DEGIS Package

**DEGIS: Disentangled Embeddings for Generative Image Synthesis**

A Python package for training disentangled representations of images using CLIP embeddings and various visual features (color histograms, edge maps, etc.).

## System Requirements

**Before installation, ensure you have:**
- **Python 3.10-3.12** (3.12.3 tested and recommended)
- **Disk Space:** At least **15GB free** (for PyTorch, CUDA, and dependencies)
- **RAM:** At least **8GB** (16GB+ recommended for training)
- **GPU:** NVIDIA GPU with CUDA support (**strongly recommended** for image generation)

**⚠️ Note:** While IP-Adapter and ControlNet can technically run on CPU, performance will be extremely slow (10-100x slower) and may encounter compatibility issues. For practical use, GPU acceleration is essential.

**💡 Check available space:**
```bash
df -h .  # Check current directory space
```

## Quick Setup (Recommended)

**For the easiest setup, run this script:**

```bash
# Linux/Mac/Server
./setup.sh
```

**Note:** The setup script automatically installs IP-Adapter with DEGIS enhancements via monkey patches. See `ip_adapter_patch/README.md` for details about the enhancements.

This script automatically:
- Installs all dependencies (including open-clip-torch)
- Sets up Jupyter notebooks with proper kernel
- Handles disk space management
- Supports both Poetry and pip installation
- Works on local machines and servers with GPU/CUDA

**⚠️ Important: After running the setup script, you need to select the kernel in VS Code:**
1. Open any `.ipynb` notebook
2. Click on the kernel name in the top-right corner
3. Select **"DEGIS Environment"** from the dropdown
4. The notebook will now use the correct environment

**📝 Note: The notebooks now have a simple environment check cell instead of automatic setup.**
- Run the first cell to verify your environment is properly configured
- No more package installation errors or conflicts

**📖 For detailed setup instructions and troubleshooting, see [SETUP_GUIDE.md](SETUP_GUIDE.md)**

## Manual Installation

### Option 1: Poetry (Recommended)

```bash
# Install Poetry if you don't have it
curl -sSL https://install.python-poetry.org | python3 -

# Install the package and dependencies
poetry install

# Install development dependencies (Jupyter, etc.)
poetry install --with dev

# Models will be downloaded automatically when first used

# Set up Jupyter kernel
poetry run python -m ipykernel install --user --name=degis --display-name="DEGIS (Python 3)"
```

### Option 2: pip (Alternative)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install in development mode
pip install -e .

# Install Jupyter
pip install jupyter jupyterlab ipykernel

# Models will be downloaded automatically when first used
```

## Quick Start

### Running Jupyter Notebooks

```bash
# Start Jupyter Lab
poetry run jupyter lab

# Or start Jupyter Notebook
poetry run jupyter notebook
```

**Available notebooks:**
- `01_data_extraction_and_training.ipynb` - Data extraction and model training
- `02_image_generation_ipadapter.ipynb` - Image generation with IP-Adapter (SD 1.5)
- `02b_image_generation_ipadapter_xl.ipynb` - Image generation with IP-Adapter XL

### CLI Usage

The package provides several CLI commands for common tasks:

```bash
# Generate CLIP embeddings
degis-embeddings --model xl --batch-size 256

# Generate visual features (histograms, edge maps)
degis-features --type all --color-space all --bins 8

# Train color disentanglement model
degis-train-color --hist-kind hcl514 --epochs 200 --batch-size 128

# Train edge decoder model
degis-train-edge --epochs 200 --batch-size 512
```

### Python API Usage

```python
import degis
from torch.utils.data import DataLoader

# Generate embeddings
embeddings = degis.generate_xl_embeddings(
    csv_path="data.csv",
    output_path="embeddings.npy",
    batch_size=256
)

# Generate features
histograms = degis.generate_hcl_histograms(
    loader=loader,
    output_path="histograms.npy",
    bins=8
)

# Train models
results = degis.train_color_model(
    embeddings_path="embeddings.npy",
    histograms_path="histograms.npy",
    epochs=200
)
```

## Package Structure

```
degis/
├── __init__.py              # Main package exports
├── core/                    # Core functionality
│   ├── embeddings.py        # CLIP embeddings generation
│   ├── features.py          # Visual features generation
│   └── training.py          # Model training
├── cli/                     # CLI tools
│   ├── generate_embeddings.py
│   ├── generate_features.py
│   ├── train_color.py
│   └── train_edge.py
└── utils/                   # Utilities
    ├── logger.py
    ├── file_utils.py
    └── summarize_runs.py
```

## CLI Commands

### `degis-embeddings`

Generate CLIP embeddings from images.

```bash
degis-embeddings --help
```

**Options:**
- `--csv-path`: Path to CSV file with image paths
- `--output-path`: Path to save embeddings
- `--model`: CLIP model to use (base, xl)
- `--batch-size`: Batch size for processing
- `--num-workers`: Number of worker processes
- `--force-recompute`: Force recomputation even if output exists

### `degis-features`

Generate visual features from images.

```bash
degis-features --help
```

**Options:**
- `--type`: Type of features (histograms, edges, all)
- `--color-space`: Color space for histograms (rgb, lab, hcl, all)
- `--bins`: Number of histogram bins per dimension
- `--batch-size`: Batch size for processing
- `--force-recompute`: Force recomputation even if output exists

### `degis-train-color`

Train color disentanglement models.

```bash
degis-train-color --help
```

**Options:**
- `--embeddings-path`: Path to CLIP embeddings
- `--histograms-path`: Path to color histograms
- `--hist-kind`: Type of histograms (rgb512, lab514, hcl514)
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--lr`: Learning rate
- `--weight-decay`: Weight decay

### `degis-train-edge`

Train edge decoder models.

```bash
degis-train-edge --help
```

**Options:**
- `--embeddings-path`: Path to CLIP embeddings
- `--edge-maps-path`: Path to edge maps
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--lr`: Learning rate
- `--patience`: Early stopping patience

## Python API

### Core Functions

#### Embeddings

```python
# Generate base CLIP embeddings
embeddings = degis.generate_clip_embeddings(
    csv_path="data.csv",
    output_path="embeddings.npy",
    model="base"
)

# Generate XL CLIP embeddings
embeddings = degis.generate_xl_embeddings(
    csv_path="data.csv",
    output_path="embeddings_xl.npy"
)
```

#### Features

```python
# Generate color histograms
histograms = degis.generate_color_histograms(
    loader=loader,
    output_path="histograms.npy",
    color_space="hcl",
    bins=8
)

# Generate edge maps
edge_maps = degis.generate_edge_maps(
    loader=loader,
    output_path="edges.npy",
    method="canny"
)
```

#### Training

```python
# Train color model
results = degis.train_color_model(
    embeddings_path="embeddings.npy",
    histograms_path="histograms.npy",
    hist_kind="hcl514",
    epochs=200,
    batch_size=128
)

# Train edge model
results = degis.train_edge_model(
    embeddings_path="embeddings.npy",
    edge_maps_path="edges.npy",
    epochs=200,
    batch_size=512
)
```

## Migration from Old Main Files

The old main files have been restructured into this package:

- `main.py` → `degis-embeddings` + `degis-features`
- `xl_main.py` → `degis-embeddings --model xl`
- `colour_train_main.py` → `degis-train-color`
- `edge_train_main.py` → `degis-train-edge`

The commented code in `main.py` for histogram and edge generation is now available as `degis-features`.

## Configuration

The package uses the existing `config.py` for default paths and settings. You can override these by passing arguments to the functions or CLI commands.

## Model Cache

This project uses a local model cache directory (`model-cache/`) to store downloaded machine learning models.

### What Gets Cached

- **Stable Diffusion models** (SD 1.5, SDXL)
- **ControlNet models** (Canny, etc.)
- **CLIP models** (ViT-H-14, ViT-bigG-14)
- **IP-Adapter models** (when downloaded via script)

### Cache Management

Models are automatically cached when you:
1. Use any HuggingFace model in your code
2. Load pre-trained models with `from_pretrained()`
3. Run generation functions (IP-Adapter checkpoints)

**Cache Details:**
- **Size**: Can grow to several GB as models are downloaded
- **Location**: Local to this project (not system-wide)
- **Cleanup**: Safe to delete - models will be re-downloaded as needed
- **Git**: Contents are ignored, but directory and README are tracked

### Environment Variables

You can override the cache location by setting:
```bash
export DEGIS_CACHE_DIR="/path/to/custom/cache"
```

### First Time Setup

Models are downloaded automatically when first used. No manual download needed!

## Examples

See `example_usage.py` for complete examples of both CLI and Python API usage.
