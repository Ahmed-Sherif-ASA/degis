# DEGIS Package

**DEGIS: Disentangled Embeddings for Generative Image Synthesis**

A Python package for training disentangled representations of images using CLIP embeddings and various visual features (color histograms, edge maps, etc.).

## Installation

```bash
# Install in development mode
pip install -e .

# Or install with poetry
poetry install
```

## Quick Start

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

## Examples

See `example_usage.py` for complete examples of both CLI and Python API usage.
