# DEGIS: Disentagled Embeddings Guided Image Synthesis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DEGIS is a novel approach for image generation that leverages pre-trained disentangled semantic specific CLIP embeddings and Earth Mover's Distance (EMD) constraints to generate high-quality images with precise color control.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the setup
bash setup.sh

# Start generating images
python -m degis.training.cli.train --help
```

## 📖 Documentation

- **[Setup Guide](docs/SETUP_GUIDE.md)** - Detailed installation and configuration
- **[Training & Inference Workflow](docs/training_inference_workflow.md)** - How to use DEGIS
- **[IP-Adapter Patch](docs/IP_ADAPTER_PATCH_README.md)** - Patch implementation details

## 🎯 Key Features

- **Color-Aware Generation**: Precise control over color palettes using EMD constraints
- **Style Transfer**: Direct image-to-image style transfer with IP-Adapter
- **Data Efficient**: Leverages pre-trained CLIP embeddings for minimal data requirements
- **Multiple Color Spaces**: Support for RGB, LAB, and HCL color representations
- **Comprehensive Metrics**: EMD and cosine similarity evaluation

## 🏗️ Architecture

![Architecture](docs/Architecture.png)

## 📊 Results

DEGIS achieves state-of-the-art results in color-constrained image generation with:
- **Low EMD scores** (0.01-0.1) for color fidelity
- **High cosine similarity** (0.2-0.8) for semantic consistency
- **Fast generation** (6-8 seconds per image)

## 🔬 Research

This project implements the DEGIS methodology for precise color control. The approach combines:

1. **CLIP-based color embeddings** for semantic understanding
2. **EMD constraints** for color distribution matching
3. **IP-Adapter integration** for style transfer capabilities

## 📝 Citation

```bibtex
@article{degis2024,
  title={DEGIS: Disentagled Embeddings Guided Image Synthesis},
  author={Ahmed Sherif Akl},
  journal={-},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 🙏 Acknowledgments

- [CLIP](https://github.com/openai/CLIP) for vision-language understanding
- [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) for image prompt adaptation
- [Diffusers](https://github.com/huggingface/diffusers) for diffusion model integration
