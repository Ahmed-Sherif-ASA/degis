#!/bin/bash
# DEGIS Setup Script for Easy Installation
# Run this script to set up the DEGIS project

set -e  # Exit on any error

echo "🚀 Setting up DEGIS project..."
echo "================================"

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry is not installed. Falling back to pip..."
    echo "📦 Installing DEGIS package and dependencies with pip..."
    
    # Create virtual environment (ignored by git)
    python3 -m venv degis-env
    source degis-env/bin/activate
    
    # Install the package
    pip install -e .
    pip install -r requirements.txt
    
    # Models will be downloaded automatically when needed
    echo "🤖 Models will be downloaded automatically when first used"
    
    # Set up Jupyter kernel
    echo "📓 Setting up Jupyter kernel..."
    python -m ipykernel install --user --name=degis --display-name="DEGIS Environment"
    
    # Update notebook kernel metadata
    echo "📝 Updating notebook kernel metadata..."
    python setup_kernel.py
    
    echo "✅ Setup complete with pip!"
    echo "To activate the environment: source degis-env/bin/activate"
    exit 0
fi

echo "✅ Poetry found"

# Try Poetry installation
echo "📦 Installing DEGIS package and dependencies..."
if poetry install; then
    echo "🔧 Installing development dependencies (Jupyter, etc.)..."
    poetry install --with dev
    
    # Models will be downloaded automatically when needed
    echo "🤖 Models will be downloaded automatically when first used"
    
    # Set up Jupyter kernel
    echo "📓 Setting up Jupyter kernel..."
    poetry run python -m ipykernel install --user --name=degis --display-name="DEGIS Environment"
    
    # Update notebook kernel metadata
    echo "📝 Updating notebook kernel metadata..."
    poetry run python setup_kernel.py
else
    echo "❌ Poetry installation failed. Falling back to pip..."
    echo "📦 Installing DEGIS package and dependencies with pip..."
    
    # Create virtual environment (ignored by git)
    python3 -m venv degis-env
    source degis-env/bin/activate
    
    # Install the package
    pip install -e .
    pip install -r requirements.txt
    
    # Models will be downloaded automatically when needed
    echo "🤖 Models will be downloaded automatically when first used"
    
    # Set up Jupyter kernel
    echo "📓 Setting up Jupyter kernel..."
    python -m ipykernel install --user --name=degis --display-name="DEGIS Environment"
    
    # Update notebook kernel metadata
    echo "📝 Updating notebook kernel metadata..."
    python setup_kernel.py
    
    echo "✅ Setup complete with pip!"
    echo "To activate the environment: source degis-env/bin/activate"
fi

echo ""
echo "🎉 Setup complete!"
echo "================================"
echo ""
echo "To get started:"
echo "1. Activate the environment: poetry shell"
echo "2. Run Jupyter: poetry run jupyter lab"
echo "3. Or run CLI commands: poetry run degis-embeddings --help"
echo ""
echo "Available notebooks:"
echo "- 01_data_extraction_and_training.ipynb"
echo "- 02_image_generation_ipadapter.ipynb"
echo "- 02b_image_generation_ipadapter_xl.ipynb"
echo ""
echo "For more help, see README_PACKAGE.md"
