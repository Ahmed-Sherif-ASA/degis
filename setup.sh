#!/bin/bash
# DEGIS Setup Script for Server Environment with GPU/CUDA

set -e  # Exit on any error

echo "Setting up DEGIS project on server (FIXED VERSION)..."
echo "========================================================"

# Check available disk space
echo "Checking disk space..."
df -h .
echo ""

# Check if we have enough space (need at least 15GB for PyTorch + dependencies)
AVAILABLE_SPACE=$(df . | tail -1 | awk '{print $4}')
if [ "$AVAILABLE_SPACE" -lt 15728640 ]; then  # 15GB in KB
    echo "Error: Insufficient disk space detected."
    echo "   Available: $(df -h . | tail -1 | awk '{print $4}')"
    echo "   Required: At least 15GB free"
    echo ""
    echo "Solutions:"
    echo "   1. Free up space: du -sh * | sort -hr | head -10"
    echo "   2. Clean up old files: find . -name '*.pyc' -delete"
    echo "   3. Remove old runs: rm -rf runs/*/checkpoints/"
    exit 1
fi

# Check Python version
echo "Checking Python version..."
python3 --version
echo ""

# Set up pip cache directory in project space
export PIP_CACHE_DIR="$(pwd)/.pip-cache"
mkdir -p "$PIP_CACHE_DIR"
echo "Using pip cache: $PIP_CACHE_DIR"

# Set up temporary directory in project space
export TMPDIR="$(pwd)/.tmp"
mkdir -p "$TMPDIR"
echo "Using temp directory: $TMPDIR"

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry is not installed. Installing Poetry..."
    
    # Install Poetry
    curl -sSL https://install.python-poetry.org | python3 -
    
    # Add to PATH
    export PATH="$HOME/.local/bin:$PATH"
    
    # Verify installation
    if ! command -v poetry &> /dev/null; then
        echo "Poetry installation failed. Falling back to pip..."
        USE_PIP=true
    else
        echo "Poetry installed successfully"
        USE_PIP=false
    fi
else
    echo "Poetry found"
    USE_PIP=false
fi

# Update Poetry lock file if using Poetry
if [ "$USE_PIP" = false ]; then
    echo "Updating Poetry lock file..."
    if poetry lock; then
        echo "Poetry lock file updated"
    else
        echo " Poetry lock failed, falling back to pip..."
        USE_PIP=true
    fi
fi

# Install using Poetry or pip
if [ "$USE_PIP" = false ]; then
    echo "Installing DEGIS package and dependencies with Poetry..."
    if poetry install; then
        echo "Installing development dependencies (Jupyter, etc.)..."
        poetry install --with dev
        
        # Set up Jupyter kernel
        echo "Setting up Jupyter kernel..."
        poetry run python -m ipykernel install --user --name=degis --display-name="DEGIS Environment"
        
        echo "Setup complete with Poetry!"
        echo "To activate: poetry shell"
        echo "To run Jupyter: poetry run jupyter lab"
    else
        echo "Poetry installation failed. Falling back to pip..."
        USE_PIP=true
    fi
fi

# Fallback to pip installation
if [ "$USE_PIP" = true ]; then
    echo "Installing DEGIS package and dependencies with pip..."
    
    # Create virtual environment (ignored by git)
    echo "Creating virtual environment..."
    python3 -m venv degis-env
    source degis-env/bin/activate
    
    # Upgrade pip first
    echo " Upgrading pip..."
    pip install --upgrade pip --cache-dir "$PIP_CACHE_DIR"
    
    # Install the package
    echo "Installing DEGIS package..."
    pip install -e . --cache-dir "$PIP_CACHE_DIR"
    
    # Install Jupyter
    echo "Installing Jupyter..."
    pip install jupyter jupyterlab ipykernel --cache-dir "$PIP_CACHE_DIR"
    
    # Set up Jupyter kernel
    echo "Setting up Jupyter kernel..."
    python -m ipykernel install --user --name=degis --display-name="DEGIS Environment"
    
    echo "Setup complete with pip!"
    echo "To activate: source degis-env/bin/activate"
    echo "To run Jupyter: jupyter lab"
fi

echo ""
echo "Setup complete!"
echo "========================================================"
echo ""
echo "System Info:"
echo "   Python: $(python3 --version)"
echo "   Available space: $(df -h . | tail -1 | awk '{print $4}')"
echo "   Pip cache: $PIP_CACHE_DIR"
echo "   Temp directory: $TMPDIR"
echo ""
echo "Next steps:"
echo "1. Activate the environment"
echo "2. Run: jupyter lab"
echo "3. Open any notebook and run the first cell"
echo ""
echo "For more help, see README_PACKAGE.md"
