@echo off
REM DEGIS Setup Script for Easy Installation (Windows)
REM Run this script to set up the DEGIS project

echo 🚀 Setting up DEGIS project...
echo ================================

REM Check if poetry is installed
poetry --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Poetry is not installed. Falling back to pip...
    echo 📦 Installing DEGIS package and dependencies with pip...
    
    REM Create virtual environment
    python -m venv degis-env
    call degis-env\Scripts\activate
    
    REM Install the package
    pip install -e .
    pip install -r requirements.txt
    
    REM Models will be downloaded automatically when needed
    echo 🤖 Models will be downloaded automatically when first used
    
    REM Set up Jupyter kernel
    echo 📓 Setting up Jupyter kernel...
    python -m ipykernel install --user --name=degis --display-name="DEGIS Environment"

REM Update notebook kernel metadata
echo 📝 Updating notebook kernel metadata...
python setup_kernel.py
    
    echo ✅ Setup complete with pip!
    echo To activate the environment: call degis-env\Scripts\activate
    pause
    exit /b 0
)

echo ✅ Poetry found

REM Try Poetry installation
echo 📦 Installing DEGIS package and dependencies...
poetry install
if %errorlevel% neq 0 (
    echo ❌ Poetry installation failed. Falling back to pip...
    echo 📦 Installing DEGIS package and dependencies with pip...
    
    REM Create virtual environment
    python -m venv degis-env
    call degis-env\Scripts\activate
    
    REM Install the package
    pip install -e .
    pip install -r requirements.txt
    
    REM Models will be downloaded automatically when needed
    echo 🤖 Models will be downloaded automatically when first used
    
    REM Set up Jupyter kernel
    echo 📓 Setting up Jupyter kernel...
    python -m ipykernel install --user --name=degis --display-name="DEGIS Environment"

REM Update notebook kernel metadata
echo 📝 Updating notebook kernel metadata...
python setup_kernel.py
    
    echo ✅ Setup complete with pip!
    echo To activate the environment: call degis-env\Scripts\activate
    pause
    exit /b 0
)

REM Install development dependencies (including Jupyter)
echo 🔧 Installing development dependencies (Jupyter, etc.)...
poetry install --with dev

REM Models will be downloaded automatically when needed
echo 🤖 Models will be downloaded automatically when first used

REM Set up Jupyter kernel
echo 📓 Setting up Jupyter kernel...
poetry run python -m ipykernel install --user --name=degis --display-name="DEGIS Environment"

REM Update notebook kernel metadata
echo 📝 Updating notebook kernel metadata...
poetry run python setup_kernel.py

echo.
echo 🎉 Setup complete!
echo ================================
echo.
echo To get started:
echo 1. Activate the environment: poetry shell
echo 2. Run Jupyter: poetry run jupyter lab
echo 3. Or run CLI commands: poetry run degis-embeddings --help
echo.
echo Available notebooks:
echo - 01_data_extraction_and_training.ipynb
echo - 02_image_generation_ipadapter.ipynb
echo - 02b_image_generation_ipadapter_xl.ipynb
echo.
echo For more help, see README_PACKAGE.md
pause
