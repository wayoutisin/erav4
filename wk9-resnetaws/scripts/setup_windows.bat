@echo off
REM Setup script for Windows local development environment

echo Setting up ResNet18 training environment on Windows...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo Error: pip is not available
    pause
    exit /b 1
)

echo Python installation found.

REM Create virtual environment
echo Creating virtual environment...
if not exist "venv" (
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch (CPU version for Windows, adjust if you have CUDA)
echo Installing PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

REM Install other requirements
echo Installing other dependencies...
pip install -r requirements.txt

REM Create necessary directories
echo Creating project directories...
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "outputs" mkdir outputs
if not exist "outputs\logs" mkdir outputs\logs

REM Create sample data directory structure
if not exist "data\sample" mkdir data\sample
if not exist "data\sample\class1" mkdir data\sample\class1
if not exist "data\sample\class2" mkdir data\sample\class2

echo.
echo === Setup Summary ===
echo ✓ Virtual environment created: venv\
echo ✓ PyTorch installed (CPU version)
echo ✓ All dependencies installed
echo ✓ Project directories created
echo.
echo === Next Steps ===
echo 1. Activate environment: venv\Scripts\activate.bat
echo 2. Place your dataset in data\ folder
echo 3. Run training: python -m src.training.train --config configs\training_config.json
echo 4. Or use Jupyter: jupyter notebook notebooks\resnet18_training.ipynb
echo.
echo === Useful Commands ===
echo Activate environment: venv\Scripts\activate.bat
echo Deactivate environment: deactivate
echo Install CUDA PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo.
echo Setup completed successfully! 🎉
pause