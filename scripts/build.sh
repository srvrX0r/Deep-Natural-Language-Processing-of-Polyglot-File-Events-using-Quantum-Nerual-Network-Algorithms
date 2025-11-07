#!/bin/bash
# Build script for Quantum Malware Detector
# Compiles Cython modules with maximum optimization

set -e  # Exit on error

echo "========================================"
echo "Quantum Malware Detector Build Script"
echo "========================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install build dependencies
echo "Installing build dependencies..."
pip install Cython numpy

# Install other dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/
rm -rf dist/
rm -rf *.egg-info
find . -name "*.so" -delete
find . -name "*.c" -path "*/src/*" -delete
find . -name "*.cpp" -path "*/src/*" -delete

# Build Cython extensions
echo "Building Cython extensions..."
export CFLAGS="-O3 -march=native -ffast-math -fopenmp"
export LDFLAGS="-fopenmp"

python3 setup.py build_ext --inplace

# Build package
echo "Building package..."
python3 setup.py sdist bdist_wheel

echo "========================================"
echo "Build completed successfully!"
echo "========================================"

# Show build artifacts
echo "Build artifacts:"
ls -lh dist/ || echo "No dist directory"
ls -lh src/**/*.so 2>/dev/null || echo "No .so files yet"

echo ""
echo "To install, run: pip install -e ."
echo "To run tests, run: pytest tests/ -v"
