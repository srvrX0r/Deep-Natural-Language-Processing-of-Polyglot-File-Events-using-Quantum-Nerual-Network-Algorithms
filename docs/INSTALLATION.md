# Installation Guide

Complete installation guide for the Quantum Neural Network Malware Detector.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Install](#quick-install)
3. [Manual Installation](#manual-installation)
4. [Platform-Specific Instructions](#platform-specific-instructions)
5. [Docker Installation](#docker-installation)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)
8. [Uninstallation](#uninstallation)

---

## System Requirements

### Minimum Requirements

- **OS**: Linux (Ubuntu 20.04+, CentOS 8+, Debian 11+)
- **CPU**: x86_64 with SSE2 support
- **RAM**: 4 GB
- **Disk**: 2 GB free space
- **Python**: 3.8 or higher
- **Compiler**: GCC 7+ or Clang 10+

### Recommended Requirements

- **OS**: Ubuntu 22.04 LTS or later
- **CPU**: x86_64 with AVX2 support, 4+ cores
- **RAM**: 8 GB or more
- **Disk**: 5 GB free space (SSD recommended)
- **Python**: 3.10 or higher
- **Compiler**: GCC 11+ with OpenMP support

### Dependencies

#### System Libraries

- `build-essential` (GCC, G++, make)
- `python3-dev`
- `libgomp1` (OpenMP runtime)
- `libmagic1` (file type detection)

#### Python Packages

All Python dependencies are listed in `requirements.txt`:

```
numpy>=1.21.0
scipy>=1.7.0
Cython>=0.29.32
scikit-learn>=1.0.0
pycryptodome>=3.15.0
cryptography>=38.0.0
python-magic>=0.4.27
psutil>=5.9.0
pyyaml>=6.0
```

---

## Quick Install

### Option 1: Automated Installation Script

```bash
# Clone repository
git clone https://github.com/srvrX0r/Deep-Natural-Language-Processing-of-Polyglot-File-Events-using-Quantum-Nerual-Network-Algorithms.git
cd Deep-Natural-Language-Processing-of-Polyglot-File-Events-using-Quantum-Nerual-Network-Algorithms

# Run installation script
bash scripts/install.sh
```

The script will:
1. Detect your OS
2. Install system dependencies
3. Create virtual environment
4. Build Cython extensions
5. Install the package
6. Verify installation

### Option 2: Using pip (from source)

```bash
# Clone repository
git clone https://github.com/srvrX0r/Deep-Natural-Language-Processing-of-Polyglot-File-Events-using-Quantum-Nerual-Network-Algorithms.git
cd Deep-Natural-Language-Processing-of-Polyglot-File-Events-using-Quantum-Nerual-Network-Algorithms

# Install
pip install .
```

---

## Manual Installation

### Step 1: Install System Dependencies

#### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    libmagic1 \
    libmagic-dev
```

#### RHEL/CentOS/Fedora

```bash
sudo yum install -y \
    python3 \
    python3-devel \
    gcc \
    gcc-c++ \
    libgomp \
    file-devel
```

#### Arch Linux

```bash
sudo pacman -S \
    python \
    python-pip \
    gcc \
    file
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install Cython and NumPy first (required for building)
pip install Cython numpy

# Install all dependencies
pip install -r requirements.txt
```

### Step 4: Build Cython Extensions

```bash
# Build with optimizations
export CFLAGS="-O3 -march=native -ffast-math -fopenmp"
export LDFLAGS="-fopenmp"

# Run build script
bash scripts/build.sh

# Or manually
python setup.py build_ext --inplace
```

### Step 5: Install Package

```bash
# Development installation (editable)
pip install -e .

# Or production installation
pip install .
```

### Step 6: Verify Installation

```bash
# Test CLI
qnn-scan --help
qnn-scan-advanced --help

# Test Python import
python -c "from src import QuantumMalwareDetector; print('Success!')"

# Run tests
pytest tests/ -v
```

---

## Platform-Specific Instructions

### Ubuntu 20.04 LTS

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y python3.8 python3.8-dev python3-pip build-essential libgomp1

# Clone and install
git clone https://github.com/srvrX0r/Deep-Natural-Language-Processing-of-Polyglot-File-Events-using-Quantum-Nerual-Network-Algorithms.git
cd Deep-Natural-Language-Processing-of-Polyglot-File-Events-using-Quantum-Nerual-Network-Algorithms
bash scripts/install.sh
```

### Ubuntu 22.04 LTS (Recommended)

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-dev python3-pip build-essential libgomp1

# Clone and install
git clone https://github.com/srvrX0r/Deep-Natural-Language-Processing-of-Polyglot-File-Events-using-Quantum-Nerual-Network-Algorithms.git
cd Deep-Natural-Language-Processing-of-Polyglot-File-Events-using-Quantum-Nerual-Network-Algorithms
bash scripts/install.sh
```

### CentOS 8 / Rocky Linux 8

```bash
# Install dependencies
sudo yum install -y python39 python39-devel gcc gcc-c++ make

# Clone and install
git clone https://github.com/srvrX0r/Deep-Natural-Language-Processing-of-Polyglot-File-Events-using-Quantum-Nerual-Network-Algorithms.git
cd Deep-Natural-Language-Processing-of-Polyglot-File-Events-using-Quantum-Nerual-Network-Algorithms
python3.9 -m venv venv
source venv/bin/activate
pip install --upgrade pip
bash scripts/build.sh
pip install -e .
```

### macOS (Intel)

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.10 gcc libomp

# Clone and install
git clone https://github.com/srvrX0r/Deep-Natural-Language-Processing-of-Polyglot-File-Events-using-Quantum-Nerual-Network-Algorithms.git
cd Deep-Natural-Language-Processing-of-Polyglot-File-Events-using-Quantum-Nerual-Network-Algorithms

# Set compiler
export CC=gcc-11
export CXX=g++-11

bash scripts/install.sh
```

### Windows (WSL2 Recommended)

For Windows, we recommend using WSL2 (Windows Subsystem for Linux):

```powershell
# Install WSL2
wsl --install -d Ubuntu-22.04

# Launch Ubuntu
wsl

# Follow Ubuntu installation instructions
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-dev build-essential
```

---

## Docker Installation

### Using Docker

```bash
# Build Docker image
docker build -t qnn-malware-detector .

# Run container
docker run -it --rm \
    -v /path/to/scan:/data \
    qnn-malware-detector \
    qnn-scan /data/suspicious_file.exe
```

### Dockerfile

```dockerfile
FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone repository
WORKDIR /app
RUN git clone https://github.com/srvrX0r/Deep-Natural-Language-Processing-of-Polyglot-File-Events-using-Quantum-Nerual-Network-Algorithms.git .

# Build and install
RUN pip3 install --upgrade pip && \
    bash scripts/build.sh && \
    pip3 install -e .

# Set entrypoint
ENTRYPOINT ["qnn-scan-advanced"]
CMD ["--help"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  qnn-detector:
    build: .
    volumes:
      - ./scan_data:/data
      - ./results:/results
    command: qnn-scan-advanced /data -r -o /results/scan_results.json
```

---

## Production Deployment

### System-Wide Installation

```bash
# Run as root
sudo bash scripts/deploy_production.sh

# This will:
# - Install to /opt/qnn_detector
# - Create service user
# - Install systemd service
# - Configure logrotate
# - Set up security policies
```

### Systemd Service

After production deployment:

```bash
# Start service
sudo systemctl start qnn-detector

# Enable on boot
sudo systemctl enable qnn-detector

# Check status
sudo systemctl status qnn-detector

# View logs
sudo journalctl -u qnn-detector -f
```

### Configuration

Production configuration:

```bash
# Edit configuration
sudo nano /etc/qnn_detector/config.yaml

# Restart service
sudo systemctl restart qnn-detector
```

---

## Verification

### Test Installation

```bash
# 1. Check CLI tools
which qnn-scan
which qnn-scan-advanced

# 2. Test Python import
python -c "
from src import QuantumMalwareDetector, AdvancedQuantumMalwareDetector
print('Standard detector: OK')
print('Advanced detector: OK')
"

# 3. Run basic scan
qnn-scan --help

# 4. Run tests
pytest tests/ -v

# 5. Check Cython compilation
python -c "
import sys
try:
    from src.qnn_core import quantum_layer
    from src.qnn_core import advanced_quantum
    print('Cython modules: OK')
except ImportError as e:
    print(f'Cython modules: FAILED - {e}')
"

# 6. Performance test
python scripts/benchmark.py
```

### Expected Output

```
✓ CLI tools installed
✓ Python imports successful
✓ Cython modules compiled
✓ Tests passing
✓ Performance benchmarks complete
```

---

## Troubleshooting

### Issue: Cython compilation fails

**Error**: `fatal error: Python.h: No such file or directory`

**Solution**:
```bash
# Install Python development headers
sudo apt-get install python3-dev  # Ubuntu
sudo yum install python3-devel     # RHEL/CentOS
```

### Issue: OpenMP not found

**Error**: `fatal error: omp.h: No such file or directory`

**Solution**:
```bash
# Install OpenMP
sudo apt-get install libgomp1      # Ubuntu
sudo yum install libgomp           # RHEL/CentOS
brew install libomp                # macOS
```

### Issue: ImportError for compiled modules

**Error**: `ImportError: No module named 'src.qnn_core.quantum_layer'`

**Solution**:
```bash
# Rebuild modules
python setup.py clean --all
bash scripts/build.sh
pip install -e .
```

### Issue: Permission denied

**Error**: `PermissionError: [Errno 13] Permission denied`

**Solution**:
```bash
# Use virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
pip install -e .

# Or install with --user flag
pip install --user -e .
```

### Issue: Out of memory during build

**Solution**:
```bash
# Reduce parallel jobs
export CYTHON_NTHREADS=1
python setup.py build_ext --inplace
```

### Issue: Tests failing

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run with verbose output
pytest tests/ -v -s

# Run specific test
pytest tests/test_qnn_core.py::test_initialization -v
```

---

## Uninstallation

### Remove Package

```bash
# Uninstall package
pip uninstall quantum-malware-detector

# Remove virtual environment
deactivate
rm -rf venv/

# Remove source (if cloned)
cd ..
rm -rf Deep-Natural-Language-Processing-of-Polyglot-File-Events-using-Quantum-Nerual-Network-Algorithms/
```

### Remove Production Installation

```bash
# Stop and disable service
sudo systemctl stop qnn-detector
sudo systemctl disable qnn-detector

# Remove files
sudo rm /etc/systemd/system/qnn-detector.service
sudo rm -rf /opt/qnn_detector
sudo rm -rf /var/lib/qnn_detector
sudo rm -rf /var/log/qnn_detector
sudo rm -rf /etc/qnn_detector

# Remove user (optional)
sudo userdel qnn_detector

# Reload systemd
sudo systemctl daemon-reload
```

---

## Getting Help

If you encounter issues:

1. **Check Documentation**: Review README.md, USAGE.md, and TROUBLESHOOTING.md
2. **Search Issues**: Check [GitHub Issues](https://github.com/srvrX0r/Deep-Natural-Language-Processing-of-Polyglot-File-Events-using-Quantum-Nerual-Network-Algorithms/issues)
3. **Report Bug**: Open a new issue with:
   - OS and version
   - Python version
   - Error messages
   - Steps to reproduce

---

## Next Steps

After installation:

1. Read [USAGE.md](../USAGE.md) for usage examples
2. Review [API.md](API.md) for API documentation
3. Check [ADVANCED_FEATURES.md](../ADVANCED_FEATURES.md) for v2.0 features
4. Run [examples/](../examples/) for practical examples

---

**Happy Scanning!** 🚀
