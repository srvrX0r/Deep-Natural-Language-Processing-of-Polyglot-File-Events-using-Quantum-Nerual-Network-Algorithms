#!/bin/bash
# Installation script for Quantum Malware Detector

set -e

echo "========================================"
echo "Quantum Malware Detector Installation"
echo "========================================"

# Check if running as root for system-wide installation
if [ "$EUID" -eq 0 ]; then
    echo "Running as root - system-wide installation"
    INSTALL_TYPE="system"
else
    echo "Running as user - user installation"
    INSTALL_TYPE="user"
fi

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    OS_VERSION=$VERSION_ID
    echo "Detected OS: $OS $OS_VERSION"
fi

# Install system dependencies
echo "Installing system dependencies..."
if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
    if [ "$INSTALL_TYPE" = "system" ]; then
        apt-get update
        apt-get install -y python3 python3-dev python3-pip \
            build-essential gcc g++ libgomp1 \
            libmagic1 libmagic-dev
    else
        echo "Please install system dependencies manually:"
        echo "  sudo apt-get install python3 python3-dev build-essential gcc g++ libgomp1"
    fi
elif [ "$OS" = "rhel" ] || [ "$OS" = "centos" ] || [ "$OS" = "fedora" ]; then
    if [ "$INSTALL_TYPE" = "system" ]; then
        yum install -y python3 python3-devel gcc gcc-c++ libgomp file-devel
    else
        echo "Please install system dependencies manually:"
        echo "  sudo yum install python3 python3-devel gcc gcc-c++ libgomp"
    fi
elif [ "$OS" = "arch" ]; then
    if [ "$INSTALL_TYPE" = "system" ]; then
        pacman -Sy --noconfirm python python-pip gcc file
    else
        echo "Please install system dependencies manually:"
        echo "  sudo pacman -S python python-pip gcc"
    fi
fi

# Build the package
echo "Building package..."
bash scripts/build.sh

# Install the package
echo "Installing package..."
if [ "$INSTALL_TYPE" = "system" ]; then
    pip3 install -e .
else
    pip3 install --user -e .
fi

# Create necessary directories
echo "Creating directories..."
if [ "$INSTALL_TYPE" = "system" ]; then
    mkdir -p /var/lib/qnn_detector
    mkdir -p /var/log/qnn_detector
    mkdir -p /var/quarantine/qnn_detector
    mkdir -p /etc/qnn_detector

    # Copy configuration
    cp config/production.yaml /etc/qnn_detector/config.yaml

    echo "System directories created:"
    echo "  Config: /etc/qnn_detector/"
    echo "  Data: /var/lib/qnn_detector/"
    echo "  Logs: /var/log/qnn_detector/"
    echo "  Quarantine: /var/quarantine/qnn_detector/"
else
    mkdir -p ~/.qnn_detector/{data,logs,quarantine}
    cp config/development.yaml ~/.qnn_detector/config.yaml

    echo "User directories created:"
    echo "  Config: ~/.qnn_detector/"
    echo "  Data: ~/.qnn_detector/data/"
    echo "  Logs: ~/.qnn_detector/logs/"
fi

echo "========================================"
echo "Installation completed successfully!"
echo "========================================"
echo ""
echo "To scan a file:"
echo "  qnn-scan /path/to/file"
echo ""
echo "To scan a directory:"
echo "  qnn-scan /path/to/directory -r"
echo ""
echo "For help:"
echo "  qnn-scan --help"
