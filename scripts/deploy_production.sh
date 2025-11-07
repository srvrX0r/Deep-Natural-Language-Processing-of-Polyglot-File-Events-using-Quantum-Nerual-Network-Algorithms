#!/bin/bash
# Production deployment script for Quantum Malware Detector

set -e

echo "========================================"
echo "Production Deployment"
echo "========================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: This script must be run as root for production deployment"
    exit 1
fi

# Configuration
INSTALL_DIR="/opt/qnn_detector"
SERVICE_USER="qnn_detector"
SERVICE_GROUP="qnn_detector"

# Create service user
echo "Creating service user..."
if ! id "$SERVICE_USER" &>/dev/null; then
    useradd -r -s /bin/false -d "$INSTALL_DIR" "$SERVICE_USER"
fi

# Create installation directory
echo "Creating installation directory..."
mkdir -p "$INSTALL_DIR"
mkdir -p /var/lib/qnn_detector
mkdir -p /var/log/qnn_detector
mkdir -p /var/quarantine/qnn_detector
mkdir -p /etc/qnn_detector

# Install the application
echo "Installing application..."
cd "$(dirname "$0")/.."
bash scripts/install.sh

# Copy application files
echo "Copying application files..."
cp -r src "$INSTALL_DIR/"
cp -r config "$INSTALL_DIR/"
cp setup.py "$INSTALL_DIR/"
cp requirements.txt "$INSTALL_DIR/"

# Set permissions
echo "Setting permissions..."
chown -R "$SERVICE_USER:$SERVICE_GROUP" "$INSTALL_DIR"
chown -R "$SERVICE_USER:$SERVICE_GROUP" /var/lib/qnn_detector
chown -R "$SERVICE_USER:$SERVICE_GROUP" /var/log/qnn_detector
chown -R "$SERVICE_USER:$SERVICE_GROUP" /var/quarantine/qnn_detector

chmod 755 "$INSTALL_DIR"
chmod 750 /var/lib/qnn_detector
chmod 750 /var/log/qnn_detector
chmod 700 /var/quarantine/qnn_detector

# Install systemd service
echo "Installing systemd service..."
cat > /etc/systemd/system/qnn-detector.service <<EOF
[Unit]
Description=Quantum Neural Network Malware Detector
After=network.target

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_GROUP
WorkingDirectory=$INSTALL_DIR
Environment="PATH=/opt/qnn_detector/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=/opt/qnn_detector/venv/bin/python3 -m src.qnn_malware_detector --config /etc/qnn_detector/config.yaml
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/qnn_detector /var/log/qnn_detector /var/quarantine/qnn_detector

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
echo "Reloading systemd..."
systemctl daemon-reload

# Enable service
echo "Enabling service..."
systemctl enable qnn-detector.service

# Install logrotate configuration
echo "Installing logrotate configuration..."
cat > /etc/logrotate.d/qnn-detector <<EOF
/var/log/qnn_detector/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0640 $SERVICE_USER $SERVICE_GROUP
    sharedscripts
    postrotate
        systemctl reload qnn-detector.service > /dev/null 2>&1 || true
    endscript
}
EOF

# Create monitoring configuration
echo "Creating monitoring configuration..."
cat > /etc/qnn_detector/monitoring.yaml <<EOF
monitoring:
  enabled: true
  port: 9090
  metrics:
    - files_scanned
    - threats_detected
    - processing_time
    - system_resources
EOF

echo "========================================"
echo "Production deployment completed!"
echo "========================================"
echo ""
echo "Service commands:"
echo "  Start:   systemctl start qnn-detector"
echo "  Stop:    systemctl stop qnn-detector"
echo "  Status:  systemctl status qnn-detector"
echo "  Logs:    journalctl -u qnn-detector -f"
echo ""
echo "Configuration: /etc/qnn_detector/config.yaml"
echo "Logs: /var/log/qnn_detector/"
echo "Data: /var/lib/qnn_detector/"
