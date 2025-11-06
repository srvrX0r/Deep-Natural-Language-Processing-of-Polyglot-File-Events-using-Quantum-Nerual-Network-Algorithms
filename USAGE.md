# Usage Guide - Quantum Malware Detector

## Table of Contents

1. [Quick Start](#quick-start)
2. [Command Line Usage](#command-line-usage)
3. [Python API](#python-api)
4. [Configuration](#configuration)
5. [Advanced Usage](#advanced-usage)
6. [Integration Examples](#integration-examples)
7. [Troubleshooting](#troubleshooting)

## Quick Start

### Installation

```bash
# Quick install
bash scripts/install.sh

# Verify installation
qnn-scan --help
```

### Basic Scanning

```bash
# Scan a single file
qnn-scan suspicious_file.exe

# Scan a directory
qnn-scan /path/to/directory -r

# Save results
qnn-scan /path/to/scan -r -o results.json
```

## Command Line Usage

### Scanning Files

```bash
# Single file with detailed output
qnn-scan malware.exe

# Multiple files
qnn-scan file1.exe file2.dll file3.bin

# With custom configuration
qnn-scan file.exe --config /path/to/config.yaml
```

### Scanning Directories

```bash
# Non-recursive scan
qnn-scan /path/to/directory

# Recursive scan (all subdirectories)
qnn-scan /path/to/directory -r

# Limit number of files
qnn-scan /path/to/directory -r --max-files 1000
```

### Output Options

```bash
# JSON output
qnn-scan /path -r -o results.json

# CSV output (requires processing)
qnn-scan /path -r -o results.json
python scripts/json_to_csv.py results.json results.csv

# Verbose output
qnn-scan file.exe -v

# Quiet mode (only show threats)
qnn-scan /path -r -q
```

### Advanced Options

```bash
# Disable encryption (faster, less secure)
qnn-scan file.exe --no-encrypt

# Custom threat threshold
qnn-scan file.exe --threshold 0.8

# Enable deep scanning
qnn-scan file.exe --deep-scan

# Show statistics
qnn-scan /path -r --stats
```

## Python API

### Basic Usage

```python
from src.qnn_malware_detector import QuantumMalwareDetector

# Initialize with default configuration
detector = QuantumMalwareDetector()

# Scan a file
result = detector.scan_file('/path/to/file.exe')

# Check results
if result['is_malicious']:
    print(f"Threat detected! Level: {result['threat_level']}")
    print(f"Polymorphism Score: {result['polymorphism_score']:.3f}")
else:
    print("File appears clean")

# Clean up
detector.shutdown()
```

### Custom Configuration

```python
import yaml
from src.qnn_malware_detector import QuantumMalwareDetector

# Load custom configuration
with open('my_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize with custom config
detector = QuantumMalwareDetector(config)

# Use detector
result = detector.scan_file('suspicious.exe')
```

### Batch Processing

```python
from pathlib import Path
from src.qnn_malware_detector import QuantumMalwareDetector

detector = QuantumMalwareDetector()

# Scan multiple files
files = list(Path('/path/to/scan').rglob('*.exe'))

results = []
for file_path in files:
    result = detector.scan_file(str(file_path))
    results.append(result)

# Filter malicious files
threats = [r for r in results if r['is_malicious']]

print(f"Scanned: {len(results)} files")
print(f"Threats: {len(threats)} detected")

# Get statistics
stats = detector.get_statistics()
print(f"Average processing time: {stats['avg_processing_time']:.3f}s")

detector.shutdown()
```

### Directory Scanning

```python
from src.qnn_malware_detector import QuantumMalwareDetector

detector = QuantumMalwareDetector()

# Scan entire directory
results = detector.scan_directory(
    '/path/to/scan',
    recursive=True,
    max_files=1000
)

# Analyze results
threat_levels = {}
for result in results:
    level = result['threat_level']
    threat_levels[level] = threat_levels.get(level, 0) + 1

print("Threat Distribution:")
for level, count in sorted(threat_levels.items()):
    print(f"  {level}: {count}")

detector.shutdown()
```

### Using Individual Components

```python
from src.malware_detection import PolymorphicDetector, MetamorphicDetector
from src.crypto import HybridEncryption
from src.file_processor import FileEventProcessor

# Polymorphic detection only
poly_detector = PolymorphicDetector()

with open('file.exe', 'rb') as f:
    data = f.read()

result = poly_detector.detect(data)
print(f"Polymorphism score: {result['polymorphism_score']}")

# File event processing
processor = FileEventProcessor()
event = processor.process_file_event('file.exe', 'scan')
print(f"Polyglot analysis: {event['polyglot_analysis']}")

# Encryption
crypto = HybridEncryption()
crypto.initialize_session()

encrypted = crypto.encrypt(b"Sensitive data")
decrypted = crypto.decrypt(encrypted)
```

## Configuration

### Configuration File Structure

```yaml
# config/my_config.yaml

system:
  max_workers: 8
  batch_size: 100
  max_queue_size: 10000

qnn:
  architecture:
    - [100, 64, 4]
    - [64, 32, 4]
    - [32, 16, 4]
    - [16, 2, 4]
  training:
    learning_rate: 0.01
    epochs: 100

detection:
  threat_threshold: 0.7
  polymorphic_threshold: 0.6
  metamorphic_threshold: 0.5
  max_file_size_mb: 100
  deep_scan: true

encryption:
  enabled: true
  algorithm: 'LATTICE_HYBRID'

logging:
  level: 'INFO'
  file: './logs/detector.log'
```

### Environment-Specific Configs

```python
import os
from src.qnn_malware_detector import QuantumMalwareDetector

# Detect environment
env = os.getenv('ENVIRONMENT', 'development')
config_file = f'config/{env}.yaml'

# Load appropriate config
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

detector = QuantumMalwareDetector(config)
```

## Advanced Usage

### Training the QNN

```python
from src.qnn_malware_detector import QuantumMalwareDetector
import json

detector = QuantumMalwareDetector()

# Prepare training dataset (format: [{data, label}, ...])
dataset = []

# Add malware samples
for malware_file in malware_samples:
    with open(malware_file, 'rb') as f:
        dataset.append({
            'data': f.read().hex(),
            'label': 1  # Malicious
        })

# Add clean samples
for clean_file in clean_samples:
    with open(clean_file, 'rb') as f:
        dataset.append({
            'data': f.read().hex(),
            'label': 0  # Clean
        })

# Save dataset
with open('training_dataset.json', 'w') as f:
    json.dump(dataset, f)

# Train
detector.train_on_dataset('training_dataset.json', epochs=100)

# Save trained model
detector.save_model('models/trained_qnn.model')
```

### Real-Time Monitoring

```python
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from src.qnn_malware_detector import QuantumMalwareDetector

class MalwareMonitor(FileSystemEventHandler):
    def __init__(self):
        self.detector = QuantumMalwareDetector()

    def on_created(self, event):
        if not event.is_directory:
            print(f"New file detected: {event.src_path}")
            result = self.detector.scan_file(event.src_path)

            if result['is_malicious']:
                print(f"THREAT DETECTED: {event.src_path}")
                print(f"Threat Level: {result['threat_level']}")
                # Take action (quarantine, alert, etc.)

# Monitor directory
monitor = MalwareMonitor()
observer = Observer()
observer.schedule(monitor, '/path/to/monitor', recursive=True)
observer.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
    monitor.detector.shutdown()

observer.join()
```

### Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor
from src.qnn_malware_detector import QuantumMalwareDetector
from pathlib import Path

def scan_file_worker(file_path):
    """Worker function for parallel scanning"""
    detector = QuantumMalwareDetector()
    result = detector.scan_file(file_path)
    detector.shutdown()
    return result

# Get all files to scan
files = list(Path('/path/to/scan').rglob('*'))
files = [str(f) for f in files if f.is_file()]

# Parallel scan
with ProcessPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(scan_file_worker, files))

# Analyze results
threats = [r for r in results if r.get('is_malicious', False)]
print(f"Scanned {len(results)} files, found {len(threats)} threats")
```

## Integration Examples

### Flask REST API

```python
from flask import Flask, request, jsonify
from src.qnn_malware_detector import QuantumMalwareDetector
import tempfile
import os

app = Flask(__name__)
detector = QuantumMalwareDetector()

@app.route('/scan', methods=['POST'])
def scan_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # Scan
        result = detector.scan_file(tmp_path)
        return jsonify(result)
    finally:
        os.unlink(tmp_path)

@app.route('/stats', methods=['GET'])
def get_stats():
    stats = detector.get_statistics()
    return jsonify(stats)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Integration with Security Tools

```python
from src.qnn_malware_detector import QuantumMalwareDetector
import syslog

detector = QuantumMalwareDetector()

def scan_and_alert(file_path):
    """Scan file and send alerts"""
    result = detector.scan_file(file_path)

    if result['is_malicious']:
        # Log to syslog
        syslog.syslog(
            syslog.LOG_ALERT,
            f"Malware detected: {file_path} "
            f"(Level: {result['threat_level']}, "
            f"Score: {result['threat_probability']:.3f})"
        )

        # Send SNMP trap (if configured)
        # send_snmp_trap(result)

        # Update threat database
        # update_threat_db(result)

    return result
```

## Troubleshooting

### Common Issues

**Issue: "Cython modules not compiled"**
```bash
# Solution: Build the Cython extensions
bash scripts/build.sh
```

**Issue: "Permission denied" errors**
```bash
# Solution: Check file permissions or run with appropriate privileges
sudo qnn-scan /protected/path
```

**Issue: High memory usage**
```python
# Solution: Reduce batch size and max workers
config = {
    'batch_size': 10,
    'max_workers': 2,
    'max_queue_size': 100
}
detector = QuantumMalwareDetector(config)
```

**Issue: Slow performance**
```bash
# Solution: Build with optimizations
export CFLAGS="-O3 -march=native -ffast-math"
bash scripts/build.sh
```

### Debug Mode

```python
import logging
from src.qnn_malware_detector import QuantumMalwareDetector

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

config = {'log_level': 'DEBUG'}
detector = QuantumMalwareDetector(config)

# Scan with debug output
result = detector.scan_file('file.exe')
```

### Performance Profiling

```python
import cProfile
import pstats
from src.qnn_malware_detector import QuantumMalwareDetector

detector = QuantumMalwareDetector()

# Profile scanning
profiler = cProfile.Profile()
profiler.enable()

detector.scan_file('large_file.exe')

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

## Support

For additional help:
- Check the main README.md
- Review test files in tests/
- Run: `qnn-scan --help`
- Open an issue on GitHub
