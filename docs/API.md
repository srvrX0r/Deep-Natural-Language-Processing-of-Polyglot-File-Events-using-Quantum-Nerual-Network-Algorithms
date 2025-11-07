# API Documentation

Complete API reference for the Quantum Neural Network Malware Detector.

## Table of Contents

1. [Main Detector Classes](#main-detector-classes)
2. [Quantum Neural Networks](#quantum-neural-networks)
3. [Detection Engines](#detection-engines)
4. [Cryptography](#cryptography)
5. [File Processing](#file-processing)
6. [Utilities](#utilities)

---

## Main Detector Classes

### QuantumMalwareDetector

Standard quantum malware detector (v1.0).

```python
from src.qnn_malware_detector import QuantumMalwareDetector

detector = QuantumMalwareDetector(config=None)
```

#### Constructor

```python
__init__(config: Optional[Dict] = None)
```

**Parameters:**
- `config` (dict, optional): Configuration dictionary. If None, uses default configuration.

**Default Configuration:**
```python
{
    'batch_size': 100,
    'max_queue_size': 10000,
    'max_workers': 8,
    'qnn_layers': [(100, 64, 4), (64, 32, 4), (32, 16, 4), (16, 2, 4)],
    'detection_threshold': 0.7,
    'encryption_enabled': True,
    'real_time_monitoring': False,
    'log_level': 'INFO'
}
```

#### Methods

##### scan_file()

Scan a single file for malware.

```python
scan_file(file_path: str, encrypt: bool = None) -> Dict
```

**Parameters:**
- `file_path` (str): Path to the file to scan
- `encrypt` (bool, optional): Whether to encrypt file data. Uses config default if None.

**Returns:**
- `dict`: Detection results containing:
  - `file_path` (str): Path to scanned file
  - `file_size` (int): File size in bytes
  - `is_malicious` (bool): Whether file is malicious
  - `threat_level` (str): CRITICAL, HIGH, MEDIUM, LOW, or CLEAN
  - `threat_probability` (float): Probability score [0-1]
  - `polymorphism_score` (float): Polymorphic detection score
  - `metamorphic_detected` (bool): Metamorphic malware detected
  - `entropy` (float): File entropy
  - `self_modifying` (bool): Self-modifying code detected
  - `processing_time` (float): Scan duration in seconds
  - `timestamp` (float): Unix timestamp
  - `encrypted` (bool): Whether file was encrypted during scan

**Example:**
```python
result = detector.scan_file('/path/to/suspicious.exe')
print(f"Malicious: {result['is_malicious']}")
print(f"Threat Level: {result['threat_level']}")
print(f"Score: {result['threat_probability']:.3f}")
```

##### scan_directory()

Scan all files in a directory.

```python
scan_directory(directory_path: str,
               recursive: bool = True,
               max_files: Optional[int] = None) -> List[Dict]
```

**Parameters:**
- `directory_path` (str): Path to directory
- `recursive` (bool): Scan subdirectories
- `max_files` (int, optional): Maximum files to scan

**Returns:**
- `list`: List of detection result dictionaries

**Example:**
```python
results = detector.scan_directory('/path/to/scan', recursive=True)
threats = [r for r in results if r['is_malicious']]
print(f"Found {len(threats)} threats in {len(results)} files")
```

##### get_statistics()

Get detection statistics.

```python
get_statistics() -> Dict
```

**Returns:**
- `dict`: Statistics containing:
  - `files_scanned` (int): Total files scanned
  - `threats_detected` (int): Total threats found
  - `avg_processing_time` (float): Average scan time
  - `detection_rate` (float): Proportion of files flagged

**Example:**
```python
stats = detector.get_statistics()
print(f"Scanned: {stats['files_scanned']}")
print(f"Avg Time: {stats['avg_processing_time']:.3f}s")
```

##### shutdown()

Cleanup and shutdown detector.

```python
shutdown() -> None
```

---

### AdvancedQuantumMalwareDetector

Advanced detector with quantum enhancements (v2.0).

```python
from src.advanced_qnn_detector import AdvancedQuantumMalwareDetector

detector = AdvancedQuantumMalwareDetector(config=None)
```

#### Constructor

```python
__init__(config: Optional[Dict] = None)
```

**Additional Configuration Options:**
```python
{
    # Advanced QNN
    'vqc_qubits': 8,
    'vqc_layers': 4,
    'qcnn_qubits': 8,
    'qcnn_kernel_size': 5,
    'qcnn_filters': 16,
    'attention_dim': 64,
    'attention_heads': 8,
    'residual_dim': 64,
    'residual_qubits': 6,
    'hybrid_input_dim': 128,

    # Ensemble
    'ensemble_models': 15,
    'detection_threshold': 0.65,

    # Features
    'use_adaptive_threshold': True,
    'use_ensemble': True,
    'use_behavioral_analysis': True,
    'use_deep_features': True
}
```

#### Methods

##### scan_file()

Advanced file scanning with all quantum components.

```python
scan_file(file_path: str, encrypt: bool = None) -> Dict
```

**Returns (Enhanced):**
Additional fields beyond standard detector:
- `vqc_score` (float): Variational Quantum Circuit score
- `qcnn_score` (float): Quantum CNN score
- `attention_score` (float): Attention mechanism score
- `residual_score` (float): Residual block score
- `hybrid_malicious_prob` (float): Hybrid network probability
- `qdnn_score` (float): Standard QDNN score
- `ensemble_score` (float): Ensemble detection score
- `ensemble_confidence` (float): Ensemble confidence
- `model_agreement` (float): Model consensus
- `behavioral_score` (float): Behavioral analysis score
- `evasion_score` (float): Evasion technique score
- `current_threshold` (float): Current adaptive threshold
- `confidence` (float): Overall confidence

**Example:**
```python
result = detector.scan_file('malware.exe')
print(f"Threat Score: {result['threat_score']:.3f}")
print(f"VQC: {result['vqc_score']:.3f}")
print(f"QCNN: {result['qcnn_score']:.3f}")
print(f"Ensemble: {result['ensemble_score']:.3f}")
print(f"Behavioral: {result['behavioral_score']:.3f}")
print(f"Confidence: {result['confidence']:.3f}")
```

---

## Quantum Neural Networks

### VariationalQuantumCircuit

Trainable quantum circuit with parameterized gates.

```python
from src.qnn_core.advanced_quantum import VariationalQuantumCircuit

vqc = VariationalQuantumCircuit(n_qubits=8, n_layers=4, learning_rate=0.01)
```

#### Methods

##### forward()

Forward pass through quantum circuit.

```python
forward(input_data: np.ndarray) -> np.ndarray
```

**Parameters:**
- `input_data` (ndarray): Input features (float64)

**Returns:**
- `ndarray`: Quantum measurements for each qubit

**Example:**
```python
inputs = np.random.randn(8).astype(np.float64)
output = vqc.forward(inputs)
print(f"Quantum output: {output}")
```

##### optimize_parameters()

Update circuit parameters using gradients.

```python
optimize_parameters(gradients: np.ndarray) -> None
```

---

### QuantumConvolutionalNetwork

Quantum convolutional layer for spatial features.

```python
from src.qnn_core.advanced_quantum import QuantumConvolutionalNetwork

qcnn = QuantumConvolutionalNetwork(n_qubits=8, kernel_size=5, n_filters=16)
```

#### Methods

##### forward()

Apply quantum convolution.

```python
forward(input_data: np.ndarray) -> np.ndarray
```

**Returns:**
- `ndarray`: Feature maps (shape: [n_filters, n_patches])

---

### QuantumAttentionMechanism

Quantum-inspired attention for feature selection.

```python
from src.qnn_core.advanced_quantum import QuantumAttentionMechanism

attention = QuantumAttentionMechanism(d_model=64, n_heads=8)
```

#### Methods

##### forward()

Apply attention mechanism.

```python
forward(input_features: np.ndarray) -> np.ndarray
```

**Returns:**
- `ndarray`: Attended features

---

## Detection Engines

### PolymorphicDetector

Detect polymorphic malware.

```python
from src.malware_detection import PolymorphicDetector

detector = PolymorphicDetector()
```

#### Methods

##### detect()

Detect polymorphic characteristics.

```python
detect(file_data: bytes) -> Dict
```

**Returns:**
- `dict`: Detection results:
  - `is_malicious` (bool)
  - `threat_level` (str)
  - `polymorphism_score` (float)
  - `entropy` (float)
  - `code_cave_ratio` (float)
  - `self_modifying` (bool)

---

### DeepBehavioralAnalyzer

Advanced behavioral analysis.

```python
from src.malware_detection.advanced_detection import DeepBehavioralAnalyzer

analyzer = DeepBehavioralAnalyzer(pattern_count=100)
```

#### Methods

##### analyze()

Perform behavioral analysis.

```python
analyze(file_data: bytes) -> Dict
```

**Returns:**
- `dict`:
  - `behavioral_score` (float): Overall behavioral score
  - `api_suspicion` (float): API call suspicion
  - `injection_indicators` (float): Code injection score
  - `memory_anomalies` (float): Memory pattern anomalies
  - `evasion_techniques` (float): Evasion detection score
  - `is_suspicious` (bool)

---

### EnsembleDetector

Multi-model ensemble detection.

```python
from src.malware_detection.advanced_detection import EnsembleDetector

ensemble = EnsembleDetector(n_models=15)
```

#### Methods

##### ensemble_predict()

Predict using ensemble voting.

```python
ensemble_predict(features: np.ndarray) -> Dict
```

**Returns:**
- `dict`:
  - `ensemble_score` (float): Weighted ensemble score
  - `is_malicious` (bool)
  - `confidence` (float): Prediction confidence
  - `votes_malicious` (int): Models voting malicious
  - `votes_benign` (int): Models voting benign
  - `model_agreement` (float): Consensus level

---

## Cryptography

### HybridEncryption

Quantum-resistant hybrid encryption.

```python
from src.crypto import HybridEncryption

crypto = HybridEncryption()
```

#### Methods

##### initialize_session()

Initialize encryption session with QKD.

```python
initialize_session() -> None
```

##### encrypt()

Encrypt data.

```python
encrypt(plaintext: bytes) -> bytes
```

**Returns:**
- `bytes`: Encrypted data with authentication tag

##### decrypt()

Decrypt data.

```python
decrypt(ciphertext: bytes) -> bytes
```

**Returns:**
- `bytes`: Decrypted plaintext

**Raises:**
- `ValueError`: If authentication fails

---

## File Processing

### FileEventProcessor

Process file system events.

```python
from src.file_processor import FileEventProcessor

processor = FileEventProcessor()
```

#### Methods

##### process_file_event()

Process a file event.

```python
process_file_event(file_path: str,
                   event_type: str,
                   file_data: bytes = None) -> Dict
```

**Parameters:**
- `file_path` (str): Path to file
- `event_type` (str): Event type ('scan', 'create', 'modify')
- `file_data` (bytes, optional): File data (loaded if None)

**Returns:**
- `dict`: Processing results with polyglot analysis

---

## Utilities

### quantum_feature_encoding()

Encode classical features to quantum state.

```python
from src.qnn_core.advanced_quantum import quantum_feature_encoding

quantum_features = quantum_feature_encoding(classical_features, n_qubits=8)
```

**Parameters:**
- `classical_features` (ndarray): Classical feature vector
- `n_qubits` (int): Number of qubits

**Returns:**
- `ndarray`: Quantum measurement results

---

## Error Handling

All methods may raise:

- `FileNotFoundError`: File not found
- `PermissionError`: Insufficient permissions
- `ValueError`: Invalid parameters
- `RuntimeError`: Processing errors

**Example:**
```python
try:
    result = detector.scan_file('file.exe')
except FileNotFoundError:
    print("File not found")
except PermissionError:
    print("Permission denied")
except Exception as e:
    print(f"Error: {e}")
```

---

## Type Hints

All public APIs use type hints:

```python
from typing import Dict, List, Optional

def scan_file(
    file_path: str,
    encrypt: bool = True
) -> Dict[str, Any]:
    """Type-annotated function."""
    pass
```

---

## Thread Safety

- `QuantumMalwareDetector`: Thread-safe (uses ThreadPoolExecutor)
- `AdvancedQuantumMalwareDetector`: Thread-safe
- Individual detectors: NOT thread-safe (create instances per thread)

---

## Performance Tips

1. **Batch Processing**: Use `scan_directory()` for multiple files
2. **Disable Encryption**: Set `encrypt=False` for faster scanning
3. **Adjust Workers**: Increase `max_workers` for more parallelism
4. **Use Standard Detector**: For speed over accuracy
5. **Cache Results**: Store results to avoid re-scanning

---

For more examples, see the [examples/](../examples/) directory.
