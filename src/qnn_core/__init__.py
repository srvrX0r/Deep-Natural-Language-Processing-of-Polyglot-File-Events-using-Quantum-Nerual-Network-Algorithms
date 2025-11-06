"""
Quantum Neural Network Core Module
High-performance quantum computing for malware detection
"""

from .quantum_layer import (
    QuantumGate,
    QuantumNeuron,
    QuantumLayer,
    QuantumDeepNeuralNetwork,
    quantum_fourier_transform,
    quantum_entanglement_measure
)

try:
    from .advanced_quantum import (
        VariationalQuantumCircuit,
        QuantumConvolutionalNetwork,
        QuantumAttentionMechanism,
        QuantumResidualBlock,
        HybridQuantumClassicalNetwork,
        quantum_feature_encoding
    )
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False

__all__ = [
    'QuantumGate',
    'QuantumNeuron',
    'QuantumLayer',
    'QuantumDeepNeuralNetwork',
    'quantum_fourier_transform',
    'quantum_entanglement_measure'
]

if ADVANCED_AVAILABLE:
    __all__.extend([
        'VariationalQuantumCircuit',
        'QuantumConvolutionalNetwork',
        'QuantumAttentionMechanism',
        'QuantumResidualBlock',
        'HybridQuantumClassicalNetwork',
        'quantum_feature_encoding'
    ])
