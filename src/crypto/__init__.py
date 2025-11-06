"""
Quantum-Resistant Cryptography Module
Hybrid encryption for secure malware analysis
"""

from .quantum_encryption import (
    LatticeBasedCrypto,
    QuantumKeyDistribution,
    HybridEncryption,
    SecureFileHandler
)

__all__ = [
    'LatticeBasedCrypto',
    'QuantumKeyDistribution',
    'HybridEncryption',
    'SecureFileHandler'
]
