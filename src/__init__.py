"""
Quantum Neural Network Malware Detection System
Deep Natural Language Processing of Polyglot File Events
"""

from .qnn_malware_detector import QuantumMalwareDetector

try:
    from .advanced_qnn_detector import AdvancedQuantumMalwareDetector
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False

__version__ = '2.0.0'
__author__ = 'Quantum Security Research Team'

__all__ = ['QuantumMalwareDetector']

if ADVANCED_AVAILABLE:
    __all__.append('AdvancedQuantumMalwareDetector')
