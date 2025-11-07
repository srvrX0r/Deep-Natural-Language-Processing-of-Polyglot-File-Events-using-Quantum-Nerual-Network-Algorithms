"""
File Event Processing Module
Polyglot file analysis and event processing
"""

from .event_processor import (
    FileSignatureDetector,
    PolyglotAnalyzer,
    FileEventProcessor,
    BatchFileProcessor
)

__all__ = [
    'FileSignatureDetector',
    'PolyglotAnalyzer',
    'FileEventProcessor',
    'BatchFileProcessor'
]
