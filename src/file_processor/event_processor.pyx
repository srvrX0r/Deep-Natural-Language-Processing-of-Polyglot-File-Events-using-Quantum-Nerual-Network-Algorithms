# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
Polyglot File Event Processor
High-performance file event processing and analysis
"""

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.string cimport strlen, strcmp, strstr
from libc.math cimport log, sqrt
import cython
import os
import mimetypes
from pathlib import Path

cnp.import_array()

ctypedef cnp.uint8_t BYTE_t
ctypedef cnp.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class FileSignatureDetector:
    """Detect file types by magic bytes"""

    cdef dict signatures
    cdef dict polyglot_patterns

    def __init__(self):
        # Common file signatures (magic bytes)
        self.signatures = {
            b'\x4D\x5A': 'PE/EXE',
            b'\x7F\x45\x4C\x46': 'ELF',
            b'\xCA\xFE\xBA\xBE': 'Mach-O',
            b'\x50\x4B\x03\x04': 'ZIP',
            b'\x1F\x8B': 'GZIP',
            b'\x42\x5A\x68': 'BZIP2',
            b'\x50\x4B\x05\x06': 'JAR',
            b'\xD0\xCF\x11\xE0': 'MS Office',
            b'\x25\x50\x44\x46': 'PDF',
            b'\x89\x50\x4E\x47': 'PNG',
            b'\xFF\xD8\xFF': 'JPEG',
            b'\x47\x49\x46\x38': 'GIF',
            b'%!PS': 'PostScript',
            b'#!/': 'Script',
            b'<?php': 'PHP',
            b'<?xml': 'XML',
            b'{\\"': 'JSON',
            b'PK\x03\x04': 'Android APK',
        }

        # Polyglot detection patterns
        self.polyglot_patterns = {
            'pdf_js': (b'%PDF', b'<script'),
            'jpg_zip': (b'\xFF\xD8\xFF', b'PK\x03\x04'),
            'gif_html': (b'GIF89a', b'<html'),
            'png_php': (b'\x89PNG', b'<?php'),
        }

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef list detect_signatures(self, unsigned char* data, int length):
        """Detect all matching file signatures"""
        cdef list matches = []
        cdef bytes sig
        cdef str file_type
        cdef int sig_len

        for sig, file_type in self.signatures.items():
            sig_len = len(sig)
            if length >= sig_len:
                if memcmp(data, <unsigned char*><char*>sig, sig_len) == 0:
                    matches.append(file_type)

        return matches

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef bint is_polyglot(self, unsigned char* data, int length):
        """Detect if file is polyglot (multiple valid formats)"""
        cdef bytes pattern1, pattern2
        cdef str poly_type

        for poly_type, (pattern1, pattern2) in self.polyglot_patterns.items():
            if self._contains_pattern(data, length, pattern1) and \
               self._contains_pattern(data, length, pattern2):
                return True

        return False

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef bint _contains_pattern(self, unsigned char* data, int length, bytes pattern) nogil:
        """Check if data contains pattern"""
        cdef int pat_len = len(pattern)
        cdef unsigned char* pat_ptr = <unsigned char*><char*>pattern
        cdef int i, j
        cdef bint match

        for i in range(length - pat_len + 1):
            match = True
            for j in range(pat_len):
                if data[i + j] != pat_ptr[j]:
                    match = False
                    break
            if match:
                return True

        return False

    cpdef dict analyze_file(self, bytes file_data):
        """Complete file signature analysis"""
        cdef unsigned char* data = <unsigned char*><char*>file_data
        cdef int length = len(file_data)

        cdef list signatures = self.detect_signatures(data, length)
        cdef bint is_polyglot_file = self.is_polyglot(data, length)

        return {
            'signatures': signatures,
            'is_polyglot': is_polyglot_file,
            'signature_count': len(signatures),
            'suspicious': is_polyglot_file or len(signatures) > 1
        }

cdef extern from "string.h":
    int memcmp(const void *s1, const void *s2, size_t n) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class PolyglotAnalyzer:
    """Advanced polyglot file analysis"""

    cdef FileSignatureDetector sig_detector
    cdef dict language_patterns
    cdef double[:] feature_vector

    def __init__(self):
        self.sig_detector = FileSignatureDetector()
        self.feature_vector = np.zeros(50, dtype=np.float64)

        # Programming language patterns
        self.language_patterns = {
            'python': [b'import ', b'def ', b'class ', b'__init__'],
            'javascript': [b'function', b'const ', b'let ', b'var '],
            'java': [b'public class', b'private ', b'import java'],
            'c': [b'#include', b'int main', b'void '],
            'cpp': [b'#include <', b'namespace ', b'std::'],
            'php': [b'<?php', b'$_GET', b'$_POST'],
            'ruby': [b'require ', b'class ', b'def '],
            'go': [b'package ', b'func ', b'import '],
            'rust': [b'fn main', b'use ', b'mod '],
            'shell': [b'#!/bin/bash', b'#!/bin/sh', b'$1'],
        }

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef dict detect_languages(self, unsigned char* data, int length):
        """Detect embedded programming languages"""
        cdef dict lang_scores = {}
        cdef str lang
        cdef list patterns
        cdef bytes pattern
        cdef int pattern_count

        for lang, patterns in self.language_patterns.items():
            pattern_count = 0
            for pattern in patterns:
                if self.sig_detector._contains_pattern(data, length, pattern):
                    pattern_count += 1
            if pattern_count > 0:
                lang_scores[lang] = pattern_count

        return lang_scores

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double calculate_entropy_sections(self, unsigned char* data, int length) nogil:
        """Calculate entropy across file sections"""
        cdef int section_size = 1024
        cdef int num_sections = (length + section_size - 1) // section_size
        cdef double total_entropy = 0.0
        cdef double entropy_variance = 0.0
        cdef int i, j
        cdef unsigned int counts[256]
        cdef double prob, section_entropy
        cdef list entropies = []

        for i in range(num_sections):
            # Reset counts
            for j in range(256):
                counts[j] = 0

            # Count bytes in section
            cdef int start = i * section_size
            cdef int end = min(start + section_size, length)
            for j in range(start, end):
                counts[data[j]] += 1

            # Calculate section entropy
            section_entropy = 0.0
            for j in range(256):
                if counts[j] > 0:
                    prob = <double>counts[j] / <double>(end - start)
                    section_entropy -= prob * log(prob) / log(2.0)

            total_entropy += section_entropy

        return total_entropy / <double>num_sections

    cpdef dict analyze_polyglot(self, bytes file_data):
        """Complete polyglot analysis"""
        cdef unsigned char* data = <unsigned char*><char*>file_data
        cdef int length = len(file_data)

        # Signature analysis
        sig_analysis = self.sig_detector.analyze_file(file_data)

        # Language detection
        languages = self.detect_languages(data, length)

        # Entropy analysis
        entropy = self.calculate_entropy_sections(data, length)

        # Suspiciousness score
        cdef double suspicion_score = 0.0
        if sig_analysis['is_polyglot']:
            suspicion_score += 0.5
        suspicion_score += len(languages) * 0.1
        if entropy > 7.5:  # High entropy indicates encryption/packing
            suspicion_score += 0.3

        return {
            'signatures': sig_analysis,
            'languages': languages,
            'language_count': len(languages),
            'entropy': float(entropy),
            'suspicion_score': min(1.0, float(suspicion_score)),
            'is_suspicious': suspicion_score > 0.5
        }

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class FileEventProcessor:
    """Main file event processing pipeline"""

    cdef PolyglotAnalyzer polyglot_analyzer
    cdef dict event_cache
    cdef long events_processed
    cdef double[:] performance_metrics

    def __init__(self):
        self.polyglot_analyzer = PolyglotAnalyzer()
        self.event_cache = {}
        self.events_processed = 0
        self.performance_metrics = np.zeros(10, dtype=np.float64)

    cpdef dict process_file_event(self, str file_path, str event_type, bytes file_data=None):
        """Process a file system event"""
        cdef dict result = {
            'file_path': file_path,
            'event_type': event_type,
            'timestamp': self._get_timestamp(),
            'success': True,
            'error': None
        }

        try:
            # Load file data if not provided
            if file_data is None:
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        file_data = f.read()
                else:
                    result['success'] = False
                    result['error'] = 'File not found'
                    return result

            # Basic file info
            result['file_size'] = len(file_data)
            result['file_extension'] = Path(file_path).suffix

            # Polyglot analysis
            polyglot_result = self.polyglot_analyzer.analyze_polyglot(file_data)
            result['polyglot_analysis'] = polyglot_result

            # Extract features for ML
            result['features'] = self._extract_ml_features(file_data, polyglot_result)

            # Update metrics
            self.events_processed += 1
            self.performance_metrics[0] = <double>self.events_processed

        except Exception as e:
            result['success'] = False
            result['error'] = str(e)

        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double[:] _extract_ml_features(self, bytes file_data, dict polyglot_result):
        """Extract features for machine learning"""
        cdef double[:] features = np.zeros(25, dtype=np.float64)

        # File size features
        features[0] = log(len(file_data) + 1.0)

        # Signature features
        features[1] = <double>polyglot_result['signatures']['signature_count']
        features[2] = 1.0 if polyglot_result['signatures']['is_polyglot'] else 0.0

        # Language features
        features[3] = <double>polyglot_result['language_count']

        # Entropy
        features[4] = polyglot_result['entropy']

        # Suspicion score
        features[5] = polyglot_result['suspicion_score']

        # Byte statistics
        cdef unsigned char* data = <unsigned char*><char*>file_data
        cdef int length = len(file_data)
        cdef int i
        cdef double mean = 0.0, variance = 0.0

        for i in range(min(length, 10000)):
            mean += <double>data[i]

        mean /= <double>min(length, 10000)
        features[6] = mean

        for i in range(min(length, 10000)):
            variance += (data[i] - mean) ** 2

        variance /= <double>min(length, 10000)
        features[7] = sqrt(variance)

        return features

    cdef double _get_timestamp(self):
        """Get current timestamp"""
        import time
        return time.time()

    cpdef dict get_statistics(self):
        """Get processing statistics"""
        return {
            'events_processed': self.events_processed,
            'cache_size': len(self.event_cache),
            'performance_metrics': np.asarray(self.performance_metrics)
        }

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class BatchFileProcessor:
    """Batch file processing for high throughput"""

    cdef FileEventProcessor processor
    cdef list processing_queue
    cdef int batch_size
    cdef int max_queue_size

    def __init__(self, int batch_size=100, int max_queue_size=10000):
        self.processor = FileEventProcessor()
        self.processing_queue = []
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size

    cpdef void add_file(self, str file_path, str event_type):
        """Add file to processing queue"""
        if len(self.processing_queue) < self.max_queue_size:
            self.processing_queue.append((file_path, event_type))

    cpdef list process_batch(self):
        """Process a batch of files"""
        cdef list results = []
        cdef int count = min(self.batch_size, len(self.processing_queue))
        cdef str file_path, event_type
        cdef dict result

        for _ in range(count):
            if not self.processing_queue:
                break

            file_path, event_type = self.processing_queue.pop(0)
            result = self.processor.process_file_event(file_path, event_type)
            results.append(result)

        return results

    cpdef int queue_size(self):
        """Get current queue size"""
        return len(self.processing_queue)
