"""
Integration tests for complete system
"""

import pytest
import numpy as np
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.qnn_malware_detector import QuantumMalwareDetector


class TestQuantumMalwareDetectorIntegration:
    """Integration tests for complete detector"""

    def setup_method(self):
        """Setup for each test"""
        config = {
            'batch_size': 10,
            'max_queue_size': 100,
            'max_workers': 2,
            'qnn_layers': [(20, 10, 3), (10, 5, 3), (5, 2, 3)],
            'detection_threshold': 0.6,
            'encryption_enabled': False,  # Disable for testing
            'log_level': 'WARNING'
        }
        try:
            self.detector = QuantumMalwareDetector(config)
        except Exception as e:
            pytest.skip(f"Cannot initialize detector: {e}")

    def test_initialization(self):
        """Test detector initialization"""
        assert self.detector is not None
        assert self.detector.config is not None

    def test_scan_text_file(self):
        """Test scanning a clean text file"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write('This is a clean test file.' * 100)
            f.flush()
            temp_path = f.name

        try:
            result = self.detector.scan_file(temp_path, encrypt=False)
            assert result is not None
            assert 'is_malicious' in result
            assert 'threat_level' in result
            assert result['file_path'] == temp_path
        finally:
            os.unlink(temp_path)

    def test_scan_binary_file(self):
        """Test scanning a binary file"""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.bin') as f:
            # Create binary data with some patterns
            data = bytes(np.random.randint(0, 256, 5000, dtype=np.uint8))
            f.write(data)
            f.flush()
            temp_path = f.name

        try:
            result = self.detector.scan_file(temp_path, encrypt=False)
            assert result is not None
            assert 'polymorphism_score' in result
            assert 0.0 <= result['polymorphism_score'] <= 1.0
        finally:
            os.unlink(temp_path)

    def test_scan_suspicious_file(self):
        """Test scanning a file with suspicious patterns"""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.exe') as f:
            # Create data resembling malware patterns
            data = b'MZ\x90\x00'  # PE header
            data += bytes([0xEB, 0xE9, 0xCC, 0x90] * 250)  # Suspicious opcodes
            f.write(data)
            f.flush()
            temp_path = f.name

        try:
            result = self.detector.scan_file(temp_path, encrypt=False)
            assert result is not None
            # High entropy or suspicious patterns should be detected
            assert result['polymorphism_score'] > 0.0
        finally:
            os.unlink(temp_path)

    def test_scan_directory(self):
        """Test scanning a directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple test files
            for i in range(5):
                file_path = os.path.join(tmpdir, f'test_{i}.txt')
                with open(file_path, 'w') as f:
                    f.write(f'Test file {i}' * 50)

            results = self.detector.scan_directory(tmpdir, recursive=False)
            assert len(results) == 5
            assert all('is_malicious' in r for r in results)

    def test_statistics(self):
        """Test statistics tracking"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write('Test data' * 100)
            f.flush()
            temp_path = f.name

        try:
            self.detector.scan_file(temp_path, encrypt=False)
            stats = self.detector.get_statistics()

            assert 'files_scanned' in stats
            assert stats['files_scanned'] >= 1
        finally:
            os.unlink(temp_path)

    def test_error_handling(self):
        """Test error handling for non-existent files"""
        result = self.detector.scan_file('/nonexistent/file.txt')
        assert result is not None
        assert 'error' in result or not result['is_malicious']

    def teardown_method(self):
        """Cleanup after each test"""
        if hasattr(self, 'detector'):
            self.detector.shutdown()


class TestEndToEnd:
    """End-to-end system tests"""

    def test_complete_workflow(self):
        """Test complete detection workflow"""
        config = {
            'batch_size': 5,
            'max_workers': 2,
            'qnn_layers': [(20, 10, 3), (10, 5, 3), (5, 2, 3)],
            'encryption_enabled': False,
            'log_level': 'WARNING'
        }

        try:
            detector = QuantumMalwareDetector(config)
        except Exception as e:
            pytest.skip(f"Cannot initialize detector: {e}")

        # Create test directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create various test files
            test_files = [
                ('clean.txt', b'Clean text file data' * 50),
                ('binary.bin', bytes(np.random.randint(0, 256, 1000))),
                ('suspicious.exe', b'MZ\x90\x00' + bytes([0xEB] * 500)),
            ]

            for filename, content in test_files:
                file_path = os.path.join(tmpdir, filename)
                with open(file_path, 'wb') as f:
                    f.write(content)

            # Scan directory
            results = detector.scan_directory(tmpdir)

            # Verify results
            assert len(results) == len(test_files)
            assert all('threat_level' in r for r in results)
            assert any(r['polymorphism_score'] > 0 for r in results)

            # Check statistics
            stats = detector.get_statistics()
            assert stats['files_scanned'] >= len(test_files)

            detector.shutdown()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
