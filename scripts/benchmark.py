#!/usr/bin/env python3
"""
Performance benchmark script for Quantum Malware Detector
"""

import sys
import os
import time
import numpy as np
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.qnn_malware_detector import QuantumMalwareDetector


def generate_test_files(num_files=100, size_range=(1024, 1024*100)):
    """Generate test files for benchmarking"""
    files = []
    tmpdir = tempfile.mkdtemp()

    print(f"Generating {num_files} test files...")

    for i in range(num_files):
        size = np.random.randint(*size_range)
        data = bytes(np.random.randint(0, 256, size, dtype=np.uint8))

        file_path = os.path.join(tmpdir, f'test_{i}.bin')
        with open(file_path, 'wb') as f:
            f.write(data)

        files.append(file_path)

    return tmpdir, files


def benchmark_single_file(detector, file_path):
    """Benchmark single file scan"""
    start = time.time()
    result = detector.scan_file(file_path, encrypt=False)
    elapsed = time.time() - start
    return elapsed, result


def benchmark_batch(detector, files):
    """Benchmark batch scanning"""
    start = time.time()

    results = []
    for file_path in files:
        result = detector.scan_file(file_path, encrypt=False)
        results.append(result)

    elapsed = time.time() - start
    return elapsed, results


def benchmark_parallel(detector, directory):
    """Benchmark parallel directory scanning"""
    start = time.time()
    results = detector.scan_directory(directory, recursive=False)
    elapsed = time.time() - start
    return elapsed, results


def print_statistics(times, label):
    """Print timing statistics"""
    times = np.array(times)
    print(f"\n{label}:")
    print(f"  Count: {len(times)}")
    print(f"  Mean: {np.mean(times):.4f}s")
    print(f"  Median: {np.median(times):.4f}s")
    print(f"  Std Dev: {np.std(times):.4f}s")
    print(f"  Min: {np.min(times):.4f}s")
    print(f"  Max: {np.max(times):.4f}s")
    print(f"  Total: {np.sum(times):.4f}s")
    print(f"  Throughput: {len(times) / np.sum(times):.2f} files/sec")


def main():
    print("=" * 60)
    print("Quantum Malware Detector Performance Benchmark")
    print("=" * 60)

    # Configuration for benchmarking
    config = {
        'batch_size': 50,
        'max_queue_size': 1000,
        'max_workers': 8,
        'qnn_layers': [(50, 32, 3), (32, 16, 3), (16, 2, 3)],
        'detection_threshold': 0.7,
        'encryption_enabled': False,
        'log_level': 'WARNING'
    }

    try:
        print("\nInitializing detector...")
        detector = QuantumMalwareDetector(config)
    except Exception as e:
        print(f"ERROR: Could not initialize detector: {e}")
        print("Make sure Cython modules are compiled:")
        print("  bash scripts/build.sh")
        return 1

    # Generate test files
    tmpdir, files = generate_test_files(num_files=100, size_range=(1024, 50*1024))

    print(f"Generated {len(files)} test files in {tmpdir}")

    # Benchmark 1: Single file scans
    print("\n" + "=" * 60)
    print("Benchmark 1: Single File Scans")
    print("=" * 60)

    single_times = []
    for i, file_path in enumerate(files[:20]):  # Test first 20
        elapsed, result = benchmark_single_file(detector, file_path)
        single_times.append(elapsed)

        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1}/20 files...")

    print_statistics(single_times, "Single File Scan Performance")

    # Benchmark 2: Batch scanning
    print("\n" + "=" * 60)
    print("Benchmark 2: Batch Scanning")
    print("=" * 60)

    batch_elapsed, batch_results = benchmark_batch(detector, files[:50])
    print(f"Scanned 50 files in {batch_elapsed:.2f}s")
    print(f"Throughput: {50 / batch_elapsed:.2f} files/sec")

    # Benchmark 3: Parallel directory scanning
    print("\n" + "=" * 60)
    print("Benchmark 3: Parallel Directory Scanning")
    print("=" * 60)

    parallel_elapsed, parallel_results = benchmark_parallel(detector, tmpdir)
    print(f"Scanned {len(parallel_results)} files in {parallel_elapsed:.2f}s")
    print(f"Throughput: {len(parallel_results) / parallel_elapsed:.2f} files/sec")

    # Detection statistics
    print("\n" + "=" * 60)
    print("Detection Statistics")
    print("=" * 60)

    stats = detector.get_statistics()
    print(f"Files Scanned: {stats['files_scanned']}")
    print(f"Threats Detected: {stats['threats_detected']}")
    if 'avg_processing_time' in stats:
        print(f"Average Processing Time: {stats['avg_processing_time']:.4f}s")
    if 'detection_rate' in stats:
        print(f"Detection Rate: {stats['detection_rate']:.2%}")

    # Feature analysis performance
    print("\n" + "=" * 60)
    print("Feature Analysis Performance")
    print("=" * 60)

    feature_times = []
    for result in batch_results[:20]:
        if 'processing_time' in result:
            feature_times.append(result['processing_time'])

    if feature_times:
        print_statistics(feature_times, "Feature Extraction Time")

    # Memory usage
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        print("\n" + "=" * 60)
        print("Memory Usage")
        print("=" * 60)
        print(f"RSS: {memory_info.rss / 1024 / 1024:.2f} MB")
        print(f"VMS: {memory_info.vms / 1024 / 1024:.2f} MB")
    except ImportError:
        pass

    # Cleanup
    print("\n" + "=" * 60)
    print("Cleaning up...")
    detector.shutdown()

    import shutil
    shutil.rmtree(tmpdir)

    print("=" * 60)
    print("Benchmark completed successfully!")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
