"""
Setup script for Quantum Neural Network Malware Detector
Compiles Cython modules for maximum performance
"""

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np
import sys
import os

# Compiler flags for optimization
if sys.platform == 'win32':
    extra_compile_args = ['/O2', '/openmp']
    extra_link_args = []
else:
    extra_compile_args = [
        '-O3',                    # Maximum optimization
        '-march=native',          # Optimize for current CPU
        '-ffast-math',           # Fast math operations
        '-fopenmp',              # OpenMP parallelization
        '-ftree-vectorize',      # Auto-vectorization
        '-funroll-loops',        # Loop unrolling
        '-fno-strict-aliasing'
    ]
    extra_link_args = ['-fopenmp']

# Define Cython extensions
extensions = [
    # Quantum Neural Network Core
    Extension(
        'src.qnn_core.quantum_layer',
        sources=['src/qnn_core/quantum_layer.pyx'],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++'
    ),

    # Polymorphic Malware Detector
    Extension(
        'src.malware_detection.polymorphic_detector',
        sources=['src/malware_detection/polymorphic_detector.pyx'],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++'
    ),

    # Quantum Encryption
    Extension(
        'src.crypto.quantum_encryption',
        sources=['src/crypto/quantum_encryption.pyx'],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++'
    ),

    # File Event Processor
    Extension(
        'src.file_processor.event_processor',
        sources=['src/file_processor/event_processor.pyx'],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++'
    ),
]

# Read requirements
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read README
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='quantum-malware-detector',
    version='1.0.0',
    author='Quantum Security Research Team',
    author_email='security@quantum-research.org',
    description='Production-ready Quantum Neural Network Malware Detection System',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/your-org/quantum-malware-detector',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Information Technology',
        'Intended Audience :: System Administrators',
        'Topic :: Security',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Cython',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': '3',
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'initializedcheck': False,
            'nonecheck': False,
            'embedsignature': True,
            'optimize.use_switch': True,
            'optimize.unpack_method_calls': True,
        },
        annotate=True,  # Generate HTML annotation files
        nthreads=os.cpu_count(),
    ),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'qnn-scan=src.qnn_malware_detector:main',
        ],
    },
    zip_safe=False,
)
