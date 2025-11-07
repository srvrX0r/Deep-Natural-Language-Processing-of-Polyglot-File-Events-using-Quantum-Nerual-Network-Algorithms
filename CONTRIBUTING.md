# Contributing to Quantum Neural Network Malware Detector

Thank you for your interest in contributing to the Quantum Neural Network Malware Detector! This document provides guidelines for contributing to this project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [How to Contribute](#how-to-contribute)
4. [Development Setup](#development-setup)
5. [Coding Standards](#coding-standards)
6. [Testing](#testing)
7. [Documentation](#documentation)
8. [Pull Request Process](#pull-request-process)
9. [Community](#community)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- GCC/G++ compiler with C++11 support
- OpenMP library
- Git

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/Deep-Natural-Language-Processing-of-Polyglot-File-Events-using-Quantum-Nerual-Network-Algorithms.git
   cd Deep-Natural-Language-Processing-of-Polyglot-File-Events-using-Quantum-Nerual-Network-Algorithms
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/srvrX0r/Deep-Natural-Language-Processing-of-Polyglot-File-Events-using-Quantum-Nerual-Network-Algorithms.git
   ```

## How to Contribute

### Types of Contributions

We welcome several types of contributions:

1. **Bug Reports**: Report bugs using GitHub Issues
2. **Feature Requests**: Suggest new features via GitHub Issues
3. **Code Contributions**: Submit code via Pull Requests
4. **Documentation**: Improve documentation, examples, tutorials
5. **Testing**: Add test cases, improve test coverage
6. **Performance**: Optimize algorithms, improve speed
7. **Research**: Contribute new quantum algorithms or detection methods

### Reporting Bugs

When reporting bugs, please include:

- **Clear title** and description
- **Steps to reproduce** the issue
- **Expected behavior** vs actual behavior
- **Environment details** (OS, Python version, etc.)
- **Error messages** and stack traces
- **Sample files** if applicable (ensure they're safe/sanitized)

**Template**:
```markdown
**Bug Description**
A clear description of the bug.

**To Reproduce**
1. Step one
2. Step two
3. See error

**Expected Behavior**
What should happen.

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.10]
- Version: [e.g., 2.0.0]

**Additional Context**
Any other relevant information.
```

### Suggesting Features

Feature requests should include:

- **Use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives**: What alternatives have you considered?
- **Implementation ideas**: If you have technical suggestions

## Development Setup

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Development Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available
```

### 3. Build Cython Extensions

```bash
bash scripts/build.sh
```

### 4. Install in Development Mode

```bash
pip install -e .
```

### 5. Run Tests

```bash
pytest tests/ -v
```

## Coding Standards

### Python Style

We follow PEP 8 with some modifications:

- **Line length**: 100 characters maximum
- **Indentation**: 4 spaces (no tabs)
- **Imports**: Organized in three groups (standard library, third-party, local)
- **Docstrings**: Google style for all public functions/classes

**Example**:
```python
def scan_file(file_path: str, encrypt: bool = True) -> Dict[str, Any]:
    """
    Scan a file for malware.

    Args:
        file_path: Path to the file to scan
        encrypt: Whether to encrypt the file data during analysis

    Returns:
        Dictionary containing detection results with keys:
        - is_malicious: bool
        - threat_level: str
        - threat_score: float

    Raises:
        FileNotFoundError: If file_path does not exist
        PermissionError: If file cannot be read

    Example:
        >>> detector = QuantumMalwareDetector()
        >>> result = detector.scan_file('/path/to/file.exe')
        >>> print(result['threat_level'])
        'HIGH'
    """
    pass
```

### Cython Style

For Cython files (.pyx):

- Use type annotations for all function parameters
- Add `nogil` where possible for parallelization
- Use memory views instead of NumPy arrays in hot loops
- Always use `boundscheck=False` and `wraparound=False` in performance-critical code

**Example**:
```cython
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double process_data(double[:] data) nogil:
    """Process data array efficiently."""
    cdef int i
    cdef double result = 0.0

    for i in range(len(data)):
        result += data[i] * data[i]

    return result
```

### Type Hints

Use type hints for all Python code:

```python
from typing import Dict, List, Optional, Tuple

def analyze_results(
    results: List[Dict[str, Any]],
    threshold: float = 0.7
) -> Tuple[int, int]:
    """Analyze detection results."""
    pass
```

### Documentation

All code must be documented:

1. **Module docstrings**: At the top of each file
2. **Class docstrings**: For all classes
3. **Function docstrings**: For all public functions
4. **Inline comments**: For complex logic
5. **Type hints**: For all functions

## Testing

### Writing Tests

- Place tests in `tests/` directory
- Name test files as `test_*.py`
- Use pytest fixtures for common setup
- Aim for >90% code coverage

**Example Test**:
```python
import pytest
from src.qnn_malware_detector import QuantumMalwareDetector

class TestQuantumMalwareDetector:
    """Tests for QuantumMalwareDetector class."""

    @pytest.fixture
    def detector(self):
        """Create detector instance for testing."""
        return QuantumMalwareDetector()

    def test_initialization(self, detector):
        """Test detector initializes correctly."""
        assert detector is not None
        assert detector.config is not None

    def test_scan_clean_file(self, detector, tmp_path):
        """Test scanning a clean file."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a clean file")

        # Scan
        result = detector.scan_file(str(test_file))

        # Assert
        assert result['is_malicious'] == False
        assert result['threat_level'] == 'CLEAN'
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_qnn_core.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run benchmarks
pytest tests/ --benchmark-only
```

### Test Categories

We use pytest marks for test categories:

```python
@pytest.mark.unit
def test_unit():
    """Unit test"""
    pass

@pytest.mark.integration
def test_integration():
    """Integration test"""
    pass

@pytest.mark.slow
def test_slow_operation():
    """Slow test"""
    pass
```

Run specific categories:
```bash
pytest -m unit          # Run only unit tests
pytest -m "not slow"    # Skip slow tests
```

## Documentation

### README Updates

When adding features, update:
- Feature list in README.md
- Usage examples
- Configuration options

### API Documentation

Document all public APIs:

```python
class NewDetector:
    """
    Brief description of the detector.

    This detector implements [algorithm name] for detecting [threat type].
    It uses [approach] to achieve [goal].

    Attributes:
        config: Configuration dictionary
        model: The underlying model

    Example:
        >>> detector = NewDetector(config={'threshold': 0.8})
        >>> result = detector.detect(data)
    """
    pass
```

### Adding Examples

Create example files in `examples/` directory:

```
examples/
├── basic_usage.py
├── advanced_detection.py
├── batch_processing.py
└── custom_config.py
```

## Pull Request Process

### Before Submitting

1. **Update from upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Create feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make changes** following coding standards

4. **Write tests** for new functionality

5. **Run test suite**:
   ```bash
   pytest tests/ -v
   ```

6. **Update documentation**

7. **Commit changes**:
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

### Commit Message Format

Follow this format:

```
<type>: <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding tests
- `chore`: Maintenance tasks

**Example**:
```
feat: Add Quantum Attention Mechanism

Implement multi-head quantum attention for feature selection.
This improves detection accuracy by focusing on relevant features.

- Add QuantumAttentionMechanism class
- Implement query-key-value architecture
- Add tests for attention mechanism
- Update documentation

Closes #123
```

### Pull Request Template

When creating a PR, include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added new tests
- [ ] Updated existing tests

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests added and passing

## Related Issues
Closes #(issue number)
```

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **Code review** by at least one maintainer
3. **Testing** on multiple platforms if applicable
4. **Documentation review**
5. **Approval** from maintainer
6. **Merge** by maintainer

### After Merge

1. **Delete your branch** (if applicable)
2. **Update your fork**:
   ```bash
   git checkout main
   git pull upstream main
   git push origin main
   ```

## Performance Considerations

When contributing performance-critical code:

1. **Profile first**: Use cProfile to identify bottlenecks
2. **Benchmark**: Measure before and after changes
3. **Document**: Explain optimizations in comments
4. **Test**: Ensure optimizations don't break functionality

**Example Benchmark**:
```python
def test_performance_improvement(benchmark):
    """Benchmark new optimization."""
    detector = QuantumMalwareDetector()
    data = generate_test_data()

    result = benchmark(detector.scan_file, data)

    # Assert performance improvement
    assert benchmark.stats['mean'] < 0.1  # 100ms target
```

## Security Considerations

When contributing security-related code:

1. **Never commit** credentials, keys, or sensitive data
2. **Sanitize** example data
3. **Document** security implications
4. **Test** security features thoroughly
5. **Report** vulnerabilities privately

## Research Contributions

For novel quantum algorithms or detection methods:

1. **Provide references** to papers or research
2. **Explain theory** in documentation
3. **Include benchmarks** comparing to existing methods
4. **Add examples** demonstrating usage
5. **Consider** publishing a research paper

## Community

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, ideas
- **Pull Requests**: Code contributions

### Getting Help

- Check [README.md](README.md) and [USAGE.md](USAGE.md)
- Search existing issues
- Review [ADVANCED_FEATURES.md](ADVANCED_FEATURES.md)
- Ask in GitHub Discussions

### Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in academic citations (if applicable)

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

## Questions?

If you have questions about contributing, please:
- Open a GitHub Discussion
- Check existing documentation
- Review closed issues for similar questions

Thank you for contributing to the Quantum Neural Network Malware Detector! 🚀
