"""
Test suite for Quantum Neural Network Core
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.qnn_core import (
        QuantumNeuron,
        QuantumLayer,
        QuantumDeepNeuralNetwork,
        quantum_fourier_transform,
        quantum_entanglement_measure
    )
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    pytestmark = pytest.mark.skip("Cython modules not compiled")


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython modules not available")
class TestQuantumNeuron:
    """Test QuantumNeuron class"""

    def test_initialization(self):
        """Test neuron initialization"""
        neuron = QuantumNeuron(n_inputs=10, n_qubits=4)
        assert neuron is not None

    def test_forward_pass(self):
        """Test forward propagation"""
        neuron = QuantumNeuron(n_inputs=10, n_qubits=4)
        inputs = np.random.randn(10).astype(np.float64)
        output = neuron.forward(inputs)
        assert isinstance(output, float)
        assert -1.0 <= output <= 1.0

    def test_backward_pass(self):
        """Test backpropagation"""
        neuron = QuantumNeuron(n_inputs=10, n_qubits=4)
        inputs = np.random.randn(10).astype(np.float64)
        gradient = 0.1
        neuron.backward(gradient, inputs)
        # Should not raise any exceptions


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython modules not available")
class TestQuantumLayer:
    """Test QuantumLayer class"""

    def test_initialization(self):
        """Test layer initialization"""
        layer = QuantumLayer(n_inputs=10, n_neurons=5, n_qubits=4)
        assert layer is not None

    def test_forward_pass(self):
        """Test forward propagation"""
        layer = QuantumLayer(n_inputs=10, n_neurons=5, n_qubits=4)
        inputs = np.random.randn(10).astype(np.float64)
        outputs = layer.forward(inputs)
        assert len(outputs) == 5
        assert all(-1.0 <= x <= 1.0 for x in outputs)

    def test_backward_pass(self):
        """Test backpropagation"""
        layer = QuantumLayer(n_inputs=10, n_neurons=5, n_qubits=4)
        inputs = np.random.randn(10).astype(np.float64)
        gradients = np.random.randn(5).astype(np.float64)
        layer.backward(gradients, inputs)


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython modules not available")
class TestQuantumDeepNeuralNetwork:
    """Test QuantumDeepNeuralNetwork class"""

    def test_initialization(self):
        """Test QDNN initialization"""
        config = [(10, 8, 3), (8, 4, 3), (4, 2, 3)]
        qdnn = QuantumDeepNeuralNetwork(config)
        assert qdnn is not None

    def test_prediction(self):
        """Test prediction"""
        config = [(10, 8, 3), (8, 4, 3), (4, 2, 3)]
        qdnn = QuantumDeepNeuralNetwork(config)
        inputs = np.random.randn(10).astype(np.float64)
        outputs = qdnn.predict(inputs)
        assert len(outputs) == 2

    def test_training(self):
        """Test training"""
        config = [(10, 8, 3), (8, 4, 3), (4, 2, 3)]
        qdnn = QuantumDeepNeuralNetwork(config)
        inputs = np.random.randn(10).astype(np.float64)
        targets = np.array([1.0, 0.0], dtype=np.float64)
        qdnn.train(inputs, targets, epochs=5)


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython modules not available")
class TestQuantumOperations:
    """Test quantum operations"""

    def test_quantum_fourier_transform(self):
        """Test QFT"""
        signal = np.random.randn(64).astype(np.float64)
        result = quantum_fourier_transform(signal)
        assert len(result) == len(signal)

    def test_quantum_entanglement_measure(self):
        """Test entanglement measurement"""
        state1 = np.random.randn(10).astype(np.float64)
        state2 = np.random.randn(10).astype(np.float64)
        correlation = quantum_entanglement_measure(state1, state2)
        assert 0.0 <= correlation <= 1.0


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython modules not available")
def test_quantum_integration():
    """Integration test for quantum components"""
    # Create a simple network
    config = [(20, 10, 3), (10, 5, 3), (5, 2, 3)]
    qdnn = QuantumDeepNeuralNetwork(config)

    # Generate synthetic data
    X_train = np.random.randn(100, 20).astype(np.float64)
    y_train = np.random.randint(0, 2, size=(100, 2)).astype(np.float64)

    # Train for a few epochs
    for i in range(10):
        idx = np.random.randint(0, len(X_train))
        qdnn.train(X_train[idx], y_train[idx], epochs=1)

    # Test prediction
    predictions = qdnn.predict(X_train[0])
    assert len(predictions) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
