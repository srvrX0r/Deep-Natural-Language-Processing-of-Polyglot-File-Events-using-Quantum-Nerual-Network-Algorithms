# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Advanced Quantum Neural Network Architectures
Variational Quantum Circuits, Quantum Convolutional Networks, and Quantum Attention
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport exp, cos, sin, sqrt, log, fabs, tanh, atan2
from libc.stdlib cimport malloc, free, rand, RAND_MAX
import cython

cnp.import_array()

ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.complex128_t CTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class VariationalQuantumCircuit:
    """Variational Quantum Circuit for feature learning"""

    cdef double[:, :] parameters
    cdef int n_qubits
    cdef int n_layers
    cdef double[:] state_real
    cdef double[:] state_imag
    cdef int state_size
    cdef double learning_rate
    cdef double[:, :] parameter_gradients

    def __init__(self, int n_qubits=8, int n_layers=4, double learning_rate=0.01):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.state_size = 2 ** n_qubits

        # Initialize parameters (rotation angles)
        self.parameters = np.random.uniform(-np.pi, np.pi,
                                           (n_layers, n_qubits * 3)).astype(np.float64)
        self.parameter_gradients = np.zeros((n_layers, n_qubits * 3), dtype=np.float64)

        # Initialize quantum state
        self.state_real = np.zeros(self.state_size, dtype=np.float64)
        self.state_imag = np.zeros(self.state_size, dtype=np.float64)
        self.state_real[0] = 1.0  # |0...0> state

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void apply_rotation(self, int qubit, double theta, double phi, double lambda_angle) nogil:
        """Apply general rotation gate U3(theta, phi, lambda)"""
        cdef int i, j, mask
        cdef double cos_half, sin_half
        cdef double temp_real, temp_imag
        cdef double phase_real, phase_imag

        mask = 1 << qubit
        cos_half = cos(theta / 2.0)
        sin_half = sin(theta / 2.0)

        for i in range(0, self.state_size, 2 * mask):
            for j in range(mask):
                # Get amplitudes
                temp_real = self.state_real[i + j]
                temp_imag = self.state_imag[i + j]

                # Apply rotation matrix
                phase_real = cos(phi) * cos_half
                phase_imag = sin(phi) * cos_half

                self.state_real[i + j] = (phase_real * temp_real -
                                          phase_imag * temp_imag)
                self.state_imag[i + j] = (phase_real * temp_imag +
                                          phase_imag * temp_real)

                # Update |1> component
                temp_real = self.state_real[i + j + mask]
                temp_imag = self.state_imag[i + j + mask]

                phase_real = cos(lambda_angle) * sin_half
                phase_imag = sin(lambda_angle) * sin_half

                self.state_real[i + j + mask] = (phase_real * temp_real -
                                                 phase_imag * temp_imag)
                self.state_imag[i + j + mask] = (phase_real * temp_imag +
                                                 phase_imag * temp_real)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void apply_entangling_layer(self) nogil:
        """Apply entangling CNOT gates"""
        cdef int i

        for i in range(self.n_qubits - 1):
            self.apply_cnot(i, i + 1)

        # Wrap around
        if self.n_qubits > 2:
            self.apply_cnot(self.n_qubits - 1, 0)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void apply_cnot(self, int control, int target) nogil:
        """Apply CNOT gate"""
        cdef int i, control_mask, target_mask
        cdef double temp_real, temp_imag

        control_mask = 1 << control
        target_mask = 1 << target

        for i in range(self.state_size):
            if (i & control_mask) != 0:  # Control qubit is 1
                # Swap target qubit states
                if (i & target_mask) == 0:
                    temp_real = self.state_real[i]
                    temp_imag = self.state_imag[i]

                    self.state_real[i] = self.state_real[i | target_mask]
                    self.state_imag[i] = self.state_imag[i | target_mask]

                    self.state_real[i | target_mask] = temp_real
                    self.state_imag[i | target_mask] = temp_imag

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double[:] forward(self, double[:] input_data):
        """Forward pass through variational circuit"""
        cdef int layer, qubit, param_idx
        cdef double theta, phi, lambda_angle
        cdef double[:] output = np.zeros(self.n_qubits, dtype=np.float64)
        cdef int i
        cdef double probability

        # Reset state
        for i in range(self.state_size):
            self.state_real[i] = 0.0
            self.state_imag[i] = 0.0
        self.state_real[0] = 1.0

        # Encode input data
        for qubit in range(min(len(input_data), self.n_qubits)):
            theta = input_data[qubit] * 3.14159265359
            self.apply_rotation(qubit, theta, 0.0, 0.0)

        # Apply variational layers
        for layer in range(self.n_layers):
            # Rotation layer
            for qubit in range(self.n_qubits):
                param_idx = qubit * 3
                theta = self.parameters[layer, param_idx]
                phi = self.parameters[layer, param_idx + 1]
                lambda_angle = self.parameters[layer, param_idx + 2]
                self.apply_rotation(qubit, theta, phi, lambda_angle)

            # Entangling layer
            self.apply_entangling_layer()

        # Measure each qubit
        for qubit in range(self.n_qubits):
            probability = 0.0
            for i in range(self.state_size):
                if (i & (1 << qubit)) != 0:
                    probability += (self.state_real[i] * self.state_real[i] +
                                   self.state_imag[i] * self.state_imag[i])
            output[qubit] = probability

        return output

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void optimize_parameters(self, double[:] gradients):
        """Update parameters using gradients"""
        cdef int layer, param

        for layer in range(self.n_layers):
            for param in range(self.n_qubits * 3):
                self.parameters[layer, param] -= self.learning_rate * gradients[layer * self.n_qubits * 3 + param]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class QuantumConvolutionalNetwork:
    """Quantum Convolutional Neural Network for spatial feature extraction"""

    cdef int n_qubits
    cdef int kernel_size
    cdef int n_filters
    cdef double[:, :, :] quantum_kernels
    cdef double[:] pooling_params

    def __init__(self, int n_qubits=8, int kernel_size=3, int n_filters=16):
        self.n_qubits = n_qubits
        self.kernel_size = kernel_size
        self.n_filters = n_filters

        # Initialize quantum kernels (rotation angles)
        self.quantum_kernels = np.random.uniform(-np.pi, np.pi,
                                                 (n_filters, kernel_size, 3)).astype(np.float64)
        self.pooling_params = np.random.uniform(-1, 1, n_filters).astype(np.float64)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double apply_quantum_kernel(self, double[:] patch, int filter_idx) nogil:
        """Apply quantum kernel to a patch"""
        cdef int i
        cdef double result = 0.0
        cdef double theta, phi, lambda_angle
        cdef double amplitude_real = 1.0, amplitude_imag = 0.0
        cdef double temp_real, temp_imag

        # Apply quantum transformations
        for i in range(min(len(patch), self.kernel_size)):
            theta = self.quantum_kernels[filter_idx, i, 0] * patch[i]
            phi = self.quantum_kernels[filter_idx, i, 1]
            lambda_angle = self.quantum_kernels[filter_idx, i, 2]

            # Apply rotation
            temp_real = amplitude_real
            temp_imag = amplitude_imag

            amplitude_real = cos(theta/2) * temp_real - sin(theta/2) * cos(phi) * temp_imag
            amplitude_imag = cos(theta/2) * temp_imag + sin(theta/2) * sin(phi) * temp_real

        # Measure
        result = amplitude_real * amplitude_real + amplitude_imag * amplitude_imag
        return tanh(result * self.pooling_params[filter_idx])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double[:, :] forward(self, double[:] input_data):
        """Forward pass through quantum convolutional layer"""
        cdef int n_patches = len(input_data) - self.kernel_size + 1
        cdef double[:, :] feature_maps = np.zeros((self.n_filters, n_patches), dtype=np.float64)
        cdef int filter_idx, patch_idx, i
        cdef double[:] patch = np.zeros(self.kernel_size, dtype=np.float64)

        for filter_idx in range(self.n_filters):
            for patch_idx in range(n_patches):
                # Extract patch
                for i in range(self.kernel_size):
                    patch[i] = input_data[patch_idx + i]

                # Apply quantum kernel
                feature_maps[filter_idx, patch_idx] = self.apply_quantum_kernel(patch, filter_idx)

        return feature_maps

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class QuantumAttentionMechanism:
    """Quantum-inspired attention mechanism for feature selection"""

    cdef int d_model
    cdef int n_heads
    cdef double[:, :] query_weights
    cdef double[:, :] key_weights
    cdef double[:, :] value_weights
    cdef double[:, :] output_weights
    cdef double[:] attention_scores

    def __init__(self, int d_model=64, int n_heads=8):
        self.d_model = d_model
        self.n_heads = n_heads

        cdef int d_head = d_model // n_heads

        # Initialize weight matrices
        self.query_weights = np.random.randn(d_model, d_model).astype(np.float64) * 0.1
        self.key_weights = np.random.randn(d_model, d_model).astype(np.float64) * 0.1
        self.value_weights = np.random.randn(d_model, d_model).astype(np.float64) * 0.1
        self.output_weights = np.random.randn(d_model, d_model).astype(np.float64) * 0.1
        self.attention_scores = np.zeros(d_model, dtype=np.float64)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double quantum_similarity(self, double[:] q, double[:] k) nogil:
        """Compute quantum-inspired similarity between query and key"""
        cdef int i
        cdef double similarity = 0.0
        cdef double norm_q = 0.0, norm_k = 0.0
        cdef double angle

        for i in range(len(q)):
            similarity += q[i] * k[i]
            norm_q += q[i] * q[i]
            norm_k += k[i] * k[i]

        if norm_q > 0.0 and norm_k > 0.0:
            similarity /= sqrt(norm_q * norm_k)
            # Quantum phase encoding
            angle = atan2(similarity, 1.0 - similarity)
            return cos(angle) * cos(angle)  # Probability

        return 0.0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void apply_attention(self, double[:] query, double[:] key, double[:] value, double[:] output) nogil:
        """Apply quantum attention mechanism"""
        cdef int i, j
        cdef double similarity, total_weight = 0.0
        cdef double[:] weights = <double[:len(key)]>malloc(len(key) * sizeof(double))

        # Compute attention weights using quantum similarity
        for i in range(len(key)):
            weights[i] = self.quantum_similarity(query, key)
            total_weight += weights[i]

        # Normalize weights
        if total_weight > 0.0:
            for i in range(len(weights)):
                weights[i] /= total_weight

        # Weighted sum of values
        for i in range(len(output)):
            output[i] = 0.0
            for j in range(len(value)):
                output[i] += weights[j] * value[j]

        free(&weights[0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double[:] forward(self, double[:] input_features):
        """Forward pass with quantum attention"""
        cdef double[:] query = np.zeros(self.d_model, dtype=np.float64)
        cdef double[:] key = np.zeros(self.d_model, dtype=np.float64)
        cdef double[:] value = np.zeros(self.d_model, dtype=np.float64)
        cdef double[:] output = np.zeros(self.d_model, dtype=np.float64)
        cdef int i, j

        # Project input to query, key, value
        for i in range(min(len(input_features), self.d_model)):
            for j in range(self.d_model):
                query[j] += self.query_weights[i, j] * input_features[i]
                key[j] += self.key_weights[i, j] * input_features[i]
                value[j] += self.value_weights[i, j] * input_features[i]

        # Apply attention
        self.apply_attention(query, key, value, output)

        # Output projection
        cdef double[:] final_output = np.zeros(self.d_model, dtype=np.float64)
        for i in range(self.d_model):
            for j in range(self.d_model):
                final_output[i] += self.output_weights[i, j] * output[j]
            final_output[i] = tanh(final_output[i])

        return final_output

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class QuantumResidualBlock:
    """Residual block with quantum transformations"""

    cdef VariationalQuantumCircuit vqc1
    cdef VariationalQuantumCircuit vqc2
    cdef double[:] residual_weights
    cdef int feature_dim

    def __init__(self, int feature_dim=64, int n_qubits=6):
        self.feature_dim = feature_dim
        self.vqc1 = VariationalQuantumCircuit(n_qubits=n_qubits, n_layers=2)
        self.vqc2 = VariationalQuantumCircuit(n_qubits=n_qubits, n_layers=2)
        self.residual_weights = np.ones(feature_dim, dtype=np.float64)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double[:] forward(self, double[:] input_features):
        """Forward pass with residual connection"""
        cdef double[:] transformed1 = self.vqc1.forward(input_features)
        cdef double[:] transformed2 = self.vqc2.forward(transformed1)
        cdef double[:] output = np.zeros(len(input_features), dtype=np.float64)
        cdef int i

        # Residual connection
        for i in range(min(len(input_features), len(transformed2))):
            output[i] = input_features[i] + self.residual_weights[i] * transformed2[i]
            output[i] = tanh(output[i])  # Activation

        return output

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class HybridQuantumClassicalNetwork:
    """Hybrid network combining quantum and classical layers"""

    cdef QuantumConvolutionalNetwork qconv
    cdef QuantumAttentionMechanism qattn
    cdef QuantumResidualBlock qres
    cdef VariationalQuantumCircuit vqc
    cdef double[:, :] classical_weights
    cdef int n_classes

    def __init__(self, int input_dim=128, int n_classes=2):
        self.n_classes = n_classes

        # Quantum layers
        self.qconv = QuantumConvolutionalNetwork(n_qubits=8, kernel_size=5, n_filters=16)
        self.qattn = QuantumAttentionMechanism(d_model=64, n_heads=8)
        self.qres = QuantumResidualBlock(feature_dim=64, n_qubits=6)
        self.vqc = VariationalQuantumCircuit(n_qubits=8, n_layers=4)

        # Classical output layer
        self.classical_weights = np.random.randn(64, n_classes).astype(np.float64) * 0.1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double[:] predict(self, double[:] input_data):
        """Full forward pass through hybrid network"""
        cdef int i, j

        # Quantum convolution
        cdef double[:, :] conv_features = self.qconv.forward(input_data)

        # Flatten conv features
        cdef int total_features = conv_features.shape[0] * conv_features.shape[1]
        cdef double[:] flattened = np.zeros(total_features, dtype=np.float64)
        cdef int idx = 0
        for i in range(conv_features.shape[0]):
            for j in range(conv_features.shape[1]):
                if idx < total_features:
                    flattened[idx] = conv_features[i, j]
                    idx += 1

        # Quantum attention
        cdef double[:] attended = self.qattn.forward(flattened[:64])

        # Residual block
        cdef double[:] residual_out = self.qres.forward(attended)

        # Final VQC
        cdef double[:] quantum_features = self.vqc.forward(residual_out[:8])

        # Classical classification head
        cdef double[:] logits = np.zeros(self.n_classes, dtype=np.float64)
        for i in range(self.n_classes):
            logits[i] = 0.0
            for j in range(min(len(quantum_features), 64)):
                logits[i] += self.classical_weights[j, i] * quantum_features[j]

        # Softmax
        cdef double max_logit = logits[0]
        for i in range(1, self.n_classes):
            if logits[i] > max_logit:
                max_logit = logits[i]

        cdef double sum_exp = 0.0
        cdef double[:] probabilities = np.zeros(self.n_classes, dtype=np.float64)
        for i in range(self.n_classes):
            probabilities[i] = exp(logits[i] - max_logit)
            sum_exp += probabilities[i]

        for i in range(self.n_classes):
            probabilities[i] /= sum_exp

        return probabilities

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:] quantum_feature_encoding(double[:] classical_features, int n_qubits=8):
    """Encode classical features into quantum state and extract measurements"""
    cdef int state_size = 2 ** n_qubits
    cdef double[:] quantum_state_real = np.zeros(state_size, dtype=np.float64)
    cdef double[:] quantum_state_imag = np.zeros(state_size, dtype=np.float64)
    cdef double[:] measurements = np.zeros(n_qubits, dtype=np.float64)
    cdef int i, qubit
    cdef double angle, cos_angle, sin_angle
    cdef double prob

    # Initialize to |0>
    quantum_state_real[0] = 1.0

    # Amplitude encoding
    for i in range(min(len(classical_features), n_qubits)):
        angle = classical_features[i] * 3.14159265359 / 2.0
        cos_angle = cos(angle)
        sin_angle = sin(angle)

        # Simple rotation encoding
        quantum_state_real[1 << i] = sin_angle
        quantum_state_real[0] *= cos_angle

    # Normalize
    cdef double norm = 0.0
    for i in range(state_size):
        norm += quantum_state_real[i] * quantum_state_real[i]
    norm = sqrt(norm)

    if norm > 0.0:
        for i in range(state_size):
            quantum_state_real[i] /= norm

    # Measure each qubit
    for qubit in range(n_qubits):
        prob = 0.0
        for i in range(state_size):
            if (i & (1 << qubit)) != 0:
                prob += quantum_state_real[i] * quantum_state_real[i]
        measurements[qubit] = prob

    return measurements
