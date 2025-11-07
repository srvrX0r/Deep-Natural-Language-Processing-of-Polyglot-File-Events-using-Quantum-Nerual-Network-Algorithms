# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
Quantum-Resistant Public Key Encryption
Implements hybrid classical-quantum encryption for secure malware analysis
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport exp, log, sqrt, pow
from libc.stdlib cimport malloc, free, rand, RAND_MAX
from libc.string cimport memcpy, memset
import cython
import hashlib
import secrets

cnp.import_array()

ctypedef cnp.uint8_t BYTE_t
ctypedef cnp.uint64_t UINT64_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class LatticeBasedCrypto:
    """Lattice-based quantum-resistant cryptography"""

    cdef long[:, :] private_key
    cdef long[:, :] public_key
    cdef int dimension
    cdef long modulus
    cdef double[:] error_distribution

    def __init__(self, int dimension=512, long modulus=12289):
        """Initialize lattice-based crypto system"""
        self.dimension = dimension
        self.modulus = modulus
        self.private_key = np.random.randint(-2, 3, size=(dimension, dimension), dtype=np.int64)
        self.error_distribution = np.random.normal(0, 1.5, dimension).astype(np.float64)
        self._generate_keypair()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _generate_keypair(self):
        """Generate public/private keypair"""
        cdef int i, j, k
        cdef long[:, :] A = np.random.randint(0, self.modulus,
                                               size=(self.dimension, self.dimension),
                                               dtype=np.int64)
        self.public_key = np.zeros((self.dimension, self.dimension), dtype=np.int64)

        # Public key = A * private_key + error
        for i in range(self.dimension):
            for j in range(self.dimension):
                self.public_key[i, j] = 0
                for k in range(self.dimension):
                    self.public_key[i, j] += A[i, k] * self.private_key[k, j]
                self.public_key[i, j] = (self.public_key[i, j] +
                                         <long>self.error_distribution[j]) % self.modulus

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef long[:] encrypt_vector(self, long[:] plaintext):
        """Encrypt a vector using lattice-based encryption"""
        cdef int i, j
        cdef long[:] ciphertext = np.zeros(self.dimension, dtype=np.int64)
        cdef long[:] randomness = np.random.randint(-1, 2, self.dimension, dtype=np.int64)

        for i in range(self.dimension):
            ciphertext[i] = plaintext[i]
            for j in range(self.dimension):
                ciphertext[i] += self.public_key[i, j] * randomness[j]
            ciphertext[i] = ciphertext[i] % self.modulus

        return ciphertext

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef long[:] decrypt_vector(self, long[:] ciphertext):
        """Decrypt a vector using private key"""
        cdef int i, j
        cdef long[:] plaintext = np.zeros(self.dimension, dtype=np.int64)

        for i in range(self.dimension):
            plaintext[i] = ciphertext[i]
            for j in range(self.dimension):
                plaintext[i] -= self.private_key[i, j] * ciphertext[j]

            # Reduce modulo and round
            plaintext[i] = plaintext[i] % self.modulus
            if plaintext[i] > self.modulus // 2:
                plaintext[i] -= self.modulus

        return plaintext

    cpdef bytes encrypt_data(self, bytes data):
        """Encrypt arbitrary data"""
        cdef int data_len = len(data)
        cdef int blocks = (data_len + self.dimension - 1) // self.dimension
        cdef list encrypted_blocks = []

        for i in range(blocks):
            start = i * self.dimension
            end = min(start + self.dimension, data_len)
            block = data[start:end]

            # Pad if necessary
            if len(block) < self.dimension:
                block = block + b'\x00' * (self.dimension - len(block))

            # Convert to long array
            plaintext = np.frombuffer(block, dtype=np.uint8).astype(np.int64)
            ciphertext = self.encrypt_vector(plaintext)
            encrypted_blocks.append(np.asarray(ciphertext).tobytes())

        return b''.join(encrypted_blocks)

    cpdef bytes decrypt_data(self, bytes encrypted_data, int original_length):
        """Decrypt arbitrary data"""
        cdef int block_size = self.dimension * 8  # 8 bytes per long
        cdef int blocks = len(encrypted_data) // block_size
        cdef list decrypted_blocks = []

        for i in range(blocks):
            start = i * block_size
            end = start + block_size
            block = encrypted_data[start:end]

            ciphertext = np.frombuffer(block, dtype=np.int64)
            plaintext = self.decrypt_vector(ciphertext)
            decrypted_blocks.append(np.asarray(plaintext).astype(np.uint8).tobytes())

        result = b''.join(decrypted_blocks)
        return result[:original_length]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class QuantumKeyDistribution:
    """Simulated Quantum Key Distribution (QKD) for secure key exchange"""

    cdef double[:] quantum_channel
    cdef int key_length
    cdef double error_rate

    def __init__(self, int key_length=256, double error_rate=0.01):
        self.key_length = key_length
        self.error_rate = error_rate
        self.quantum_channel = np.zeros(key_length, dtype=np.float64)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void prepare_qubits(self, unsigned char[:] key_bits, unsigned char[:] bases) nogil:
        """Prepare qubits in random bases"""
        cdef int i
        cdef double angle

        for i in range(self.key_length):
            # Encode bit in chosen basis
            if bases[i] == 0:  # Rectilinear basis
                angle = 0.0 if key_bits[i] == 0 else 1.5708  # 0 or π/2
            else:  # Diagonal basis
                angle = 0.7854 if key_bits[i] == 0 else 2.3562  # π/4 or 3π/4

            # Add quantum noise
            angle += ((<double>rand() / <double>RAND_MAX) - 0.5) * self.error_rate

            self.quantum_channel[i] = angle

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void measure_qubits(self, unsigned char[:] measured_bits, unsigned char[:] bases) nogil:
        """Measure qubits in random bases"""
        cdef int i
        cdef double angle, threshold

        for i in range(self.key_length):
            angle = self.quantum_channel[i]

            # Measure in chosen basis
            if bases[i] == 0:  # Rectilinear
                threshold = 0.7854  # π/4
                measured_bits[i] = 1 if angle > threshold else 0
            else:  # Diagonal
                if angle < 1.5708:  # π/2
                    measured_bits[i] = 0
                else:
                    measured_bits[i] = 1

    cpdef bytes generate_shared_key(self):
        """Generate shared key using BB84 protocol"""
        cdef unsigned char[:] alice_bits = np.random.randint(0, 2, self.key_length, dtype=np.uint8)
        cdef unsigned char[:] alice_bases = np.random.randint(0, 2, self.key_length, dtype=np.uint8)
        cdef unsigned char[:] bob_bases = np.random.randint(0, 2, self.key_length, dtype=np.uint8)
        cdef unsigned char[:] bob_bits = np.zeros(self.key_length, dtype=np.uint8)

        # Alice prepares and sends qubits
        self.prepare_qubits(alice_bits, alice_bases)

        # Bob measures qubits
        self.measure_qubits(bob_bits, bob_bases)

        # Sift key: keep only bits where bases match
        cdef list sifted_key = []
        cdef int i
        for i in range(self.key_length):
            if alice_bases[i] == bob_bases[i]:
                sifted_key.append(alice_bits[i])

        # Convert to bytes
        if len(sifted_key) < 128:
            sifted_key.extend([0] * (128 - len(sifted_key)))

        return bytes(sifted_key[:128])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class HybridEncryption:
    """Hybrid quantum-resistant encryption system"""

    cdef LatticeBasedCrypto lattice_crypto
    cdef QuantumKeyDistribution qkd
    cdef bytes session_key
    cdef object hasher

    def __init__(self):
        self.lattice_crypto = LatticeBasedCrypto(dimension=256, modulus=12289)
        self.qkd = QuantumKeyDistribution(key_length=256)
        self.session_key = None

    cpdef void initialize_session(self):
        """Initialize secure session with quantum key distribution"""
        # Generate quantum-safe session key
        qkd_key = self.qkd.generate_shared_key()

        # Mix with additional entropy
        entropy = secrets.token_bytes(32)
        combined = qkd_key + entropy

        # Derive session key using SHA3-256
        self.session_key = hashlib.sha3_256(combined).digest()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef bytes xor_cipher(self, bytes data, bytes key):
        """Fast XOR cipher"""
        cdef int data_len = len(data)
        cdef int key_len = len(key)
        cdef unsigned char* data_ptr = <unsigned char*><char*>data
        cdef unsigned char* key_ptr = <unsigned char*><char*>key
        cdef unsigned char[:] result = np.zeros(data_len, dtype=np.uint8)
        cdef int i

        for i in range(data_len):
            result[i] = data_ptr[i] ^ key_ptr[i % key_len]

        return bytes(result)

    cpdef bytes encrypt(self, bytes plaintext):
        """Encrypt data using hybrid approach"""
        if self.session_key is None:
            self.initialize_session()

        # Layer 1: XOR with session key (fast)
        layer1 = self.xor_cipher(plaintext, self.session_key)

        # Layer 2: Lattice-based encryption (quantum-resistant)
        layer2 = self.lattice_crypto.encrypt_data(layer1)

        # Add authentication tag
        tag = hashlib.sha3_256(layer2).digest()[:16]

        return tag + layer2

    cpdef bytes decrypt(self, bytes ciphertext):
        """Decrypt data using hybrid approach"""
        if self.session_key is None:
            raise ValueError("Session not initialized")

        # Verify authentication tag
        tag = ciphertext[:16]
        encrypted_data = ciphertext[16:]

        expected_tag = hashlib.sha3_256(encrypted_data).digest()[:16]
        if tag != expected_tag:
            raise ValueError("Authentication failed - data may be corrupted")

        # Layer 2: Lattice decryption
        original_len = len(encrypted_data) // (256 * 8) * 256
        layer1 = self.lattice_crypto.decrypt_data(encrypted_data, original_len)

        # Layer 1: XOR decryption
        plaintext = self.xor_cipher(layer1, self.session_key)

        return plaintext

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class SecureFileHandler:
    """Secure file handling with encryption"""

    cdef HybridEncryption crypto
    cdef dict file_cache

    def __init__(self):
        self.crypto = HybridEncryption()
        self.crypto.initialize_session()
        self.file_cache = {}

    cpdef bytes encrypt_file_data(self, bytes file_data, str file_path):
        """Encrypt file data before analysis"""
        cdef bytes encrypted = self.crypto.encrypt(file_data)

        # Cache encrypted data
        file_hash = hashlib.sha256(file_data).hexdigest()
        self.file_cache[file_hash] = {
            'path': file_path,
            'size': len(file_data),
            'encrypted_size': len(encrypted)
        }

        return encrypted

    cpdef bytes decrypt_file_data(self, bytes encrypted_data):
        """Decrypt file data for analysis"""
        return self.crypto.decrypt(encrypted_data)

    cpdef dict get_file_info(self, str file_hash):
        """Get cached file information"""
        return self.file_cache.get(file_hash, {})
