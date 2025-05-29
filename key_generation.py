"""
Key Generation Module for Multi Medical Image Encryption
Implements SHA-512 based key generation from combined binary bits
"""

import hashlib
import numpy as np
from typing import List, Tuple


class KeyGenerator:
    """
    Generates cryptographic keys using SHA-512 from combined binary bits
    """

    def __init__(self, seed: int = None):
        """
        Initialize the key generator

        Args:
            seed: Optional seed for reproducible key generation
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def combine_binary_bits(self, before_blur: np.ndarray, after_blur: np.ndarray) -> np.ndarray:
        """
        Combine binary bits from before and after blur images

        Args:
            before_blur: Original image array
            after_blur: Blurred image array

        Returns:
            Combined binary representation
        """
        if before_blur.shape != after_blur.shape:
            raise ValueError("Before and after blur images must have the same shape")

        # XOR operation to combine the bits
        combined = np.bitwise_xor(before_blur, after_blur)

        # Additional bit manipulation for enhanced randomness
        # Rotate bits and combine with original
        rotated_before = np.roll(before_blur, 1, axis=0)
        rotated_after = np.roll(after_blur, 1, axis=1)

        # Create complex combination
        complex_combined = np.bitwise_xor(
            combined,
            np.bitwise_and(rotated_before, rotated_after)
        )

        return complex_combined

    def generate_sha512_keys(self, combined_bits: np.ndarray, num_keys: int = 4) -> List[str]:
        """
        Generate multiple SHA-512 keys from combined binary bits

        Args:
            combined_bits: Combined binary data
            num_keys: Number of keys to generate

        Returns:
            List of SHA-512 hash strings
        """
        keys = []

        # Convert array to bytes for hashing
        base_data = combined_bits.tobytes()

        for i in range(num_keys):
            # Create unique input for each key
            key_input = base_data + i.to_bytes(4, byteorder='big')

            # Add timestamp-like component for additional entropy
            entropy = int(np.sum(combined_bits)) + i * 1337
            key_input += entropy.to_bytes(8, byteorder='big')

            # Generate SHA-512 hash
            sha512_hash = hashlib.sha512(key_input).hexdigest()
            keys.append(sha512_hash)

        return keys

    def keys_to_numeric_arrays(self, keys: List[str], target_shape: Tuple[int, ...]) -> List[np.ndarray]:
        """
        Convert SHA-512 keys to numeric arrays for encryption operations

        Args:
            keys: List of SHA-512 hash strings
            target_shape: Target shape for the numeric arrays

        Returns:
            List of numeric arrays derived from keys
        """
        numeric_arrays = []
        total_elements = np.prod(target_shape)

        for key in keys:
            # Convert hex string to bytes
            key_bytes = bytes.fromhex(key)

            # Expand key bytes to match target size
            expanded_bytes = []
            for i in range(total_elements):
                expanded_bytes.append(key_bytes[i % len(key_bytes)])

            # Convert to numpy array and reshape
            numeric_array = np.array(expanded_bytes, dtype=np.uint8).reshape(target_shape)
            numeric_arrays.append(numeric_array)

        return numeric_arrays

    def generate_random_sequence(self, key: str, length: int) -> np.ndarray:
        """
        Generate a pseudo-random sequence from a key

        Args:
            key: SHA-512 key string
            length: Length of sequence to generate

        Returns:
            Random sequence array
        """
        # Use key as seed for reproducible randomness
        key_int = int(key[:16], 16)  # Use first 16 hex chars
        np.random.seed(key_int % (2**32))

        random_vals = np.random.randint(0, 256, size=length)
        return np.clip(random_vals, 0, 255).astype(np.uint8)

    def create_permutation_matrix(self, key: str, size: int) -> np.ndarray:
        """
        Create a permutation matrix from a key for scrambling operations

        Args:
            key: SHA-512 key string
            size: Size of the permutation matrix

        Returns:
            Permutation indices array
        """
        # Use key to generate deterministic permutation
        key_int = int(key[:16], 16)
        np.random.seed(key_int % (2**32))

        indices = np.arange(size)
        np.random.shuffle(indices)

        return indices


def test_key_generation():
    """Test the key generation functionality"""
    print("Testing Key Generation Module...")

    # Create test images
    before_blur = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
    after_blur = np.random.randint(0, 256, (10, 10), dtype=np.uint8)

    # Initialize key generator
    key_gen = KeyGenerator(seed=42)

    # Test bit combination
    combined = key_gen.combine_binary_bits(before_blur, after_blur)
    print(f"Combined bits shape: {combined.shape}")

    # Test key generation
    keys = key_gen.generate_sha512_keys(combined, num_keys=4)
    print(f"Generated {len(keys)} keys")
    print(f"First key (truncated): {keys[0][:32]}...")

    # Test numeric array conversion
    numeric_arrays = key_gen.keys_to_numeric_arrays(keys, (10, 10))
    print(f"Generated {len(numeric_arrays)} numeric arrays")

    # Test random sequence generation
    sequence = key_gen.generate_random_sequence(keys[0], 100)
    print(f"Random sequence length: {len(sequence)}")

    # Test permutation matrix
    perm = key_gen.create_permutation_matrix(keys[0], 100)
    print(f"Permutation matrix length: {len(perm)}")

    print("Key generation tests completed successfully!")


if __name__ == "__main__":
    test_key_generation()
