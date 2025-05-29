"""
Multi Medical Image Encryption System
Comprehensive implementation of the 6-step encryption process
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Any
import os

# Import our custom modules
from img_bluring import randomly_change_specified_bits, process_image_data
from key_generation import KeyGenerator
from dna_operations import DNAOperations
from fisher_scrambling import FisherScrambling
from bit_plane_operations import BitPlaneOperations


class MedicalImageEncryption:
    """
    Complete medical image encryption system implementing all 6 steps
    """

    def __init__(self, seed: int = None):
        """
        Initialize the encryption system

        Args:
            seed: Optional seed for reproducible encryption
        """
        self.seed = seed
        self.key_generator = KeyGenerator(seed)
        self.dna_operations = DNAOperations()
        self.fisher_scrambling = FisherScrambling(seed)
        self.bit_plane_operations = BitPlaneOperations(seed)

        # Store encryption metadata
        self.encryption_metadata = {}

    def load_medical_image(self, image_path: str) -> np.ndarray:
        """
        Load medical image from file

        Args:
            image_path: Path to the medical image

        Returns:
            Loaded image as numpy array
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load as grayscale for medical images
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Could not load image from: {image_path}")

        return image

    def step1_random_bit_modification(self, image: np.ndarray) -> np.ndarray:
        """
        Step 1: Randomly change the 3 bit binary information of a medical image

        Args:
            image: Input medical image

        Returns:
            Image with randomly modified bits
        """
        print("Step 1: Randomly modifying 3 bits of medical image...")
        return process_image_data(image)

    def step2_combine_binary_bits(self, before_blur: np.ndarray, after_blur: np.ndarray) -> np.ndarray:
        """
        Step 2: Create new pixels by combining binary bits before and after blur

        Args:
            before_blur: Original image
            after_blur: Blurred image

        Returns:
            Combined binary representation
        """
        print("Step 2: Combining binary bits before and after blur...")
        return self.key_generator.combine_binary_bits(before_blur, after_blur)

    def step3_generate_sha512_keys(self, combined_bits: np.ndarray) -> List[str]:
        """
        Step 3: Integrate combined bits with SHA-512 to generate random keys

        Args:
            combined_bits: Combined binary data

        Returns:
            List of SHA-512 keys
        """
        print("Step 3: Generating SHA-512 keys from combined bits...")
        return self.key_generator.generate_sha512_keys(combined_bits, num_keys=8)

    def step4_dna_encoding_and_scrambling(self, image: np.ndarray, keys: List[str]) -> Tuple[np.ndarray, Dict]:
        """
        Step 4: Convert first 3 bits to 0-7 range, DNA encoding, and 3D Fisher scrambling

        Args:
            image: Input image
            keys: SHA-512 keys for operations

        Returns:
            Tuple of (scrambled_dna_image, scrambling_info)
        """
        print("Step 4: DNA encoding and 3D Fisher-Yates scrambling...")

        # Convert first 3 bits to 0-7 range
        converted_image = self.dna_operations.convert_first_3bits_to_range(image)

        # DNA encoding using multiple rules
        dna_images = []
        for i, key in enumerate(keys[:4]):  # Use first 4 keys for 4 DNA rules
            rule_name = f'rule{(i % 8) + 1}'
            dna_encoded = self.dna_operations.encode_to_dna(converted_image, rule_name)
            dna_images.append(dna_encoded)

        # Create 3D stack for cross-plane scrambling
        # Convert DNA strings to numeric for 3D operations
        numeric_images = []
        for dna_img in dna_images:
            # Convert DNA sequences to numeric representation
            numeric_img = np.zeros(dna_img.shape, dtype=np.uint8)
            for i in range(dna_img.shape[0]):
                for j in range(dna_img.shape[1]):
                    # Convert DNA sequence to number (A=0, T=1, G=2, C=3)
                    dna_seq = dna_img[i, j]
                    numeric_val = 0
                    for k, base in enumerate(dna_seq):
                        base_val = {'A': 0, 'T': 1, 'G': 2, 'C': 3}[base]
                        numeric_val += base_val * (4 ** k)
                    numeric_img[i, j] = min(255, numeric_val % 256)
            numeric_images.append(numeric_img)

        # 3D Fisher-Yates cross-plane scrambling
        image_3d = self.fisher_scrambling.create_3d_image_stack(numeric_images)
        scrambled_3d, scrambling_info = self.fisher_scrambling.cross_plane_scrambling_3d(
            image_3d, keys[0]
        )

        return scrambled_3d, scrambling_info

    def step5_asymmetric_dna_diffusion(self, scrambled_3d: np.ndarray, keys: List[str]) -> List[np.ndarray]:
        """
        Step 5: Asymmetric DNA coding/decoding for high-quality diffusion

        Args:
            scrambled_3d: 3D scrambled image
            keys: SHA-512 keys for diffusion

        Returns:
            List of multiple ciphertext images of different sizes
        """
        print("Step 5: Asymmetric DNA diffusion for multiple ciphertext images...")

        multiple_ciphers = []
        depth, height, width = scrambled_3d.shape

        # Create different sized cipher images
        for i in range(depth):
            plane = scrambled_3d[i, :, :]

            # Convert back to DNA for diffusion operations
            rule_name = f'rule{(i % 8) + 1}'
            dna_plane = self.dna_operations.encode_to_dna(plane, rule_name)

            # Generate key sequence for diffusion
            key_sequence = self.key_generator.generate_random_sequence(keys[i % len(keys)], height * width)
            dna_key_seq = ''.join(['ATGC'[val % 4] for val in key_sequence])

            # Apply asymmetric DNA diffusion
            diffused_dna = self.dna_operations.asymmetric_dna_diffusion(dna_plane, dna_key_seq)

            # Decode back to numeric
            diffused_numeric = self.dna_operations.decode_from_dna(diffused_dna, rule_name)

            # Apply additional DNA operations for enhanced security
            if i % 3 == 0:
                # Apply DNA XOR with key-derived pattern
                key_array = np.clip(key_sequence[:height*width], 0, 255).reshape(height, width).astype(np.uint8)
                key_pattern = self.dna_operations.encode_to_dna(key_array, rule_name)
                for h in range(height):
                    for w in range(width):
                        diffused_dna[h, w] = self.dna_operations.dna_xor_operation(
                            diffused_dna[h, w], key_pattern[h, w]
                        )
                diffused_numeric = self.dna_operations.decode_from_dna(diffused_dna, rule_name)

            multiple_ciphers.append(diffused_numeric)

        return multiple_ciphers

    def step6_generate_same_size_ciphers(self, original_image: np.ndarray,
                                       blurred_image: np.ndarray,
                                       multiple_ciphers: List[np.ndarray]) -> List[np.ndarray]:
        """
        Step 6: Use bit plane information to generate multiple same-size ciphertext images

        Args:
            original_image: Original image
            blurred_image: Blurred image
            multiple_ciphers: Multiple cipher images from step 5

        Returns:
            List of same-size ciphertext images
        """
        print("Step 6: Generating multiple same-size ciphertext images...")

        # Use bit plane operations to create multiple same-size cipher images
        same_size_ciphers = self.bit_plane_operations.create_multiple_cipher_images(
            original_image, blurred_image, num_images=len(multiple_ciphers)
        )

        # Combine with the multiple ciphers from step 5 for enhanced security
        final_ciphers = []
        target_shape = original_image.shape

        for i, (cipher, same_size_cipher) in enumerate(zip(multiple_ciphers, same_size_ciphers)):
            # Resize cipher to match target shape if needed
            if cipher.shape != target_shape:
                cipher_resized = cv2.resize(cipher, (target_shape[1], target_shape[0]))
            else:
                cipher_resized = cipher.copy()

            # Ensure both arrays have the same shape and data type
            cipher_resized = cipher_resized.astype(np.uint8)
            same_size_cipher = same_size_cipher.astype(np.uint8)

            # Resize same_size_cipher if needed
            if same_size_cipher.shape != target_shape:
                same_size_cipher = cv2.resize(same_size_cipher, (target_shape[1], target_shape[0]))

            # Combine using XOR operation
            combined_cipher = np.bitwise_xor(cipher_resized, same_size_cipher)

            # Apply final bit plane transformation
            bit_planes = self.bit_plane_operations.extract_bit_planes(combined_cipher)

            # Shuffle bit planes for additional security
            np.random.seed(self.seed + i if self.seed else None)
            shuffled_indices = np.random.permutation(8)
            shuffled_bit_planes = [bit_planes[idx] for idx in shuffled_indices]

            final_cipher = self.bit_plane_operations.reconstruct_from_bit_planes(shuffled_bit_planes)
            final_ciphers.append(final_cipher)

            # Store shuffling information for decryption
            self.encryption_metadata[f'bit_plane_shuffle_{i}'] = shuffled_indices

        return final_ciphers

    def encrypt_medical_image(self, image_path: str) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Complete encryption process implementing all 6 steps

        Args:
            image_path: Path to the medical image

        Returns:
            Tuple of (list_of_cipher_images, encryption_metadata)
        """
        print(f"Starting medical image encryption for: {image_path}")

        # Load the medical image
        original_image = self.load_medical_image(image_path)
        print(f"Loaded image with shape: {original_image.shape}")

        # Step 1: Random bit modification
        blurred_image = self.step1_random_bit_modification(original_image)

        # Step 2: Combine binary bits
        combined_bits = self.step2_combine_binary_bits(original_image, blurred_image)

        # Step 3: Generate SHA-512 keys
        sha512_keys = self.step3_generate_sha512_keys(combined_bits)

        # Step 4: DNA encoding and 3D Fisher scrambling
        scrambled_3d, scrambling_info = self.step4_dna_encoding_and_scrambling(
            blurred_image, sha512_keys
        )

        # Step 5: Asymmetric DNA diffusion
        multiple_ciphers = self.step5_asymmetric_dna_diffusion(scrambled_3d, sha512_keys)

        # Step 6: Generate same-size ciphertext images
        final_ciphers = self.step6_generate_same_size_ciphers(
            original_image, blurred_image, multiple_ciphers
        )

        # Store metadata for decryption
        self.encryption_metadata.update({
            'original_shape': original_image.shape,
            'sha512_keys': sha512_keys,
            'scrambling_info': scrambling_info,
            'num_ciphers': len(final_ciphers),
            'seed': self.seed
        })

        print(f"Encryption completed! Generated {len(final_ciphers)} cipher images.")
        return final_ciphers, self.encryption_metadata

    def save_cipher_images(self, cipher_images: List[np.ndarray], output_dir: str = "cipher_outputs"):
        """
        Save cipher images to files

        Args:
            cipher_images: List of cipher images
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)

        for i, cipher in enumerate(cipher_images):
            output_path = os.path.join(output_dir, f"cipher_image_{i+1}.png")
            cv2.imwrite(output_path, cipher)
            print(f"Saved cipher image {i+1} to: {output_path}")


def test_medical_encryption():
    """Test the complete medical image encryption system"""
    print("Testing Multi Medical Image Encryption System...")

    # Initialize encryption system
    encryption_system = MedicalImageEncryption(seed=42)

    # Test with a sample image (create if doesn't exist)
    test_image_path = "braincd.png"

    if os.path.exists(test_image_path):
        # Encrypt the medical image
        cipher_images, metadata = encryption_system.encrypt_medical_image(test_image_path)

        # Save cipher images
        encryption_system.save_cipher_images(cipher_images)

        print(f"Encryption test completed successfully!")
        print(f"Generated {len(cipher_images)} cipher images")
        print(f"Metadata keys: {list(metadata.keys())}")
    else:
        print(f"Test image {test_image_path} not found. Creating dummy test...")

        # Create dummy test image
        dummy_image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        cv2.imwrite(test_image_path, dummy_image)

        # Run encryption on dummy image
        cipher_images, metadata = encryption_system.encrypt_medical_image(test_image_path)
        encryption_system.save_cipher_images(cipher_images)

        print("Dummy encryption test completed successfully!")


if __name__ == "__main__":
    test_medical_encryption()
