"""
Comprehensive Test Suite for Multi Medical Image Encryption System
Tests all components and the complete encryption pipeline
"""

import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# Import all modules
from core_implementation.medical_image_encryption import MedicalImageEncryption
from core_implementation.key_generation import KeyGenerator
from core_implementation.dna_operations import DNAOperations
from core_implementation.fisher_scrambling import FisherScrambling
from core_implementation.bit_plane_operations import BitPlaneOperations


class EncryptionTestSuite:
    """
    Comprehensive test suite for the medical image encryption system
    """

    def __init__(self):
        """Initialize the test suite"""
        self.test_results = {}
        self.test_images = {}

    def create_test_images(self):
        """Create various test images for testing"""
        print("Creating test images...")

        # 1. Simple gradient image
        gradient = np.zeros((64, 64), dtype=np.uint8)
        for i in range(64):
            for j in range(64):
                gradient[i, j] = (i + j) % 256
        self.test_images['gradient'] = gradient

        # 2. Checkerboard pattern
        checkerboard = np.zeros((64, 64), dtype=np.uint8)
        for i in range(64):
            for j in range(64):
                if (i // 8 + j // 8) % 2 == 0:
                    checkerboard[i, j] = 255
        self.test_images['checkerboard'] = checkerboard

        # 3. Random noise image
        np.random.seed(42)
        noise = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        self.test_images['noise'] = noise

        # 4. Simulated medical image (brain-like structure)
        medical_sim = np.zeros((128, 128), dtype=np.uint8)
        center_x, center_y = 64, 64
        for i in range(128):
            for j in range(128):
                dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                if dist < 50:
                    medical_sim[i, j] = np.clip(int(200 - dist * 2), 0, 255)
                elif dist < 60:
                    medical_sim[i, j] = np.clip(int(100 + np.sin(dist) * 50), 0, 255)
        self.test_images['medical_sim'] = medical_sim

        # Save test images
        os.makedirs("test_images", exist_ok=True)
        for name, image in self.test_images.items():
            cv2.imwrite(f"test_images/{name}.png", image)

        print(f"Created {len(self.test_images)} test images")

    def test_key_generation(self):
        """Test the key generation module"""
        print("\n=== Testing Key Generation Module ===")

        try:
            key_gen = KeyGenerator(seed=42)

            # Test with dummy images
            before = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
            after = np.random.randint(0, 256, (32, 32), dtype=np.uint8)

            # Test bit combination
            combined = key_gen.combine_binary_bits(before, after)
            assert combined.shape == before.shape, "Combined bits shape mismatch"

            # Test key generation
            keys = key_gen.generate_sha512_keys(combined, num_keys=4)
            assert len(keys) == 4, "Wrong number of keys generated"
            assert all(len(key) == 128 for key in keys), "Invalid key length"

            # Test numeric array conversion
            numeric_arrays = key_gen.keys_to_numeric_arrays(keys, (32, 32))
            assert len(numeric_arrays) == 4, "Wrong number of numeric arrays"

            self.test_results['key_generation'] = "PASSED"
            print("✓ Key generation tests passed")

        except Exception as e:
            self.test_results['key_generation'] = f"FAILED: {str(e)}"
            print(f"✗ Key generation tests failed: {e}")

    def test_dna_operations(self):
        """Test the DNA operations module"""
        print("\n=== Testing DNA Operations Module ===")

        try:
            dna_ops = DNAOperations()

            # Test DNA encoding/decoding
            test_image = np.array([[100, 150], [200, 50]], dtype=np.uint8)

            for rule in ['rule1', 'rule2', 'rule3', 'rule4']:
                encoded = dna_ops.encode_to_dna(test_image, rule)
                decoded = dna_ops.decode_from_dna(encoded, rule)
                assert np.array_equal(test_image, decoded), f"DNA encoding/decoding failed for {rule}"

            # Test DNA operations
            seq1, seq2 = "ATGC", "CGTA"
            xor_result = dna_ops.dna_xor_operation(seq1, seq2)
            add_result = dna_ops.dna_add_operation(seq1, seq2)
            sub_result = dna_ops.dna_sub_operation(seq1, seq2)

            assert len(xor_result) == 4, "DNA XOR result length incorrect"
            assert len(add_result) == 4, "DNA ADD result length incorrect"
            assert len(sub_result) == 4, "DNA SUB result length incorrect"

            # Test first 3 bits conversion
            converted = dna_ops.convert_first_3bits_to_range(test_image)
            assert converted.shape == test_image.shape, "Shape mismatch in bit conversion"

            self.test_results['dna_operations'] = "PASSED"
            print("✓ DNA operations tests passed")

        except Exception as e:
            self.test_results['dna_operations'] = f"FAILED: {str(e)}"
            print(f"✗ DNA operations tests failed: {e}")

    def test_fisher_scrambling(self):
        """Test the Fisher scrambling module"""
        print("\n=== Testing Fisher Scrambling Module ===")

        try:
            fisher = FisherScrambling(seed=42)

            # Create test 3D image
            test_images = [np.random.randint(0, 256, (16, 16), dtype=np.uint8) for _ in range(3)]
            image_3d = fisher.create_3d_image_stack(test_images)

            # Test 3D scrambling
            scrambled_3d, scrambling_info = fisher.cross_plane_scrambling_3d(image_3d, "test_key")
            unscrambled_3d = fisher.reverse_cross_plane_scrambling(scrambled_3d, scrambling_info)

            assert np.array_equal(image_3d, unscrambled_3d), "3D scrambling/unscrambling failed"

            # Test adaptive scrambling
            adaptive_scrambled, adaptive_info = fisher.adaptive_scrambling(image_3d)
            assert adaptive_scrambled.shape == image_3d.shape, "Adaptive scrambling shape mismatch"

            self.test_results['fisher_scrambling'] = "PASSED"
            print("✓ Fisher scrambling tests passed")

        except Exception as e:
            self.test_results['fisher_scrambling'] = f"FAILED: {str(e)}"
            print(f"✗ Fisher scrambling tests failed: {e}")

    def test_bit_plane_operations(self):
        """Test the bit plane operations module"""
        print("\n=== Testing Bit Plane Operations Module ===")

        try:
            bit_ops = BitPlaneOperations(seed=42)

            # Test bit plane extraction/reconstruction
            test_image = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
            bit_planes = bit_ops.extract_bit_planes(test_image)
            reconstructed = bit_ops.reconstruct_from_bit_planes(bit_planes)

            assert np.array_equal(test_image, reconstructed), "Bit plane extraction/reconstruction failed"

            # Test multiple cipher image creation
            original = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
            blurred = np.random.randint(0, 256, (32, 32), dtype=np.uint8)

            cipher_images = bit_ops.create_multiple_cipher_images(original, blurred, num_images=4)
            assert len(cipher_images) == 4, "Wrong number of cipher images created"

            self.test_results['bit_plane_operations'] = "PASSED"
            print("✓ Bit plane operations tests passed")

        except Exception as e:
            self.test_results['bit_plane_operations'] = f"FAILED: {str(e)}"
            print(f"✗ Bit plane operations tests failed: {e}")

    def test_complete_encryption_pipeline(self):
        """Test the complete encryption pipeline"""
        print("\n=== Testing Complete Encryption Pipeline ===")

        try:
            encryption_system = MedicalImageEncryption(seed=42)

            # Test with each test image
            for image_name, test_image in self.test_images.items():
                print(f"  Testing with {image_name} image...")

                # Save test image temporarily
                temp_path = f"temp_{image_name}.png"
                cv2.imwrite(temp_path, test_image)

                # Run complete encryption
                start_time = time.time()
                cipher_images, metadata = encryption_system.encrypt_medical_image(temp_path)
                encryption_time = time.time() - start_time

                # Validate results
                assert len(cipher_images) > 0, "No cipher images generated"
                assert 'sha512_keys' in metadata, "Missing SHA-512 keys in metadata"
                assert 'scrambling_info' in metadata, "Missing scrambling info in metadata"

                # Check cipher image properties
                for i, cipher in enumerate(cipher_images):
                    assert cipher.dtype == np.uint8, f"Cipher {i} has wrong data type"
                    assert cipher.shape == test_image.shape, f"Cipher {i} has wrong shape"

                print(f"    ✓ {image_name}: {len(cipher_images)} ciphers in {encryption_time:.3f}s")

                # Clean up
                os.remove(temp_path)

            self.test_results['complete_pipeline'] = "PASSED"
            print("✓ Complete encryption pipeline tests passed")

        except Exception as e:
            self.test_results['complete_pipeline'] = f"FAILED: {str(e)}"
            print(f"✗ Complete encryption pipeline tests failed: {e}")

    def test_security_properties(self):
        """Test security properties of the encryption"""
        print("\n=== Testing Security Properties ===")

        try:
            encryption_system = MedicalImageEncryption(seed=42)

            # Use the medical simulation image
            test_image = self.test_images['medical_sim']
            temp_path = "temp_security_test.png"
            cv2.imwrite(temp_path, test_image)

            # Encrypt the image
            cipher_images, metadata = encryption_system.encrypt_medical_image(temp_path)

            temp_path_modified = "temp_security_test_modified.png"
            try:
                # Test 1: Avalanche effect (small change in input should cause large change in output)
                test_image_modified = test_image.copy().astype(np.uint8)
                test_image_modified[0, 0] = np.clip((test_image_modified[0, 0] + 1) % 256, 0, 255)  # Change one pixel

                cv2.imwrite(temp_path_modified, test_image_modified)

                encryption_system_2 = MedicalImageEncryption(seed=42)  # Same seed
                cipher_images_modified, _ = encryption_system_2.encrypt_medical_image(temp_path_modified)

                # Calculate difference
                diff_ratio = np.mean(cipher_images[0] != cipher_images_modified[0])
                assert diff_ratio > 0.3, f"Avalanche effect too weak: {diff_ratio:.3f}"
                print(f"    ✓ Avalanche effect: {diff_ratio:.3f} (>0.3)")

                # Test 2: Key sensitivity (different seeds should produce different results)
                encryption_system_3 = MedicalImageEncryption(seed=123)  # Different seed
                cipher_images_diff_key, _ = encryption_system_3.encrypt_medical_image(temp_path)

                key_diff_ratio = np.mean(cipher_images[0] != cipher_images_diff_key[0])
                assert key_diff_ratio > 0.9, f"Key sensitivity too weak: {key_diff_ratio:.3f}"
                print(f"    ✓ Key sensitivity: {key_diff_ratio:.3f} (>0.9)")

                # Test 3: Statistical properties (cipher should look random)
                cipher_entropy = self._calculate_entropy(cipher_images[0])
                assert cipher_entropy > 7.0, f"Cipher entropy too low: {cipher_entropy:.3f}"
                print(f"    ✓ Cipher entropy: {cipher_entropy:.3f} (>7.0)")

            except Exception as e:
                raise e
            finally:
                # Clean up
                if os.path.exists(temp_path_modified):
                    os.remove(temp_path_modified)

            # Clean up
            os.remove(temp_path)

            self.test_results['security_properties'] = "PASSED"
            print("✓ Security properties tests passed")

        except Exception as e:
            self.test_results['security_properties'] = f"FAILED: {str(e)}"
            print(f"✗ Security properties tests failed: {e}")

    def _calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate Shannon entropy of an image"""
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist = hist[hist > 0]  # Remove zero entries
        prob = hist / np.sum(hist)
        entropy = -np.sum(prob * np.log2(prob))
        return entropy

    def run_all_tests(self):
        """Run all tests in the suite"""
        print("=" * 60)
        print("MEDICAL IMAGE ENCRYPTION - COMPREHENSIVE TEST SUITE")
        print("=" * 60)

        # Create test images
        self.create_test_images()

        # Run individual module tests
        self.test_key_generation()
        self.test_dna_operations()
        self.test_fisher_scrambling()
        self.test_bit_plane_operations()

        # Run integration tests
        self.test_complete_encryption_pipeline()
        self.test_security_properties()

        # Print summary
        self.print_test_summary()

    def print_test_summary(self):
        """Print test results summary"""
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)

        passed = 0
        total = len(self.test_results)

        for test_name, result in self.test_results.items():
            status = "✓ PASSED" if result == "PASSED" else f"✗ {result}"
            print(f"{test_name:25} : {status}")
            if result == "PASSED":
                passed += 1

        print("-" * 60)
        print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

        if passed == total:
            print("🎉 ALL TESTS PASSED! The encryption system is working correctly.")
        else:
            print("⚠️  Some tests failed. Please review the implementation.")


if __name__ == "__main__":
    test_suite = EncryptionTestSuite()
    test_suite.run_all_tests()
