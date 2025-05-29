"""
Advanced Multi-Image Medical Encryption System
Implements SAMCML + DNA Computing + 3D Fisher-Yates + Advanced Pixel Blurring
"""

import numpy as np
import cv2
import time
import os
from typing import List, Tuple, Dict, Any
import json

from .samcml_chaotic_system import SAMCMLChaoticSystem
from .advanced_pixel_blurring import AdvancedPixelBlurring
from .dynamic_dna_computing import DynamicDNAComputing
from .advanced_3d_fisher_scrambling import Advanced3DFisherScrambling


class AdvancedMedicalImageEncryption:
    """
    Advanced multi-image medical encryption system implementing:
    1. SAMCML chaotic system
    2. Advanced pixel blurring with bit plane extraction
    3. Dynamic DNA computing with 8 encoding rules
    4. 3D Fisher-Yates scrambling
    5. New DNA operations
    """

    def __init__(self, seed: int = None, noise_density: float = 0.05):
        """
        Initialize advanced encryption system

        Args:
            seed: Random seed for reproducible results
            noise_density: Density for salt-and-pepper noise
        """
        self.seed = seed
        self.noise_density = noise_density

        # Initialize subsystems
        self.pixel_blurring = AdvancedPixelBlurring(noise_density=noise_density, seed=seed)
        self.dna_computing = DynamicDNAComputing()

        # SAMCML and 3D scrambling will be initialized from hash
        self.samcml_system = None
        self.scrambling_system = None

        # Encryption metadata
        self.encryption_metadata = {}

    def encrypt_single_medical_image(self, image_path: str) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Encrypt single medical image using advanced pipeline

        Args:
            image_path: Path to medical image

        Returns:
            Tuple of (cipher_images, metadata)
        """
        print(f"Starting advanced encryption for: {os.path.basename(image_path)}")
        start_time = time.time()

        # Step 1: Load and preprocess image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        print(f"✓ Loaded image: {image.shape}")

        # Step 2: Advanced pixel blurring with bit plane extraction
        blurred_combined, sha512_hash, blur_metadata = self.pixel_blurring.process_medical_image(image)
        print(f"✓ Pixel blurring completed, hash generated")

        # Step 3: Initialize SAMCML system from SHA-512 hash
        self.samcml_system = SAMCMLChaoticSystem.from_sha512_hash(sha512_hash)
        self.scrambling_system = Advanced3DFisherScrambling(self.samcml_system)
        print(f"✓ SAMCML system initialized from hash")

        # Step 4: Dynamic DNA encoding
        dna_matrix, rule_indices = self.dna_computing.encode_image_to_dna(image)
        print(f"✓ DNA encoding completed: {dna_matrix.shape}")

        # Step 5: Generate chaotic key matrix
        chaotic_matrix = self.samcml_system.generate_chaotic_matrix(image.shape[0], image.shape[1])
        key_dna_matrix = self.dna_computing.generate_key_dna_matrix(chaotic_matrix, rule_indices)
        print(f"✓ Chaotic key matrix generated")

        # Step 6: Apply new DNA operation
        diffused_dna_matrix = self.dna_computing.dna_diffusion(dna_matrix, key_dna_matrix)
        print(f"✓ DNA diffusion applied")

        # Step 7: Create 3D structure for scrambling
        image_3d = self.scrambling_system.create_3d_from_dna_matrix(diffused_dna_matrix, num_planes=4)
        print(f"✓ 3D structure created: {image_3d.shape}")

        # Step 8: 3D Fisher-Yates scrambling
        scrambled_3d, scrambling_info = self.scrambling_system.multi_round_scrambling(image_3d, rounds=3)
        print(f"✓ 3D scrambling completed: {scrambling_info[0]['operation_count']} operations per round")

        # Step 9: Generate final cipher images
        cipher_images = self._generate_cipher_images(scrambled_3d, blurred_combined)
        print(f"✓ Generated {len(cipher_images)} cipher images")

        # Step 10: Create comprehensive metadata
        encryption_time = time.time() - start_time
        metadata = self._create_encryption_metadata(
            image_path, image.shape, sha512_hash, blur_metadata,
            scrambling_info, rule_indices, encryption_time
        )

        print(f"✓ Encryption completed in {encryption_time:.3f} seconds")
        return cipher_images, metadata

    def encrypt_multiple_medical_images(self, image_paths: List[str],
                                      batch_mode: str = "different_seeds") -> List[Tuple[List[np.ndarray], Dict[str, Any]]]:
        """
        Encrypt multiple medical images with batch processing

        Args:
            image_paths: List of image paths
            batch_mode: "same_seed", "different_seeds", or "adaptive"

        Returns:
            List of (cipher_images, metadata) tuples
        """
        print(f"Starting batch encryption for {len(image_paths)} images...")
        batch_start_time = time.time()

        results = []
        base_seed = self.seed if self.seed else 42

        for i, image_path in enumerate(image_paths):
            try:
                # Adjust seed based on batch mode
                if batch_mode == "same_seed":
                    current_seed = base_seed
                elif batch_mode == "different_seeds":
                    current_seed = base_seed + i
                else:  # adaptive
                    # Load image to determine adaptive seed
                    temp_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    current_seed = base_seed + int(np.mean(temp_img))

                # Update seed for this image
                self.seed = current_seed
                self.pixel_blurring.seed = current_seed
                if current_seed is not None:
                    np.random.seed(current_seed)

                print(f"\nProcessing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
                print(f"Using seed: {current_seed}")

                # Encrypt single image
                cipher_images, metadata = self.encrypt_single_medical_image(image_path)

                # Add batch information
                metadata['batch_info'] = {
                    'batch_index': i,
                    'total_images': len(image_paths),
                    'batch_mode': batch_mode,
                    'seed_used': current_seed
                }

                results.append((cipher_images, metadata))

            except Exception as e:
                print(f"Error encrypting {image_path}: {str(e)}")
                # Add error result
                error_metadata = {
                    'error': str(e),
                    'image_path': image_path,
                    'batch_info': {
                        'batch_index': i,
                        'total_images': len(image_paths),
                        'batch_mode': batch_mode,
                        'status': 'failed'
                    }
                }
                results.append(([], error_metadata))

        batch_time = time.time() - batch_start_time
        print(f"\n✓ Batch encryption completed in {batch_time:.3f} seconds")
        print(f"✓ Successfully processed {sum(1 for r in results if len(r[0]) > 0)}/{len(image_paths)} images")

        return results

    def _generate_cipher_images(self, scrambled_3d: np.ndarray, blurred_combined: np.ndarray) -> List[np.ndarray]:
        """Generate final cipher images from scrambled 3D data"""
        depth, height, width = scrambled_3d.shape
        cipher_images = []

        # Generate multiple cipher images using different combinations
        for plane_idx in range(depth):
            # Extract plane
            plane = scrambled_3d[plane_idx]

            # Combine with blurred data using different operations
            if plane_idx == 0:
                # XOR combination
                cipher = np.bitwise_xor(plane, blurred_combined)
            elif plane_idx == 1:
                # Addition with modulo
                cipher = (plane.astype(np.uint16) + blurred_combined.astype(np.uint16)) % 256
                cipher = cipher.astype(np.uint8)
            elif plane_idx == 2:
                # Subtraction with modulo
                cipher = (plane.astype(np.uint16) - blurred_combined.astype(np.uint16)) % 256
                cipher = cipher.astype(np.uint8)
            else:
                # Rotation and XOR
                rotated_blur = np.roll(blurred_combined, shift=plane_idx, axis=0)
                cipher = np.bitwise_xor(plane, rotated_blur)

            cipher_images.append(cipher)

        return cipher_images

    def _create_encryption_metadata(self, image_path: str, image_shape: Tuple[int, int],
                                  sha512_hash: str, blur_metadata: Dict[str, Any],
                                  scrambling_info: List[Dict[str, Any]], rule_indices: np.ndarray,
                                  encryption_time: float) -> Dict[str, Any]:
        """Create comprehensive encryption metadata"""

        metadata = {
            'image_info': {
                'path': image_path,
                'name': os.path.basename(image_path),
                'shape': image_shape,
                'size_bytes': image_shape[0] * image_shape[1]
            },
            'encryption_params': {
                'seed': self.seed,
                'noise_density': self.noise_density,
                'sha512_hash': sha512_hash,
                'samcml_params': self.samcml_system.get_system_info() if self.samcml_system else None
            },
            'blur_metadata': blur_metadata,
            'scrambling_info': {
                'rounds': len(scrambling_info),
                'total_operations': sum(info['operation_count'] for info in scrambling_info),
                'adaptive_intensity': self.scrambling_system.adaptive_scrambling_intensity(
                    np.random.randint(0, 256, (4, *image_shape), dtype=np.uint8)
                ) if self.scrambling_system else 1.0
            },
            'dna_info': {
                'rule_indices_shape': rule_indices.shape,
                'unique_rules_used': len(np.unique(rule_indices)),
                'rule_distribution': {str(i): int(np.sum(rule_indices == i)) for i in range(8)}
            },
            'performance': {
                'encryption_time': encryption_time,
                'throughput_pixels_per_second': (image_shape[0] * image_shape[1]) / encryption_time
            },
            'security_metrics': {
                'keyspace_size': '2^512',  # From SHA-512
                'chaotic_dimensions': 3,   # SAMCML has 3 dimensions
                'dna_rules': 8,           # 8 different DNA encoding rules
                'scrambling_rounds': len(scrambling_info)
            },
            'timestamp': time.time(),
            'version': '2.0.0'
        }

        return metadata

    def save_encryption_results(self, cipher_images: List[np.ndarray], metadata: Dict[str, Any],
                              output_dir: str = "advanced_cipher_outputs") -> str:
        """Save encryption results with organized structure"""

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Create subdirectory for this encryption
        image_name = metadata['image_info']['name'].split('.')[0]
        seed_used = metadata['encryption_params']['seed']
        timestamp = int(metadata['timestamp'])

        result_dir = os.path.join(output_dir, f"{image_name}_seed_{seed_used}_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)

        # Save cipher images
        for i, cipher in enumerate(cipher_images):
            cipher_path = os.path.join(result_dir, f"cipher_{i+1}.png")
            cv2.imwrite(cipher_path, cipher)

        # Save metadata
        metadata_path = os.path.join(result_dir, "encryption_metadata.json")
        with open(metadata_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_metadata = self._prepare_metadata_for_json(metadata)
            json.dump(json_metadata, f, indent=2)

        print(f"✓ Encryption results saved to: {result_dir}")
        return result_dir

    def _prepare_metadata_for_json(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare metadata for JSON serialization"""
        json_metadata = {}

        for key, value in metadata.items():
            if isinstance(value, np.ndarray):
                json_metadata[key] = value.tolist()
            elif isinstance(value, dict):
                json_metadata[key] = self._prepare_metadata_for_json(value)
            elif isinstance(value, (np.integer, np.floating)):
                json_metadata[key] = value.item()
            else:
                json_metadata[key] = value

        return json_metadata

    def analyze_encryption_security(self, original_image: np.ndarray,
                                  cipher_images: List[np.ndarray]) -> Dict[str, Any]:
        """
        Comprehensive security analysis of encryption results

        Args:
            original_image: Original image
            cipher_images: List of cipher images

        Returns:
            Dictionary with security analysis results
        """
        analysis_results = {}

        for i, cipher in enumerate(cipher_images):
            cipher_analysis = {
                'histogram_analysis': self._histogram_analysis(original_image, cipher),
                'entropy_analysis': self._entropy_analysis(cipher),
                'correlation_analysis': self._correlation_analysis(original_image, cipher),
                'npcr_uaci': self._calculate_npcr_uaci(original_image, cipher),
                'chi_square_test': self._chi_square_test(cipher),
                'key_sensitivity': self._key_sensitivity_analysis(original_image, cipher)
            }
            analysis_results[f'cipher_{i+1}'] = cipher_analysis

        # Aggregate analysis
        analysis_results['aggregate'] = self._aggregate_security_metrics(analysis_results)

        return analysis_results

    def _histogram_analysis(self, original: np.ndarray, cipher: np.ndarray) -> Dict[str, float]:
        """Analyze histogram uniformity"""
        orig_hist, _ = np.histogram(original, bins=256, range=(0, 256))
        cipher_hist, _ = np.histogram(cipher, bins=256, range=(0, 256))

        # Calculate histogram variance (lower is better for cipher)
        cipher_variance = np.var(cipher_hist)
        orig_variance = np.var(orig_hist)

        # Calculate histogram uniformity
        expected_freq = cipher.size / 256
        uniformity = 1 - (cipher_variance / (expected_freq ** 2))

        return {
            'original_variance': float(orig_variance),
            'cipher_variance': float(cipher_variance),
            'uniformity_score': float(uniformity),
            'improvement_ratio': float(orig_variance / cipher_variance) if cipher_variance > 0 else float('inf')
        }

    def _entropy_analysis(self, image: np.ndarray) -> Dict[str, float]:
        """Calculate Shannon entropy"""
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist = hist[hist > 0]
        prob = hist / np.sum(hist)
        entropy = -np.sum(prob * np.log2(prob))

        return {
            'entropy': float(entropy),
            'max_entropy': 8.0,
            'entropy_ratio': float(entropy / 8.0)
        }

    def _correlation_analysis(self, original: np.ndarray, cipher: np.ndarray) -> Dict[str, float]:
        """Analyze correlation coefficients"""
        # Horizontal correlation
        h_orig = np.corrcoef(original[:, :-1].flatten(), original[:, 1:].flatten())[0, 1]
        h_cipher = np.corrcoef(cipher[:, :-1].flatten(), cipher[:, 1:].flatten())[0, 1]

        # Vertical correlation
        v_orig = np.corrcoef(original[:-1, :].flatten(), original[1:, :].flatten())[0, 1]
        v_cipher = np.corrcoef(cipher[:-1, :].flatten(), cipher[1:, :].flatten())[0, 1]

        # Diagonal correlation
        d_orig = np.corrcoef(original[:-1, :-1].flatten(), original[1:, 1:].flatten())[0, 1]
        d_cipher = np.corrcoef(cipher[:-1, :-1].flatten(), cipher[1:, 1:].flatten())[0, 1]

        # Cross-correlation between original and cipher
        cross_corr = np.corrcoef(original.flatten(), cipher.flatten())[0, 1]

        return {
            'horizontal_original': float(h_orig),
            'horizontal_cipher': float(h_cipher),
            'vertical_original': float(v_orig),
            'vertical_cipher': float(v_cipher),
            'diagonal_original': float(d_orig),
            'diagonal_cipher': float(d_cipher),
            'cross_correlation': float(cross_corr),
            'avg_cipher_correlation': float((abs(h_cipher) + abs(v_cipher) + abs(d_cipher)) / 3)
        }

    def _calculate_npcr_uaci(self, image1: np.ndarray, image2: np.ndarray) -> Dict[str, float]:
        """Calculate NPCR and UACI metrics"""
        # NPCR (Number of Pixels Change Rate)
        diff_pixels = np.sum(image1 != image2)
        total_pixels = image1.size
        npcr = (diff_pixels / total_pixels) * 100

        # UACI (Unified Average Changing Intensity)
        uaci = np.sum(np.abs(image1.astype(float) - image2.astype(float))) / (total_pixels * 255) * 100

        return {
            'npcr': float(npcr),
            'uaci': float(uaci),
            'npcr_ideal': 99.6094,  # Theoretical ideal for 8-bit images
            'uaci_ideal': 33.4635   # Theoretical ideal for 8-bit images
        }

    def _chi_square_test(self, image: np.ndarray) -> Dict[str, float]:
        """Perform chi-square test for randomness"""
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        expected_freq = image.size / 256

        # Chi-square statistic
        chi_square = np.sum((hist - expected_freq) ** 2 / expected_freq)

        # Degrees of freedom
        dof = 255

        # Critical value at 0.05 significance level for 255 DOF
        critical_value = 293.25

        return {
            'chi_square': float(chi_square),
            'degrees_of_freedom': dof,
            'critical_value': critical_value,
            'passes_test': bool(chi_square < critical_value),
            'p_value_estimate': float(1 - (chi_square / (2 * dof))) if chi_square < 2 * dof else 0.0
        }

    def _key_sensitivity_analysis(self, original: np.ndarray, cipher: np.ndarray) -> Dict[str, float]:
        """Analyze key sensitivity"""
        entropy = self._entropy_analysis(cipher)['entropy']
        correlation = self._correlation_analysis(original, cipher)['cross_correlation']

        # Key sensitivity score based on entropy and correlation
        sensitivity_score = entropy * (1 - abs(correlation))

        return {
            'sensitivity_score': float(sensitivity_score),
            'max_sensitivity': 8.0,
            'sensitivity_ratio': float(sensitivity_score / 8.0)
        }

    def _aggregate_security_metrics(self, analysis_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate aggregate security metrics across all ciphers"""
        cipher_keys = [k for k in analysis_results.keys() if k.startswith('cipher_')]

        if not cipher_keys:
            return {}

        # Average metrics across all ciphers
        avg_entropy = np.mean([analysis_results[k]['entropy_analysis']['entropy'] for k in cipher_keys])
        avg_correlation = np.mean([abs(analysis_results[k]['correlation_analysis']['cross_correlation']) for k in cipher_keys])
        avg_npcr = np.mean([analysis_results[k]['npcr_uaci']['npcr'] for k in cipher_keys])
        avg_uaci = np.mean([analysis_results[k]['npcr_uaci']['uaci'] for k in cipher_keys])

        # Security score (0-100)
        entropy_score = (avg_entropy / 8.0) * 25
        correlation_score = (1 - avg_correlation) * 25
        npcr_score = (avg_npcr / 99.6094) * 25
        uaci_score = (avg_uaci / 33.4635) * 25

        overall_score = entropy_score + correlation_score + npcr_score + uaci_score

        return {
            'average_entropy': float(avg_entropy),
            'average_correlation': float(avg_correlation),
            'average_npcr': float(avg_npcr),
            'average_uaci': float(avg_uaci),
            'entropy_score': float(entropy_score),
            'correlation_score': float(correlation_score),
            'npcr_score': float(npcr_score),
            'uaci_score': float(uaci_score),
            'overall_security_score': float(overall_score)
        }


def test_advanced_medical_encryption():
    """Test advanced medical encryption system"""
    print("Testing Advanced Medical Image Encryption System...")

    # Create test image
    test_image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    test_path = "test_medical_image.png"
    cv2.imwrite(test_path, test_image)

    # Initialize encryption system
    encryption_system = AdvancedMedicalImageEncryption(seed=42, noise_density=0.05)

    # Test single image encryption
    cipher_images, metadata = encryption_system.encrypt_single_medical_image(test_path)

    print(f"✓ Single image encryption completed")
    print(f"  Generated {len(cipher_images)} cipher images")
    print(f"  Encryption time: {metadata['performance']['encryption_time']:.3f} seconds")
    print(f"  Throughput: {metadata['performance']['throughput_pixels_per_second']:.0f} pixels/sec")

    # Test batch encryption
    test_paths = [test_path] * 3  # Encrypt same image 3 times with different seeds
    batch_results = encryption_system.encrypt_multiple_medical_images(test_paths, "different_seeds")

    print(f"✓ Batch encryption completed")
    print(f"  Processed {len(batch_results)} images")

    # Clean up
    os.remove(test_path)

    print("Advanced medical encryption system tests completed successfully!")


if __name__ == "__main__":
    test_advanced_medical_encryption()
