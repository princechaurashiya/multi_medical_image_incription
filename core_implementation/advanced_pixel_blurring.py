"""
Advanced Pixel Blurring with Bit Plane Extraction
Implements salt-and-pepper noise blurring on specific bit planes
"""

import numpy as np
import cv2
from typing import Tuple, List
import hashlib


class AdvancedPixelBlurring:
    """
    Advanced pixel blurring using bit plane extraction and salt-and-pepper noise
    Focuses on 1st, 7th, and 8th bit planes for key generation
    """
    
    def __init__(self, noise_density: float = 0.05, seed: int = None):
        """
        Initialize pixel blurring system
        
        Args:
            noise_density: Density of salt-and-pepper noise [0, 1]
            seed: Random seed for reproducible results
        """
        self.noise_density = noise_density
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def extract_bit_plane(self, image: np.ndarray, bit_position: int) -> np.ndarray:
        """
        Extract specific bit plane from image
        
        Args:
            image: Input image (grayscale)
            bit_position: Bit position to extract (0-7, where 0 is LSB)
            
        Returns:
            Binary image representing the bit plane
        """
        if not (0 <= bit_position <= 7):
            raise ValueError("Bit position must be between 0 and 7")
        
        # Extract bit plane using bitwise operations
        bit_plane = (image >> bit_position) & 1
        
        # Convert to 8-bit image (0 or 255)
        bit_plane = bit_plane * 255
        
        return bit_plane.astype(np.uint8)
    
    def apply_salt_pepper_noise(self, image: np.ndarray, density: float = None) -> np.ndarray:
        """
        Apply salt-and-pepper noise to image
        
        Args:
            image: Input image
            density: Noise density (if None, uses self.noise_density)
            
        Returns:
            Noisy image
        """
        if density is None:
            density = self.noise_density
        
        noisy_image = image.copy()
        
        # Generate random noise mask
        noise_mask = np.random.random(image.shape)
        
        # Apply salt noise (white pixels)
        salt_mask = noise_mask < density / 2
        noisy_image[salt_mask] = 255
        
        # Apply pepper noise (black pixels)
        pepper_mask = noise_mask > (1 - density / 2)
        noisy_image[pepper_mask] = 0
        
        return noisy_image
    
    def blur_bit_planes(self, image: np.ndarray, target_bits: List[int] = [0, 6, 7]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Extract and blur specific bit planes
        
        Args:
            image: Input grayscale image
            target_bits: List of bit positions to extract and blur (default: 1st, 7th, 8th bits)
            
        Returns:
            Tuple of (original_bit_planes, blurred_bit_planes)
        """
        original_bit_planes = []
        blurred_bit_planes = []
        
        for bit_pos in target_bits:
            # Extract bit plane
            bit_plane = self.extract_bit_plane(image, bit_pos)
            original_bit_planes.append(bit_plane)
            
            # Apply salt-and-pepper noise
            blurred_bit_plane = self.apply_salt_pepper_noise(bit_plane)
            blurred_bit_planes.append(blurred_bit_plane)
        
        return original_bit_planes, blurred_bit_planes
    
    def combine_blurred_bits(self, blurred_bit_planes: List[np.ndarray], 
                           target_bits: List[int] = [0, 6, 7]) -> np.ndarray:
        """
        Combine blurred bit planes to create new pixel values
        
        Args:
            blurred_bit_planes: List of blurred bit plane images
            target_bits: Corresponding bit positions
            
        Returns:
            Combined image with new pixel values
        """
        if len(blurred_bit_planes) != len(target_bits):
            raise ValueError("Number of bit planes must match number of target bits")
        
        # Initialize result image
        height, width = blurred_bit_planes[0].shape
        combined_image = np.zeros((height, width), dtype=np.uint8)
        
        # Combine bit planes
        for bit_plane, bit_pos in zip(blurred_bit_planes, target_bits):
            # Convert back to binary (0 or 1)
            binary_plane = (bit_plane > 127).astype(np.uint8)
            
            # Place in correct bit position
            combined_image |= (binary_plane << bit_pos)
        
        return combined_image
    
    def generate_key_material(self, original_image: np.ndarray, 
                            blurred_image: np.ndarray) -> np.ndarray:
        """
        Generate key material by combining original and blurred images
        
        Args:
            original_image: Original image
            blurred_image: Blurred image from bit plane operations
            
        Returns:
            Combined key material
        """
        # XOR combination for enhanced entropy
        key_material = np.bitwise_xor(original_image, blurred_image)
        
        # Additional mixing using rotation and shifting
        rotated = np.roll(key_material, shift=1, axis=0)
        shifted = np.roll(key_material, shift=1, axis=1)
        
        # Final combination
        final_key_material = np.bitwise_xor(key_material, rotated)
        final_key_material = np.bitwise_xor(final_key_material, shifted)
        
        return final_key_material
    
    def create_sha512_hash(self, key_material: np.ndarray) -> str:
        """
        Create SHA-512 hash from key material
        
        Args:
            key_material: Key material array
            
        Returns:
            SHA-512 hash string
        """
        # Convert to bytes
        key_bytes = key_material.tobytes()
        
        # Create SHA-512 hash
        hash_object = hashlib.sha512(key_bytes)
        hash_hex = hash_object.hexdigest()
        
        return hash_hex
    
    def process_medical_image(self, image: np.ndarray) -> Tuple[np.ndarray, str, dict]:
        """
        Complete processing pipeline for medical image
        
        Args:
            image: Input medical image (grayscale)
            
        Returns:
            Tuple of (blurred_combined_image, sha512_hash, metadata)
        """
        # Step 1: Extract and blur bit planes
        original_bits, blurred_bits = self.blur_bit_planes(image)
        
        # Step 2: Combine blurred bit planes
        blurred_combined = self.combine_blurred_bits(blurred_bits)
        
        # Step 3: Generate key material
        key_material = self.generate_key_material(image, blurred_combined)
        
        # Step 4: Create SHA-512 hash
        sha512_hash = self.create_sha512_hash(key_material)
        
        # Step 5: Create metadata
        metadata = {
            'original_shape': image.shape,
            'noise_density': self.noise_density,
            'target_bits': [0, 6, 7],  # 1st, 7th, 8th bits
            'key_material_entropy': self._calculate_entropy(key_material),
            'blurred_entropy': self._calculate_entropy(blurred_combined),
            'seed': self.seed
        }
        
        return blurred_combined, sha512_hash, metadata
    
    def _calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate Shannon entropy of image"""
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist = hist[hist > 0]  # Remove zero entries
        prob = hist / np.sum(hist)
        entropy = -np.sum(prob * np.log2(prob))
        return entropy
    
    def analyze_bit_plane_statistics(self, image: np.ndarray) -> dict:
        """
        Analyze statistics of all bit planes
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with bit plane statistics
        """
        stats = {}
        
        for bit_pos in range(8):
            bit_plane = self.extract_bit_plane(image, bit_pos)
            
            stats[f'bit_{bit_pos}'] = {
                'entropy': self._calculate_entropy(bit_plane),
                'mean': np.mean(bit_plane),
                'std': np.std(bit_plane),
                'ones_ratio': np.sum(bit_plane > 127) / bit_plane.size
            }
        
        return stats
    
    def visualize_bit_planes(self, image: np.ndarray, save_path: str = None):
        """
        Visualize all bit planes of an image
        
        Args:
            image: Input image
            save_path: Optional path to save visualization
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Bit Plane Decomposition', fontsize=16)
        
        for bit_pos in range(8):
            row = bit_pos // 4
            col = bit_pos % 4
            
            bit_plane = self.extract_bit_plane(image, bit_pos)
            
            axes[row, col].imshow(bit_plane, cmap='gray')
            axes[row, col].set_title(f'Bit Plane {bit_pos}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


def test_advanced_pixel_blurring():
    """Test advanced pixel blurring system"""
    print("Testing Advanced Pixel Blurring System...")
    
    # Create test image
    test_image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    
    # Initialize blurring system
    blurring = AdvancedPixelBlurring(noise_density=0.05, seed=42)
    
    # Test bit plane extraction
    bit_plane = blurring.extract_bit_plane(test_image, 0)
    print(f"✓ Bit plane extraction: shape {bit_plane.shape}")
    
    # Test salt-and-pepper noise
    noisy_image = blurring.apply_salt_pepper_noise(test_image)
    print(f"✓ Salt-and-pepper noise applied")
    
    # Test complete processing
    blurred_combined, sha512_hash, metadata = blurring.process_medical_image(test_image)
    print(f"✓ Complete processing: hash length {len(sha512_hash)}")
    print(f"  Metadata keys: {list(metadata.keys())}")
    
    # Test bit plane statistics
    stats = blurring.analyze_bit_plane_statistics(test_image)
    print(f"✓ Bit plane statistics calculated for {len(stats)} planes")
    
    print("Advanced pixel blurring tests completed successfully!")


if __name__ == "__main__":
    test_advanced_pixel_blurring()
