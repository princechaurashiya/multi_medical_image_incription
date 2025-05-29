"""
Bit Plane Operations Module for Multi Medical Image Encryption
Implements bit plane manipulation and pixel generation
"""

import numpy as np
from typing import List, Tuple, Dict
import random


class BitPlaneOperations:
    """
    Implements bit plane manipulation and pixel generation operations
    """
    
    def __init__(self, seed: int = None):
        """
        Initialize bit plane operations
        
        Args:
            seed: Optional seed for reproducible operations
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def extract_bit_planes(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Extract all 8 bit planes from an image
        
        Args:
            image: Input grayscale image
            
        Returns:
            List of 8 bit planes (from LSB to MSB)
        """
        bit_planes = []
        
        for bit_position in range(8):
            # Extract bit plane using bitwise AND with appropriate mask
            bit_plane = (image >> bit_position) & 1
            bit_planes.append(bit_plane.astype(np.uint8))
        
        return bit_planes
    
    def reconstruct_from_bit_planes(self, bit_planes: List[np.ndarray]) -> np.ndarray:
        """
        Reconstruct image from bit planes
        
        Args:
            bit_planes: List of 8 bit planes
            
        Returns:
            Reconstructed image
        """
        if len(bit_planes) != 8:
            raise ValueError("Exactly 8 bit planes are required")
        
        reconstructed = np.zeros_like(bit_planes[0], dtype=np.uint8)
        
        for bit_position, bit_plane in enumerate(bit_planes):
            reconstructed += (bit_plane << bit_position)
        
        return reconstructed
    
    def generate_new_pixels_from_bit_planes(self, bit_planes_before: List[np.ndarray], 
                                          bit_planes_after: List[np.ndarray]) -> np.ndarray:
        """
        Generate new pixels by combining bit planes before and after blur
        
        Args:
            bit_planes_before: Bit planes from original image
            bit_planes_after: Bit planes from blurred image
            
        Returns:
            New image with generated pixels
        """
        if len(bit_planes_before) != 8 or len(bit_planes_after) != 8:
            raise ValueError("Both bit plane lists must contain exactly 8 planes")
        
        height, width = bit_planes_before[0].shape
        new_image = np.zeros((height, width), dtype=np.uint8)
        
        # Combine bit planes using different strategies for each bit position
        for bit_pos in range(8):
            before_plane = bit_planes_before[bit_pos]
            after_plane = bit_planes_after[bit_pos]
            
            if bit_pos < 3:  # Lower bits: XOR combination
                combined_plane = np.bitwise_xor(before_plane, after_plane)
            elif bit_pos < 6:  # Middle bits: AND combination
                combined_plane = np.bitwise_and(before_plane, after_plane)
            else:  # Higher bits: OR combination
                combined_plane = np.bitwise_or(before_plane, after_plane)
            
            # Add some randomness based on position
            if bit_pos % 2 == 0:
                # Even bit positions: add spatial dependency
                shifted_plane = np.roll(combined_plane, 1, axis=0)
                combined_plane = np.bitwise_xor(combined_plane, shifted_plane)
            
            new_image += (combined_plane << bit_pos)
        
        return new_image
    
    def create_multiple_cipher_images(self, original_image: np.ndarray, 
                                    blurred_image: np.ndarray, 
                                    num_images: int = 4) -> List[np.ndarray]:
        """
        Create multiple cipher images of different sizes using bit plane information
        
        Args:
            original_image: Original image
            blurred_image: Blurred image
            num_images: Number of cipher images to generate
            
        Returns:
            List of cipher images with different sizes
        """
        bit_planes_orig = self.extract_bit_planes(original_image)
        bit_planes_blur = self.extract_bit_planes(blurred_image)
        
        cipher_images = []
        height, width = original_image.shape
        
        for i in range(num_images):
            # Create different sized images
            scale_factor = 0.5 + (i * 0.25)  # 0.5, 0.75, 1.0, 1.25
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)
            
            # Generate base cipher image
            base_cipher = self.generate_new_pixels_from_bit_planes(bit_planes_orig, bit_planes_blur)
            
            # Resize and modify based on bit plane information
            if scale_factor < 1.0:
                # Downscale: use bit plane compression
                cipher_image = self._compress_using_bit_planes(base_cipher, (new_height, new_width), i)
            elif scale_factor > 1.0:
                # Upscale: use bit plane expansion
                cipher_image = self._expand_using_bit_planes(base_cipher, (new_height, new_width), i)
            else:
                # Same size: apply bit plane transformation
                cipher_image = self._transform_using_bit_planes(base_cipher, i)
            
            cipher_images.append(cipher_image)
        
        return cipher_images
    
    def _compress_using_bit_planes(self, image: np.ndarray, target_shape: Tuple[int, int], 
                                 variant: int) -> np.ndarray:
        """
        Compress image using bit plane information
        
        Args:
            image: Input image
            target_shape: Target (height, width)
            variant: Variant number for different compression strategies
            
        Returns:
            Compressed cipher image
        """
        target_height, target_width = target_shape
        bit_planes = self.extract_bit_planes(image)
        
        # Different compression strategies based on variant
        if variant % 2 == 0:
            # Strategy 1: Average pooling on each bit plane
            compressed_planes = []
            for plane in bit_planes:
                compressed_plane = self._average_pool_2d(plane, target_shape)
                compressed_planes.append(compressed_plane)
        else:
            # Strategy 2: Max pooling on each bit plane
            compressed_planes = []
            for plane in bit_planes:
                compressed_plane = self._max_pool_2d(plane, target_shape)
                compressed_planes.append(compressed_plane)
        
        return self.reconstruct_from_bit_planes(compressed_planes)
    
    def _expand_using_bit_planes(self, image: np.ndarray, target_shape: Tuple[int, int], 
                               variant: int) -> np.ndarray:
        """
        Expand image using bit plane information
        
        Args:
            image: Input image
            target_shape: Target (height, width)
            variant: Variant number for different expansion strategies
            
        Returns:
            Expanded cipher image
        """
        target_height, target_width = target_shape
        bit_planes = self.extract_bit_planes(image)
        
        # Different expansion strategies based on variant
        expanded_planes = []
        for i, plane in enumerate(bit_planes):
            if variant % 3 == 0:
                # Strategy 1: Nearest neighbor interpolation
                expanded_plane = self._nearest_neighbor_expand(plane, target_shape)
            elif variant % 3 == 1:
                # Strategy 2: Bilinear-like expansion with bit operations
                expanded_plane = self._bilinear_bit_expand(plane, target_shape)
            else:
                # Strategy 3: Pattern-based expansion
                expanded_plane = self._pattern_expand(plane, target_shape, i)
            
            expanded_planes.append(expanded_plane)
        
        return self.reconstruct_from_bit_planes(expanded_planes)
    
    def _transform_using_bit_planes(self, image: np.ndarray, variant: int) -> np.ndarray:
        """
        Transform image using bit plane operations (same size)
        
        Args:
            image: Input image
            variant: Variant number for different transformation strategies
            
        Returns:
            Transformed cipher image
        """
        bit_planes = self.extract_bit_planes(image)
        
        # Apply different transformations based on variant
        transformed_planes = []
        for i, plane in enumerate(bit_planes):
            if variant % 4 == 0:
                # Rotation-based transformation
                transformed_plane = np.rot90(plane, k=i % 4)
            elif variant % 4 == 1:
                # Flip-based transformation
                if i % 2 == 0:
                    transformed_plane = np.flipud(plane)
                else:
                    transformed_plane = np.fliplr(plane)
            elif variant % 4 == 2:
                # Shift-based transformation
                transformed_plane = np.roll(plane, shift=i, axis=i % 2)
            else:
                # XOR with pattern
                pattern = self._generate_bit_pattern(plane.shape, i)
                transformed_plane = np.bitwise_xor(plane, pattern)
            
            transformed_planes.append(transformed_plane)
        
        return self.reconstruct_from_bit_planes(transformed_planes)
    
    def _average_pool_2d(self, array: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Average pooling for 2D array"""
        height, width = array.shape
        target_height, target_width = target_shape
        
        pool_h = height // target_height
        pool_w = width // target_width
        
        pooled = np.zeros(target_shape, dtype=np.uint8)
        
        for i in range(target_height):
            for j in range(target_width):
                start_h, end_h = i * pool_h, (i + 1) * pool_h
                start_w, end_w = j * pool_w, (j + 1) * pool_w
                pooled[i, j] = np.mean(array[start_h:end_h, start_w:end_w]) > 0.5
        
        return pooled
    
    def _max_pool_2d(self, array: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Max pooling for 2D array"""
        height, width = array.shape
        target_height, target_width = target_shape
        
        pool_h = height // target_height
        pool_w = width // target_width
        
        pooled = np.zeros(target_shape, dtype=np.uint8)
        
        for i in range(target_height):
            for j in range(target_width):
                start_h, end_h = i * pool_h, (i + 1) * pool_h
                start_w, end_w = j * pool_w, (j + 1) * pool_w
                pooled[i, j] = np.max(array[start_h:end_h, start_w:end_w])
        
        return pooled
    
    def _nearest_neighbor_expand(self, array: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Nearest neighbor expansion"""
        height, width = array.shape
        target_height, target_width = target_shape
        
        expanded = np.zeros(target_shape, dtype=np.uint8)
        
        for i in range(target_height):
            for j in range(target_width):
                orig_i = int(i * height / target_height)
                orig_j = int(j * width / target_width)
                expanded[i, j] = array[orig_i, orig_j]
        
        return expanded
    
    def _bilinear_bit_expand(self, array: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Bilinear-like expansion for binary data"""
        height, width = array.shape
        target_height, target_width = target_shape
        
        expanded = np.zeros(target_shape, dtype=np.uint8)
        
        for i in range(target_height):
            for j in range(target_width):
                # Map to original coordinates
                orig_i = i * height / target_height
                orig_j = j * width / target_width
                
                # Get surrounding pixels
                i1, i2 = int(orig_i), min(int(orig_i) + 1, height - 1)
                j1, j2 = int(orig_j), min(int(orig_j) + 1, width - 1)
                
                # For binary data, use majority voting
                neighbors = [array[i1, j1], array[i1, j2], array[i2, j1], array[i2, j2]]
                expanded[i, j] = 1 if sum(neighbors) >= 2 else 0
        
        return expanded
    
    def _pattern_expand(self, array: np.ndarray, target_shape: Tuple[int, int], bit_pos: int) -> np.ndarray:
        """Pattern-based expansion"""
        height, width = array.shape
        target_height, target_width = target_shape
        
        # Create base expansion
        expanded = self._nearest_neighbor_expand(array, target_shape)
        
        # Add pattern based on bit position
        pattern = self._generate_bit_pattern(target_shape, bit_pos)
        expanded = np.bitwise_xor(expanded, pattern)
        
        return expanded
    
    def _generate_bit_pattern(self, shape: Tuple[int, int], seed: int) -> np.ndarray:
        """Generate a bit pattern for transformation"""
        height, width = shape
        np.random.seed(seed)
        
        # Create different patterns based on seed
        if seed % 3 == 0:
            # Checkerboard pattern
            pattern = np.zeros(shape, dtype=np.uint8)
            pattern[::2, ::2] = 1
            pattern[1::2, 1::2] = 1
        elif seed % 3 == 1:
            # Random pattern
            pattern = np.random.randint(0, 2, shape, dtype=np.uint8)
        else:
            # Diagonal pattern
            pattern = np.zeros(shape, dtype=np.uint8)
            for i in range(height):
                for j in range(width):
                    if (i + j) % 2 == 0:
                        pattern[i, j] = 1
        
        return pattern


def test_bit_plane_operations():
    """Test the bit plane operations functionality"""
    print("Testing Bit Plane Operations Module...")
    
    bit_ops = BitPlaneOperations(seed=42)
    
    # Create test images
    original = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
    blurred = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
    
    # Test bit plane extraction and reconstruction
    bit_planes = bit_ops.extract_bit_planes(original)
    reconstructed = bit_ops.reconstruct_from_bit_planes(bit_planes)
    print(f"Bit plane extraction/reconstruction test: {np.array_equal(original, reconstructed)}")
    
    # Test new pixel generation
    bit_planes_orig = bit_ops.extract_bit_planes(original)
    bit_planes_blur = bit_ops.extract_bit_planes(blurred)
    new_pixels = bit_ops.generate_new_pixels_from_bit_planes(bit_planes_orig, bit_planes_blur)
    print(f"New pixel generation completed, shape: {new_pixels.shape}")
    
    # Test multiple cipher image creation
    cipher_images = bit_ops.create_multiple_cipher_images(original, blurred, num_images=4)
    print(f"Created {len(cipher_images)} cipher images with shapes:")
    for i, img in enumerate(cipher_images):
        print(f"  Cipher {i}: {img.shape}")
    
    print("Bit plane operations tests completed successfully!")


if __name__ == "__main__":
    test_bit_plane_operations()
