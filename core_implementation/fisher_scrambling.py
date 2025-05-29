"""
3D Fisher-Yates Scrambling Module for Multi Medical Image Encryption
Implements cross-plane scrambling using Fisher-Yates algorithm
"""

import numpy as np
from typing import Tuple, List
import random


class FisherScrambling:
    """
    Implements 3D Fisher-Yates scrambling for cross-plane operations
    """
    
    def __init__(self, seed: int = None):
        """
        Initialize the Fisher scrambling system
        
        Args:
            seed: Optional seed for reproducible scrambling
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def create_3d_image_stack(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Create a 3D stack from multiple 2D images
        
        Args:
            images: List of 2D image arrays
            
        Returns:
            3D array with shape (depth, height, width)
        """
        if not images:
            raise ValueError("Images list cannot be empty")
        
        # Ensure all images have the same shape
        base_shape = images[0].shape
        for i, img in enumerate(images):
            if img.shape != base_shape:
                raise ValueError(f"Image {i} has different shape {img.shape} vs {base_shape}")
        
        return np.stack(images, axis=0)
    
    def fisher_yates_shuffle_1d(self, array: np.ndarray, key: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Fisher-Yates shuffle on 1D array
        
        Args:
            array: 1D array to shuffle
            key: Optional key for deterministic shuffling
            
        Returns:
            Tuple of (shuffled_array, permutation_indices)
        """
        if key:
            # Use key to create deterministic shuffle
            key_int = int(key[:16], 16) if len(key) >= 16 else hash(key)
            random.seed(key_int % (2**32))
        
        indices = np.arange(len(array))
        shuffled_array = array.copy()
        
        # Fisher-Yates shuffle algorithm
        for i in range(len(indices) - 1, 0, -1):
            j = random.randint(0, i)
            # Swap elements
            indices[i], indices[j] = indices[j], indices[i]
            shuffled_array[i], shuffled_array[j] = shuffled_array[j], shuffled_array[i]
        
        return shuffled_array, indices
    
    def inverse_fisher_yates_shuffle(self, shuffled_array: np.ndarray, 
                                   permutation_indices: np.ndarray) -> np.ndarray:
        """
        Reverse Fisher-Yates shuffle using permutation indices
        
        Args:
            shuffled_array: Shuffled array
            permutation_indices: Indices used for shuffling
            
        Returns:
            Original unshuffled array
        """
        original_array = np.zeros_like(shuffled_array)
        for i, original_pos in enumerate(permutation_indices):
            original_array[original_pos] = shuffled_array[i]
        
        return original_array
    
    def cross_plane_scrambling_3d(self, image_3d: np.ndarray, key: str = None) -> Tuple[np.ndarray, dict]:
        """
        Perform 3D cross-plane scrambling using Fisher-Yates algorithm
        
        Args:
            image_3d: 3D image array with shape (depth, height, width)
            key: Optional key for deterministic scrambling
            
        Returns:
            Tuple of (scrambled_3d_image, scrambling_info)
        """
        depth, height, width = image_3d.shape
        scrambled_3d = np.copy(image_3d)
        scrambling_info = {}
        
        # 1. Scramble along depth dimension (cross-plane)
        for h in range(height):
            for w in range(width):
                pixel_column = image_3d[:, h, w]  # Extract column across all planes
                shuffled_column, indices = self.fisher_yates_shuffle_1d(pixel_column, key)
                scrambled_3d[:, h, w] = shuffled_column
                scrambling_info[f'depth_{h}_{w}'] = indices
        
        # 2. Scramble within each plane (intra-plane)
        for d in range(depth):
            plane = scrambled_3d[d, :, :]
            
            # Scramble rows
            for h in range(height):
                row = plane[h, :]
                shuffled_row, indices = self.fisher_yates_shuffle_1d(row, key)
                scrambled_3d[d, h, :] = shuffled_row
                scrambling_info[f'row_{d}_{h}'] = indices
            
            # Scramble columns
            for w in range(width):
                col = scrambled_3d[d, :, w]
                shuffled_col, indices = self.fisher_yates_shuffle_1d(col, key)
                scrambled_3d[d, :, w] = shuffled_col
                scrambling_info[f'col_{d}_{w}'] = indices
        
        # 3. Diagonal scrambling for additional security
        for d in range(depth):
            plane = scrambled_3d[d, :, :]
            
            # Main diagonal
            if height == width:
                diagonal = np.diag(plane)
                shuffled_diag, indices = self.fisher_yates_shuffle_1d(diagonal, key)
                np.fill_diagonal(scrambled_3d[d, :, :], shuffled_diag)
                scrambling_info[f'main_diag_{d}'] = indices
                
                # Anti-diagonal
                anti_diagonal = np.diag(np.fliplr(plane))
                shuffled_anti_diag, indices = self.fisher_yates_shuffle_1d(anti_diagonal, key)
                np.fill_diagonal(np.fliplr(scrambled_3d[d, :, :]), shuffled_anti_diag)
                scrambling_info[f'anti_diag_{d}'] = indices
        
        return scrambled_3d, scrambling_info
    
    def reverse_cross_plane_scrambling(self, scrambled_3d: np.ndarray, 
                                     scrambling_info: dict) -> np.ndarray:
        """
        Reverse the 3D cross-plane scrambling
        
        Args:
            scrambled_3d: Scrambled 3D image
            scrambling_info: Information needed to reverse scrambling
            
        Returns:
            Original 3D image
        """
        depth, height, width = scrambled_3d.shape
        unscrambled_3d = np.copy(scrambled_3d)
        
        # Reverse in opposite order of scrambling
        
        # 3. Reverse diagonal scrambling
        for d in range(depth):
            if height == width:
                # Reverse anti-diagonal
                if f'anti_diag_{d}' in scrambling_info:
                    anti_diagonal = np.diag(np.fliplr(unscrambled_3d[d, :, :]))
                    original_anti_diag = self.inverse_fisher_yates_shuffle(
                        anti_diagonal, scrambling_info[f'anti_diag_{d}']
                    )
                    np.fill_diagonal(np.fliplr(unscrambled_3d[d, :, :]), original_anti_diag)
                
                # Reverse main diagonal
                if f'main_diag_{d}' in scrambling_info:
                    diagonal = np.diag(unscrambled_3d[d, :, :])
                    original_diag = self.inverse_fisher_yates_shuffle(
                        diagonal, scrambling_info[f'main_diag_{d}']
                    )
                    np.fill_diagonal(unscrambled_3d[d, :, :], original_diag)
        
        # 2. Reverse intra-plane scrambling
        for d in range(depth):
            # Reverse column scrambling
            for w in range(width):
                if f'col_{d}_{w}' in scrambling_info:
                    col = unscrambled_3d[d, :, w]
                    original_col = self.inverse_fisher_yates_shuffle(
                        col, scrambling_info[f'col_{d}_{w}']
                    )
                    unscrambled_3d[d, :, w] = original_col
            
            # Reverse row scrambling
            for h in range(height):
                if f'row_{d}_{h}' in scrambling_info:
                    row = unscrambled_3d[d, h, :]
                    original_row = self.inverse_fisher_yates_shuffle(
                        row, scrambling_info[f'row_{d}_{h}']
                    )
                    unscrambled_3d[d, h, :] = original_row
        
        # 1. Reverse cross-plane scrambling
        for h in range(height):
            for w in range(width):
                if f'depth_{h}_{w}' in scrambling_info:
                    pixel_column = unscrambled_3d[:, h, w]
                    original_column = self.inverse_fisher_yates_shuffle(
                        pixel_column, scrambling_info[f'depth_{h}_{w}']
                    )
                    unscrambled_3d[:, h, w] = original_column
        
        return unscrambled_3d
    
    def adaptive_scrambling(self, image_3d: np.ndarray, intensity_factor: float = 1.0) -> Tuple[np.ndarray, dict]:
        """
        Perform adaptive scrambling based on image content
        
        Args:
            image_3d: 3D image array
            intensity_factor: Factor to control scrambling intensity
            
        Returns:
            Tuple of (adaptively_scrambled_image, scrambling_info)
        """
        depth, height, width = image_3d.shape
        
        # Calculate local variance for each region
        block_size = min(8, height // 4, width // 4)
        scrambling_intensity = np.zeros((depth, height // block_size, width // block_size))
        
        for d in range(depth):
            for i in range(0, height - block_size + 1, block_size):
                for j in range(0, width - block_size + 1, block_size):
                    block = image_3d[d, i:i+block_size, j:j+block_size]
                    variance = np.var(block)
                    scrambling_intensity[d, i//block_size, j//block_size] = variance
        
        # Normalize scrambling intensity
        max_intensity = np.max(scrambling_intensity)
        if max_intensity > 0:
            scrambling_intensity = (scrambling_intensity / max_intensity) * intensity_factor
        
        # Apply variable scrambling based on intensity
        scrambled_3d = np.copy(image_3d)
        scrambling_info = {'adaptive': True, 'intensity_map': scrambling_intensity}
        
        # Higher variance regions get more scrambling
        for d in range(depth):
            for i in range(scrambling_intensity.shape[1]):
                for j in range(scrambling_intensity.shape[2]):
                    intensity = scrambling_intensity[d, i, j]
                    if intensity > 0.5:  # High variance region
                        # Apply multiple rounds of scrambling
                        start_h, end_h = i * block_size, min((i + 1) * block_size, height)
                        start_w, end_w = j * block_size, min((j + 1) * block_size, width)
                        
                        region = scrambled_3d[d, start_h:end_h, start_w:end_w]
                        flattened = region.flatten()
                        
                        # Apply Fisher-Yates shuffle multiple times
                        rounds = int(intensity * 3) + 1
                        for round_num in range(rounds):
                            shuffled, indices = self.fisher_yates_shuffle_1d(flattened)
                            flattened = shuffled
                            scrambling_info[f'adaptive_{d}_{i}_{j}_{round_num}'] = indices
                        
                        scrambled_3d[d, start_h:end_h, start_w:end_w] = flattened.reshape(region.shape)
        
        return scrambled_3d, scrambling_info


def test_fisher_scrambling():
    """Test the Fisher scrambling functionality"""
    print("Testing Fisher Scrambling Module...")
    
    fisher = FisherScrambling(seed=42)
    
    # Create test 3D image
    test_images = [
        np.random.randint(0, 256, (4, 4), dtype=np.uint8) for _ in range(3)
    ]
    image_3d = fisher.create_3d_image_stack(test_images)
    print(f"Created 3D image with shape: {image_3d.shape}")
    
    # Test 1D Fisher-Yates shuffle
    test_array = np.array([1, 2, 3, 4, 5])
    shuffled, indices = fisher.fisher_yates_shuffle_1d(test_array, "test_key")
    unshuffled = fisher.inverse_fisher_yates_shuffle(shuffled, indices)
    print(f"1D shuffle test: {np.array_equal(test_array, unshuffled)}")
    
    # Test 3D cross-plane scrambling
    scrambled_3d, scrambling_info = fisher.cross_plane_scrambling_3d(image_3d, "test_key")
    unscrambled_3d = fisher.reverse_cross_plane_scrambling(scrambled_3d, scrambling_info)
    print(f"3D scrambling test: {np.array_equal(image_3d, unscrambled_3d)}")
    
    # Test adaptive scrambling
    adaptive_scrambled, adaptive_info = fisher.adaptive_scrambling(image_3d)
    print(f"Adaptive scrambling completed with intensity map shape: {adaptive_info['intensity_map'].shape}")
    
    print("Fisher scrambling tests completed successfully!")


if __name__ == "__main__":
    test_fisher_scrambling()
