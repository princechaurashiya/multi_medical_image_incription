"""
Advanced 3D Fisher-Yates Scrambling with SAMCML Integration
Cross-plane scrambling for medical image encryption
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from .samcml_chaotic_system import SAMCMLChaoticSystem


class Advanced3DFisherScrambling:
    """
    Advanced 3D Fisher-Yates scrambling using SAMCML chaotic system
    Implements cross-plane scrambling with enhanced security
    """
    
    def __init__(self, samcml_system: SAMCMLChaoticSystem = None):
        """
        Initialize 3D Fisher-Yates scrambling system
        
        Args:
            samcml_system: SAMCML chaotic system for generating scrambling sequences
        """
        self.samcml_system = samcml_system
        if samcml_system is None:
            self.samcml_system = SAMCMLChaoticSystem()
    
    def create_3d_image_stack(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Create 3D stack from list of 2D images
        
        Args:
            images: List of 2D images
            
        Returns:
            3D array (depth, height, width)
        """
        if not images:
            raise ValueError("Image list cannot be empty")
        
        # Ensure all images have the same shape
        base_shape = images[0].shape
        for i, img in enumerate(images):
            if img.shape != base_shape:
                raise ValueError(f"Image {i} has different shape: {img.shape} vs {base_shape}")
        
        # Stack images along depth dimension
        image_3d = np.stack(images, axis=0)
        
        return image_3d
    
    def create_3d_from_dna_matrix(self, dna_matrix: np.ndarray, num_planes: int = 4) -> np.ndarray:
        """
        Create 3D structure from DNA matrix for scrambling
        
        Args:
            dna_matrix: 2D DNA sequence matrix
            num_planes: Number of planes to create
            
        Returns:
            3D array for scrambling
        """
        height, width = dna_matrix.shape
        
        # Create multiple planes by rotating and transforming the DNA matrix
        planes = []
        
        for plane_idx in range(num_planes):
            # Generate chaotic matrix for this plane
            chaotic_matrix = self.samcml_system.generate_chaotic_matrix(height, width, 'combined')
            
            # Convert DNA sequences to numerical values for scrambling
            numerical_plane = np.zeros((height, width), dtype=np.uint8)
            
            for i in range(height):
                for j in range(width):
                    dna_seq = dna_matrix[i, j]
                    # Convert DNA sequence to numerical value
                    numerical_value = self._dna_to_numerical(dna_seq, plane_idx)
                    numerical_plane[i, j] = numerical_value
            
            # Apply chaotic transformation
            transformed_plane = self._apply_chaotic_transformation(numerical_plane, chaotic_matrix)
            planes.append(transformed_plane)
        
        return np.stack(planes, axis=0)
    
    def _dna_to_numerical(self, dna_seq: str, plane_idx: int) -> int:
        """Convert DNA sequence to numerical value based on plane index"""
        base_values = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
        
        # Different conversion for each plane
        if plane_idx == 0:
            # Standard conversion
            value = sum(base_values[base] * (4 ** i) for i, base in enumerate(dna_seq))
        elif plane_idx == 1:
            # Reverse order
            value = sum(base_values[base] * (4 ** i) for i, base in enumerate(reversed(dna_seq)))
        elif plane_idx == 2:
            # XOR with pattern
            value = sum(base_values[base] ^ (i % 4) for i, base in enumerate(dna_seq))
        else:
            # Weighted sum
            weights = [1, 3, 5, 7]
            value = sum(base_values[base] * weights[i] for i, base in enumerate(dna_seq))
        
        return value % 256
    
    def _apply_chaotic_transformation(self, plane: np.ndarray, chaotic_matrix: np.ndarray) -> np.ndarray:
        """Apply chaotic transformation to plane"""
        # Normalize chaotic matrix to [0, 255]
        chaotic_normalized = (chaotic_matrix * 255).astype(np.uint8)
        
        # XOR with chaotic matrix
        transformed = np.bitwise_xor(plane, chaotic_normalized)
        
        return transformed
    
    def generate_3d_permutation_indices(self, shape_3d: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate 3D permutation indices using SAMCML
        
        Args:
            shape_3d: 3D shape (depth, height, width)
            
        Returns:
            Tuple of (depth_indices, height_indices, width_indices)
        """
        depth, height, width = shape_3d
        total_elements = depth * height * width
        
        # Generate chaotic sequence for permutation
        chaotic_sequence = self.samcml_system.generate_chaotic_sequence(total_elements)
        
        # Create permutation indices for each dimension
        depth_indices = np.argsort(chaotic_sequence[:depth, 0])
        height_indices = np.argsort(chaotic_sequence[:height, 1] if height <= total_elements else chaotic_sequence[:height, 0])
        width_indices = np.argsort(chaotic_sequence[:width, 2] if width <= total_elements else chaotic_sequence[:width, 0])
        
        return depth_indices, height_indices, width_indices
    
    def cross_plane_scrambling_3d(self, image_3d: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply 3D cross-plane Fisher-Yates scrambling
        
        Args:
            image_3d: 3D image array (depth, height, width)
            
        Returns:
            Tuple of (scrambled_3d, scrambling_info)
        """
        depth, height, width = image_3d.shape
        scrambled_3d = image_3d.copy()
        
        # Generate permutation indices
        depth_perm, height_perm, width_perm = self.generate_3d_permutation_indices((depth, height, width))
        
        # Apply Fisher-Yates scrambling in 3D
        scrambling_operations = []
        
        # 1. Depth-wise scrambling
        for d in range(depth - 1, 0, -1):
            # Generate chaotic index
            chaotic_val = self.samcml_system.generate_chaotic_sequence(1)[0, 0]
            j = int(chaotic_val * (d + 1))
            
            # Swap entire planes
            if j != d:
                scrambled_3d[[d, j]] = scrambled_3d[[j, d]]
                scrambling_operations.append(('depth', d, j))
        
        # 2. Height-wise scrambling (within each plane)
        for d in range(depth):
            for h in range(height - 1, 0, -1):
                chaotic_val = self.samcml_system.generate_chaotic_sequence(1)[0, 1]
                j = int(chaotic_val * (h + 1))
                
                if j != h:
                    scrambled_3d[d, [h, j]] = scrambled_3d[d, [j, h]]
                    scrambling_operations.append(('height', d, h, j))
        
        # 3. Width-wise scrambling (within each row)
        for d in range(depth):
            for h in range(height):
                for w in range(width - 1, 0, -1):
                    chaotic_val = self.samcml_system.generate_chaotic_sequence(1)[0, 2]
                    j = int(chaotic_val * (w + 1))
                    
                    if j != w:
                        scrambled_3d[d, h, [w, j]] = scrambled_3d[d, h, [j, w]]
                        scrambling_operations.append(('width', d, h, w, j))
        
        # 4. Cross-plane element scrambling
        total_elements = depth * height * width
        flat_scrambled = scrambled_3d.flatten()
        
        for i in range(total_elements - 1, 0, -1):
            chaotic_val = self.samcml_system.generate_chaotic_sequence(1)[0, 0]
            j = int(chaotic_val * (i + 1))
            
            if j != i:
                flat_scrambled[i], flat_scrambled[j] = flat_scrambled[j], flat_scrambled[i]
                scrambling_operations.append(('element', i, j))
        
        scrambled_3d = flat_scrambled.reshape((depth, height, width))
        
        # Store scrambling information for reversal
        scrambling_info = {
            'operations': scrambling_operations,
            'original_shape': (depth, height, width),
            'samcml_params': self.samcml_system.get_system_info(),
            'operation_count': len(scrambling_operations)
        }
        
        return scrambled_3d, scrambling_info
    
    def reverse_cross_plane_scrambling(self, scrambled_3d: np.ndarray, scrambling_info: Dict[str, Any]) -> np.ndarray:
        """
        Reverse 3D cross-plane scrambling
        
        Args:
            scrambled_3d: Scrambled 3D array
            scrambling_info: Information needed for reversal
            
        Returns:
            Original 3D array
        """
        depth, height, width = scrambling_info['original_shape']
        unscrambled_3d = scrambled_3d.copy()
        
        # Reverse operations in reverse order
        operations = scrambling_info['operations']
        
        for operation in reversed(operations):
            op_type = operation[0]
            
            if op_type == 'element':
                # Reverse element scrambling
                _, i, j = operation
                flat_unscrambled = unscrambled_3d.flatten()
                flat_unscrambled[i], flat_unscrambled[j] = flat_unscrambled[j], flat_unscrambled[i]
                unscrambled_3d = flat_unscrambled.reshape((depth, height, width))
            
            elif op_type == 'width':
                # Reverse width scrambling
                _, d, h, w, j = operation
                unscrambled_3d[d, h, [w, j]] = unscrambled_3d[d, h, [j, w]]
            
            elif op_type == 'height':
                # Reverse height scrambling
                _, d, h, j = operation
                unscrambled_3d[d, [h, j]] = unscrambled_3d[d, [j, h]]
            
            elif op_type == 'depth':
                # Reverse depth scrambling
                _, d, j = operation
                unscrambled_3d[[d, j]] = unscrambled_3d[[j, d]]
        
        return unscrambled_3d
    
    def adaptive_scrambling_intensity(self, image_3d: np.ndarray) -> float:
        """
        Calculate adaptive scrambling intensity based on image characteristics
        
        Args:
            image_3d: 3D image array
            
        Returns:
            Scrambling intensity factor [0.5, 2.0]
        """
        # Calculate image variance across all planes
        total_variance = np.var(image_3d)
        
        # Calculate entropy-like measure
        hist, _ = np.histogram(image_3d.flatten(), bins=256, range=(0, 256))
        hist = hist[hist > 0]
        prob = hist / np.sum(hist)
        entropy = -np.sum(prob * np.log2(prob))
        
        # Combine metrics to determine intensity
        # Higher variance and entropy -> higher intensity
        normalized_variance = min(total_variance / 10000, 1.0)  # Normalize to [0, 1]
        normalized_entropy = entropy / 8.0  # Normalize to [0, 1]
        
        intensity = 0.5 + 1.5 * (normalized_variance + normalized_entropy) / 2
        
        return intensity
    
    def multi_round_scrambling(self, image_3d: np.ndarray, rounds: int = 3) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Apply multiple rounds of scrambling for enhanced security
        
        Args:
            image_3d: 3D image array
            rounds: Number of scrambling rounds
            
        Returns:
            Tuple of (final_scrambled, list_of_scrambling_info)
        """
        current_3d = image_3d.copy()
        all_scrambling_info = []
        
        for round_idx in range(rounds):
            # Update SAMCML system state for each round
            self.samcml_system.x1 = (self.samcml_system.x1 + 0.1 * round_idx) % 1
            self.samcml_system.x2 = (self.samcml_system.x2 + 0.1 * round_idx) % 1
            self.samcml_system.x3 = (self.samcml_system.x3 + 0.1 * round_idx) % 1
            
            # Apply scrambling
            current_3d, scrambling_info = self.cross_plane_scrambling_3d(current_3d)
            scrambling_info['round'] = round_idx
            all_scrambling_info.append(scrambling_info)
        
        return current_3d, all_scrambling_info
    
    def reverse_multi_round_scrambling(self, scrambled_3d: np.ndarray, 
                                     all_scrambling_info: List[Dict[str, Any]]) -> np.ndarray:
        """
        Reverse multiple rounds of scrambling
        
        Args:
            scrambled_3d: Final scrambled array
            all_scrambling_info: List of scrambling info for each round
            
        Returns:
            Original 3D array
        """
        current_3d = scrambled_3d.copy()
        
        # Reverse rounds in reverse order
        for scrambling_info in reversed(all_scrambling_info):
            current_3d = self.reverse_cross_plane_scrambling(current_3d, scrambling_info)
        
        return current_3d


def test_advanced_3d_fisher_scrambling():
    """Test advanced 3D Fisher-Yates scrambling"""
    print("Testing Advanced 3D Fisher-Yates Scrambling...")
    
    # Create SAMCML system
    samcml = SAMCMLChaoticSystem(seed=42)
    
    # Initialize scrambling system
    scrambler = Advanced3DFisherScrambling(samcml)
    
    # Create test 3D data
    test_images = [np.random.randint(0, 256, (32, 32), dtype=np.uint8) for _ in range(4)]
    image_3d = scrambler.create_3d_image_stack(test_images)
    
    print(f"✓ Created 3D image stack: {image_3d.shape}")
    
    # Test single round scrambling
    scrambled_3d, scrambling_info = scrambler.cross_plane_scrambling_3d(image_3d)
    unscrambled_3d = scrambler.reverse_cross_plane_scrambling(scrambled_3d, scrambling_info)
    
    print(f"✓ Single round scrambling: {scrambling_info['operation_count']} operations")
    print(f"✓ Perfect reconstruction: {np.array_equal(image_3d, unscrambled_3d)}")
    
    # Test multi-round scrambling
    multi_scrambled, multi_info = scrambler.multi_round_scrambling(image_3d, rounds=3)
    multi_unscrambled = scrambler.reverse_multi_round_scrambling(multi_scrambled, multi_info)
    
    print(f"✓ Multi-round scrambling: {len(multi_info)} rounds")
    print(f"✓ Perfect multi-round reconstruction: {np.array_equal(image_3d, multi_unscrambled)}")
    
    # Test adaptive intensity
    intensity = scrambler.adaptive_scrambling_intensity(image_3d)
    print(f"✓ Adaptive scrambling intensity: {intensity:.3f}")
    
    print("Advanced 3D Fisher-Yates scrambling tests completed successfully!")


if __name__ == "__main__":
    test_advanced_3d_fisher_scrambling()
