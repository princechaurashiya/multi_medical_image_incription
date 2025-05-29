"""
SAMCML (Sin-Arcsin-Arnold Multi-Dynamic Coupled Map Lattice) Chaotic System
Advanced chaotic model for medical image encryption
"""

import numpy as np
import hashlib
from typing import Tuple, List, Dict, Any


class SAMCMLChaoticSystem:
    """
    Sin-Arcsin-Arnold Multi-Dynamic Coupled Map Lattice (SAMCML) implementation
    Combines sine, arcsin, Arnold transform, and coupled map lattice for enhanced chaos
    """
    
    def __init__(self, mu1: float = 3.9, mu2: float = 3.8, mu3: float = 3.7,
                 x1: float = 0.1, x2: float = 0.2, x3: float = 0.3,
                 e1: float = 0.1, e2: float = 0.2, e3: float = 0.3):
        """
        Initialize SAMCML system with parameters
        
        Args:
            mu1, mu2, mu3: Control parameters for chaotic maps
            x1, x2, x3: Initial values for chaotic sequences
            e1, e2, e3: Coupling strengths
        """
        self.mu1, self.mu2, self.mu3 = mu1, mu2, mu3
        self.x1, self.x2, self.x3 = x1, x2, x3
        self.e1, self.e2, self.e3 = e1, e2, e3
        
        # Arnold transform parameters
        self.arnold_a = 1
        self.arnold_b = 1
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate chaotic system parameters"""
        # Check mu parameters (should be in chaotic range)
        for mu in [self.mu1, self.mu2, self.mu3]:
            if not (3.57 <= mu <= 4.0):
                print(f"Warning: mu={mu} may not be in optimal chaotic range [3.57, 4.0]")
        
        # Check initial values (should be in [0, 1])
        for x in [self.x1, self.x2, self.x3]:
            if not (0 < x < 1):
                raise ValueError(f"Initial value {x} must be in range (0, 1)")
        
        # Check coupling strengths
        for e in [self.e1, self.e2, self.e3]:
            if not (0 <= e <= 1):
                raise ValueError(f"Coupling strength {e} must be in range [0, 1]")
    
    @classmethod
    def from_sha512_hash(cls, hash_data: str):
        """
        Create SAMCML system from SHA-512 hash
        
        Args:
            hash_data: SHA-512 hash string (128 hex characters)
            
        Returns:
            SAMCMLChaoticSystem instance with parameters derived from hash
        """
        if len(hash_data) != 128:
            raise ValueError("SHA-512 hash must be 128 hex characters")
        
        # Extract parameters from different parts of hash
        # Use different segments for different parameters
        mu1 = 3.57 + (int(hash_data[0:8], 16) / 0xFFFFFFFF) * 0.43  # [3.57, 4.0]
        mu2 = 3.57 + (int(hash_data[8:16], 16) / 0xFFFFFFFF) * 0.43
        mu3 = 3.57 + (int(hash_data[16:24], 16) / 0xFFFFFFFF) * 0.43
        
        x1 = int(hash_data[24:32], 16) / 0xFFFFFFFF  # [0, 1]
        x2 = int(hash_data[32:40], 16) / 0xFFFFFFFF
        x3 = int(hash_data[40:48], 16) / 0xFFFFFFFF
        
        e1 = int(hash_data[48:56], 16) / 0xFFFFFFFF  # [0, 1]
        e2 = int(hash_data[56:64], 16) / 0xFFFFFFFF
        e3 = int(hash_data[64:72], 16) / 0xFFFFFFFF
        
        return cls(mu1, mu2, mu3, x1, x2, x3, e1, e2, e3)
    
    def sine_map(self, x: float, mu: float) -> float:
        """Sine chaotic map: x_{n+1} = mu * sin(π * x_n)"""
        return mu * np.sin(np.pi * x)
    
    def arcsin_map(self, x: float, mu: float) -> float:
        """Arcsin chaotic map: x_{n+1} = (2/π) * arcsin(sqrt(mu * x))"""
        if x < 0 or x > 1:
            x = abs(x) % 1  # Ensure x is in [0, 1]
        return (2 / np.pi) * np.arcsin(np.sqrt(min(mu * x, 1.0)))
    
    def arnold_transform(self, x: float, y: float) -> Tuple[float, float]:
        """
        Arnold cat map transformation
        
        Args:
            x, y: Input coordinates
            
        Returns:
            Transformed coordinates
        """
        x_new = (x + self.arnold_a * y) % 1
        y_new = (self.arnold_b * x + (self.arnold_a * self.arnold_b + 1) * y) % 1
        return x_new, y_new
    
    def coupled_map_lattice_step(self, x1: float, x2: float, x3: float) -> Tuple[float, float, float]:
        """
        Single step of coupled map lattice evolution
        
        Args:
            x1, x2, x3: Current state values
            
        Returns:
            Next state values
        """
        # Apply individual chaotic maps
        f1 = self.sine_map(x1, self.mu1)
        f2 = self.arcsin_map(x2, self.mu2)
        f3 = self.sine_map(x3, self.mu3)
        
        # Apply Arnold transform to f1 and f2
        f1_arnold, f2_arnold = self.arnold_transform(f1, f2)
        
        # Coupled evolution with coupling strengths
        x1_new = (1 - self.e1) * f1_arnold + self.e1 * (f2_arnold + f3) / 2
        x2_new = (1 - self.e2) * f2_arnold + self.e2 * (f1_arnold + f3) / 2
        x3_new = (1 - self.e3) * f3 + self.e3 * (f1_arnold + f2_arnold) / 2
        
        # Ensure values stay in [0, 1]
        x1_new = abs(x1_new) % 1
        x2_new = abs(x2_new) % 1
        x3_new = abs(x3_new) % 1
        
        return x1_new, x2_new, x3_new
    
    def generate_chaotic_sequence(self, length: int, skip_transient: int = 1000) -> np.ndarray:
        """
        Generate chaotic sequence using SAMCML
        
        Args:
            length: Length of sequence to generate
            skip_transient: Number of initial iterations to skip (for stability)
            
        Returns:
            Chaotic sequence array of shape (length, 3)
        """
        # Initialize state
        x1, x2, x3 = self.x1, self.x2, self.x3
        
        # Skip transient behavior
        for _ in range(skip_transient):
            x1, x2, x3 = self.coupled_map_lattice_step(x1, x2, x3)
        
        # Generate sequence
        sequence = np.zeros((length, 3))
        for i in range(length):
            x1, x2, x3 = self.coupled_map_lattice_step(x1, x2, x3)
            sequence[i] = [x1, x2, x3]
        
        return sequence
    
    def generate_chaotic_matrix(self, rows: int, cols: int, matrix_type: str = 'combined') -> np.ndarray:
        """
        Generate chaotic matrix for image operations
        
        Args:
            rows, cols: Matrix dimensions
            matrix_type: Type of matrix ('x1', 'x2', 'x3', 'combined')
            
        Returns:
            Chaotic matrix
        """
        total_elements = rows * cols
        sequence = self.generate_chaotic_sequence(total_elements)
        
        if matrix_type == 'x1':
            matrix = sequence[:, 0].reshape(rows, cols)
        elif matrix_type == 'x2':
            matrix = sequence[:, 1].reshape(rows, cols)
        elif matrix_type == 'x3':
            matrix = sequence[:, 2].reshape(rows, cols)
        elif matrix_type == 'combined':
            # Combine all three sequences
            matrix = (sequence[:, 0] + sequence[:, 1] + sequence[:, 2]) / 3
            matrix = matrix.reshape(rows, cols)
        else:
            raise ValueError(f"Unknown matrix_type: {matrix_type}")
        
        return matrix
    
    def generate_permutation_indices(self, size: int) -> np.ndarray:
        """
        Generate permutation indices using chaotic sequence
        
        Args:
            size: Size of permutation
            
        Returns:
            Permutation indices array
        """
        sequence = self.generate_chaotic_sequence(size)
        
        # Use first chaotic sequence to generate permutation
        chaotic_values = sequence[:, 0]
        
        # Sort indices based on chaotic values
        indices = np.argsort(chaotic_values)
        
        return indices
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system parameters and information"""
        return {
            'parameters': {
                'mu1': self.mu1, 'mu2': self.mu2, 'mu3': self.mu3,
                'x1': self.x1, 'x2': self.x2, 'x3': self.x3,
                'e1': self.e1, 'e2': self.e2, 'e3': self.e3
            },
            'arnold_params': {
                'a': self.arnold_a, 'b': self.arnold_b
            },
            'system_type': 'SAMCML',
            'description': 'Sin-Arcsin-Arnold Multi-Dynamic Coupled Map Lattice'
        }


def test_samcml_system():
    """Test SAMCML chaotic system"""
    print("Testing SAMCML Chaotic System...")
    
    # Test 1: Basic initialization
    samcml = SAMCMLChaoticSystem()
    print(f"✓ System initialized with default parameters")
    
    # Test 2: Generate chaotic sequence
    sequence = samcml.generate_chaotic_sequence(1000)
    print(f"✓ Generated chaotic sequence: shape {sequence.shape}")
    print(f"  Sequence range: [{np.min(sequence):.6f}, {np.max(sequence):.6f}]")
    
    # Test 3: Generate chaotic matrix
    matrix = samcml.generate_chaotic_matrix(64, 64)
    print(f"✓ Generated chaotic matrix: shape {matrix.shape}")
    
    # Test 4: From SHA-512 hash
    test_hash = hashlib.sha512(b"test_medical_image").hexdigest()
    samcml_hash = SAMCMLChaoticSystem.from_sha512_hash(test_hash)
    print(f"✓ Created system from SHA-512 hash")
    print(f"  Parameters: μ1={samcml_hash.mu1:.6f}, x1={samcml_hash.x1:.6f}")
    
    # Test 5: Permutation indices
    perm_indices = samcml.generate_permutation_indices(100)
    print(f"✓ Generated permutation indices: {len(perm_indices)} elements")
    
    print("SAMCML system tests completed successfully!")


if __name__ == "__main__":
    test_samcml_system()
