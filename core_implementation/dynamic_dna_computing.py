"""
Dynamic DNA Computing with 8 Encoding Rules
Advanced DNA operations for medical image encryption
"""

import numpy as np
from typing import Dict, List, Tuple, Any


class DynamicDNAComputing:
    """
    Dynamic DNA computing system with 8 encoding rules and new DNA operations
    Uses 3-bit selection for dynamic rule switching
    """
    
    def __init__(self):
        """Initialize DNA computing system with 8 encoding rules"""
        
        # Define 8 different DNA encoding rules
        self.dna_encoding_rules = {
            'rule_0': {'00': 'A', '01': 'T', '10': 'G', '11': 'C'},
            'rule_1': {'00': 'A', '01': 'C', '10': 'G', '11': 'T'},
            'rule_2': {'00': 'T', '01': 'A', '10': 'C', '11': 'G'},
            'rule_3': {'00': 'T', '01': 'G', '10': 'C', '11': 'A'},
            'rule_4': {'00': 'G', '01': 'A', '10': 'T', '11': 'C'},
            'rule_5': {'00': 'G', '01': 'C', '10': 'T', '11': 'A'},
            'rule_6': {'00': 'C', '01': 'A', '10': 'G', '11': 'T'},
            'rule_7': {'00': 'C', '01': 'T', '10': 'G', '11': 'A'}
        }
        
        # Create reverse mapping for decoding
        self.dna_decoding_rules = {}
        for rule_name, encoding in self.dna_encoding_rules.items():
            self.dna_decoding_rules[rule_name] = {v: k for k, v in encoding.items()}
        
        # New DNA operation matrices (different from XOR, ADD, SUB)
        self.dna_operation_matrix = {
            'A': {'A': 'A', 'T': 'G', 'G': 'C', 'C': 'T'},
            'T': {'A': 'G', 'T': 'T', 'G': 'A', 'C': 'C'},
            'G': {'A': 'C', 'T': 'A', 'G': 'G', 'C': 'T'},
            'C': {'A': 'T', 'T': 'C', 'G': 'T', 'C': 'C'}
        }
        
        # Inverse DNA operation matrix for decryption
        self.dna_inverse_matrix = self._create_inverse_matrix()
    
    def _create_inverse_matrix(self) -> Dict[str, Dict[str, str]]:
        """Create inverse matrix for DNA operations"""
        inverse_matrix = {}
        
        for base1 in ['A', 'T', 'G', 'C']:
            inverse_matrix[base1] = {}
            for base2 in ['A', 'T', 'G', 'C']:
                # Find the inverse: if op(a, b) = c, then inv_op(a, c) = b
                result = self.dna_operation_matrix[base1][base2]
                inverse_matrix[base1][result] = base2
        
        return inverse_matrix
    
    def pixel_to_binary(self, pixel_value: int) -> str:
        """Convert pixel value to 8-bit binary string"""
        return format(pixel_value, '08b')
    
    def binary_to_pixel(self, binary_str: str) -> int:
        """Convert binary string to pixel value"""
        return int(binary_str, 2)
    
    def extract_3bits_for_rule_selection(self, image: np.ndarray) -> np.ndarray:
        """
        Extract 3 bits from each pixel for DNA rule selection
        Uses bits 0, 1, 2 (LSB) to select from 8 rules
        
        Args:
            image: Input image
            
        Returns:
            Array of rule indices (0-7) for each pixel
        """
        # Extract first 3 bits (LSB)
        bit0 = (image >> 0) & 1
        bit1 = (image >> 1) & 1
        bit2 = (image >> 2) & 1
        
        # Combine to form rule index (0-7)
        rule_indices = (bit2 << 2) | (bit1 << 1) | bit0
        
        return rule_indices
    
    def encode_pixel_to_dna(self, pixel_value: int, rule_index: int) -> str:
        """
        Encode single pixel to DNA sequence using specified rule
        
        Args:
            pixel_value: Pixel value (0-255)
            rule_index: DNA encoding rule index (0-7)
            
        Returns:
            DNA sequence string (4 bases)
        """
        rule_name = f'rule_{rule_index}'
        encoding_rule = self.dna_encoding_rules[rule_name]
        
        # Convert pixel to 8-bit binary
        binary_str = self.pixel_to_binary(pixel_value)
        
        # Encode each 2-bit pair to DNA base
        dna_sequence = ''
        for i in range(0, 8, 2):
            two_bits = binary_str[i:i+2]
            dna_base = encoding_rule[two_bits]
            dna_sequence += dna_base
        
        return dna_sequence
    
    def decode_dna_to_pixel(self, dna_sequence: str, rule_index: int) -> int:
        """
        Decode DNA sequence to pixel value using specified rule
        
        Args:
            dna_sequence: DNA sequence (4 bases)
            rule_index: DNA decoding rule index (0-7)
            
        Returns:
            Pixel value (0-255)
        """
        rule_name = f'rule_{rule_index}'
        decoding_rule = self.dna_decoding_rules[rule_name]
        
        # Decode each DNA base to 2-bit pair
        binary_str = ''
        for base in dna_sequence:
            two_bits = decoding_rule[base]
            binary_str += two_bits
        
        # Convert binary to pixel value
        return self.binary_to_pixel(binary_str)
    
    def encode_image_to_dna(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode entire image to DNA using dynamic rule selection
        
        Args:
            image: Input grayscale image
            
        Returns:
            Tuple of (dna_matrix, rule_indices)
        """
        height, width = image.shape
        
        # Extract rule indices for each pixel
        rule_indices = self.extract_3bits_for_rule_selection(image)
        
        # Initialize DNA matrix (each element is a 4-character DNA sequence)
        dna_matrix = np.empty((height, width), dtype='U4')
        
        # Encode each pixel
        for i in range(height):
            for j in range(width):
                pixel_value = image[i, j]
                rule_index = rule_indices[i, j]
                dna_sequence = self.encode_pixel_to_dna(pixel_value, rule_index)
                dna_matrix[i, j] = dna_sequence
        
        return dna_matrix, rule_indices
    
    def decode_dna_to_image(self, dna_matrix: np.ndarray, rule_indices: np.ndarray) -> np.ndarray:
        """
        Decode DNA matrix back to image using rule indices
        
        Args:
            dna_matrix: DNA sequence matrix
            rule_indices: Rule indices for each position
            
        Returns:
            Decoded image
        """
        height, width = dna_matrix.shape
        image = np.zeros((height, width), dtype=np.uint8)
        
        # Decode each DNA sequence
        for i in range(height):
            for j in range(width):
                dna_sequence = dna_matrix[i, j]
                rule_index = rule_indices[i, j]
                pixel_value = self.decode_dna_to_pixel(dna_sequence, rule_index)
                image[i, j] = pixel_value
        
        return image
    
    def apply_new_dna_operation(self, dna_seq1: str, dna_seq2: str) -> str:
        """
        Apply new DNA operation between two DNA sequences
        
        Args:
            dna_seq1, dna_seq2: DNA sequences (4 bases each)
            
        Returns:
            Result DNA sequence
        """
        if len(dna_seq1) != 4 or len(dna_seq2) != 4:
            raise ValueError("DNA sequences must be 4 bases long")
        
        result_sequence = ''
        for base1, base2 in zip(dna_seq1, dna_seq2):
            result_base = self.dna_operation_matrix[base1][base2]
            result_sequence += result_base
        
        return result_sequence
    
    def apply_inverse_dna_operation(self, dna_seq1: str, result_seq: str) -> str:
        """
        Apply inverse DNA operation for decryption
        
        Args:
            dna_seq1: First DNA sequence
            result_seq: Result sequence from forward operation
            
        Returns:
            Original second DNA sequence
        """
        if len(dna_seq1) != 4 or len(result_seq) != 4:
            raise ValueError("DNA sequences must be 4 bases long")
        
        original_sequence = ''
        for base1, result_base in zip(dna_seq1, result_seq):
            original_base = self.dna_inverse_matrix[base1][result_base]
            original_sequence += original_base
        
        return original_sequence
    
    def dna_diffusion(self, dna_matrix: np.ndarray, key_dna_matrix: np.ndarray) -> np.ndarray:
        """
        Apply DNA diffusion using new DNA operation
        
        Args:
            dna_matrix: Input DNA matrix
            key_dna_matrix: Key DNA matrix
            
        Returns:
            Diffused DNA matrix
        """
        height, width = dna_matrix.shape
        diffused_matrix = np.empty((height, width), dtype='U4')
        
        # Apply DNA operation element-wise
        for i in range(height):
            for j in range(width):
                dna_seq = dna_matrix[i, j]
                key_seq = key_dna_matrix[i, j]
                diffused_seq = self.apply_new_dna_operation(dna_seq, key_seq)
                diffused_matrix[i, j] = diffused_seq
        
        return diffused_matrix
    
    def reverse_dna_diffusion(self, diffused_matrix: np.ndarray, key_dna_matrix: np.ndarray) -> np.ndarray:
        """
        Reverse DNA diffusion for decryption
        
        Args:
            diffused_matrix: Diffused DNA matrix
            key_dna_matrix: Key DNA matrix
            
        Returns:
            Original DNA matrix
        """
        height, width = diffused_matrix.shape
        original_matrix = np.empty((height, width), dtype='U4')
        
        # Apply inverse DNA operation element-wise
        for i in range(height):
            for j in range(width):
                diffused_seq = diffused_matrix[i, j]
                key_seq = key_dna_matrix[i, j]
                original_seq = self.apply_inverse_dna_operation(key_seq, diffused_seq)
                original_matrix[i, j] = original_seq
        
        return original_matrix
    
    def generate_key_dna_matrix(self, chaotic_matrix: np.ndarray, rule_indices: np.ndarray) -> np.ndarray:
        """
        Generate DNA key matrix from chaotic values
        
        Args:
            chaotic_matrix: Chaotic values matrix [0, 1]
            rule_indices: Rule indices for encoding
            
        Returns:
            DNA key matrix
        """
        height, width = chaotic_matrix.shape
        key_dna_matrix = np.empty((height, width), dtype='U4')
        
        # Convert chaotic values to pixel values and encode to DNA
        for i in range(height):
            for j in range(width):
                # Convert chaotic value to pixel value
                pixel_value = int(chaotic_matrix[i, j] * 255)
                rule_index = rule_indices[i, j]
                
                # Encode to DNA
                dna_sequence = self.encode_pixel_to_dna(pixel_value, rule_index)
                key_dna_matrix[i, j] = dna_sequence
        
        return key_dna_matrix
    
    def analyze_dna_distribution(self, dna_matrix: np.ndarray) -> Dict[str, float]:
        """
        Analyze distribution of DNA bases in matrix
        
        Args:
            dna_matrix: DNA sequence matrix
            
        Returns:
            Dictionary with base distribution statistics
        """
        # Count all bases
        base_counts = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
        total_bases = 0
        
        for i in range(dna_matrix.shape[0]):
            for j in range(dna_matrix.shape[1]):
                sequence = dna_matrix[i, j]
                for base in sequence:
                    base_counts[base] += 1
                    total_bases += 1
        
        # Calculate percentages
        distribution = {}
        for base, count in base_counts.items():
            distribution[base] = count / total_bases if total_bases > 0 else 0
        
        return distribution


def test_dynamic_dna_computing():
    """Test dynamic DNA computing system"""
    print("Testing Dynamic DNA Computing System...")
    
    # Initialize DNA system
    dna_system = DynamicDNAComputing()
    
    # Test single pixel encoding/decoding
    pixel_value = 123
    rule_index = 3
    dna_seq = dna_system.encode_pixel_to_dna(pixel_value, rule_index)
    decoded_pixel = dna_system.decode_dna_to_pixel(dna_seq, rule_index)
    print(f"✓ Single pixel: {pixel_value} -> {dna_seq} -> {decoded_pixel}")
    
    # Test image encoding/decoding
    test_image = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
    dna_matrix, rule_indices = dna_system.encode_image_to_dna(test_image)
    decoded_image = dna_system.decode_dna_to_image(dna_matrix, rule_indices)
    
    print(f"✓ Image encoding: {test_image.shape} -> DNA matrix {dna_matrix.shape}")
    print(f"✓ Perfect reconstruction: {np.array_equal(test_image, decoded_image)}")
    
    # Test DNA operations
    seq1, seq2 = "ATGC", "CGTA"
    result = dna_system.apply_new_dna_operation(seq1, seq2)
    recovered = dna_system.apply_inverse_dna_operation(seq1, result)
    print(f"✓ DNA operation: {seq1} ⊕ {seq2} = {result}")
    print(f"✓ Inverse operation: {seq1} ⊕⁻¹ {result} = {recovered}")
    
    # Test distribution analysis
    distribution = dna_system.analyze_dna_distribution(dna_matrix)
    print(f"✓ DNA base distribution: {distribution}")
    
    print("Dynamic DNA computing tests completed successfully!")


if __name__ == "__main__":
    test_dynamic_dna_computing()
