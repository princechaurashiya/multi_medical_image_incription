"""
DNA Operations Module for Multi Medical Image Encryption
Implements DNA encoding/decoding and asymmetric DNA operations
"""

import numpy as np
from typing import Dict, List, Tuple
import random


class DNAOperations:
    """
    Implements DNA encoding, decoding, and cryptographic operations
    """
    
    # DNA encoding rules - multiple rule sets for asymmetric operations
    DNA_ENCODING_RULES = {
        'rule1': {'00': 'A', '01': 'T', '10': 'G', '11': 'C'},
        'rule2': {'00': 'T', '01': 'A', '10': 'C', '11': 'G'},
        'rule3': {'00': 'G', '01': 'C', '10': 'A', '11': 'T'},
        'rule4': {'00': 'C', '01': 'G', '10': 'T', '11': 'A'},
        'rule5': {'00': 'A', '01': 'G', '10': 'T', '11': 'C'},
        'rule6': {'00': 'T', '01': 'C', '10': 'A', '11': 'G'},
        'rule7': {'00': 'G', '01': 'A', '10': 'C', '11': 'T'},
        'rule8': {'00': 'C', '01': 'T', '10': 'G', '11': 'A'}
    }
    
    # DNA decoding rules (reverse of encoding)
    DNA_DECODING_RULES = {}
    
    # DNA operation rules for XOR, ADD, SUB operations
    DNA_XOR_RULES = {
        'A': {'A': 'A', 'T': 'T', 'G': 'G', 'C': 'C'},
        'T': {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'},
        'G': {'A': 'G', 'T': 'C', 'G': 'A', 'C': 'T'},
        'C': {'A': 'C', 'T': 'G', 'G': 'T', 'C': 'A'}
    }
    
    DNA_ADD_RULES = {
        'A': {'A': 'A', 'T': 'T', 'G': 'G', 'C': 'C'},
        'T': {'A': 'T', 'T': 'G', 'G': 'C', 'C': 'A'},
        'G': {'A': 'G', 'T': 'C', 'G': 'T', 'C': 'A'},
        'C': {'A': 'C', 'T': 'A', 'G': 'A', 'C': 'T'}
    }
    
    DNA_SUB_RULES = {
        'A': {'A': 'A', 'T': 'C', 'G': 'T', 'C': 'G'},
        'T': {'A': 'G', 'T': 'A', 'G': 'C', 'C': 'T'},
        'G': {'A': 'C', 'T': 'G', 'G': 'A', 'C': 'T'},
        'C': {'A': 'T', 'T': 'C', 'G': 'G', 'C': 'A'}
    }
    
    def __init__(self):
        """Initialize DNA operations and create decoding rules"""
        self._create_decoding_rules()
    
    def _create_decoding_rules(self):
        """Create decoding rules from encoding rules"""
        for rule_name, encoding_rule in self.DNA_ENCODING_RULES.items():
            self.DNA_DECODING_RULES[rule_name] = {v: k for k, v in encoding_rule.items()}
    
    def pixel_to_binary_pairs(self, pixel_value: int) -> List[str]:
        """
        Convert pixel value to binary pairs for DNA encoding
        
        Args:
            pixel_value: 8-bit pixel value (0-255)
            
        Returns:
            List of 2-bit binary strings
        """
        if not (0 <= pixel_value <= 255):
            raise ValueError("Pixel value must be between 0 and 255")
        
        # Convert to 8-bit binary string
        binary_str = format(pixel_value, '08b')
        
        # Split into pairs
        pairs = [binary_str[i:i+2] for i in range(0, 8, 2)]
        return pairs
    
    def binary_pairs_to_pixel(self, binary_pairs: List[str]) -> int:
        """
        Convert binary pairs back to pixel value
        
        Args:
            binary_pairs: List of 2-bit binary strings
            
        Returns:
            8-bit pixel value
        """
        binary_str = ''.join(binary_pairs)
        return int(binary_str, 2)
    
    def encode_to_dna(self, image: np.ndarray, rule_name: str = 'rule1') -> np.ndarray:
        """
        Encode image to DNA sequences using specified rule
        
        Args:
            image: Input image array
            rule_name: DNA encoding rule to use
            
        Returns:
            DNA encoded image as string array
        """
        if rule_name not in self.DNA_ENCODING_RULES:
            raise ValueError(f"Unknown encoding rule: {rule_name}")
        
        encoding_rule = self.DNA_ENCODING_RULES[rule_name]
        dna_image = np.empty(image.shape, dtype='U4')  # Each pixel -> 4 DNA bases
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                pixel_value = image[i, j]
                binary_pairs = self.pixel_to_binary_pairs(pixel_value)
                dna_sequence = ''.join([encoding_rule[pair] for pair in binary_pairs])
                dna_image[i, j] = dna_sequence
        
        return dna_image
    
    def decode_from_dna(self, dna_image: np.ndarray, rule_name: str = 'rule1') -> np.ndarray:
        """
        Decode DNA sequences back to image using specified rule
        
        Args:
            dna_image: DNA encoded image
            rule_name: DNA decoding rule to use
            
        Returns:
            Decoded image array
        """
        if rule_name not in self.DNA_DECODING_RULES:
            raise ValueError(f"Unknown decoding rule: {rule_name}")
        
        decoding_rule = self.DNA_DECODING_RULES[rule_name]
        image = np.zeros(dna_image.shape, dtype=np.uint8)
        
        for i in range(dna_image.shape[0]):
            for j in range(dna_image.shape[1]):
                dna_sequence = dna_image[i, j]
                binary_pairs = [decoding_rule[base] for base in dna_sequence]
                pixel_value = self.binary_pairs_to_pixel(binary_pairs)
                image[i, j] = pixel_value
        
        return image
    
    def dna_xor_operation(self, dna_seq1: str, dna_seq2: str) -> str:
        """
        Perform DNA XOR operation between two DNA sequences
        
        Args:
            dna_seq1: First DNA sequence
            dna_seq2: Second DNA sequence
            
        Returns:
            Result of DNA XOR operation
        """
        if len(dna_seq1) != len(dna_seq2):
            raise ValueError("DNA sequences must have the same length")
        
        result = ''
        for base1, base2 in zip(dna_seq1, dna_seq2):
            result += self.DNA_XOR_RULES[base1][base2]
        
        return result
    
    def dna_add_operation(self, dna_seq1: str, dna_seq2: str) -> str:
        """
        Perform DNA ADD operation between two DNA sequences
        
        Args:
            dna_seq1: First DNA sequence
            dna_seq2: Second DNA sequence
            
        Returns:
            Result of DNA ADD operation
        """
        if len(dna_seq1) != len(dna_seq2):
            raise ValueError("DNA sequences must have the same length")
        
        result = ''
        for base1, base2 in zip(dna_seq1, dna_seq2):
            result += self.DNA_ADD_RULES[base1][base2]
        
        return result
    
    def dna_sub_operation(self, dna_seq1: str, dna_seq2: str) -> str:
        """
        Perform DNA SUB operation between two DNA sequences
        
        Args:
            dna_seq1: First DNA sequence
            dna_seq2: Second DNA sequence
            
        Returns:
            Result of DNA SUB operation
        """
        if len(dna_seq1) != len(dna_seq2):
            raise ValueError("DNA sequences must have the same length")
        
        result = ''
        for base1, base2 in zip(dna_seq1, dna_seq2):
            result += self.DNA_SUB_RULES[base1][base2]
        
        return result
    
    def asymmetric_dna_diffusion(self, dna_image: np.ndarray, key_sequence: str) -> np.ndarray:
        """
        Perform asymmetric DNA diffusion for high-quality encryption
        
        Args:
            dna_image: DNA encoded image
            key_sequence: DNA key sequence for diffusion
            
        Returns:
            Diffused DNA image
        """
        diffused_image = np.copy(dna_image)
        rows, cols = dna_image.shape
        
        # Expand key sequence to match image size
        key_length = len(key_sequence)
        
        # Forward diffusion (left to right, top to bottom)
        for i in range(rows):
            for j in range(cols):
                key_idx = (i * cols + j) % key_length
                key_base = key_sequence[key_idx]
                
                # Create 4-base key sequence for pixel
                pixel_key = key_base * 4
                
                # Apply DNA XOR with key
                diffused_image[i, j] = self.dna_xor_operation(
                    diffused_image[i, j], pixel_key
                )
                
                # Add dependency on previous pixel for diffusion
                if j > 0:
                    diffused_image[i, j] = self.dna_add_operation(
                        diffused_image[i, j], diffused_image[i, j-1]
                    )
                elif i > 0:
                    diffused_image[i, j] = self.dna_add_operation(
                        diffused_image[i, j], diffused_image[i-1, cols-1]
                    )
        
        return diffused_image
    
    def convert_first_3bits_to_range(self, image: np.ndarray) -> np.ndarray:
        """
        Convert first 3 bits of each pixel to 0-7 range before DNA encoding
        
        Args:
            image: Input image array
            
        Returns:
            Image with first 3 bits converted to 0-7 range
        """
        converted_image = np.copy(image)
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                pixel = image[i, j]
                # Extract first 3 bits (MSB)
                first_3_bits = (pixel >> 5) & 0x07  # 0x07 = 0b111
                # Map to 0-7 range (already in range, but ensure)
                mapped_value = first_3_bits % 8
                # Replace the first 3 bits with mapped value
                converted_image[i, j] = (mapped_value << 5) | (pixel & 0x1F)
        
        return converted_image


def test_dna_operations():
    """Test the DNA operations functionality"""
    print("Testing DNA Operations Module...")
    
    dna_ops = DNAOperations()
    
    # Test pixel to binary conversion
    pixel = 150  # 10010110 in binary
    binary_pairs = dna_ops.pixel_to_binary_pairs(pixel)
    print(f"Pixel {pixel} -> Binary pairs: {binary_pairs}")
    
    # Test DNA encoding
    test_image = np.array([[100, 150], [200, 50]], dtype=np.uint8)
    dna_encoded = dna_ops.encode_to_dna(test_image, 'rule1')
    print(f"DNA encoded image:\n{dna_encoded}")
    
    # Test DNA decoding
    decoded_image = dna_ops.decode_from_dna(dna_encoded, 'rule1')
    print(f"Decoded image:\n{decoded_image}")
    print(f"Original == Decoded: {np.array_equal(test_image, decoded_image)}")
    
    # Test DNA operations
    seq1, seq2 = "ATGC", "CGTA"
    xor_result = dna_ops.dna_xor_operation(seq1, seq2)
    add_result = dna_ops.dna_add_operation(seq1, seq2)
    print(f"DNA XOR: {seq1} ⊕ {seq2} = {xor_result}")
    print(f"DNA ADD: {seq1} + {seq2} = {add_result}")
    
    # Test first 3 bits conversion
    converted = dna_ops.convert_first_3bits_to_range(test_image)
    print(f"First 3 bits converted:\n{converted}")
    
    print("DNA operations tests completed successfully!")


if __name__ == "__main__":
    test_dna_operations()
