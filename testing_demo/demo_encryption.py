"""
Demonstration Script for Multi Medical Image Encryption System
Shows how to use the complete 6-step encryption process
"""

import numpy as np
import cv2
import os
import json
import matplotlib.pyplot as plt
from core_implementation.medical_image_encryption import MedicalImageEncryption


def create_sample_medical_images():
    """Create sample medical images for demonstration"""
    print("Creating sample medical images...")

    os.makedirs("sample_images", exist_ok=True)

    # 1. Brain MRI simulation
    brain_mri = np.zeros((256, 256), dtype=np.uint8)
    center_x, center_y = 128, 128

    # Create brain-like structure
    for i in range(256):
        for j in range(256):
            dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)

            # Brain tissue
            if dist < 100:
                intensity = 180 - dist * 0.8
                # Add some texture
                texture = np.sin(i * 0.1) * np.cos(j * 0.1) * 20
                brain_mri[i, j] = max(0, min(255, int(intensity + texture)))

            # Skull
            elif dist < 120:
                brain_mri[i, j] = np.clip(220 + int(np.random.normal(0, 10)), 0, 255)

            # Background
            else:
                brain_mri[i, j] = np.clip(30 + int(np.random.normal(0, 5)), 0, 255)

    cv2.imwrite("sample_images/brain_mri.png", brain_mri)

    # 2. CT Scan simulation
    ct_scan = np.zeros((256, 256), dtype=np.uint8)

    # Create organs and bones
    for i in range(256):
        for j in range(256):
            # Bone structures (high intensity)
            if (i - 128)**2 + (j - 128)**2 < 50**2:
                if abs(i - 128) < 10 or abs(j - 128) < 10:  # Cross pattern for bones
                    ct_scan[i, j] = 240
                else:
                    ct_scan[i, j] = 120  # Soft tissue

            # Lung areas (low intensity)
            elif ((i - 100)**2 + (j - 100)**2 < 30**2) or ((i - 156)**2 + (j - 156)**2 < 30**2):
                ct_scan[i, j] = 50

            # Other soft tissues
            else:
                ct_scan[i, j] = np.clip(100 + int(np.random.normal(0, 15)), 0, 255)

    cv2.imwrite("sample_images/ct_scan.png", ct_scan)

    # 3. X-Ray simulation
    xray = np.zeros((256, 256), dtype=np.uint8)

    # Create bone structures
    for i in range(256):
        for j in range(256):
            # Spine
            if abs(j - 128) < 8:
                xray[i, j] = 200

            # Ribs
            elif i % 30 < 5 and 50 < j < 206:
                xray[i, j] = 180

            # Soft tissue
            else:
                xray[i, j] = np.clip(80 + int(np.random.normal(0, 10)), 0, 255)

    cv2.imwrite("sample_images/xray.png", xray)

    print("Sample medical images created in 'sample_images/' directory")
    return ["sample_images/brain_mri.png", "sample_images/ct_scan.png", "sample_images/xray.png"]


def demonstrate_encryption_steps(image_path: str):
    """Demonstrate each step of the encryption process"""
    print(f"\n{'='*60}")
    print(f"DEMONSTRATING ENCRYPTION STEPS FOR: {os.path.basename(image_path)}")
    print(f"{'='*60}")

    # Initialize encryption system
    encryption_system = MedicalImageEncryption(seed=42)

    # Load original image
    original_image = encryption_system.load_medical_image(image_path)
    print(f"Original image shape: {original_image.shape}")

    # Step 1: Random bit modification
    print("\nStep 1: Random 3-bit modification...")
    blurred_image = encryption_system.step1_random_bit_modification(original_image)
    bit_change_ratio = np.mean(original_image != blurred_image)
    print(f"  - Pixels changed: {bit_change_ratio:.3f} ({bit_change_ratio*100:.1f}%)")

    # Step 2: Combine binary bits
    print("\nStep 2: Combining binary bits...")
    combined_bits = encryption_system.step2_combine_binary_bits(original_image, blurred_image)
    print(f"  - Combined bits shape: {combined_bits.shape}")
    print(f"  - Combined bits entropy: {calculate_entropy(combined_bits):.3f}")

    # Step 3: Generate SHA-512 keys
    print("\nStep 3: Generating SHA-512 keys...")
    sha512_keys = encryption_system.step3_generate_sha512_keys(combined_bits)
    print(f"  - Generated {len(sha512_keys)} keys")
    print(f"  - First key (truncated): {sha512_keys[0][:32]}...")

    # Step 4: DNA encoding and scrambling
    print("\nStep 4: DNA encoding and 3D Fisher-Yates scrambling...")
    scrambled_3d, scrambling_info = encryption_system.step4_dna_encoding_and_scrambling(
        blurred_image, sha512_keys
    )
    print(f"  - 3D scrambled shape: {scrambled_3d.shape}")
    print(f"  - Scrambling operations: {len(scrambling_info)}")

    # Step 5: Asymmetric DNA diffusion
    print("\nStep 5: Asymmetric DNA diffusion...")
    multiple_ciphers = encryption_system.step5_asymmetric_dna_diffusion(scrambled_3d, sha512_keys)
    print(f"  - Generated {len(multiple_ciphers)} different-sized ciphers")
    for i, cipher in enumerate(multiple_ciphers):
        print(f"    Cipher {i+1}: {cipher.shape}")

    # Step 6: Generate same-size ciphertext images
    print("\nStep 6: Generating same-size ciphertext images...")
    final_ciphers = encryption_system.step6_generate_same_size_ciphers(
        original_image, blurred_image, multiple_ciphers
    )
    print(f"  - Generated {len(final_ciphers)} final cipher images")
    print(f"  - All cipher shapes: {[cipher.shape for cipher in final_ciphers]}")

    return original_image, final_ciphers, encryption_system.encryption_metadata


def analyze_encryption_quality(original: np.ndarray, ciphers: list):
    """Analyze the quality of encryption"""
    print(f"\n{'='*40}")
    print("ENCRYPTION QUALITY ANALYSIS")
    print(f"{'='*40}")

    for i, cipher in enumerate(ciphers):
        print(f"\nCipher Image {i+1}:")

        # 1. Correlation with original
        correlation = np.corrcoef(original.flatten(), cipher.flatten())[0, 1]
        print(f"  - Correlation with original: {correlation:.6f}")

        # 2. Entropy
        entropy = calculate_entropy(cipher)
        print(f"  - Entropy: {entropy:.3f} (ideal: ~8.0)")

        # 3. Pixel change rate
        change_rate = np.mean(original != cipher)
        print(f"  - Pixel change rate: {change_rate:.3f} ({change_rate*100:.1f}%)")

        # 4. Mean and standard deviation
        print(f"  - Mean: {np.mean(cipher):.2f}, Std: {np.std(cipher):.2f}")

        # 5. Histogram uniformity
        hist, _ = np.histogram(cipher, bins=256, range=(0, 256))
        chi_square = np.sum((hist - np.mean(hist))**2 / np.mean(hist))
        print(f"  - Histogram uniformity (χ²): {chi_square:.2f}")


def calculate_entropy(image: np.ndarray) -> float:
    """Calculate Shannon entropy of an image"""
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    hist = hist[hist > 0]
    prob = hist / np.sum(hist)
    entropy = -np.sum(prob * np.log2(prob))
    return entropy


def visualize_encryption_results(original: np.ndarray, ciphers: list, image_name: str):
    """Create visualization of encryption results"""
    print(f"\nCreating visualization for {image_name}...")

    # Create subplot layout
    num_ciphers = len(ciphers)
    cols = min(4, num_ciphers + 1)
    rows = (num_ciphers + 1 + cols - 1) // cols

    plt.figure(figsize=(15, 4 * rows))

    # Original image
    plt.subplot(rows, cols, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Cipher images
    for i, cipher in enumerate(ciphers):
        plt.subplot(rows, cols, i + 2)
        plt.imshow(cipher, cmap='gray')
        plt.title(f'Cipher {i+1}')
        plt.axis('off')

    plt.tight_layout()

    # Save visualization
    os.makedirs("encryption_results", exist_ok=True)
    plt.savefig(f"encryption_results/{image_name}_encryption_results.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Create histogram comparison
    plt.figure(figsize=(15, 5))

    # Original histogram
    plt.subplot(1, num_ciphers + 1, 1)
    plt.hist(original.flatten(), bins=50, alpha=0.7, color='blue')
    plt.title('Original Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    # Cipher histograms
    for i, cipher in enumerate(ciphers):
        plt.subplot(1, num_ciphers + 1, i + 2)
        plt.hist(cipher.flatten(), bins=50, alpha=0.7, color='red')
        plt.title(f'Cipher {i+1} Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f"encryption_results/{image_name}_histograms.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  - Saved visualization: encryption_results/{image_name}_encryption_results.png")
    print(f"  - Saved histograms: encryption_results/{image_name}_histograms.png")


def save_encryption_metadata(metadata: dict, image_name: str):
    """Save encryption metadata to JSON file"""
    # Convert numpy arrays to lists for JSON serialization
    json_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            json_metadata[key] = value.tolist()
        elif isinstance(value, dict):
            json_metadata[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v
                                for k, v in value.items()}
        else:
            json_metadata[key] = value

    os.makedirs("encryption_metadata", exist_ok=True)
    with open(f"encryption_metadata/{image_name}_metadata.json", 'w') as f:
        json.dump(json_metadata, f, indent=2)

    print(f"  - Saved metadata: encryption_metadata/{image_name}_metadata.json")


def main():
    """Main demonstration function"""
    print("MEDICAL IMAGE ENCRYPTION SYSTEM - DEMONSTRATION")
    print("=" * 60)

    # Create sample medical images
    sample_images = create_sample_medical_images()

    # Add existing images if they exist
    existing_images = ["braincd.png", "ctscan.png"]
    for img_path in existing_images:
        if os.path.exists(img_path):
            sample_images.append(img_path)

    # Demonstrate encryption for each image
    for image_path in sample_images:
        try:
            image_name = os.path.splitext(os.path.basename(image_path))[0]

            # Run complete encryption demonstration
            original, ciphers, metadata = demonstrate_encryption_steps(image_path)

            # Analyze encryption quality
            analyze_encryption_quality(original, ciphers)

            # Create visualizations
            visualize_encryption_results(original, ciphers, image_name)

            # Save metadata
            save_encryption_metadata(metadata, image_name)

            # Save cipher images
            os.makedirs("cipher_outputs", exist_ok=True)
            for i, cipher in enumerate(ciphers):
                output_path = f"cipher_outputs/{image_name}_cipher_{i+1}.png"
                cv2.imwrite(output_path, cipher)

            print(f"  - Saved {len(ciphers)} cipher images to cipher_outputs/")

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    print(f"\n{'='*60}")
    print("DEMONSTRATION COMPLETED!")
    print("Check the following directories for results:")
    print("  - encryption_results/    : Visualizations")
    print("  - encryption_metadata/   : Encryption metadata")
    print("  - cipher_outputs/        : Encrypted images")
    print("  - sample_images/         : Sample medical images")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
