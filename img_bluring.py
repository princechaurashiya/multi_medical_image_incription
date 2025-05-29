import random
import hashlib
# Import NumPy for image array manipulation.
# If you don't have it, install it using: pip install numpy
# For actual image file loading/saving, you'd also use a library like OpenCV: pip install opencv-python
import numpy as np

# --- Helper Functions for Bit Manipulation ---

def get_bit(value, bit_index):
    """
    Gets the bit at a specific index (0-indexed from LSB).
    For an 8-bit number:
    - bit 0 is the 1st bit (LSB)
    - bit 6 is the 7th bit
    - bit 7 is the 8th bit (MSB)
    """
    return (value >> bit_index) & 1

def set_bit_to_value(value, bit_index, new_bit_value):
    if new_bit_value not in (0, 1):
        raise ValueError("New bit value must be 0 or 1.")
    if new_bit_value == 1:
        return value | (1 << bit_index)
    else:
        mask = 0xFF ^ (1 << bit_index)
        return value & mask
# Set bit to 0 using AND with inverted mask

# --- Core Function for Blurring Specific Bits ---

def randomly_change_specified_bits(pixel_value):
    """
    Randomly changes the 1st, 7th, and 8th bits of an 8-bit pixel_value.
    This simulates a "salt and pepper" attack on these specific bit-planes,
    by randomly setting them to 0 or 1.
    """
    if not (0 <= pixel_value <= 255):
        # Assuming 8-bit pixel values for grayscale images
        raise ValueError("Pixel value must be between 0 and 255 for 8-bit representation.")

    modified_value = pixel_value

    # Bit indices to change (0-indexed from LSB):
    # 1st bit corresponds to index 0
    # 7th bit corresponds to index 6
    # 8th bit corresponds to index 7
    bits_to_change_indices = [0, 6, 7]

    for bit_index in bits_to_change_indices:
        # Randomly decide the new value for the bit (0 or 1)
        new_random_bit = random.randint(0, 1)
        modified_value = set_bit_to_value(modified_value, bit_index, new_random_bit)
        # print(f"  Changing bit {bit_index}: original bit {get_bit(pixel_value, bit_index)}, new random bit {new_random_bit}")


    return modified_value

# --- Main Application Logic ---

def process_image_data(image_data_array):
    """
    Applies the random bit changing function to each pixel in an image data array.
    Args:
        image_data_array (numpy.ndarray): A NumPy array representing the image (e.g., grayscale).
    Returns:
        numpy.ndarray: A new NumPy array with the specified bits randomly changed.
    """
    if not isinstance(image_data_array, np.ndarray):
        raise TypeError("Input image_data_array must be a NumPy array.")

    # Create a copy to avoid modifying the original array directly
    blurred_image_array = np.copy(image_data_array)

    # Iterate over each pixel in the image array
    # For a 2D grayscale image:
    if blurred_image_array.ndim == 2:
        for i in range(blurred_image_array.shape[0]):  # Iterate over rows
            for j in range(blurred_image_array.shape[1]):  # Iterate over columns
                original_pixel = blurred_image_array[i, j]
                blurred_image_array[i, j] = randomly_change_specified_bits(original_pixel)
    # For a 3D color image (e.g., RGB), you might apply it to each channel or just specific ones.
    # This example assumes grayscale as medical images are often grayscale.
    # If applying to color, you'd iterate through channels as well.
    # Example for RGB (assuming channel is the last dimension):
    # elif blurred_image_array.ndim == 3:
    #     for i in range(blurred_image_array.shape[0]):
    #         for j in range(blurred_image_array.shape[1]):
    #             for k in range(blurred_image_array.shape[2]): # Iterate over channels
    #                 original_pixel = blurred_image_array[i, j, k]
    #                 blurred_image_array[i, j, k] = randomly_change_specified_bits(original_pixel)
    else:
        raise ValueError("Image data array must be 2D (grayscale) or 3D (color).")


    return blurred_image_array

# --- Example Usage ---

if __name__ == "__main__":
    # 1. Example with a single pixel value
    print("--- Single Pixel Example ---")
    original_pixel_val = 150  # Binary: 10010110
    print(f"Original Pixel Value: {original_pixel_val} (Binary: {original_pixel_val:08b})")

    # Show bits to be changed
    print(f"  Bit 0 (1st bit): {get_bit(original_pixel_val, 0)}")
    print(f"  Bit 6 (7th bit): {get_bit(original_pixel_val, 6)}")
    print(f"  Bit 7 (8th bit): {get_bit(original_pixel_val, 7)}")


    blurred_pixel_val = randomly_change_specified_bits(original_pixel_val)
    print(f"Blurred Pixel Value:  {blurred_pixel_val} (Binary: {blurred_pixel_val:08b})")
    print("-" * 30)

    # 2. Example with a dummy image (NumPy array)
    print("\n--- Image Array Example ---")
    # Create a small dummy grayscale image (2x3 pixels) with uint8 data type
    # Values are between 0-255
    dummy_image = np.array([[100, 150, 200],
                            [50, 75, 125]], dtype=np.uint8)

    print("Original Dummy Image Data:")
    print(dummy_image)
    print("Binary Representation (Original):")
    for row in dummy_image:
        print([f"{pixel:08b}" for pixel in row])

    # Process the dummy image
    blurred_dummy_image = process_image_data(dummy_image)

    print("\nBlurred Dummy Image Data:")
    print(blurred_dummy_image)
    print("Binary Representation (Blurred):")
    for row in blurred_dummy_image:
        print([f"{pixel:08b}" for pixel in row])
    print("-" * 30)

    # 3. How to use with a real image (requires OpenCV)
    print("\n--- Real Image Processing (Conceptual - Requires OpenCV) ---")
    print("To process a real image file:")
    print("1. Make sure you have OpenCV installed: pip install opencv-python")
    print("2. Uncomment and adapt the following conceptual code:")

    try:
        import cv2 # OpenCV library

        # Load a grayscale image
        # Replace 'your_medical_image.png' with the actual path to your image
        image_path = './braincd.png' # or .jpg, .pgm, .dicom (might need pydicom for DICOM)
        original_cv_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if original_cv_image is None:
            print(f"Error: Could not load image from {image_path}")
        else:
            print(f"Successfully loaded image: {image_path} with shape {original_cv_image.shape}")

            # Process the loaded image
            blurred_cv_image = process_image_data(original_cv_image)

            #Display images (optional)
            cv2.imshow('Original Medical Image', original_cv_image)
            cv2.imshow('Blurred Medical Image (Bits Changed)', blurred_cv_image)
            cv2.waitKey(0) # Wait for a key press to close windows
            cv2.destroyAllWindows()

            #Save the blurred image
            output_path = 'blurred_medical_image.png'
            cv2.imwrite(output_path, blurred_cv_image)
            print(f"Blurred image saved to {output_path}")

    except ImportError:
        print("OpenCV (cv2) is not installed. Please install it to run real image processing.")
    except Exception as e:
        print(f"An error occurred during real image processing: {e}")

        #source venv/bin/activate

