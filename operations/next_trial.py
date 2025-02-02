import cv2
import numpy as np
from PIL import Image as PIL, ImageOps
from pdf417decoder import PDF417Decoder
import os

def preprocess_image(image_path):
    # Read image using OpenCV
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance contrast by applying histogram equalization
    equalized_image = cv2.equalizeHist(gray_image)

    # Apply GaussianBlur to reduce noise
    blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)

    # Apply binary thresholding to make the barcode stand out
    _, binarized_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Optionally, invert the binary image (white on black) for better recognition
    binarized_image = cv2.bitwise_not(binarized_image)

    # Convert back to PIL Image
    pil_image = PIL.fromarray(binarized_image)

    return pil_image

def decode_pdf417(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
        
    try:
        # Preprocess image before decoding
        pil_image = preprocess_image(image_path)
        
        # Initialize PDF417 decoder
        decoder = PDF417Decoder(pil_image)
        
        # Decode barcode
        decode_result = decoder.decode()
        
        if decode_result > 0:
            decoded_data = decoder.barcode_data_index_to_string(0)
            return decoded_data
        else:
            return "No barcode detected"
            
    except PIL.UnidentifiedImageError:
        raise ValueError("Invalid image format")
    except Exception as e:
        raise RuntimeError(f"Decoding error: {str(e)}")

if __name__ == "__main__":
    try:
        result = decode_pdf417("image1_pdf417.jpg")
        print(f"Decoded Data: \n{result}")
    except Exception as e:
        print(f"Error: {str(e)}")
