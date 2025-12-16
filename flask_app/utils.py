
import numpy as np
from PIL import Image
import io
import base64

def preprocess_image(image_file, target_size=(128, 128)):
    """
    Load and preprocess an image for the model.
    """
    try:
        # Open image
        img = Image.open(image_file)
        
        # Convert to RGB (in case of RGBA/Grayscale)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Resize
        img = img.resize(target_size)
        
        # Convert to array and normalize (0-1 range)
        img_array = np.array(img) / 255.0
        
        # Add batch dimension (1, 128, 128, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None



def generate_fft_visualization(image_file, size=128):
    """
    Generate FFT magnitude spectrum visualization.
    GANs often show grid-like patterns in FFT.
    """
    try:
        image_file.seek(0)  # Reset file pointer
        img = Image.open(image_file).convert('L')  # Convert to grayscale
        img = img.resize((size, size))
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Apply FFT
        f_transform = np.fft.fft2(img_array)
        f_shift = np.fft.fftshift(f_transform)  # Center the spectrum
        
        # Get magnitude spectrum (log scale for better visualization)
        magnitude = np.abs(f_shift)
        magnitude = np.log1p(magnitude)  # log(1 + x) to handle zeros
        
        # Normalize to 0-255 range
        magnitude = ((magnitude - magnitude.min()) / (magnitude.max() - magnitude.min()) * 255).astype(np.uint8)
        
        # Convert to PIL Image and then to base64
        fft_img = Image.fromarray(magnitude)
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        fft_img.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Encode as base64
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            "success": True,
            "image": f"data:image/png;base64,{img_base64}",
            "description": "FFT magnitude spectrum - GANs may show grid patterns"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "description": "Could not generate FFT"
        }
