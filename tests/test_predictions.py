"""
Quick test to verify model predictions on sample images.
"""
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')


def load_model(model_path='flask_app/model/deepfake_model.keras'):
fake_files = sorted(os.listdir(fake_dir))[:5]    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return tf.keras.models.load_model(model_path)


def predict_images(model, directory, num_images=5):
    """
    Make predictions on images in a directory.
    Returns list of predictions with filenames.
    """
    if not os.path.exists(directory):
        print(f"  Warning: Directory not found: {directory}")
        return []
    
    files = sorted(os.listdir(directory))[:num_images]  # Sort for deterministic ordering
    
    if not files:
        print(f"  Warning: No files found in {directory}")
        return []
    
    predictions = []
    for f in files:
        try:
            img_path = os.path.join(directory, f)
            img = Image.open(img_path).convert('RGB').resize((128, 128))
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = np.expand_dims(arr, axis=0)
            pred = float(model.predict(arr, verbose=0)[0][0])
            predictions.append((f, pred))
            print(f"  {f}: prediction = {pred:.6f}")
        except Exception as e:
            print(f"  Error loading {f}: {e}")
    
    return predictions


def main():
    """Run prediction tests."""
    print("Loading model...")
    
    try:
        model = load_model()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return
    
    print("Model loaded!\n")
    
    # Test on FAKE images
    fake_dir = 'data/raw/test/FAKE'
    print('=== Images from FAKE folder ===')
    fake_results = predict_images(model, fake_dir, 5)
    
    # Test on REAL images
    real_dir = 'data/raw/test/REAL'
    print('\n=== Images from REAL folder ===')
    real_results = predict_images(model, real_dir, 5)
    
    # Summary
    print('\n' + '=' * 50)
    print('SUMMARY:')
    
    if fake_results:
        fake_preds = [p for _, p in fake_results]
        print(f'  FAKE folder images avg prediction: {np.mean(fake_preds):.6f}')
    else:
        print('  FAKE folder: No predictions available')
    
    if real_results:
        real_preds = [p for _, p in real_results]
        print(f'  REAL folder images avg prediction: {np.mean(real_preds):.6f}')
    else:
        print('  REAL folder: No predictions available')
    
    print('=' * 50)
    print('\nIf FAKE avg is HIGH (>0.5) and REAL avg is LOW (<0.5):')
    print('  -> Model is correct, FAKE=1, REAL=0')
    print('If FAKE avg is LOW (<0.5) and REAL avg is HIGH (>0.5):')
    print('  -> Labels are inverted, need to flip logic')


if __name__ == "__main__":
    main()
