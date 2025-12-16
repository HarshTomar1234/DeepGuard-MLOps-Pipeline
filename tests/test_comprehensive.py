"""
Comprehensive model evaluation on test dataset
to understand prediction consistency and accuracy
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
    """Load model with error handling."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return tf.keras.models.load_model(model_path)


def test_images_in_directory(model, directory, expected_label, num_images=50):
    """
    Test images in a directory and return results.
    
    Args:
        model: Loaded Keras model
        directory: Path to image directory
        expected_label: 'FAKE' (expects < 0.5) or 'REAL' (expects > 0.5)
        num_images: Number of images to test
        
    Returns:
        dict with correct, incorrect counts and predictions
    """
    if not os.path.exists(directory):
        print(f"  Warning: Directory not found: {directory}")
        return {"correct": 0, "incorrect": 0, "predictions": [], "errors": 1}
    
    files = sorted(os.listdir(directory))[:num_images]  # Sort for deterministic ordering
    
    if not files:
        print(f"  Warning: No files found in {directory}")
        return {"correct": 0, "incorrect": 0, "predictions": [], "errors": 1}
    
    correct = 0
    incorrect = 0
    predictions = []
    
    for f in files:
        try:
            img_path = os.path.join(directory, f)
            img = Image.open(img_path).convert('RGB').resize((128, 128))
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = np.expand_dims(arr, axis=0)
            pred = float(model.predict(arr, verbose=0)[0][0])
            predictions.append(pred)
            
            if expected_label == 'FAKE':
                is_correct = pred < 0.5
            else:  # REAL
                is_correct = pred > 0.5
            
            if is_correct:
                correct += 1
            else:
                incorrect += 1
                threshold = "< 0.5" if expected_label == 'FAKE' else "> 0.5"
                print(f"  WRONG: {f} -> {pred:.4f} (should be {threshold})")
                
        except Exception as e:
            print(f"  Error loading {f}: {e}")
    
    return {"correct": correct, "incorrect": incorrect, "predictions": predictions, "errors": 0}


def main():
    """Run comprehensive model tests."""
    print("Loading model from flask_app/model/deepfake_model.keras...")
    
    try:
        model = load_model()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return
    
    print("Model loaded!\n")
    
    # Test FAKE images
    fake_dir = 'data/raw/test/FAKE'
    print("=" * 60)
    print("Testing 50 FAKE images (should all predict < 0.5 for correct)")
    print("=" * 60)
    
    fake_results = test_images_in_directory(model, fake_dir, 'FAKE', 50)
    
    if fake_results["predictions"]:
        print(f"\nFAKE images: {fake_results['correct']}/50 correct, {fake_results['incorrect']}/50 incorrect")
        print(f"FAKE predictions - min: {min(fake_results['predictions']):.6f}, "
              f"max: {max(fake_results['predictions']):.6f}, "
              f"avg: {np.mean(fake_results['predictions']):.6f}")
    
    # Test REAL images
    real_dir = 'data/raw/test/REAL'
    print("\n" + "=" * 60)
    print("Testing 50 REAL images (should all predict > 0.5 for correct)")
    print("=" * 60)
    
    real_results = test_images_in_directory(model, real_dir, 'REAL', 50)
    
    if real_results["predictions"]:
        print(f"\nREAL images: {real_results['correct']}/50 correct, {real_results['incorrect']}/50 incorrect")
        print(f"REAL predictions - min: {min(real_results['predictions']):.6f}, "
              f"max: {max(real_results['predictions']):.6f}, "
              f"avg: {np.mean(real_results['predictions']):.6f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_correct = fake_results['correct'] + real_results['correct']
    total_images = len(fake_results['predictions']) + len(real_results['predictions'])
    
    if total_images > 0:
        accuracy = total_correct / total_images * 100
        print(f"Overall Accuracy: {total_correct}/{total_images} = {accuracy:.1f}%")
        print(f"\nIf accuracy is around 88%, model is working as expected (matches training metrics)")
        print(f"If accuracy is random (~50%), there's a preprocessing or model loading issue")
    else:
        print("No images were tested. Check that data directories exist.")


if __name__ == "__main__":
    main()
