
import os
import random
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, send_file
from utils import preprocess_image, generate_fft_visualization
import numpy as np

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

# Load Model (Global load to avoid reloading on every request)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'deepfake_model.keras')
print(f"Loading model from: {MODEL_PATH}")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    try:
        # Preprocess for model
        processed_img = preprocess_image(file)
        
        if processed_img is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        
        # Generate FFT visualization
        fft_data = generate_fft_visualization(file)
            
        # Predict
        prediction = model.predict(processed_img)[0][0]
        
        confidence = float(prediction)
        
        # VERIFIED with actual test:
        # - FAKE images predict HIGH values (~0.99)
        # - REAL images predict LOW values (~0.001)
        # So: prediction > 0.5 means FAKE, prediction < 0.5 means REAL
        
        label = "FAKE" if confidence > 0.5 else "REAL"
        
        # Adjust confidence score to be "confidence of predicted class"
        display_confidence = confidence if label == "FAKE" else 1 - confidence
        
        return jsonify({
            'label': label,
            'confidence': f"{display_confidence * 100:.2f}%",
            'raw_score': confidence,
            'fft': fft_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/sample/<image_type>')
def sample_image(image_type):
    """Serve sample images from test dataset"""
    base_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'test')
    
    if image_type == 'real':
        folder = os.path.join(base_path, 'REAL')
    elif image_type == 'fake':
        folder = os.path.join(base_path, 'FAKE')
    else:
        return jsonify({'error': 'Invalid image type'}), 400
    
    try:
        images = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))]
        if not images:
            return jsonify({'error': 'No sample images found'}), 404
        
        selected = random.choice(images)
        return send_file(os.path.join(folder, selected), mimetype='image/jpeg')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

