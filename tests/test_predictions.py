import tensorflow as tf
import numpy as np
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

print("Loading model...")
model = tf.keras.models.load_model('models/XceptionTransfer_best.keras')
print("Model loaded!\n")

# Test on FAKE images
fake_dir = 'data/raw/test/FAKE'
fake_files = os.listdir(fake_dir)[:5]
print('=== Images from FAKE folder ===')
fake_preds = []
for f in fake_files:
    img = Image.open(os.path.join(fake_dir, f)).convert('RGB').resize((128, 128))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr, verbose=0)[0][0]
    fake_preds.append(pred)
    print(f"  {f}: prediction = {pred:.6f}")

# Test on REAL images 
real_dir = 'data/raw/test/REAL'
real_files = os.listdir(real_dir)[:5]
print('\n=== Images from REAL folder ===')
real_preds = []
for f in real_files:
    img = Image.open(os.path.join(real_dir, f)).convert('RGB').resize((128, 128))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr, verbose=0)[0][0]
    real_preds.append(pred)
    print(f"  {f}: prediction = {pred:.6f}")

print('\n' + '='*50)
print('SUMMARY:')
print(f'  FAKE folder images avg prediction: {np.mean(fake_preds):.6f}')
print(f'  REAL folder images avg prediction: {np.mean(real_preds):.6f}')
print('='*50)
print('\nIf FAKE avg is HIGH (>0.5) and REAL avg is LOW (<0.5):')
print('  -> Model is correct, FAKE=1, REAL=0')
print('If FAKE avg is LOW (<0.5) and REAL avg is HIGH (>0.5):')
print('  -> Labels are inverted, need to flip logic')
