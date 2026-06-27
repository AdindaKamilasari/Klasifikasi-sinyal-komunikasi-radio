import os
import cv2
import numpy as np
from pathlib import Path

# Disable TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import load_model

_CNN_MODEL = None

def get_cnn_model():
    global _CNN_MODEL
    if _CNN_MODEL is None:
        model_path = Path(__file__).parent / 'cnn_signal_model.h5'
        if model_path.exists():
            _CNN_MODEL = load_model(str(model_path))
        else:
            _CNN_MODEL = None
    return _CNN_MODEL

def predict_signal_type(img_bgr, panel):
    """
    Predict signal type (AM or FM) using the trained CNN model.
    """
    model = get_cnn_model()
    if model is None:
        # Fallback to simple bandwidth heuristic if CNN model is not found
        # (This is important for initial state before train_cnn.py is run)
        print("Warning: cnn_signal_model.h5 not found. Using fallback heuristic.")
        # If we can estimate bandwidth (using signal_processor logic)
        # For simplicity, default to FM or use name heuristic
        return "FM", 0.50
        
    try:
        x0, y0, x1, y1 = panel
        h, w = img_bgr.shape[:2]
        
        # Crop the spectrum panel
        crop = img_bgr[y0:y1, 0:w]
        
        # Grayscale and Resize to 128x128
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128, 128))
        
        # Normalize and Reshape to (1, 128, 128, 1)
        normalized = resized.astype('float32') / 255.0
        input_data = np.expand_dims(normalized, axis=(0, -1))
        
        # Predict probability (Sigmoid output: 0=FM, 1=AM)
        prob = float(model.predict(input_data, verbose=0)[0][0])
        
        if prob < 0.5:
            confidence = (1.0 - prob) * 100.0
            return "FM", confidence
        else:
            confidence = prob * 100.0
            return "AM", confidence
            
    except Exception as e:
        print(f"Error in CNN prediction: {e}")
        return "FM", 50.0

if __name__ == '__main__':
    # Test fallback
    print("Testing cnn_classifier.py...")
    test_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    label, conf = predict_signal_type(test_img, (0, 86, 1587, 540))
    print(f"Predicted class: {label} (Confidence: {conf:.2f}%)")
