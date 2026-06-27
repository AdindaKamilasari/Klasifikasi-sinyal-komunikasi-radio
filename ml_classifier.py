import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import signal_processor as sp

MODEL_PATH = Path(__file__).parent / 'signal_model.pkl'

def extract_image_features(image_path):
    """
    Extract feature vector from a single image.
    Returns: (features_dict, success)
    """
    try:
        res = sp.analyze_image(image_path)
        if res['signals']:
            # Take the strongest signal
            sig = res['signals'][0]
            features = {
                'bandwidth_px': sig['bandwidth_px'],
                'variance': sig['variance'],
                'snr_db': sig['snr_db'],
                'peak_db': sig['peak_db']
            }
            return features, True
    except Exception as e:
        pass
    return None, False

def build_and_train_model(dataset_root=None, max_samples_per_class=25):
    """
    Scan a subset of images from DATASET_GAMBAR:
    - 1. FM folder -> label 'FM'
    - 2. AIRBAND folder -> label 'AM' (Airband uses AM modulation)
    Train a Random Forest classifier and save it.
    """
    if dataset_root is None:
        # Try relative to the script location
        dataset_root = Path(__file__).parent / 'DATASET_GAMBAR'
        if not dataset_root.exists():
            dataset_root = Path(r'c:\TA BARUUUUUUU\DATASET_GAMBAR')
            
    root = Path(dataset_root)
    if not root.exists():
        print(f"Dataset root {dataset_root} not found. Training skipped.")
        return None

    data = []
    
    # We will look for images in both FM and AIRBAND folders
    classes = {
        '1. FM': 'FM',
        '2. AIRBAND': 'AM'
    }
    
    print("Gathering training samples...")
    for folder_name, label in classes.items():
        count = 0
        # Search recursively for target folder
        for p in root.rglob(folder_name):
            if p.is_dir():
                # Find all images
                images = sorted(img for img in p.iterdir() if img.suffix.lower() in ('.jpg', '.jpeg', '.png'))
                for img in images:
                    if count >= max_samples_per_class:
                        break
                    features, success = extract_image_features(img)
                    if success:
                        features['label'] = label
                        data.append(features)
                        count += 1
                        print(f"  [{label}] Extracted features from {img.name}")
                if count >= max_samples_per_class:
                    break
                    
    if len(data) < 5:
        print("Not enough training data found. Using fallback model.")
        return None
        
    df = pd.DataFrame(data)
    X = df[['bandwidth_px', 'variance', 'snr_db', 'peak_db']]
    y = df['label']
    
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y)
    
    # Save the model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)
        
    print(f"Successfully trained Random Forest Classifier and saved to {MODEL_PATH}")
    return clf

def load_or_train_model():
    """
    Load the trained model, or train a new one if it doesn't exist.
    """
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
            
    # Try to train a new one
    clf = build_and_train_model()
    return clf

def predict_signal_type(bandwidth_px, variance, snr_db, peak_db, clf=None):
    """
    Predict signal modulation type (AM vs FM) using the trained classifier.
    If no model is available, uses a robust rule-based fallback based on bandwidth.
    """
    if clf is None:
        clf = load_or_train_model()
        
    if clf is not None:
        try:
            X = pd.DataFrame([{
                'bandwidth_px': bandwidth_px,
                'variance': variance,
                'snr_db': snr_db,
                'peak_db': peak_db
            }])
            return clf.predict(X)[0]
        except Exception:
            pass
            
    # --- Robust Rule-based Fallback ---
    # FM signals have wide bandwidths (typically > 35 pixels on these screens).
    # AM/Airband signals have very narrow bandwidths (typically < 30 pixels).
    if bandwidth_px > 35:
        return 'FM'
    else:
        return 'AM'

if __name__ == '__main__':
    # Train the model if run directly
    build_and_train_model()
