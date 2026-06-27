import os
import cv2
import numpy as np
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

def load_dataset(dataset_path, max_samples=250):
    X = []
    y = []
    root = Path(dataset_path)
    
    # Label: FM = 0, AM = 1
    classes = {'1. FM': 0, '2. AIRBAND': 1}
    
    for folder_name, label in classes.items():
        print(f"Membaca data untuk label: {folder_name}...")
        img_paths = list(root.rglob(f"{folder_name}/*.jpg"))
        
        # Shuffle paths to get a varied training set
        np.random.seed(42)
        np.random.shuffle(img_paths)
        
        count = 0
        for p in img_paths:
            if count >= max_samples:
                break
                
            img = cv2.imread(str(p))
            if img is None:
                continue
                
            # Crop area spektrum saja (menggunakan koordinat panel standar Anda)
            h, w = img.shape[:2]
            # Untuk menjaga kestabilan, jika tinggi gambar standar (1080x1920), crop koordinat panel 86 s/d 540
            if h == 1080 and w == 1920:
                crop = img[86:540, 0:w]
            else:
                crop = img[int(0.08*h):int(0.5*h), 0:w]
            
            # Resize dan Grayscale
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (128, 128))
            
            X.append(resized)
            y.append(label)
            count += 1
            
    X = np.array(X, dtype='float32') / 255.0  # Normalisasi
    X = np.expand_dims(X, axis=-1)            # Reshape ke (N, 128, 128, 1)
    y = np.array(y)
    return X, y

def main():
    dataset_dir = Path(__file__).parent / 'DATASET_GAMBAR'
    if not dataset_dir.exists():
        print(f"Dataset tidak ditemukan di: {dataset_dir}")
        return
        
    print("Memulai pemrosesan dataset...")
    X, y = load_dataset(dataset_dir, max_samples=250)
    print(f"Dataset berhasil dimuat. Total data: {len(X)}")
    
    # Split 80% train, 20% validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Definisikan Arsitektur CNN Ringan
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Output binary (Sigmoid)
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n--- Ringkasan Model CNN ---")
    model.summary()
    
    print("\nMemulai pelatihan CNN (10 Epochs)...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=16,
        validation_data=(X_val, y_val)
    )
    
    # Simpan model
    model_path = Path(__file__).parent / 'cnn_signal_model.h5'
    model.save(str(model_path))
    print(f"\nModel CNN sukses disimpan di: {model_path}")

if __name__ == '__main__':
    main()
