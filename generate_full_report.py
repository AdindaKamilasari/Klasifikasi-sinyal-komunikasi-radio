import os
import sys
import csv
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(r'c:\TA BARUUUUUUU')
import signal_processor as sp
import ml_classifier as ml

# Helper function for amplifier recommendations (same logic as app.py)
def get_amplifier_rec_type(signal_type, peak_db, snr_db):
    if snr_db is None or np.isnan(snr_db):
        return "Tidak Terdeteksi (N/A)"
    
    if snr_db > 29:
        return "Tidak Memerlukan Penguat (Bypass)"
        
    if signal_type == 'FM':
        if peak_db < -60:
            if snr_db >= 12:
                return "Low Noise Amplifier (LNA) Khusus VHF/FM"
            else:
                return "VHF Bandpass Filter + LNA"
        else:
            return "Penguat High-Linearity / Attenuator Pasif"
    else: # AM / Airband
        if peak_db < -60:
            if snr_db >= 12:
                return "Low Noise Amplifier (LNA) HF/VHF"
            else:
                return "Tuned Bandpass Filter (Preselector) + LNA"
        else:
            return "Filter Pasif / Bypass"

def process_image(img_path):
    img_path = Path(img_path)
    image_name = img_path.name
    
    # Parse path to get ground-truth labels
    # Path: DATASET_GAMBAR / <wilayah> / <jarak> / <folder_kategori> / <name>
    parts = img_path.parts
    try:
        idx = parts.index('DATASET_GAMBAR')
        wilayah = parts[idx + 1].lower()
        jarak = parts[idx + 2].lower()
        cat_folder = parts[idx + 3]
    except Exception:
        wilayah = "unknown"
        jarak = "unknown"
        cat_folder = "unknown"
        
    if "1. FM" in cat_folder:
        signal_type = "FM"
    elif "2. AIRBAND" in cat_folder:
        signal_type = "AM"
    else:
        signal_type = "ETC"
        
    # Analyze using signal processor
    try:
        res = sp.analyze_image(img_path)
        if res['signals']:
            sig = res['signals'][0]
            predicted_peak = sig['peak_db']
            noise_floor = res['noise_floor']
            snr = sig['snr_db']
            condition = sig['category']
            
            # Predict type using trained classifier
            pred_class = ml.predict_signal_type(
                sig['bandwidth_px'], 
                sig['variance'], 
                sig['snr_db'], 
                sig['peak_db']
            )
            predicted_type = pred_class
            rec = get_amplifier_rec_type(predicted_type, sig['peak_db'], sig['snr_db'])
        else:
            predicted_type = "Tidak Terdeteksi"
            predicted_peak = None
            noise_floor = res.get('noise_floor', -50.0)
            snr = 0.0
            condition = "Not detected"
            rec = "Tidak Terdeteksi (N/A)"
    except Exception:
        predicted_type = "Error"
        predicted_peak = None
        noise_floor = -50.0
        snr = 0.0
        condition = "Error"
        rec = "Error"
        
    return {
        'image_name': image_name,
        'wilayah': wilayah,
        'jarak': jarak,
        'signal_type': signal_type,
        'predicted_type': predicted_type,
        'predicted_peak_dbfs': predicted_peak,
        'noise_floor': noise_floor,
        'snr': snr,
        'signal_condition': condition,
        'recommended_amplifier': rec
    }

def main():
    dataset_root = Path(r'c:\TA BARUUUUUUU\DATASET_GAMBAR')
    if not dataset_root.exists():
        print("Dataset folder not found!")
        return
        
    # Find all images
    print("Scanning dataset directory...")
    img_paths = sorted(
        p for p in dataset_root.rglob('*') 
        if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png')
    )
    total_images = len(img_paths)
    print(f"Found {total_images} images to process.")
    
    results = []
    processed_count = 0
    
    # Process in parallel using ThreadPoolExecutor
    print("Processing images in parallel...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_image, p): p for p in img_paths}
        for future in as_completed(futures):
            try:
                data = future.result()
                results.append(data)
            except Exception as e:
                path = futures[future]
                print(f"Failed to process {path.name}: {e}")
                
            processed_count += 1
            if processed_count % 100 == 0 or processed_count == total_images:
                print(f"Progress: {processed_count}/{total_images} images processed.")
                
    # Sort results for clean output (by wilayah, jarak, image_name)
    results = sorted(results, key=lambda x: (x['wilayah'], x['jarak'], x['image_name']))
    
    # Write to CSV
    output_path = Path(r'c:\TA BARUUUUUUU\hasil_analisis_keseluruhan.csv')
    fieldnames = [
        'image_name', 'wilayah', 'jarak', 'signal_type', 'predicted_type', 
        'predicted_peak_dbfs', 'noise_floor', 'snr', 'signal_condition', 'recommended_amplifier'
    ]
    
    print(f"Writing results to {output_path}...")
    with open(output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
            
    print("All tasks completed successfully!")

if __name__ == '__main__':
    main()
