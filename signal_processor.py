import cv2
import numpy as np
import re
import os
from pathlib import Path
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.signal import find_peaks

# --- Tesseract Configuration ---
import pytesseract

def configure_tesseract(user_path=None):
    """
    Configure pytesseract path dynamically by checking standard locations or user path.
    """
    if user_path and Path(user_path).exists():
        pytesseract.pytesseract.tesseract_cmd = str(user_path)
        return True
    
    # Common paths on Windows
    common_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%USERPROFILE%\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe")
    ]
    for p in common_paths:
        if Path(p).exists():
            pytesseract.pytesseract.tesseract_cmd = p
            return True
            
    # Try running directly (hoping it's in PATH)
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

# Run initial search
HAS_TESSERACT = configure_tesseract()

# --- Constant Parameters ---
DB_TOP_DEFAULT = -20
DB_BOTTOM_DEFAULT = -90

def classify_tiphon(snr_db):
    if snr_db is None or np.isnan(snr_db): 
        return 'Not detected'
    if snr_db > 29: 
        return 'Outstanding'
    if snr_db >= 20.0: 
        return 'Excellent'
    if snr_db >= 12.0:  # Using the updated cell 9 threshold values
        return 'Good'
    if snr_db >= 6.0: 
        return 'Fair'
    return 'Bad'

def detect_spectrum_panel(img_bgr):
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    dark = gray < 80
    
    # If the image is already cropped to just the dark spectrum (dark pixels > 85%), return full dimensions
    if dark.mean() > 0.85:
        return (0, 0, w, h)
        
    row_frac = dark[:, :int(w * 0.82)].mean(axis=1)
    row_mask = row_frac > 0.55
    row_mask[:int(0.08 * h)] = False
    
    ranges, i = [], 0
    while i < h:
        if row_mask[i]:
            s = i
            while i < h and row_mask[i]: i += 1
            if i - 1 - s >= 5: ranges.append([s, i - 1])
        else: 
            i += 1
            
    merged = []
    for s, e in ranges:
        if not merged or s - merged[-1][1] > 25: 
            merged.append([s, e])
        else: 
            merged[-1][1] = e
            
    candidates = [r for r in merged if r[1] - r[0] > 150 and r[0] < 0.35 * h and r[1] < 0.85 * h]
    if not candidates: 
        candidates = [r for r in merged if r[1] - r[0] > 50]
    if not candidates: 
        raise RuntimeError('Spectrum panel not detected.')
        
    y0, y1 = max(candidates, key=lambda r: r[1] - r[0])
    col_frac = dark[y0:y1, :].mean(axis=0)
    
    # Check if the dark spectrum extends all the way to the right (no settings sidebar)
    if np.mean(col_frac[int(w * 0.85):]) > 0.65:
        x1 = w - 2
    else:
        x1 = int(w * 0.78)
        for x in range(int(w * 0.55), w - 40):
            if np.mean(col_frac[x:x + 40]) < 0.25:
                x1 = x - 1
                break
                
    return (0, int(y0), int(x1), int(y1))

def ocr_axis_values(img_bgr, panel):
    if not HAS_TESSERACT: 
        return [], ''
    x0, y0, x1, y1 = panel
    w = img_bgr.shape[1]
    crop = img_bgr[y0:y1, 0:max(60, int(w * 0.04))]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    th = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
    th = cv2.resize(th, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    try:
        text = pytesseract.image_to_string(th, config='--psm 6 -c tessedit_char_whitelist=-0123456789')
    except Exception: 
        return [], ''
    vals = [int(m) for m in re.findall(r'-\d+', text)]
    vals = [v for v in vals if -150 <= v <= -1 and v % 5 == 0]
    return vals, text

def estimate_tick_step(vals):
    vals = sorted(set(vals), reverse=True)
    diffs = []
    for a, b in zip(vals, vals[1:]):
        d = abs(a - b)
        if 3 <= d <= 25:
            d5 = int(round(d / 5) * 5)
            if d5 in (5, 10, 15, 20, 25): 
                diffs.append(d5)
    if diffs: 
        m = min(diffs)
        if m in (5, 10):
            return m
        return 10
    return 10

def detect_grid_axis_y(img_bgr, panel):
    x0, y0, x1, y1 = panel
    w = img_bgr.shape[1]
    left_margin = max(45, int(w * 0.027))
    crop = img_bgr[y0:y1, left_margin:x1 - 5]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    med = np.median(gray, axis=1).astype(float)
    sm = gaussian_filter1d(med, 1)
    bg = median_filter(sm, size=31)
    score = sm - bg
    peaks, _ = find_peaks(score, distance=20, prominence=0.5)
    peaks = peaks + y0
    candidates = [int(p) for p in peaks if p > y0 + 0.10 * (y1 - y0) and p < y1 - 0.05 * (y1 - y0)]
    if len(candidates) >= 2: 
        return min(candidates), max(candidates), candidates
    return int(y0 + 2), int(y1 - 2), candidates

def calibrate_db_axis(img_bgr, panel):
    y_top, y_bottom, grid = detect_grid_axis_y(img_bgr, panel)
    vals, _ = ocr_axis_values(img_bgr, panel)
    if len(vals) == 0:
        return DB_TOP_DEFAULT, DB_BOTTOM_DEFAULT, y_top, y_bottom, grid, 'fallback default'
    db_top = max(vals)
    tick_step = estimate_tick_step(vals)
    grid_in_range = sorted([p for p in grid if y_top - 2 <= p <= y_bottom + 2])
    
    if len(grid_in_range) >= 3:
        diffs = np.diff(grid_in_range)
        med = np.median(diffs)
        normal_diffs = [d for d in diffs if 0.5 * med < d < 1.5 * med]
        spacing = np.median(normal_diffs) if normal_diffs else med
        
        # Sanity check: spacing in Gqrx is usually between 35 and 120 pixels
        if 35 <= spacing <= 120:
            n_intervals = max(1, int(round((y_bottom - y_top) / spacing)))
        else:
            bottom_target = -120 if db_top <= -90 else -90
            n_intervals = max(1, int(round(abs(db_top - bottom_target) / tick_step)))
    else:
        bottom_target = -120 if db_top <= -90 else -90
        n_intervals = max(1, int(round(abs(db_top - bottom_target) / tick_step)))
        
    db_bottom_from_grid = db_top - tick_step * n_intervals
    db_bottom_from_ocr = min(vals) if len(vals) >= 2 else db_bottom_from_grid
    db_bottom = min(db_bottom_from_ocr, db_bottom_from_grid)
    
    return db_top, db_bottom, y_top, y_bottom, grid, 'ocr/grid'

def extract_spectrum_trace(img_bgr, inner):
    x0, y0, x1, y1 = inner
    crop = img_bgr[y0:y1, x0:x1]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    
    white_trace = (V > 150) & (S < 90)
    yellow_trace = (rgb[:, :, 0] > 130) & (rgb[:, :, 1] > 120) & (rgb[:, :, 2] < 120)
    trace_mask = white_trace | yellow_trace
    
    Hh, Ww = trace_mask.shape
    
    # Mask out the Gqrx frequency text area (top left of the grid)
    # to prevent letters/numbers from being detected as curve pixels
    text_y1 = int(Hh * 0.12)
    text_x0 = int(Ww * 0.08)
    text_x1 = int(Ww * 0.48)
    trace_mask[:text_y1, text_x0:text_x1] = False
    y_trace = np.full(Ww, np.nan)
    for x in range(Ww):
        yy = np.where(trace_mask[:, x])[0]
        if yy.size: 
            y_trace[x] = yy.min() + y0
            
    valid = np.isfinite(y_trace)
    if valid.sum() < 10: 
        return y_trace
        
    idx = np.arange(Ww)
    y_interp = np.interp(idx, idx[valid], y_trace[valid])
    y_smooth = median_filter(y_interp, size=5)
    return gaussian_filter1d(y_smooth, 1)

def estimate_noise_floor(db_trace, method='median'):
    valid_vals = db_trace[np.isfinite(db_trace)]
    if valid_vals.size == 0: 
        return np.nan
    if method == 'median': 
        return float(np.nanpercentile(valid_vals, 30))
    if method == 'percentile75': 
        return float(np.nanpercentile(valid_vals, 75))
    return float(np.nanmax(valid_vals))

def y_to_db(y, y_top, y_bottom, db_top, db_bottom):
    return db_top + (y - y_top) * (db_bottom - db_top) / (y_bottom - y_top)

def db_to_y(db, y_top, y_bottom, db_top, db_bottom):
    return y_top + (db - db_top) * (y_bottom - y_top) / (db_bottom - db_top)

def detect_signals(db_trace, noise_floor, min_above_noise=12.0):
    valid = np.isfinite(db_trace)
    if valid.sum() < 10: 
        return []
    db_filled = np.where(valid, db_trace, noise_floor)
    
    # Use a lighter smoothing (sigma=2) to detect peak candidates without flattening them too much
    db_smooth = gaussian_filter1d(db_filled, sigma=2)
    W = len(db_trace)
    
    # Detect peaks on the smoothed curve
    peaks, _ = find_peaks(db_smooth, height=noise_floor + 8.0, prominence=6.0, distance=max(20, W // 30))
    signals = []
    for peak_idx in peaks:
        # Refine the peak position using the raw unsmoothed trace (db_filled)
        win_size = 15
        left_win = max(0, peak_idx - win_size)
        right_win = min(W, peak_idx + win_size + 1)
        refined_idx = left_win + np.argmax(db_filled[left_win:right_win])
        
        # Use the raw peak power for accuracy
        peak_db = float(db_filled[refined_idx])
        
        # Expand left/right on the smoothed curve to determine signal bandwidth (at -8 dB from smoothed peak)
        smoothed_peak_db = float(db_smooth[refined_idx])
        thresh = smoothed_peak_db - 8.0
        
        left = refined_idx
        while left > 0 and db_smooth[left] > thresh: 
            left -= 1
        right = refined_idx
        while right < W - 1 and db_smooth[right] > thresh: 
            right += 1
            
        width = right - left
        if width < 4: 
            continue
            
        local_noise = float(np.nanpercentile(db_trace[valid], 25))
        snr = peak_db - local_noise
        if snr < min_above_noise: 
            continue
            
        # Calculate curve features in the signal area for ML classification
        signal_slice = db_smooth[left:right]
        variance = float(np.var(signal_slice)) if len(signal_slice) > 1 else 0.0
        
        signals.append({
            'peak_idx': int(refined_idx), 
            'peak_db': round(peak_db, 2),
            'noise_db': round(local_noise, 2), 
            'snr_db': round(snr, 2),
            'band_start': int(left), 
            'band_end': int(right),
            'bandwidth_px': int(width),
            'variance': round(variance, 4),
            'category': classify_tiphon(snr),
        })
    return sorted(signals, key=lambda x: x['snr_db'], reverse=True)



def analyze_image(image_path, tesseract_path=None):
    """
    Full pipeline to load and analyze an image, extracting trace parameters.
    """
    if tesseract_path:
        configure_tesseract(tesseract_path)
        
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None: 
        raise ValueError(f'Could not read image: {image_path}')
        
    panel = detect_spectrum_panel(img_bgr)
    db_top, db_bottom, y_top, y_bottom, grid, axis_source = calibrate_db_axis(img_bgr, panel)
    
    w = img_bgr.shape[1]
    x0, y0, x1, y1 = panel
    left_margin = max(50, int(w * 0.027))
    inner = (x0 + left_margin, y_top, x1 - 5, y_bottom)
    
    y_trace = extract_spectrum_trace(img_bgr, inner)
    db_trace = y_to_db(y_trace, y_top, y_bottom, db_top, db_bottom)
    
    valid = np.isfinite(db_trace)
    if valid.sum() < 10: 
        raise RuntimeError(f'Spectrum curve not detected in: {Path(image_path).name}')
        
    noise_floor = estimate_noise_floor(db_trace, 'median')
    signals = detect_signals(db_trace, noise_floor, 12.0)
    
    return {
        'filename': Path(image_path).name,
        'db_top': db_top,
        'db_bottom': db_bottom,
        'y_top': y_top,
        'y_bottom': y_bottom,
        'inner': inner,
        'noise_floor': round(noise_floor, 2),
        'signals': signals,
        'img_shape': img_bgr.shape,
        'db_trace': db_trace,
        'img_bgr': img_bgr
    }

CATEGORY_COLORS = {
    'Outstanding': (0, 215, 255), 
    'Excellent': (0, 255, 0),
    'Good': (255, 255, 0), 
    'Fair': (0, 165, 255), 
    'Bad': (0, 0, 255),
    'Not detected': (150, 150, 150)
}

def annotate_signals(img_bgr, inner, signals, db_top, db_bottom, y_top, y_bottom):
    img = img_bgr.copy()
    x0, y0, x1, y1 = inner
    
    if signals:
        global_noise = np.median([s['noise_db'] for s in signals])
        ny = int(round(db_to_y(global_noise, y_top, y_bottom, db_top, db_bottom)))
        for xd in range(x0, x1, 12):
            cv2.line(img, (xd, ny), (min(xd + 6, x1), ny), (200, 200, 200), 1)
            
    panel_h = 26 * (len(signals) + 1)
    panel_y0 = max(0, y0 - panel_h - 4)
    panel_y1 = y0 - 2
    if panel_y0 < 0: 
        panel_y0, panel_y1 = y0 + 2, y0 + panel_h + 2
        
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, panel_y0), (x1, panel_y1), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    
    cv2.putText(img, f"{'#':<3} {'Peak Signal':>14} {'Peak Noise':>12} {'SNR':>8} {'Category':<12}",
                (x0 + 8, panel_y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
                
    for i, sig in enumerate(signals):
        color = CATEGORY_COLORS.get(sig['category'], (255, 255, 255))
        sig_y = int(round(db_to_y(sig['peak_db'], y_top, y_bottom, db_top, db_bottom)))
        noise_y = int(round(db_to_y(sig['noise_db'], y_top, y_bottom, db_top, db_bottom)))
        # Center the vertical marker line at the midpoint of the bandwidth
        sig_x = x0 + (sig['band_start'] + sig['band_end']) // 2
        
        cv2.line(img, (x0 + sig['band_start'], sig_y), (x0 + sig['band_end'], sig_y), color, 2)
        cv2.circle(img, (sig_x, sig_y), 5, color, -1)
        cv2.line(img, (sig_x, sig_y), (sig_x, noise_y), color, 1)
        cv2.circle(img, (sig_x, noise_y), 4, color, 1)
        cv2.putText(img, str(i + 1), (sig_x + 6, sig_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        
        row_y = panel_y0 + 18 + (i + 1) * 22
        info = f"#{i+1:<2}  {sig['peak_db']:>10.2f} dBFS   {sig['noise_db']:>8.2f} dBFS   {sig['snr_db']:>6.2f} dB  {sig['category']:<12}"
        cv2.putText(img, info, (x0 + 8, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        
    if not signals:
        cv2.putText(img, 'No signal detected (SNR below threshold)', (x0 + 8, panel_y0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 100, 255), 1, cv2.LINE_AA)
                    
    return img
