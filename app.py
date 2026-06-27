import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pytesseract

import signal_processor as sp
import ml_classifier as ml

# Set page configuration
st.set_page_config(
    page_title="Signal Analyzer & Amplifier Recommender",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Sleek CSS Styles for premium look
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    
    /* Header gradient and glow styling */
    .title-text {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #3498db, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-shadow: 0px 4px 20px rgba(52, 152, 219, 0.15);
    }
    
    .subtitle-text {
        font-family: 'Inter', sans-serif;
        color: #8892b0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Custom Card Styling */
    .metric-card {
        background: rgba(23, 28, 41, 0.65);
        border: 1px solid rgba(52, 152, 219, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(52, 152, 219, 0.5);
    }
    
    .metric-title {
        color: #8892b0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }
    
    .metric-value {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
    }
    
    .metric-unit {
        font-size: 1rem;
        color: #3498db;
        margin-left: 0.2rem;
    }
    
    /* Class badges */
    .badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: 700;
        text-align: center;
        font-size: 1rem;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .badge-am {
        background-color: rgba(243, 156, 18, 0.2);
        color: #f39c12;
        border: 1px solid #f39c12;
    }
    
    .badge-fm {
        background-color: rgba(52, 152, 219, 0.2);
        color: #3498db;
        border: 1px solid #3498db;
    }
    
    /* Quality badges */
    .q-outstanding { background-color: rgba(255, 215, 0, 0.2); color: #ffd700; border: 1px solid #ffd700; }
    .q-excellent { background-color: rgba(46, 204, 113, 0.2); color: #2ecc71; border: 1px solid #2ecc71; }
    .q-good { background-color: rgba(26, 188, 156, 0.2); color: #1abc9c; border: 1px solid #1abc9c; }
    .q-fair { background-color: rgba(230, 126, 34, 0.2); color: #e67e22; border: 1px solid #e67e22; }
    .q-bad { background-color: rgba(231, 76, 60, 0.2); color: #e74c3c; border: 1px solid #e74c3c; }
    .q-none { background-color: rgba(127, 140, 141, 0.2); color: #95a5a6; border: 1px solid #95a5a6; }
    
    /* Recommendation panel */
    .recom-panel {
        background: linear-gradient(135deg, rgba(23, 28, 41, 0.95), rgba(15, 23, 42, 0.95));
        border-left: 5px solid #2ecc71;
        border-radius: 0 12px 12px 0;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
    }
    
    .recom-header {
        font-weight: 700;
        font-size: 1.2rem;
        color: #2ecc71;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .recom-specs {
        background: rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 6px;
        padding: 0.8rem;
        font-family: 'Courier New', Courier, monospace;
        color: #a0aec0;
        margin-top: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function for amplifier recommendations
def get_amplifier_recommendation(signal_type, peak_db, snr_db):
    if snr_db is None or np.isnan(snr_db):
        return {
            "type": "Tidak Terdeteksi (N/A)",
            "desc": "Sinyal tidak terdeteksi atau SNR berada di bawah ambang batas minimal. Periksa koneksi antena, orientasi arah antena, atau level noise lingkungan sebelum menggunakan penguat aktif.",
            "specs": "Gain: N/A\nNoise Figure: N/A\nImpedansi: N/A"
        }
    
    if snr_db > 29:
        return {
            "type": "Tidak Memerlukan Penguat (Bypass)",
            "desc": "Kualitas sinyal luar biasa (Outstanding) dan sangat bersih. Menambahkan penguat (amplifier) justru berisiko membebani receiver (overload), memicu kejenuhan (saturation), dan merusak kualitas sinyal akibat distorsi intermodulasi (IMD).",
            "specs": "Saran: Bypass langsung ke receiver atau gunakan attenuator pasif jika daya terlalu tinggi."
        }
        
    if signal_type == 'FM':
        if peak_db < -60:
            if snr_db >= 12:
                return {
                    "type": "Low Noise Amplifier (LNA) Khusus VHF/FM",
                    "desc": "Tingkat kekuatan sinyal lemah tetapi rasio sinyal terhadap noise bersih (SNR bagus). Diperlukan penguat bersuara rendah (LNA) untuk mendongkrak daya sinyal tanpa menambah noise tambahan secara signifikan.",
                    "specs": "Rentang Frekuensi : 88 - 108 MHz\nGain              : 12 - 15 dB\nNoise Figure (NF) : < 1.5 dB\nImpedansi         : 75 Ohm"
                }
            else:
                return {
                    "type": "VHF Bandpass Filter + LNA",
                    "desc": "Sinyal lemah dan noise tinggi (SNR rendah). Mengamplifikasi sinyal tanpa filter berisiko ikut memperkuat noise di luar pita. Diperlukan Bandpass Filter FM sebelum sinyal masuk ke LNA.",
                    "specs": "Bandpass Filter   : 88 - 108 MHz (Redaman luar pita > 30 dB)\nGain LNA          : 10 - 12 dB\nNoise Figure (NF) : < 1.8 dB"
                }
        else:
            return {
                "type": "Penguat High-Linearity / Attenuator Pasif",
                "desc": "Level daya sinyal memadai tetapi kualitas sedang (SNR moderat). Gunakan penguat linearitas tinggi (OIP3 tinggi) jika diperlukan untuk mengatasi redaman kabel panjang.",
                "specs": "Gain      : 6 - 10 dB (adjustable)\nOIP3      : > 30 dBm\nImpedansi : 75 Ohm"
            }
    else: # AM / Airband (HF/VHF)
        if peak_db < -60:
            if snr_db >= 12:
                return {
                    "type": "Low Noise Amplifier (LNA) HF/VHF",
                    "desc": "Kekuatan sinyal Airband/AM lemah dengan kondisi noise rendah. Gunakan LNA berspesifikasi VHF penerima udara agar sensitivitas sinyal meningkat.",
                    "specs": "Rentang Frekuensi : 108 - 137 MHz (Airband) / 0.5 - 30 MHz (AM)\nGain              : 10 - 12 dB\nNoise Figure (NF) : < 2.0 dB"
                }
            else:
                return {
                    "type": "Tuned Bandpass Filter (Preselector) + LNA",
                    "desc": "Sinyal sangat lemah dan rentan terhadap interferensi dari sinyal FM pemancar besar di sekitarnya. Diperlukan filter bandpass tertala (preselector) untuk membuang interferensi sinyal luar sebelum diperkuat LNA.",
                    "specs": "Bandpass Filter   : 108 - 137 MHz (Airband)\nGain LNA          : 8 - 10 dB\nNoise Figure (NF) : < 2.2 dB"
                }
        else:
            return {
                "type": "Filter Pasif / Bypass",
                "desc": "Daya sinyal AM/Airband sudah mencukupi. Penggunaan amplifier aktif tambahan tidak disarankan untuk mencegah modulasi silang (cross-modulation) dari noise sekitar. Gunakan filter pasif jika terdapat gangguan.",
                "specs": "Saran: Gunakan filter pasif bandpass pita 108-137 MHz."
            }

# Sidebar Navigation
st.sidebar.markdown("### 📡 MENU NAVIGASI")
page = st.sidebar.radio("Pilih Halaman:", ["Analisis Sinyal", "Analisis Spasial & Statistik"])

# Load Machine Learning Model
clf = ml.load_or_train_model()

# Page 1: Analisis Sinyal
if page == "Analisis Sinyal":
    st.markdown('<div class="title-text">📡 Analisis & Klasifikasi Sinyal</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-text">Gunakan pengolahan citra komputer untuk menganalisis parameter sinyal spektrum frekuensi atau masukkan data secara manual untuk mendapatkan jenis modulasi dan saran penguat.</div>', unsafe_allow_html=True)
    
    # Mode input
    mode = st.radio("Pilih Metode Input Data:", ["Unggah Gambar Spektrum", "Input Parameter Manual"], horizontal=True)
    
    if mode == "Unggah Gambar Spektrum":
        col_up, col_conf = st.columns([2, 1])
        
        with col_up:
            uploaded_file = st.file_uploader("Pilih file gambar screenshot spektrum (.jpg, .jpeg, .png):", type=["jpg", "jpeg", "png"])
            
        with col_conf:
            # Inform user about Tesseract
            if sp.HAS_TESSERACT:
                st.success("🤖 Tesseract OCR terdeteksi dan aktif.")
                tess_path = st.text_input("Path Tesseract (Opsional):", value=pytesseract.pytesseract.tesseract_cmd)
            else:
                st.warning("⚠️ Tesseract OCR tidak terdeteksi. Sumbu Y akan menggunakan kalibrasi default (-20 s/d -90 dBFS).")
                tess_path = st.text_input("Tentukan Path Tesseract.exe jika sudah diinstal:")
                if tess_path:
                    if sp.configure_tesseract(tess_path):
                        st.success("Tesseract berhasil dikonfigurasi!")
                        sp.HAS_TESSERACT = True
                        st.rerun()

        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = Path(tmp_file.name)
                
            # Perform Analysis
            with st.spinner("Sedang memproses gambar spektrum..."):
                try:
                    res = sp.analyze_image(tmp_path, tesseract_path=tess_path if tess_path else None)
                    
                    # Layout output
                    col_metrics, col_plot = st.columns([1, 1])
                    
                    with col_metrics:
                        st.markdown("### 📊 Hasil Ekstraksi Parameter")
                        
                        # Check if signals are detected
                        if res['signals']:
                            # Get strongest signal
                            sig = res['signals'][0]
                            
                            # Predict signal type (AM vs FM) using Random Forest model
                            sig_type = ml.predict_signal_type(
                                bandwidth_px=sig['bandwidth_px'],
                                variance=sig['variance'],
                                snr_db=sig['snr_db'],
                                peak_db=sig['peak_db'],
                                clf=clf
                            )
                            
                            # Metrics Cards Grid
                            m_col1, m_col2 = st.columns(2)
                            with m_col1:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-title">Peak Power</div>
                                    <div class="metric-value">{sig['peak_db']}<span class="metric-unit">dBFS</span></div>
                                </div>
                                """, unsafe_allow_html=True)
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-title">SNR (Signal-Noise)</div>
                                    <div class="metric-value">{sig['snr_db']}<span class="metric-unit">dB</span></div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                            with m_col2:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-title">Noise Floor</div>
                                    <div class="metric-value">{sig['noise_db']}<span class="metric-unit">dBFS</span></div>
                                </div>
                                """, unsafe_allow_html=True)
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-title">Bandwidth</div>
                                    <div class="metric-value">{sig['bandwidth_px']}<span class="metric-unit">px</span></div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Display Classification
                            st.markdown("#### 🎯 Hasil Klasifikasi & Kualitas")
                            
                            badge_type = 'badge-am' if sig_type == 'AM' else 'badge-fm'
                            q_class = f"q-{sig['category'].lower()}"
                            
                            st.markdown(f"""
                            <div style="display: flex; gap: 1rem; margin-bottom: 1.5rem;">
                                <div>
                                    <span style="font-size:0.9rem; color:#8892b0; display:block; margin-bottom:0.2rem;">Tipe Sinyal</span>
                                    <span class="badge {badge_type}">{sig_type} (via AI Model)</span>
                                </div>
                                <div>
                                    <span style="font-size:0.9rem; color:#8892b0; display:block; margin-bottom:0.2rem;">Kualitas TIPHON</span>
                                    <span class="badge {q_class}">{sig['category']}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Amplifier recommendation
                            recom = get_amplifier_recommendation(sig_type, sig['peak_db'], sig['snr_db'])
                            st.markdown(f"""
                            <div class="recom-panel">
                                <div class="recom-header">🔌 Rekomendasi Amplifier: {recom['type']}</div>
                                <div>{recom['desc']}</div>
                                <div class="recom-specs">
                                    <strong>SPESIFIKASI ACUAN:</strong><br/>
                                    {recom['specs'].replace(chr(10), '<br/>')}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.warning("⚠️ Tidak ada puncak sinyal yang melebihi batas SNR minimal.")
                            # Standard Metrics showing Noise Floor only
                            m_col1, m_col2 = st.columns(2)
                            with m_col1:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-title">Noise Floor</div>
                                    <div class="metric-value">{res['noise_floor']}<span class="metric-unit">dBFS</span></div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            recom = get_amplifier_recommendation('FM', -90, None)
                            st.markdown(f"""
                            <div class="recom-panel" style="border-left-color: #7f8c8d;">
                                <div class="recom-header" style="color: #95a5a6;">🔌 Rekomendasi: {recom['type']}</div>
                                <div>{recom['desc']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                    with col_plot:
                        st.markdown("### 🖼️ Citra Hasil Deteksi & Anotasi")
                        # Annotate image
                        annotated_bgr = sp.annotate_signals(
                            res['img_bgr'], res['inner'], res['signals'],
                            res['db_top'], res['db_bottom'], res['y_top'], res['y_bottom']
                        )
                        # Convert BGR to RGB for Streamlit
                        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                        st.image(annotated_rgb, width='stretch', caption="Spektrum teranotasi")
                        
                except Exception as e:
                    st.error(f"Gagal memproses gambar: {e}")
                    
            # Cleanup temp file
            if tmp_path.exists():
                os.remove(tmp_path)
                
    elif mode == "Input Parameter Manual":
        col_inputs, col_results = st.columns([1, 1])
        
        with col_inputs:
            st.markdown("### 📥 Form Input Parameter Sinyal")
            
            preset = st.selectbox(
                "Pilih Preset Contoh Sinyal (Autofill):",
                ["Kustom (Isi Sendiri)", "FM Bersih & Kuat", "FM Lemah", "AM/Airband Bersih & Kuat", "AM/Airband Lemah & Bising"]
            )
            
            # Map presets
            if preset == "FM Bersih & Kuat":
                p_val, n_val, snr_init, bw_val, var_val = -30, -80, 50.0, 120, 85.0
            elif preset == "FM Lemah":
                p_val, n_val, snr_init, bw_val, var_val = -70, -85, 15.0, 95, 60.0
            elif preset == "AM/Airband Bersih & Kuat":
                p_val, n_val, snr_init, bw_val, var_val = -35, -80, 45.0, 15, 4.5
            elif preset == "AM/Airband Lemah & Bising":
                p_val, n_val, snr_init, bw_val, var_val = -75, -82, 7.0, 12, 3.0
            else:
                p_val, n_val, snr_init, bw_val, var_val = -50, -85, 35.0, 50, 8.5
                
            peak_val = st.slider("Tingkat Kekuatan Puncak (Peak Power) [dBFS]:", -120, 0, p_val)
            noise_val = st.slider("Batas Bawah Kebisingan (Noise Floor) [dBFS]:", -120, -30, n_val)
            
            # Calculate dynamic default SNR if "Kustom"
            if preset == "Kustom (Isi Sendiri)":
                snr_default = float(max(0, peak_val - noise_val))
            else:
                snr_default = snr_init
                
            snr_val = st.number_input("Rasio Sinyal Terhadap Noise (SNR) [dB]:", min_value=0.0, max_value=120.0, value=snr_default)
            
            bandwidth_val = st.slider("Lebar Bandwidth [Pixel]:", 5, 300, bw_val, help="Lebar puncak sinyal pada layar. AM biasanya kecil (<30px), FM biasanya besar (>40px).")
            variance_val = st.number_input("Varians Spektrum:", min_value=0.0, max_value=100.0, value=var_val, format="%.4f")
            
        with col_results:
            st.markdown("### 📊 Hasil Analisis Parameter Manual")
            
            # Predict quality based on SNR
            q_category = sp.classify_tiphon(snr_val)
            
            # Predict AM/FM using Random Forest
            sig_type = ml.predict_signal_type(
                bandwidth_px=bandwidth_val,
                variance=variance_val,
                snr_db=snr_val,
                peak_db=peak_val,
                clf=clf
            )
            
            # Display metrics Cards
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Peak Power</div>
                    <div class="metric-value">{peak_val}<span class="metric-unit">dBFS</span></div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">SNR (Signal-Noise)</div>
                    <div class="metric-value">{snr_val}<span class="metric-unit">dB</span></div>
                </div>
                """, unsafe_allow_html=True)
                
            with m_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Noise Floor</div>
                    <div class="metric-value">{noise_val}<span class="metric-unit">dBFS</span></div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Bandwidth</div>
                    <div class="metric-value">{bandwidth_val}<span class="metric-unit">px</span></div>
                </div>
                """, unsafe_allow_html=True)
                
            # Badges
            badge_type = 'badge-am' if sig_type == 'AM' else 'badge-fm'
            q_class = f"q-{q_category.lower()}"
            st.markdown(f"""
            <div style="display: flex; gap: 1rem; margin-bottom: 1.5rem;">
                <div>
                    <span style="font-size:0.9rem; color:#8892b0; display:block; margin-bottom:0.2rem;">Tipe Sinyal</span>
                    <span class="badge {badge_type}">{sig_type} (Model Classifier)</span>
                </div>
                <div>
                    <span style="font-size:0.9rem; color:#8892b0; display:block; margin-bottom:0.2rem;">Kualitas TIPHON</span>
                    <span class="badge {q_class}">{q_category}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommendations
            recom = get_amplifier_recommendation(sig_type, peak_val, snr_val)
            st.markdown(f"""
            <div class="recom-panel">
                <div class="recom-header">🔌 Rekomendasi Amplifier: {recom['type']}</div>
                <div>{recom['desc']}</div>
                <div class="recom-specs">
                    <strong>SPESIFIKASI ACUAN:</strong><br/>
                    {recom['specs'].replace(chr(10), '<br/>')}
                </div>
            </div>
            """, unsafe_allow_html=True)

# Page 2: Analisis Spasial & Statistik
elif page == "Analisis Spasial & Statistik":
    st.markdown('<div class="title-text">🗺️ Analisis Spasial Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-text">Menampilkan sebaran jumlah file dan analisis kualitas sinyal berdasarkan data spasial Arah (Barat, Selatan, Timur, Utara) serta Jarak (2km, 4km, 6km, 8km, 10km).</div>', unsafe_allow_html=True)
    
    # Dataset Folder Scan
    dataset_root = Path(__file__).parent / 'DATASET_GAMBAR'
    
    if not dataset_root.exists():
        st.error("Folder DATASET_GAMBAR tidak ditemukan di direktori proyek.")
    else:
        # Scan folders to gather statistics
        directions = ['Barat', 'Selatan', 'Timur', 'Utara']
        distances = [2, 4, 6, 8, 10]
        
        # 1. Gather file counts
        counts = []
        for direction in directions:
            dir_path = dataset_root / direction
            if not dir_path.exists():
                continue
            for dist in distances:
                fm_path = dir_path / f'{dist}km'
                if not fm_path.exists():
                    fm_path = dir_path / f'{dist}KM'
                
                fm_count = 0
                am_count = 0
                etc_count = 0
                
                # Scan subfolders if they exist
                if fm_path.exists() and fm_path.is_dir():
                    # Check for 1. FM
                    fm_sub = fm_path / '1. FM'
                    if fm_sub.exists():
                        fm_count = len(list(fm_sub.glob('*.jpg')))
                    
                    # Check for 2. AIRBAND
                    am_sub = fm_path / '2. AIRBAND'
                    if am_sub.exists():
                        am_count = len(list(am_sub.glob('*.jpg')))
                        
                    # Check for 3. ETC
                    etc_sub = fm_path / '3. ETC'
                    if etc_sub.exists():
                        etc_count = len(list(etc_sub.glob('*.jpg')))
                
                counts.append({
                    'Arah': direction,
                    'Jarak': f"{dist}km",
                    'FM (Images)': fm_count,
                    'AM (Images)': am_count,
                    'ETC (Images)': etc_count,
                    'Total Images': fm_count + am_count + etc_count
                })
                
        df_counts = pd.DataFrame(counts)
        
        # Display Stats Table
        st.markdown("### 📁 Distribusi Dataset Riil (File Counts)")
        
        # Pivot table for total images
        pivot_total = df_counts.pivot(index='Arah', columns='Jarak', values='Total Images').fillna(0).astype(int)
        
        col_tab, col_graph = st.columns([1, 1])
        
        with col_tab:
            st.write("Jumlah Total Gambar per Wilayah (Arah & Jarak):")
            st.dataframe(pivot_total, width='stretch')
            
            # Show sub-counts details
            st.write("Detail Kategori Sinyal:")
            cat_sum = df_counts[['FM (Images)', 'AM (Images)', 'ETC (Images)']].sum()
            st.write(pd.DataFrame(cat_sum, columns=['Jumlah Gambar']))
            
            # Download button for the raw counts table
            csv_counts = df_counts.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Unduh Data Distribusi (CSV)",
                data=csv_counts,
                file_name="distribusi_dataset.csv",
                mime="text/csv",
                key="download_counts"
            )
            
        with col_graph:
            # Bar plot of counts
            fig, ax = plt.subplots(figsize=(8, 4.5))
            df_melted = df_counts.melt(id_vars=['Arah', 'Jarak'], value_vars=['FM (Images)', 'AM (Images)', 'ETC (Images)'], var_name='Kategori', value_name='Jumlah')
            
            # Aggregate by direction
            sns.barplot(data=df_melted, x='Arah', y='Jumlah', hue='Kategori', errorbar=None, ax=ax, palette='Blues_d')
            ax.set_title('Distribusi Citra per Arah dan Kategori')
            ax.set_ylabel('Jumlah Citra')
            plt.tight_layout()
            st.pyplot(fig)
            
        st.markdown("---")
        
        # Batch analysis generator
        st.markdown("### ⚡ Demo Batch Analysis")
        st.write("Jalankan analisis cepat terhadap beberapa contoh gambar spektrum dari setiap folder arah dan jarak untuk melihat visualisasi kategori kualitas sinyal secara spasial.")
        
        if st.button("Jalankan Demo Batch Analisis"):
            with st.spinner("Memproses sampel gambar spektrum..."):
                results = []
                for direction in directions:
                    dir_path = dataset_root / direction
                    if not dir_path.exists():
                        continue
                    for dist in distances:
                        fm_path = dir_path / f'{dist}km'
                        if not fm_path.exists():
                            fm_path = dir_path / f'{dist}KM'
                        
                        # Process first image in 1. FM (representing FM) if available
                        fm_sub = fm_path / '1. FM'
                        if fm_sub.exists():
                            images = list(fm_sub.glob('*.jpg'))
                            if images:
                                # Pick 1 sample image
                                try:
                                    res = sp.analyze_image(images[0])
                                    if res['signals']:
                                        sig = res['signals'][0]
                                        recom = get_amplifier_recommendation('FM', sig['peak_db'], sig['snr_db'])
                                        results.append({
                                            'Arah': direction,
                                            'Jarak': dist,
                                            'SNR': sig['snr_db'],
                                            'Quality': sig['category'],
                                            'Recommendation': recom['type'],
                                            'Type': 'FM'
                                        })
                                    else:
                                        recom = get_amplifier_recommendation('FM', -90.0, None)
                                        results.append({
                                            'Arah': direction,
                                            'Jarak': dist,
                                            'SNR': 0.0,
                                            'Quality': 'Not detected',
                                            'Recommendation': recom['type'],
                                            'Type': 'FM'
                                        })
                                except Exception:
                                    pass
                
                if results:
                    df_res = pd.DataFrame(results)
                    
                    st.success("Batch analisis selesai!")
                    
                    # Display metrics
                    col_res1, col_res2 = st.columns([1, 1])
                    
                    with col_res1:
                        st.write("Rata-rata SNR (dB) per Wilayah:")
                        pivot_snr = df_res.pivot(index='Arah', columns='Jarak', values='SNR').round(2)
                        st.dataframe(pivot_snr, width='stretch')
                        
                        # Download button for the SNR pivot table
                        csv_snr = pivot_snr.to_csv().encode('utf-8')
                        st.download_button(
                            label="📥 Unduh Data Rata-rata SNR (CSV)",
                            data=csv_snr,
                            file_name="rata_rata_snr.csv",
                            mime="text/csv",
                            key="download_snr"
                        )
                        
                    with col_res2:
                        # Heatmap
                        fig_hm, ax_hm = plt.subplots(figsize=(7, 4))
                        sns.heatmap(pivot_snr, annot=True, cmap='YlGnBu', fmt='.1f', cbar=True, ax=ax_hm)
                        ax_hm.set_title('Peta Kalor SNR Rata-rata (dB)')
                        ax_hm.set_xlabel('Jarak (km)')
                        ax_hm.set_ylabel('Arah')
                        plt.tight_layout()
                        st.pyplot(fig_hm)
                        
                    # Display quality and recommendations spatially
                    st.markdown("---")
                    st.markdown("### 📊 Peta Kualitas Sinyal & Rekomendasi Spasial")
                    col_q, col_r = st.columns([1, 1])
                    
                    with col_q:
                        st.write("Kualitas Sinyal (Kategori TIPHON) per Wilayah:")
                        pivot_quality = df_res.pivot(index='Arah', columns='Jarak', values='Quality').fillna("N/A")
                        st.dataframe(pivot_quality, width='stretch')
                        
                        csv_q = pivot_quality.to_csv().encode('utf-8')
                        st.download_button(
                            label="📥 Unduh Kualitas Spasial (CSV)",
                            data=csv_q,
                            file_name="kualitas_spasial.csv",
                            mime="text/csv",
                            key="download_quality"
                        )
                        
                    with col_r:
                        st.write("Rekomendasi Alat/Amplifier per Wilayah:")
                        pivot_recom = df_res.pivot(index='Arah', columns='Jarak', values='Recommendation').fillna("N/A")
                        st.dataframe(pivot_recom, width='stretch')
                        
                        csv_r = pivot_recom.to_csv().encode('utf-8')
                        st.download_button(
                            label="📥 Unduh Rekomendasi Spasial (CSV)",
                            data=csv_r,
                            file_name="rekomendasi_spasial.csv",
                            mime="text/csv",
                            key="download_recom"
                        )
                else:
                    st.warning("Gagal memproses gambar. Pastikan Tesseract sudah terkonfigurasi dengan benar di sistem Anda.")
