import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Konfigurasi Halaman
st.set_page_config(
    page_title="Health Condition Prediction App",
    page_icon="🩺",
    layout="centered"
)

# Judul Aplikasi
st.title("🩺 Aplikasi Prediksi Kondisi Kesehatan")
st.markdown("Masukkan data kesehatan Anda di bawah ini untuk melihat analisis dan prediksi kondisi kesehatan.")
st.write("---")

# 1. Fungsi untuk memuat Scaler, Label Encoder, dan Model
@st.cache_resource
def load_ml_components():
    try:
        # Load Scaler
        with open("scaler (6).pkl", "rb") as f:
            scaler = pickle.load(f)
        
        # Load Label Encoder
        with open("label_encoder (1).pkl", "rb") as f:
            le = pickle.load(f)
            
        # Memuat file model ML Anda
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
            
        return scaler, le, model
    except FileNotFoundError as e:
        st.error(f"Gagal memuat komponen ML: {e}. Pastikan file .pkl ada di folder yang sama.")
        return None, None, None

scaler, le, model = load_ml_components()

# Hanya jalankan form jika semua komponen berhasil dimuat
if scaler and le and model:
    
    st.subheader("📋 Form Data Kesehatan")
    st.caption("Arahkan kursor atau sentuh tanda tanya (?) di samping kolom untuk melihat penjelasan parameter.")
    
    # 2. Membuat Input Form dengan Parameter 'help' sebagai keterangan tambahan
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Usia (Tahun)", min_value=1, max_value=120, value=30,
                              help="Usia pasien saat ini dalam satuan tahun.")
        
        height_cm = st.number_input("Tinggi Badan (cm)", min_value=50, max_value=250, value=165,
                                    help="Tinggi fisik tubuh tanpa alas kaki.")
        
        weight_kg = st.number_input("Berat Badan (kg)", min_value=10, max_value=200, value=60,
                                    help="Berat badan aktual saat ini dalam satuan kilogram.")
        
        bmi = st.number_input("BMI (Indeks Massa Tubuh)", min_value=5.0, max_value=50.0, value=22.0,
                              help="Rasio berat badan terhadap tinggi badan (BB (kg) / TB² (m)). Normal: 18.5 - 24.9.")
        
        duration_minutes = st.number_input("Durasi Latihan Harian (Menit)", min_value=0, max_value=480, value=30,
                                           help="Total lama waktu yang dihabiskan untuk berolahraga atau aktivitas fisik intens per hari.")
        
        calories_burned = st.number_input("Kalori Terbakar (kkal)", min_value=0.0, max_value=5000.0, value=200.0,
                                          help="Perkiraan jumlah energi/kalori yang terbakar dari aktivitas fisik harian.")
        
        daily_steps = st.number_input("Jumlah Langkah Harian", min_value=0, max_value=50000, value=7000,
                                      help="Total akumulasi langkah kaki yang ditempuh dalam satu hari penuh.")
        
        avg_heart_rate = st.number_input("Rata-rata Detak Jantung", min_value=40, max_value=220, value=80,
                                         help="Rata-rata detak jantung per menit (bpm) selama beraktivitas sehari-hari.")

    with col2:
        resting_heart_rate = st.number_input("Detak Jantung Istirahat", min_value=30, max_value=150, value=65,
                                             help="Detak jantung per menit (bpm) saat tubuh dalam kondisi rileks atau baru bangun tidur.")
        
        blood_pressure_systolic = st.number_input("Tekanan Darah Sistolik", min_value=70, max_value=250, value=120,
                                                  help="Tekanan saat jantung memompa darah (angka atas pada tensimeter). Normalnya sekitar 120 mmHg.")
        
        blood_pressure_diastolic = st.number_input("Tekanan Darah Diastolik", min_value=40, max_value=150, value=80,
                                                   help="Tekanan saat jantung beristirahat di antara detakan (angka bawah pada tensimeter). Normalnya sekitar 80 mmHg.")
        
        endurance_level = st.number_input("Tingkat Daya Tahan (Endurance)", min_value=0.0, max_value=20.0, value=10.0,
                                          help="Skor kapasitas kardiorespirasi atau daya tahan stamina fisik Anda.")
        
        sleep_hours = st.number_input("Durasi Tidur (Jam)", min_value=1.0, max_value=24.0, value=7.0,
                                      help="Durasi tidur atau istirahat di malam hari dalam satuan jam.")
        
        stress_level = st.slider("Tingkat Stres", min_value=1, max_value=10, value=5,
                                 help="Skor subjektif tingkat stres psikologis Anda (1 = Sangat Rileks, 10 = Sangat Stres).")
        
        hydration_level = st.number_input("Tingkat Hidrasi (Liter)", min_value=0.0, max_value=10.0, value=2.5,
                                          help="Volume air putih yang dikonsumsi per hari dalam satuan liter.")
        
        fitness_level = st.number_input("Tingkat Kebugaran (Fitness)", min_value=0.0, max_value=20.0, value=10.0,
                                        help="Skor kebugaran fisik atau kapasitas kebugaran tubuh secara keseluruhan.")

    st.write("---")
    
    # Tombol Prediksi
    if st.button("🔮 Lakukan Prediksi", type="primary"):
        # Susun input menjadi DataFrame sesuai urutan yang dibutuhkan Scaler
        input_data = pd.DataFrame([{
            'age': age, 'height_cm': height_cm, 'weight_kg': weight_kg, 'bmi': bmi,
            'duration_minutes': duration_minutes, 'calories_burned': calories_burned,
            'daily_steps': daily_steps, 'avg_heart_rate': avg_heart_rate,
            'resting_heart_rate': resting_heart_rate, 'blood_pressure_systolic': blood_pressure_systolic,
            'blood_pressure_diastolic': blood_pressure_diastolic, 'endurance_level': endurance_level,
            'sleep_hours': sleep_hours, 'stress_level': stress_level,
            'hydration_level': hydration_level, 'fitness_level': fitness_level
        }])
        
        # 3. Lakukan Scaling terhadap Input
        input_scaled = scaler.transform(input_data)
        
        # 4. Prediksi Menggunakan Model
        prediction = model.predict(input_scaled)
        
        # 5. Decode Hasil Prediksi dengan Label Encoder
        prediction_label = le.inverse_transform(prediction)
        hasil = prediction_label[0]
        
        # Tampilkan Hasil & Keterangan Output Berdasarkan Kelas Label Encoder
        st.subheader("📊 Hasil Analisis Prediksi:")
        
        if hasil == "Asthma":
            st.warning(f"Kondisi Kesehatan yang Diprediksi: **{hasil} (Asma)**")
            st.markdown("""
            **💡 Keterangan & Saran:**
            Sistem mendeteksi indikasi masalah pada saluran pernapasan. 
            * Hindari faktor pemicu seperti debu, asap rokok, atau udara yang terlalu dingin.
            * Jaga intensitas olahraga agar tidak memicu sesak napas secara mendadak.
            * *Segera konsultasikan dengan dokter spesialis paru untuk penanganan medis lebih lanjut.*
            """)
            
        elif hasil == "Diabetes":
            st.error(f"Kondisi Kesehatan yang Diprediksi: **{hasil} (Diabetes)**")
            st.markdown("""
            **💡 Keterangan & Saran:**
            Sistem mendeteksi adanya pola data yang berkorelasi dengan risiko diabetes atau gangguan gula darah.
            * Batasi konsumsi karbohidrat sederhana dan makanan/minuman tinggi gula.
            * Pertahankan aktivitas fisik teratur (minimal jalan kaki 30 menit sehari).
            * *Sangat disarankan untuk melakukan pemeriksaan kadar gula darah (Puasa & HbA1c) di laboratorium klinis terdekat.*
            """)
            
        elif hasil == "Hypertension":
            st.error(f"Kondisi Kesehatan yang Diprediksi: **{hasil} (Hipertensi)**")
            st.markdown("""
            **💡 Keterangan & Saran:**
            Sistem mendeteksi indikasi tekanan darah tinggi berdasarkan kombinasi parameter klinis dan gaya hidup Anda.
            * Kurangi konsumsi garam/natrium dan makanan olahan (fast food).
            * Lakukan manajemen stres dengan baik dan pastikan istirahat/tidur cukup.
            * *Pantau tekanan darah Anda secara berkala dan konsultasikan dengan dokter untuk evaluasi jantung/pembuluh darah.*
            """)
        else:
            st.success(f"Kondisi Kesehatan yang Diprediksi: **{hasil}**")
            st.markdown("""
            **💡 Keterangan & Saran:**
            Data Anda berada dalam rentang indikasi normal dari target penyakit utama. Tetap pertahankan pola makan seimbang, olahraga teratur, dan istirahat yang cukup untuk menjaga imunitas tubuh Anda!
            """)
else:
    st.info("💡 Sediakan file `model.pkl` Anda di dalam folder agar tombol prediksi dapat berfungsi dengan baik.")
