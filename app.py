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
st.markdown("Masukkan data kesehatan Anda di bawah ini untuk melihat hasil prediksi.")
st.write("---")

# 1. Fungsi untuk memuat Scaler, Label Encoder, dan Model
@st.cache_resource
def load_ml_components():
    try:
        # Load Scaler
        with open("scaler (6).pkl", "pickle" if "pickle" in globals() else "rb") as f:
            scaler = pickle.load(f)
        
        # Load Label Encoder
        with open("label_encoder (1).pkl", "rb") as f:
            le = pickle.load(f)
            
        # [OPSIONAL] Silakan ganti nama file model sesuai dengan model Anda (misal: model.pkl)
        # di sini kita asumsikan Anda punya file model.pkl
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
    
    # 2. Membuat Input Form sesuai dengan feature_names_in_ dari scaler Anda
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Usia (Tahun)", min_value=1, max_value=120, value=30)
        height_cm = st.number_input("Tinggi Badan (cm)", min_value=50, max_value=250, value=165)
        weight_kg = st.number_input("Berat Badan (kg)", min_value=10, max_value=200, value=60)
        bmi = st.number_input("BMI (Indeks Massa Tubuh)", min_value=5.0, max_value=50.0, value=22.0)
        duration_minutes = st.number_input("Durasi Latihan Harian (Menit)", min_value=0, max_value=480, value=30)
        calories_burned = st.number_input("Kalori Terbakar (kkal)", min_value=0.0, max_value=5000.0, value=200.0)
        daily_steps = st.number_input("Jumlah Langkah Harian", min_value=0, max_value=50000, value=7000)
        avg_heart_rate = st.number_input("Rata-rata Detak Jantung", min_value=40, max_value=220, value=80)

    with col2:
        resting_heart_rate = st.number_input("Detak Jantung Istirahat", min_value=30, max_value=150, value=65)
        blood_pressure_systolic = st.number_input("Tekanan Darah Sistolik", min_value=70, max_value=250, value=120)
        blood_pressure_diastolic = st.number_input("Tekanan Darah Diastolik", min_value=40, max_value=150, value=80)
        endurance_level = st.number_input("Tingkat Daya Tahan (Endurance)", min_value=0.0, max_value=20.0, value=10.0)
        sleep_hours = st.number_input("Durasi Tidur (Jam)", min_value=1.0, max_value=24.0, value=7.0)
        stress_level = st.slider("Tingkat Stres", min_value=1, max_value=10, value=5)
        hydration_level = st.number_input("Tingkat Hidrasi (Liter)", min_value=0.0, max_value=10.0, value=2.5)
        fitness_level = st.number_input("Tingkat Kebugaran (Fitness)", min_value=0.0, max_value=20.0, value=10.0)

    st.write("---")
    
    # Tombol Prediksi
    if st.button("🔮 Lakukan Prediksi", type="primary"):
        # Susun input menjadi DataFrame sesuai urutan yang dibutuhkan Scaler
        input_data = pd.DataFrame([{
            'age': age,
            'height_cm': height_cm,
            'weight_kg': weight_kg,
            'bmi': bmi,
            'duration_minutes': duration_minutes,
            'calories_burned': calories_burned,
            'daily_steps': daily_steps,
            'avg_heart_rate': avg_heart_rate,
            'resting_heart_rate': resting_heart_rate,
            'blood_pressure_systolic': blood_pressure_systolic,
            'blood_pressure_diastolic': blood_pressure_diastolic,
            'endurance_level': endurance_level,
            'sleep_hours': sleep_hours,
            'stress_level': stress_level,
            'hydration_level': hydration_level,
            'fitness_level': fitness_level
        }])
        
        # 3. Lakukan Scaling terhadap Input
        input_scaled = scaler.transform(input_data)
        
        # 4. Prediksi Menggunakan Model
        prediction = model.predict(input_scaled)
        
        # 5. Decode Hasil Prediksi dengan Label Encoder
        prediction_label = le.inverse_transform(prediction)
        
        # Tampilkan Hasil
        st.subheader("📊 Hasil Prediksi:")
        st.success(f"Kondisi Kesehatan yang Terdeteksi/Diprediksi: **{prediction_label[0]}**")
else:
    st.info("💡 Sediakan file `model.pkl` Anda di dalam folder agar tombol prediksi dapat berfungsi dengan baik.")
