import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- 1. LOAD MODELS ---
@st.cache_resource
def load_models():
    # Pastikan nama file sesuai dengan yang ada di folder kamu
    with open('scaler (6).pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('decision_tree_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return scaler, model

scaler, model = load_models()

# --- 2. JUDUL APLIKASI ---
st.title("Prediksi Fitness & Aktivitas Fisik")
st.write("Masukkan data kamu di bawah ini untuk melihat hasil prediksi model.")

# --- 3. INPUT FORM ---
st.header("Data Pengguna")

col1, col2 = st.columns(2)

# Mengumpulkan 16 fitur numerik (untuk di-scale)
with col1:
    age = st.number_input("Umur (Age)", min_value=10, max_value=100, value=25)
    height_cm = st.number_input("Tinggi Badan (cm)", value=170.0)
    weight_kg = st.number_input("Berat Badan (kg)", value=65.0)
    bmi = st.number_input("BMI", value=22.5)
    duration_minutes = st.number_input("Durasi Olahraga (menit)", value=30)
    calories_burned = st.number_input("Kalori Terbakar", value=200)
    daily_steps = st.number_input("Langkah Harian", value=5000)
    avg_heart_rate = st.number_input("Rata-rata Detak Jantung", value=110)

with col2:
    resting_heart_rate = st.number_input("Detak Jantung Istirahat", value=70)
    blood_pressure_systolic = st.number_input("Sistolik", value=120)
    blood_pressure_diastolic = st.number_input("Diastolik", value=80)
    endurance_level = st.number_input("Level Ketahanan", value=5.0)
    sleep_hours = st.number_input("Jam Tidur", value=7.0)
    stress_level = st.number_input("Level Stres", value=5.0)
    hydration_level = st.number_input("Level Hidrasi", value=5.0)
    fitness_level = st.number_input("Level Kebugaran", value=5.0)

st.header("Konteks Waktu & Kategori")
col3, col4 = st.columns(2)

with col3:
    year = st.number_input("Tahun", value=2024)
    month = st.number_input("Bulan (1-12)", min_value=1, max_value=12, value=1)
    day = st.number_input("Tanggal (1-31)", min_value=1, max_value=31, value=1)
    day_of_week = st.number_input("Hari dalam Minggu (0-6)", min_value=0, max_value=6, value=0)
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    
with col4:
    activity_type = st.selectbox("Tipe Aktivitas", ["Cycling", "Dancing", "HIIT", "Running", "Swimming", "Tennis", "Walking", "Weight Training", "Yoga", "Lainnya"])
    intensity = st.selectbox("Intensitas", ["High", "Medium", "Low"])
    smoking_status = st.selectbox("Status Merokok", ["Current", "Former", "Never"])

# --- 4. PREDIKSI ---
if st.button("Lakukan Prediksi"):
    # Siapkan data numerik
    num_features = pd.DataFrame([[
        age, height_cm, weight_kg, bmi, duration_minutes, calories_burned, daily_steps,
        avg_heart_rate, resting_heart_rate, blood_pressure_systolic, blood_pressure_diastolic,
        endurance_level, sleep_hours, stress_level, hydration_level, fitness_level
    ]], columns=scaler.feature_names_in_)
    
    # Scale data numerik
    num_scaled = scaler.transform(num_features)
    num_scaled_df = pd.DataFrame(num_scaled, columns=scaler.feature_names_in_)
    
    # Siapkan data kategorikal/waktu dengan One-Hot Encoding manual sesuai struktur model
    cat_features = {
        'year': year, 'month': month, 'day': day, 'day_of_week': day_of_week,
        'gender_M': 1 if gender == "Male" else 0,
        'gender_Other': 1 if gender == "Other" else 0,
        'activity_type_Cycling': 1 if activity_type == "Cycling" else 0,
        'activity_type_Dancing': 1 if activity_type == "Dancing" else 0,
        'activity_type_HIIT': 1 if activity_type == "HIIT" else 0,
        'activity_type_Running': 1 if activity_type == "Running" else 0,
        'activity_type_Swimming': 1 if activity_type == "Swimming" else 0,
        'activity_type_Tennis': 1 if activity_type == "Tennis" else 0,
        'activity_type_Walking': 1 if activity_type == "Walking" else 0,
        'activity_type_Weight Training': 1 if activity_type == "Weight Training" else 0,
        'activity_type_Yoga': 1 if activity_type == "Yoga" else 0,
        'intensity_Low': 1 if intensity == "Low" else 0,
        'intensity_Medium': 1 if intensity == "Medium" else 0,
        'smoking_status_Former': 1 if smoking_status == "Former" else 0,
        'smoking_status_Never': 1 if smoking_status == "Never" else 0
    }
    cat_df = pd.DataFrame([cat_features])
    
    # Gabungkan data untuk prediksi
    final_input = pd.concat([num_scaled_df, cat_df], axis=1)
    
    # Lakukan prediksi
    prediction = model.predict(final_input)
    
    st.success(f"Hasil Prediksi Model: {prediction[0]}")
