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
        with open("scaler (6).pkl", "rb") as f:
            scaler = pickle.load(f)
        
        # Load Label Encoder
        with open("label_encoder (1).pkl", "rb") as f:
            le = pickle.load(f)
            
        # [OPSIONAL] Silakan ganti nama file model sesuai dengan model Anda
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
    st.caption("Arahkan kursor atau sentuh ikon tanda tanya (?) pada setiap input untuk melihat keterangan.")
    
    # 2. Membuat Input Form dengan Fitur Keterangan (parameter 'help')
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Usia (Tahun)", min_value=1, max_value=120, value=30, 
                              help="Usia pasien saat ini dalam satuan tahun.")
        height_cm = st.number_input("Tinggi Badan (cm)", min_value=50, max_value=250, value=165,
                                    help="Tinggi badan dalam satuan sentimeter.")
        weight_kg = st.number_input("Berat Badan (kg)", min_value=10, max_value=200, value=60,
                                    help="Berat badan dalam satuan kilogram.")
        bmi = st.number_input("BMI (Indeks Massa Tubuh)", min_value=5.0, max_value=50.0, value=22.0,
                              help="Body Mass Index (BMI). Dihitung dari Berat (kg) / Tinggi (m) kuadrat.")
        duration_minutes = st.number_input("Durasi Latihan Harian (Menit)", min_value=0, max_value=480, value=30,
                                           help="Total durasi aktivitas fisik atau olahraga dalam sehari.")
        calories_burned = st.number_input("Kalori Terbakar (kkal)", min_value=0.0, max_value=5000.0, value=200.0,
                                          help="Estimasi jumlah energi/kalori yang dibakar per hari.")
        daily_steps = st.number_input("Jumlah Langkah Harian", min_value=0, max_value=50000, value=7000,
                                      help="Total jumlah langkah kaki yang ditempuh dalam satu hari penuh.")
        avg_heart_rate = st.number_input("Rata-rata Detak Jantung", min_value=40, max_value=220, value=80,
                                         help="Detak jantung rata-rata selama beraktivitas (BPM).")

    with col2:
        resting_heart_rate = st.number_input("Detak Jantung Istirahat", min_value=30, max_value=150, value=65,
                                             help="Detak jantung saat kondisi tubuh rileks/bangun tidur (BPM).")
        blood_pressure_systolic = st.number_input("Tekanan Darah Sistolik", min_value=70, max_value=250, value=120,
                                                  help="Tekanan saat jantung memompa darah (angka atas pada tensi).")
        blood_pressure_diastolic = st.number_input("Tekanan Darah Diastolik", min_value=40, max_value=150, value=80,
                                                   help="Tekanan saat jantung beristirahat (angka bawah pada tensi).")
        endurance_level = st.number_input("Tingkat Daya Tahan (Endurance)", min_value=0.0, max_value=20.0, value=10.0,
                                          help="Skor kapasitas kardiovaskular atau ketahanan fisik subjek.")
        sleep_hours = st.number_input("Durasi Tidur (Jam)", min_value=1.0, max_value=24.0, value=7.0,
                                      help="Total waktu tidur atau istirahat di malam hari.")
        stress_level = st.slider("Tingkat Stres", min_value=1, max_value=10, value=5,
                                 help="Skor tingkat stres psikologis atau beban pikiran (1 Rendah - 10 Tinggi).")
        hydration_level = st.number_input("Tingkat Hidrasi (Liter)", min_value=0.0, max_value=10.0, value=2.5,
                                          help="Jumlah konsumsi air putih dalam satu hari.")
        fitness_level = st.number_input("Tingkat Kebugaran (Fitness)", min_value=0.0, max_value=20.0, value=10.0,
                                        help="Skor keseluruhan indeks kebugaran fisik Anda.")

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
        result = prediction_label[0]
        
        # Tampilkan Hasil
        st.subheader("📊 Hasil Prediksi:")
        st.success(f"Kondisi Kesehatan yang Terdeteksi/Diprediksi: **{result}**")
        
        # Fitur Tambahan: Keterangan Berdasarkan Output Prediksi
        with st.expander("ℹ️ Lihat Keterangan Hasil Prediksi"):
            if result == "Asthma":
                st.markdown("""
                **Mengenai Asma (Asthma):**
                Kondisi kronis pada saluran pernapasan yang ditandai dengan peradangan dan penyempitan saluran napas, sehingga menimbulkan sesak napas, batuk, atau mengorok. 
                *Disarankan untuk menjaga kebersihan udara sekitar, menghindari pemicu alergi, dan berkonsultasi dengan dokter.*
                """)
            elif result == "Diabetes":
                st.markdown("""
                **Mengenai Diabetes:**
                Penyakit jangka panjang yang ditandai dengan tingginya kadar gula (glukosa) darah akibat tubuh tidak dapat memproduksi atau menggunakan insulin secara efektif.
                *Disarankan untuk membatasi konsumsi gula berlebih, memantau pola makan, serta rutin berolahraga ringan.*
                """)
            elif result == "Hypertension":
                st.markdown("""
                **Mengenai Hipertensi (Hypertension):**
                Kondisi medis di mana tekanan darah di arteri meningkat secara kronis (tekanan darah tinggi). Hal ini meningkatkan beban kerja jantung.
                *Disarankan untuk mengurangi konsumsi garam (natrium), mengelola tingkat stres, dan menjaga berat badan ideal.*
                """)
            else:
                st.markdown("""
                **Kondisi Lainnya / Normal:**
                Model memprediksi status kesehatan di luar klasifikasi utama atau dalam batas parameter tertentu. 
                *Tetap jaga pola makan seimbang, hidrasi cukup, tidur teratur, dan lakukan aktivitas fisik secara konsisten.*
                """)
            
            st.info("⚠️ **Disclaimer:** Hasil prediksi ini murni berbasis data/model Machine Learning untuk tujuan edukasi atau skrining awal, dan **bukan** pengganti diagnosis medis resmi dari dokter.")
else:
    st.info("💡 Sediakan file `model.pkl` Anda di dalam folder agar tombol prediksi dapat berfungsi dengan baik.")
