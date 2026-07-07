import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Health & Fitness Dashboard",
    page_icon="🏃‍♂️",
    layout="wide"
)

# 2. Fungsi untuk Memuat Data (dengan Cache agar Cepat)
@st.cache_data
def load_data():
    # Membaca data dan memastikan kolom tanggal terbaca dengan benar
    df = pd.read_csv("health_fitness_dataset.csv")
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("File 'health_fitness_dataset.csv' tidak ditemukan. Pastikan file tersebut berada di folder yang sama dengan app.py!")
    st.stop()

# 3. Judul Aplikasi
st.title("🏃‍♂️ Health & Fitness Analytics Dashboard")
st.markdown("Aplikasi interaktif untuk menganalisis data kesehatan, aktivitas fisik, dan kebugaran pengguna.")
st.write("---")

# 4. Sidebar untuk Filter Interaktif
st.sidebar.header("Filter Data")

# Filter Jenis Kelamin
gender_options = ["Semua"] + list(df['gender'].dropna().unique())
selected_gender = st.sidebar.selectbox("Pilih Jenis Kelamin:", gender_options)

# Filter Jenis Aktivitas
activity_options = ["Semua"] + list(df['activity_type'].dropna().unique())
selected_activity = st.sidebar.multiselect("Pilih Jenis Aktivitas:", activity_options, default="Semua")

# Filter Umur (Slider)
min_age = int(df['age'].min())
max_age = int(df['age'].max())
selected_age = st.sidebar.slider("Pilih Rentang Usia:", min_age, max_age, (min_age, max_age))

# Menerapkan Filter ke Dataframe
df_filtered = df.copy()

if selected_gender != "Semua":
    df_filtered = df_filtered[df_filtered['gender'] == selected_gender]

if "Semua" not in selected_activity and len(selected_activity) > 0:
    df_filtered = df_filtered[df_filtered['activity_type'].isin(selected_activity)]

df_filtered = df_filtered[(df_filtered['age'] >= selected_age[0]) & (df_filtered['age'] <= selected_age[1])]


# 5. Baris Metrik Utama (KPIs)
st.subheader("📊 Ringkasan Metrik Kesehatan")
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_steps = df_filtered['daily_steps'].mean()
    st.metric(label="Rata-rata Langkah Harian", value=f"{avg_steps:,.0f} langkah")

with col2:
    avg_calories = df_filtered['calories_burned'].mean()
    # Menggunakan f-string sederhana tanpa ekspresi kompleks di dalam kurung kurawal
    st.metric(label="Rata-rata Kalori Terbakar", value=f"{avg_calories:.1f} kkal")

with col3:
    avg_sleep = df_filtered['sleep_hours'].mean()
    st.metric(label="Rata-rata Tidur", value=f"{avg_sleep:.1f} Jam")

with col4:
    avg_bmi = df_filtered['bmi'].mean()
    st.metric(label="Rata-rata BMI", value=f"{avg_bmi:.1f}")

st.write("---")


# 6. Bagian Visualisasi Grafik
st.subheader("📈 Analisis & Tren Visual")

tab1, tab2, tab3 = st.tabs(["Aktivitas & Kalori", "Distribusi Kesehatan", "Data Mentah"])

with tab1:
    st.markdown("### Hubungan Aktivitas dan Kalori Terbakar")
    # Bar Chart: Rata-rata kalori per jenis aktivitas
    cal_per_activity = df_filtered.groupby('activity_type')['calories_burned'].mean().reset_index()
    fig_bar = px.bar(
        cal_per_activity, 
        x='activity_type', 
        y='calories_burned',
        title="Rata-rata Kalori yang Terbakar Berdasarkan Jenis Aktivitas",
        labels={'activity_type': 'Jenis Aktivitas', 'calories_burned': 'Kalori (kkal)'},
        color='calories_burned',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Scatter Plot: Durasi vs Kalori
    fig_scatter = px.scatter(
        df_filtered,
        x='duration_minutes',
        y='calories_burned',
        color='intensity',
        title="Hubungan Durasi Latihan dan Kalori Terbakar Berdasarkan Intensitas",
        labels={'duration_minutes': 'Durasi (Menit)', 'calories_burned': 'Kalori (kkal)', 'intensity': 'Intensitas'}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    st.markdown("### Distribusi Parameter Kesehatan")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Histogram: Distribusi Umur
        fig_age = px.histogram(
            df_filtered, 
            x='age', 
            nbins=20, 
            title="Distribusi Umur Pengguna",
            labels={'age': 'Usia'},
            color_discrete_sequence=['#4A90E2']
        )
        st.plotly_chart(fig_age, use_container_width=True)
        
    with col_chart2:
        # Boxplot: Tidur berdasarkan Tingkat Stress
        fig_box = px.box(
            df_filtered,
            x='stress_level',
            y='sleep_hours',
            title="Hubungan Tingkat Stres dengan Durasi Tidur",
            labels={'stress_level': 'Tingkat Stres', 'sleep_hours': 'Jam Tidur'},
            color='stress_level'
        )
        st.plotly_chart(fig_box, use_container_width=True)

with tab3:
    st.markdown("### Eksplorasi Data Mentah Terfilter")
    st.dataframe(df_filtered, use_container_width=True)
    
    # Fitur Download Data Terfilter
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Unduh Data Terfilter sebagai CSV",
        data=csv,
        file_name='health_fitness_filtered.csv',
        mime='text/csv',
    )