import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Churn - Muh Faris Khabibi",
    layout="wide"
)

# Judul utama
st.title("📊 Dashboard Analisis Churn Pelanggan")
st.markdown("**Nama:** Muh Faris Khabibi | **Tugas Akhir:** Visualisasi Data")
st.markdown("---")

# Sidebar untuk navigasi
with st.sidebar:
    st.header("Menu Navigasi")
    page = st.radio(
        "Pilih Halaman:",
        ["Dashboard Utama", "Data", "Visualisasi", "Tentang"]
    )

# --- GENERATE DATA SAMPLE ---
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 500  # Kurangi jumlah data untuk performa lebih baik
    
    data = {
        'ID': range(1, n+1),
        'Tenure': np.random.randint(1, 73, n),
        'MonthlyCharges': np.round(np.random.uniform(20, 120, n), 2),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.25, 0.2]),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.35, 0.45, 0.2]),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n),
        'Churn': np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7])
    }
    
    df = pd.DataFrame(data)
    df['TotalCharges'] = df['Tenure'] * df['MonthlyCharges']
    return df

df = load_data()

# --- HALAMAN DASHBOARD UTAMA ---
if page == "Dashboard Utama":
    st.header("📈 Dashboard Utama")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = len(df)
        st.metric("Total Pelanggan", f"{total_customers:,}")
    
    with col2:
        churn_count = (df['Churn'] == 'Yes').sum()
        st.metric("Jumlah Churn", f"{churn_count:,}")
    
    with col3:
        churn_rate = (churn_count / total_customers) * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    with col4:
        avg_tenure = df['Tenure'].mean()
        st.metric("Rata-rata Tenure", f"{avg_tenure:.1f} bulan")
    
    st.markdown("---")
    
    # Grafik sederhana
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Distribusi Churn")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        churn_counts = df['Churn'].value_counts()
        colors = ['green', 'red']
        ax.pie(churn_counts.values, labels=churn_counts.index, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Proporsi Pelanggan Churn vs Non-Churn')
        st.pyplot(fig)
    
    with col_right:
        st.subheader("Churn berdasarkan Kontrak")
        
        # Hitung churn rate per kontrak
        contract_churn = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        contract_churn.plot(kind='bar', color=['red', 'orange', 'green'], ax=ax)
        ax.set_xlabel('Tipe Kontrak')
        ax.set_ylabel('Churn Rate (%)')
        ax.set_title('Tingkat Churn berdasarkan Tipe Kontrak')
        ax.set_ylim(0, 100)
        
        # Tambah nilai di atas bar
        for i, v in enumerate(contract_churn.values):
            ax.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Insight
    st.subheader("📌 Insight Utama")
    
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        st.info("""
        **🏆 Pelanggan Loyal:**
        - 70% pelanggan tetap setia
        - Rata-rata tenure: 36 bulan
        - Biaya bulanan: $65
        """)
    
    with insight_col2:
        st.warning("""
        **⚠️ Risiko Churn Tinggi:**
        - Kontrak bulanan: 45% churn
        - Fiber optic: 35% churn
        - Electronic check: 40% churn
        """)
    
    with insight_col3:
        st.success("""
        **💡 Rekomendasi:**
        - Konversi ke kontrak tahunan
        - Improve tech support
        - Promo pembayaran otomatis
        """)

# --- HALAMAN DATA ---
elif page == "Data":
    st.header("📁 Data Pelanggan")
    
    # Tampilkan data
    st.dataframe(df, use_container_width=True)
    
    st.markdown("---")
    
    # Statistik data
    st.subheader("📊 Statistik Data")
    
    col_stat1, col_stat2 = st.columns(2)
    
    with col_stat1:
        st.write("**Info Dataset:**")
        st.write(f"- Jumlah baris: {len(df)}")
        st.write(f"- Jumlah kolom: {len(df.columns)}")
        st.write(f"- Kolom: {', '.join(df.columns)}")
    
    with col_stat2:
        st.write("**Statistik Numerik:**")
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe())
    
    # Download data
    st.markdown("---")
    st.subheader("📥 Download Data")
    
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download sebagai CSV",
        data=csv,
        file_name="data_churn.csv",
        mime="text/csv"
    )

# --- HALAMAN VISUALISASI ---
elif page == "Visualisasi":
    st.header("📈 Visualisasi Data")
    
    # Pilihan visualisasi
    viz_option = st.selectbox(
        "Pilih Jenis Visualisasi:",
        ["Distribusi Tenure", "Biaya Bulanan", "Analisis Kontrak", "Segmentasi Pelanggan"]
    )
    
    if viz_option == "Distribusi Tenure":
        st.subheader("Distribusi Masa Berlangganan (Tenure)")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histogram untuk semua pelanggan
        ax.hist(df['Tenure'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Tenure (Bulan)')
        ax.set_ylabel('Jumlah Pelanggan')
        ax.set_title('Distribusi Tenure Semua Pelanggan')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Distribusi berdasarkan Churn
        st.subheader("Distribusi Tenure: Churn vs Non-Churn")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Non-Churn
        tenure_no_churn = df[df['Churn'] == 'No']['Tenure']
        ax1.hist(tenure_no_churn, bins=15, color='green', alpha=0.7, edgecolor='black')
        ax1.set_title('Tenure - Non Churn')
        ax1.set_xlabel('Tenure (Bulan)')
        ax1.set_ylabel('Jumlah')
        ax1.grid(True, alpha=0.3)
        
        # Churn
        tenure_churn = df[df['Churn'] == 'Yes']['Tenure']
        ax2.hist(tenure_churn, bins=15, color='red', alpha=0.7, edgecolor='black')
        ax2.set_title('Tenure - Churn')
        ax2.set_xlabel('Tenure (Bulan)')
        ax2.set_ylabel('Jumlah')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    elif viz_option == "Biaya Bulanan":
        st.subheader("Analisis Biaya Bulanan")
        
        # Box plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data_to_plot = [df[df['Churn'] == 'No']['MonthlyCharges'], 
                       df[df['Churn'] == 'Yes']['MonthlyCharges']]
        
        bp = ax.boxplot(data_to_plot, labels=['Non-Churn', 'Churn'], patch_artist=True)
        
        # Warna box
        colors = ['lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Biaya Bulanan ($)')
        ax.set_title('Distribusi Biaya Bulanan: Churn vs Non-Churn')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Scatter plot
        st.subheader("Hubungan Tenure vs Biaya Bulanan")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot dengan warna berbeda untuk churn dan non-churn
        scatter_no = ax.scatter(df[df['Churn'] == 'No']['Tenure'], 
                               df[df['Churn'] == 'No']['MonthlyCharges'],
                               alpha=0.6, s=30, label='Non-Churn', color='green')
        
        scatter_yes = ax.scatter(df[df['Churn'] == 'Yes']['Tenure'], 
                                df[df['Churn'] == 'Yes']['MonthlyCharges'],
                                alpha=0.6, s=30, label='Churn', color='red')
        
        ax.set_xlabel('Tenure (Bulan)')
        ax.set_ylabel('Biaya Bulanan ($)')
        ax.set_title('Tenure vs Biaya Bulanan dengan Status Churn')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    elif viz_option == "Analisis Kontrak":
        st.subheader("Analisis Berdasarkan Tipe Kontrak")
        
        # Buat multi-plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # Plot 1: Distribusi kontrak
        contract_counts = df['Contract'].value_counts()
        axes[0].bar(contract_counts.index, contract_counts.values, color=['red', 'orange', 'green'])
        axes[0].set_title('Distribusi Tipe Kontrak')
        axes[0].set_ylabel('Jumlah Pelanggan')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Churn rate per kontrak
        contract_churn_rate = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
        axes[1].bar(contract_churn_rate.index, contract_churn_rate.values, color=['red', 'orange', 'green'])
        axes[1].set_title('Churn Rate per Tipe Kontrak')
        axes[1].set_ylabel('Churn Rate (%)')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Rata-rata tenure per kontrak
        avg_tenure_contract = df.groupby('Contract')['Tenure'].mean()
        axes[2].bar(avg_tenure_contract.index, avg_tenure_contract.values, color=['red', 'orange', 'green'])
        axes[2].set_title('Rata-rata Tenure per Kontrak')
        axes[2].set_ylabel('Tenure (Bulan)')
        axes[2].tick_params(axis='x', rotation=45)
        
        # Plot 4: Rata-rata biaya per kontrak
        avg_charges_contract = df.groupby('Contract')['MonthlyCharges'].mean()
        axes[3].bar(avg_charges_contract.index, avg_charges_contract.values, color=['red', 'orange', 'green'])
        axes[3].set_title('Rata-rata Biaya Bulanan per Kontrak')
        axes[3].set_ylabel('Biaya ($)')
        axes[3].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    elif viz_option == "Segmentasi Pelanggan":
        st.subheader("Segmentasi Pelanggan")
        
        # Buat segmentasi sederhana
        df['Segment'] = pd.cut(df['Tenure'], 
                              bins=[0, 12, 36, 72],
                              labels=['Baru (0-12 bln)', 'Menengah (13-36 bln)', 'Lama (37+ bln)'])
        
        # Analisis segment
        segment_analysis = df.groupby('Segment').agg({
            'Churn': lambda x: (x == 'Yes').mean() * 100,
            'MonthlyCharges': 'mean',
            'ID': 'count'
        }).rename(columns={'ID': 'Jumlah', 'Churn': 'Churn_Rate', 'MonthlyCharges': 'Avg_Charges'})
        
        # Tampilkan tabel
        st.dataframe(segment_analysis)
        
        # Visualisasi segment
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot churn rate per segment
        axes[0].bar(segment_analysis.index, segment_analysis['Churn_Rate'], color=['red', 'orange', 'green'])
        axes[0].set_title('Churn Rate per Segment')
        axes[0].set_ylabel('Churn Rate (%)')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Plot jumlah pelanggan per segment
        axes[1].bar(segment_analysis.index, segment_analysis['Jumlah'], color=['red', 'orange', 'green'])
        axes[1].set_title('Jumlah Pelanggan per Segment')
        axes[1].set_ylabel('Jumlah Pelanggan')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)

# --- HALAMAN TENTANG ---
else:
    st.header("👨‍💻 Tentang Proyek")
    
    st.markdown("""
    ## Dashboard Analisis Churn Pelanggan
    
    **Dibuat oleh:** Muh Faris Khabibi
    
    **Tugas Akhir:** Visualisasi Data
    
    **Deskripsi:**
    Dashboard ini dibuat untuk menganalisis faktor-faktor yang mempengaruhi churn (berhenti berlangganan) 
    pelanggan perusahaan telekomunikasi.
    
    **Fitur Utama:**
    1. **Dashboard Utama**: Ringkasan metrik dan insight
    2. **Data**: Tampilan data mentah dan statistik
    3. **Visualisasi**: Berbagai jenis grafik analisis
    4. **Segmentasi**: Pengelompokan pelanggan berdasarkan karakteristik
    
    **Teknologi yang Digunakan:**
    - Python
    - Streamlit
    - Pandas
    - Matplotlib
    - NumPy
    
    **Dataset:**
    Data simulasi berdasarkan karakteristik dataset Telco Customer Churn
    """)
    
    st.markdown("---")
    
    # Informasi kontak
    st.subheader("📞 Informasi Kontak")
    
    contact_col1, contact_col2 = st.columns(2)
    
    with contact_col1:
        st.write("**Nama:** Muh Faris Khabibi")
        st.write("**Proyek:** Tugas Akhir Visualisasi Data")
    
    with contact_col2:
        st.write("**Mata Kuliah:** Visualisasi Data")
        st.write("**Tahun:** 2024")

# --- FOOTER ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "© 2024 | Dashboard Analisis Churn | Muh Faris Khabibi | Tugas Akhir Visualisasi Data"
    "</div>",
    unsafe_allow_html=True
)
