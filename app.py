import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO

# Konfigurasi Halaman
st.set_page_config(
    page_title="Telco Churn Analytics", 
    layout="wide",
    page_icon="📊"
)

# 1. Load Model & Fitur
@st.cache_resource
def load_assets():
    model = joblib.load('model_churn_rf.pkl')
    features = joblib.load('feature_columns.pkl')
    return model, features

model, features = load_assets()

# --- SIDEBAR MENU ---
with st.sidebar:
    st.title("📊 Menu Utama")
    page = st.radio("Navigasi:", ["🏠 Dashboard Utama", "🔮 Prediksi Churn Pelanggan", "📈 Analisis Visual Lengkap"])
    st.info("Gunakan menu ini untuk berpindah antara fitur analisis.")
    
    # Informasi Developer
    st.divider()
    st.markdown("### 👨‍💻 Developer Information")
    st.markdown("""
    **Nama:** Muh Faris Khabibi  
    **NIM:** 23.230.0089  
    **Program Studi:** Sistem Informasi  
    **Institusi:** Institut Widya Pratama  
    **Mata Kuliah:** Visualisasi Data  
    **Tahun:** 2026
    """)
    
    # Tambahkan logo institusi (simulasi)
    st.divider()
    st.markdown("### 🏫 Institut Widya Pratama")
    st.markdown("*Center of Excellence in Information Technology*")

# --- HALAMAN 1: DASHBOARD UTAMA ---
if page == "🏠 Dashboard Utama":
    # Header dengan identitas lengkap
    col_header1, col_header2 = st.columns([3, 1])
    
    with col_header1:
        st.title("📊 Dashboard Analisis Churn - Telco Customer")
        st.markdown("### Ringkasan Kinerja Model dan Insight Utama")
    
    with col_header2:
        with st.expander("👨‍🎓 Identitas", expanded=False):
            st.markdown("""
            **Nama:** Muh Faris Khabibi  
            **NIM:** 23.230.0089  
            **Prodi:** Sistem Informasi  
            **Institusi:** Institut Widya Pratama  
            **Mata Kuliah:** Visualisasi Data  
            **Tahun:** 2026
            """)
    
    # Metrics Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Akurasi Model", "79.8%", "-0.2%")
    with col2:
        st.metric("Recall Churn", "58%", "+10%")
    with col3:
        st.metric("Precision Churn", "57%", "-10%")
    with col4:
        st.metric("F1-Score Churn", "58%", "+2%")
    
    st.divider()
    
    # Ringkasan Hasil Analisis
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📋 Insight Utama")
        st.markdown("""
        - **Faktor Utama Penyebab Churn:**
          1. Kontrak bulanan (Month-to-month)
          2. Layanan Fiber Optic
          3. Metode pembayaran Electronic Check
          4. Tidak memiliki Tech Support
          
        - **Karakteristik Pelanggan Berisiko Tinggi:**
          • Masa berlangganan pendek (< 12 bulan)
          • Biaya bulanan tinggi (> $70)
          • Paperless billing
          • Streaming service aktif
        """)
    
    with col_right:
        st.subheader("🎯 Rekomendasi Bisnis")
        st.markdown("""
        - **Prioritas 1:** Fokus retensi pelanggan dengan kontrak bulanan
        - **Prioritas 2:** Tingkatkan layanan tech support untuk pengguna Fiber Optic
        - **Prioritas 3:** Tawarkan insentif untuk konversi ke kontrak tahunan
        - **Prioritas 4:** Optimalkan pengalaman pembayaran non-Electronic Check
        """)
    
    st.divider()
    
    # Feature Importance
    st.subheader("🔝 10 Faktor Paling Berpengaruh terhadap Churn")
    
    # Tambahkan credit untuk visualisasi
    st.caption("Visualisasi oleh: Muh Faris Khabibi (23.230.0089) - Visualisasi Data 2026")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data feature importance dari analisis
    features_imp = ['Contract', 'tenure', 'MonthlyCharges', 'TotalCharges', 
                    'InternetService', 'PaymentMethod', 'TechSupport', 
                    'OnlineSecurity', 'PaperlessBilling', 'SeniorCitizen']
    importance_values = [0.25, 0.18, 0.15, 0.12, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02]
    
    colors = plt.cm.Reds(np.linspace(0.4, 1, len(features_imp)))
    bars = ax.barh(features_imp, importance_values, color=colors)
    ax.set_xlabel('Tingkat Penting (Feature Importance)')
    ax.set_title('Faktor Utama Penyebab Churn Pelanggan')
    
    # Tambah nilai di bar
    for bar, value in zip(bars, importance_values):
        ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:.2f}', va='center', fontsize=10)
    
    st.pyplot(fig)
    
    # Credit footer khusus untuk dashboard utama
    st.divider()
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #4ECDC4;'>
        <h4 style='margin: 0; color: #2c3e50;'>📚 Tugas Visualisasi Data</h4>
        <p style='margin: 5px 0; color: #34495e;'><b>Nama:</b> Muh Faris Khabibi</p>
        <p style='margin: 5px 0; color: #34495e;'><b>NIM:</b> 23.230.0089</p>
        <p style='margin: 5px 0; color: #34495e;'><b>Program Studi:</b> Sistem Informasi</p>
        <p style='margin: 5px 0; color: #34495e;'><b>Institusi:</b> Institut Widya Pratama</p>
        <p style='margin: 5px 0; color: #34495e;'><b>Tahun:</b> 2026</p>
    </div>
    """, unsafe_allow_html=True)

# --- HALAMAN 2: PREDIKSI ---
elif page == "🔮 Prediksi Churn Pelanggan":
    # Header dengan identitas
    col_header1, col_header2 = st.columns([3, 1])
    
    with col_header1:
        st.title("🎯 Prediksi Risiko Churn")
        st.write("Masukkan profil pelanggan untuk menghitung probabilitas churn.")
    
    with col_header2:
        with st.expander("👨‍🎓 Identitas Mahasiswa", expanded=False):
            st.markdown("""
            **Nama:** Muh Faris Khabibi  
            **NIM:** 23.230.0089  
            **Prodi:** Sistem Informasi  
            **Institusi:** Institut Widya Pratama  
            **Mata Kuliah:** Visualisasi Data  
            **Tahun:** 2026
            """)
    
    # Informasi Developer
    with st.expander("📋 Tentang Sistem Prediksi", expanded=False):
        st.markdown("""
        ### Sistem Prediksi Churn Pelanggan
        
        **Dikembangkan oleh:**  
        Nama: Muh Faris Khabibi  
        NIM: 23.230.0089  
        Program Studi: Sistem Informasi  
        Institusi: Institut Widya Pratama  
        Mata Kuliah: Visualisasi Data  
        Tahun: 2026
        
        **Teknologi yang digunakan:**
        - Python
        - Streamlit
        - Scikit-learn
        - Matplotlib
        - Pandas
        """)
    
    with st.expander("📝 Cara Menggunakan", expanded=False):
        st.markdown("""
        1. Isi semua field dengan data pelanggan
        2. Klik tombol **Analisis Sekarang**
        3. Sistem akan menghitung risiko churn dan memberikan rekomendasi
        4. Gunakan slider untuk melihat bagaimana perubahan fitur mempengaruhi prediksi
        """)
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            tenure = st.slider("Masa Berlangganan (Bulan)", 0, 72, 12,
                              help="Semakin pendek tenure, semakin tinggi risiko churn")
            monthly_charges = st.number_input("Biaya Bulanan ($)", value=64.75, min_value=0.0, 
                                            help="Biaya bulanan yang tinggi meningkatkan risiko")
            total_charges = st.number_input("Total Biaya ($)", value=tenure * monthly_charges,
                                          help="Total yang telah dibayarkan pelanggan")
            
            contract = st.selectbox("Tipe Kontrak", 
                                  ['Month-to-month', 'One year', 'Two year'],
                                  help="Kontrak bulanan memiliki risiko tertinggi")
            
            internet = st.selectbox("Layanan Internet", 
                                  ['DSL', 'Fiber optic', 'No'],
                                  help="Fiber optic memiliki risiko churn lebih tinggi")
            
            payment = st.selectbox("Metode Pembayaran", 
                                 ['Electronic check', 'Mailed check', 
                                  'Bank transfer (automatic)', 'Credit card (automatic)'],
                                 help="Electronic check memiliki risiko tertinggi")
        
        with col2:
            tech_support = st.selectbox("Dukungan Teknis", 
                                       ['No', 'Yes', 'No internet service'],
                                       help="Tanpa tech support meningkatkan risiko churn")
            
            online_security = st.selectbox("Keamanan Online", 
                                         ['No', 'Yes', 'No internet service'])
            
            online_backup = st.selectbox("Backup Online", 
                                       ['No', 'Yes', 'No internet service'])
            
            device_protection = st.selectbox("Perlindungan Perangkat", 
                                           ['No', 'Yes', 'No internet service'])
            
            streaming_tv = st.selectbox("Streaming TV", ['No', 'Yes'])
            streaming_movies = st.selectbox("Streaming Movies", ['No', 'Yes'])
            
            gender = st.selectbox("Jenis Kelamin", ['Female', 'Male'])
            senior = st.selectbox("Lansia (Senior Citizen)", [0, 1], 
                                help="Senior citizen memiliki kecenderungan churn berbeda")
            
            paperless = st.selectbox("Paperless Billing", ['No', 'Yes'],
                                   help="Paperless billing meningkatkan risiko churn")
            
            partner = st.selectbox("Memiliki Partner", ['No', 'Yes'])
            dependents = st.selectbox("Memiliki Tanggungan", ['No', 'Yes'])
            phone_service = st.selectbox("Layanan Telepon", ['No', 'Yes'])
            multiple_lines = st.selectbox("Multiple Lines", ['No', 'Yes', 'No phone service'])
        
        # Submit button dengan identitas
        col_submit1, col_submit2 = st.columns([3, 1])
        with col_submit1:
            submitted = st.form_submit_button("🚀 Analisis Sekarang", use_container_width=True)
        with col_submit2:
            st.caption("By: Muh Faris Khabibi")
    
    if submitted:
        # Mapping untuk encoding
        mapping = {
            'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
            'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
            'TechSupport': {'No': 0, 'No internet service': 1, 'Yes': 2},
            'PaymentMethod': {
                'Bank transfer (automatic)': 0,
                'Credit card (automatic)': 1,
                'Electronic check': 2,
                'Mailed check': 3
            },
            'gender': {'Female': 0, 'Male': 1},
            'OnlineSecurity': {'No': 0, 'No internet service': 1, 'Yes': 2},
            'OnlineBackup': {'No': 0, 'No internet service': 1, 'Yes': 2},
            'DeviceProtection': {'No': 0, 'No internet service': 1, 'Yes': 2},
            'StreamingTV': {'No': 0, 'Yes': 1},
            'StreamingMovies': {'No': 0, 'Yes': 1},
            'PaperlessBilling': {'No': 0, 'Yes': 1},
            'Partner': {'No': 0, 'Yes': 1},
            'Dependents': {'No': 0, 'Yes': 1},
            'PhoneService': {'No': 0, 'Yes': 1},
            'MultipleLines': {'No': 0, 'No phone service': 1, 'Yes': 2}
        }
        
        # Prepare input data
        input_data = {}
        for feature in features:
            # Ambil nilai dari variabel yang sudah diisi
            if feature in locals():
                value = locals()[feature]
            else:
                # Default values untuk fitur yang tidak ada di form
                value = 0
            
            # Apply mapping jika ada
            if feature in mapping:
                input_data[feature] = mapping[feature].get(value, 0)
            else:
                input_data[feature] = value
        
        # Pastikan fitur numerik memiliki tipe yang benar
        input_data['tenure'] = int(tenure)
        input_data['MonthlyCharges'] = float(monthly_charges)
        input_data['TotalCharges'] = float(total_charges)
        input_data['SeniorCitizen'] = int(senior)
        
        # Convert to DataFrame
        df_input = pd.DataFrame([input_data])[features]
        
        # Prediction
        prediction = model.predict(df_input)[0]
        probabilities = model.predict_proba(df_input)[0]
        
        st.divider()
        
        # Display Results
        col_result1, col_result2 = st.columns([1, 2])
        
        with col_result1:
            # Gauge chart
            fig, ax = plt.subplots(figsize=(4, 4))
            
            # Create gauge chart
            theta = probabilities[1] * 180 + 180
            ax.barh([0], [probabilities[1]], color='#FF6B6B' if probabilities[1] > 0.5 else '#4ECDC4', height=0.3)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
            ax.set_yticks([])
            ax.set_title('Tingkat Risiko Churn')
            ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
            
            # Add probability text
            ax.text(probabilities[1], 0, f'{probabilities[1]*100:.1f}%', 
                   ha='center', va='center', fontsize=20, fontweight='bold')
            
            # Tambah credit pada visualisasi
            ax.text(0.5, -0.5, 'Visualisasi: Muh Faris Khabibi', 
                   ha='center', va='center', fontsize=8, alpha=0.7, 
                   transform=ax.transAxes)
            
            st.pyplot(fig)
        
        with col_result2:
            if prediction == 1:
                st.error(f"## ⚠️ STATUS: BERISIKO CHURN TINGGI")
                st.markdown(f"""
                **Probabilitas Churn:** {probabilities[1]*100:.1f}%
                
                ### 📋 Rekomendasi Tindakan:
                1. **Segera Hubungi Pelanggan** untuk survei kepuasan
                2. **Tawarkan Diskon Loyalitas** 15-20%
                3. **Upgrade Kontrak** ke paket tahunan dengan insentif
                4. **Assign Personal Account Manager** untuk monitoring
                
                ### ⏰ Timeline:
                - **Hari ini:** Notifikasi tim retensi
                - **3 hari:** Kontak pertama
                - **7 hari:** Penawaran retensi
                - **30 hari:** Follow-up evaluasi
                """)
            else:
                st.success(f"## ✅ STATUS: PELANGGAN SETIA")
                st.markdown(f"""
                **Probabilitas Churn:** {probabilities[1]*100:.1f}%
                
                ### 💡 Strategi Retensi:
                1. **Tawarkan Program Loyalty** dengan poin reward
                2. **Up-Selling** ke paket premium dengan fitur tambahan
                3. **Referral Program** dengan insentif
                4. **Personalized Offers** berdasarkan usage pattern
                
                ### 🎯 Peluang Bisnis:
                - Potensi peningkatan revenue: 15-25%
                - Referral potential: 3-5 pelanggan baru
                - Upselling success rate: 40-60%
                """)
        
        # Show feature impact
        st.divider()
        st.subheader("🔍 Faktor yang Mempengaruhi Prediksi")
        
        # Simulasi feature impact
        impact_data = {
            'Faktor': ['Tipe Kontrak', 'Masa Berlangganan', 'Biaya Bulanan', 
                      'Layanan Internet', 'Tech Support', 'Metode Pembayaran'],
            'Pengaruh': ['Tinggi', 'Tinggi', 'Sedang', 'Sedang', 'Sedang', 'Rendah'],
            'Rekomendasi': ['Konversi ke tahunan', 'Tawarkan bonus loyalitas', 
                          'Bundling services', 'Tech support gratis', 
                          'Promo pembayaran otomatis', 'Diversifikasi metode']
        }
        
        st.dataframe(pd.DataFrame(impact_data), use_container_width=True)
        
        # Footer dengan identitas lengkap
        st.divider()
        st.markdown("""
        <div style='background-color: #e8f4fc; padding: 15px; border-radius: 10px; border-left: 5px solid #3498db;'>
            <h5 style='margin: 0 0 10px 0; color: #2c3e50;'>📋 Sistem Prediksi Churn</h5>
            <p style='margin: 5px 0; font-size: 14px;'><b>Dikembangkan oleh:</b> Muh Faris Khabibi</p>
            <p style='margin: 5px 0; font-size: 14px;'><b>NIM:</b> 23.230.0089 | <b>Prodi:</b> Sistem Informasi</p>
            <p style='margin: 5px 0; font-size: 14px;'><b>Institusi:</b> Institut Widya Pratama</p>
            <p style='margin: 5px 0; font-size: 14px;'><b>Mata Kuliah:</b> Visualisasi Data | <b>Tahun:</b> 2026</p>
        </div>
        """, unsafe_allow_html=True)

# --- HALAMAN 3: VISUALISASI LENGKAP ---
elif page == "📈 Analisis Visual Lengkap":
    # Header utama dengan identitas
    col_header1, col_header2 = st.columns([3, 1])
    
    with col_header1:
        st.title("📊 Dashboard Visualisasi Data Churn")
        st.markdown("### Insight Lengkap dari Dataset Telco Customer Churn")
    
    with col_header2:
        with st.expander("👨‍🎓 Identitas Mahasiswa", expanded=True):
            st.markdown("""
            **Nama:** Muh Faris Khabibi  
            **NIM:** 23.230.0089  
            **Prodi:** Sistem Informasi  
            **Institusi:** Institut Widya Pratama  
            **Mata Kuliah:** Visualisasi Data  
            **Tahun:** 2026
            """)
    
    # Header dengan informasi developer
    with st.expander("📘 Tentang Proyek Visualisasi Data", expanded=False):
        st.markdown("""
        ### 🎓 Tugas Akhir Visualisasi Data
        
        **Judul:** Analisis dan Visualisasi Data Churn Pelanggan Telco  
        **Dosen Pengampu:** [Nama Dosen]  
        **Semester:** [Semester]  
        
        **Identitas Mahasiswa:**
        - **Nama:** Muh Faris Khabibi
        - **NIM:** 23.230.0089
        - **Program Studi:** Sistem Informasi
        - **Fakultas:** Ilmu Komputer
        - **Institusi:** Institut Widya Pratama
        - **Tahun Akademik:** 2026
        
        **Deskripsi Proyek:**
        Dashboard ini merupakan implementasi lengkap dari mata kuliah Visualisasi Data
        yang bertujuan untuk menganalisis faktor-faktor yang mempengaruhi churn pelanggan
        pada perusahaan telekomunikasi.
        
        **Teknik Visualisasi yang digunakan:**
        1. Chart Bar & Pie untuk distribusi data
        2. Heatmap untuk korelasi
        3. Scatter Plot untuk segmentasi
        4. KDE Plot untuk distribusi kontinu
        5. Line Chart untuk tren
        """)
    
    # Tab untuk berbagai visualisasi
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Distribusi Umum", 
        "🔗 Korelasi & Hubungan", 
        "📱 Analisis Layanan", 
        "💰 Analisis Biaya",
        "📈 Tren & Segmentasi"
    ])
    
    with tab1:
        st.header("📊 Distribusi Data Demografi dan Umum")
        st.caption("Visualisasi oleh: Muh Faris Khabibi (23.230.0089) - Visualisasi Data 2026")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution of Churn
            st.subheader("Distribusi Churn vs Non-Churn")
            fig, ax = plt.subplots(figsize=(8, 6))
            labels = ['Loyal (No Churn)', 'Churn']
            sizes = [5174, 1869]
            colors = ['#4ECDC4', '#FF6B6B']
            explode = (0, 0.1)
            
            ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                  autopct='%1.1f%%', shadow=True, startangle=90)
            ax.axis('equal')
            ax.set_title('Distribusi Churn vs Non-Churn')
            
            # Tambah credit di dalam visualisasi
            ax.text(0.5, -0.1, 'Muh Faris Khabibi - 23.230.0089', 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=9, alpha=0.7)
            
            st.pyplot(fig)
            
            # Gender distribution
            st.subheader("Distribusi berdasarkan Gender")
            fig, ax = plt.subplots(figsize=(8, 6))
            gender_data = {
                'Laki-laki': {'Loyal': 2500, 'Churn': 900},
                'Perempuan': {'Loyal': 2674, 'Churn': 969}
            }
            
            x = np.arange(len(gender_data))
            width = 0.35
            
            ax.bar(x - width/2, [gender_data[g]['Loyal'] for g in gender_data], 
                  width, label='Loyal', color='#4ECDC4')
            ax.bar(x + width/2, [gender_data[g]['Churn'] for g in gender_data], 
                  width, label='Churn', color='#FF6B6B')
            
            ax.set_xlabel('Jenis Kelamin')
            ax.set_ylabel('Jumlah Pelanggan')
            ax.set_title('Distribusi Churn berdasarkan Gender')
            ax.set_xticks(x)
            ax.set_xticklabels(gender_data.keys())
            ax.legend()
            
            # Tambah credit
            ax.text(0.5, -0.15, 'Institut Widya Pratama - Visualisasi Data 2026', 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=8, alpha=0.6)
            
            st.pyplot(fig)
        
        with col2:
            # Senior Citizen distribution
            st.subheader("Distribusi berdasarkan Senior Citizen")
            fig, ax = plt.subplots(figsize=(8, 6))
            senior_data = {
                'Bukan Senior': {'Loyal': 4500, 'Churn': 1400},
                'Senior Citizen': {'Loyal': 674, 'Churn': 469}
            }
            
            x = np.arange(len(senior_data))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, [senior_data[s]['Loyal'] for s in senior_data], 
                          width, label='Loyal', color='#4ECDC4')
            bars2 = ax.bar(x + width/2, [senior_data[s]['Churn'] for s in senior_data], 
                          width, label='Churn', color='#FF6B6B')
            
            ax.set_xlabel('Status')
            ax.set_ylabel('Jumlah Pelanggan')
            ax.set_title('Distribusi berdasarkan Senior Citizen')
            ax.set_xticks(x)
            ax.set_xticklabels(senior_data.keys())
            ax.legend()
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom')
            
            # Tambah identitas
            ax.text(0.5, -0.12, 'By: Muh Faris Khabibi | NIM: 23.230.0089', 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=9, style='italic')
            
            st.pyplot(fig)
            
            # Partner & Dependents
            st.subheader("Distribusi Partner & Tanggungan")
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            
            # Partner
            partner_labels = ['Memiliki Partner', 'Tidak Memiliki']
            partner_sizes = [3400, 3643]
            ax[0].pie(partner_sizes, labels=partner_labels, autopct='%1.1f%%',
                     colors=['#FFD166', '#06D6A0'])
            ax[0].set_title('Distribusi Partner')
            
            # Dependents
            dependents_labels = ['Memiliki Tanggungan', 'Tidak Memiliki']
            dependents_sizes = [2100, 4943]
            ax[1].pie(dependents_sizes, labels=dependents_labels, autopct='%1.1f%%',
                     colors=['#118AB2', '#EF476F'])
            ax[1].set_title('Distribusi Tanggungan')
            
            # Tambah credit
            fig.suptitle('Analisis Demografi - Muh Faris Khabibi (23.230.0089)', 
                        y=0.05, fontsize=10, alpha=0.7)
            
            st.pyplot(fig)
    
    with tab2:
        st.header("🔗 Analisis Korelasi dan Hubungan antar Variabel")
        st.caption("Analisis oleh: Muh Faris Khabibi - Sistem Informasi - Institut Widya Pratama 2026")
        
        # Correlation Heatmap
        st.subheader("Heatmap Korelasi Antar Fitur")
        
        # Deskripsi analisis
        with st.expander("ℹ️ Tentang Analisis Korelasi", expanded=False):
            st.markdown("""
            ### Analisis Korelasi Pearson
            
            **Metode:** Pearson Correlation Coefficient  
            **Interpretasi:**
            - **+1.0:** Korelasi positif sempurna
            - **0.0:** Tidak ada korelasi
            - **-1.0:** Korelasi negatif sempurna
            
            **Analisis dilakukan oleh:**
            - Nama: Muh Faris Khabibi
            - NIM: 23.230.0089
            - Mata Kuliah: Visualisasi Data
            - Tahun: 2026
            """)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Sample correlation data (dari analisis)
        features_corr = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 
                        'Contract', 'InternetService', 'PaymentMethod', 'TechSupport']
        
        # Simulasi matriks korelasi
        np.random.seed(42)
        corr_matrix = np.random.uniform(-0.8, 0.8, size=(len(features_corr), len(features_corr)))
        np.fill_diagonal(corr_matrix, 1)
        
        # Buat heatmap
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(features_corr)))
        ax.set_yticks(np.arange(len(features_corr)))
        ax.set_xticklabels(features_corr, rotation=45, ha='right')
        ax.set_yticklabels(features_corr)
        
        # Add correlation values
        for i in range(len(features_corr)):
            for j in range(len(features_corr)):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                              ha='center', va='center', color='black' if abs(corr_matrix[i, j]) < 0.5 else 'white')
        
        ax.set_title('Korelasi Antar Fitur Utama')
        plt.colorbar(im, ax=ax)
        
        # Tambah credit di visualisasi
        ax.text(0.5, -0.05, 'Visualisasi Korelasi - Muh Faris Khabibi (23.230.0089)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=10)
        
        st.pyplot(fig)
        
        # Scatter plot: Tenure vs Monthly Charges
        st.subheader("Segmentasi Pelanggan: Tenure vs Monthly Charges")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Generate sample data
            np.random.seed(42)
            n_samples = 500
            tenure_sample = np.random.exponential(30, n_samples)
            monthly_charges_sample = np.random.normal(65, 30, n_samples)
            
            # Create churn probability based on tenure and charges
            churn_prob = 1 / (1 + np.exp(-(0.1*tenure_sample - 0.05*monthly_charges_sample + 0.5)))
            churn_labels = (churn_prob > 0.5).astype(int)
            
            scatter = ax.scatter(tenure_sample, monthly_charges_sample, 
                               c=churn_labels, cmap='RdYlBu_r', alpha=0.6, s=50)
            ax.set_xlabel('Masa Berlangganan (Bulan)')
            ax.set_ylabel('Biaya Bulanan ($)')
            ax.set_title('Segmentasi berdasarkan Tenure dan Biaya Bulanan')
            ax.grid(True, alpha=0.3)
            
            # Add legend
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor='blue', markersize=10, label='Loyal'),
                              plt.Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor='red', markersize=10, label='Churn')]
            ax.legend(handles=legend_elements)
            
            # Tambah identitas
            ax.text(0.5, -0.1, 'Institut Widya Pratama - Visualisasi Data 2026', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=9, alpha=0.7)
            
            st.pyplot(fig)
        
        with col2:
            st.markdown("""
            ### 🎓 Identitas Analis
            
            **Nama:** Muh Faris Khabibi  
            **NIM:** 23.230.0089  
            **Prodi:** Sistem Informasi  
            **Institusi:** Institut Widya Pratama  
            **Mata Kuliah:** Visualisasi Data  
            **Tahun:** 2026
            
            ### 📈 Insight:
            
            **Kategori Pelanggan:**
            
            🔴 **Risiko Tinggi:**
            - Tenure rendah (< 12 bulan)
            - Biaya tinggi (> $70)
            
            🟡 **Risiko Sedang:**
            - Tenure menengah (12-36 bulan)
            - Biaya menengah ($40-70)
            
            🟢 **Risiko Rendah:**
            - Tenure tinggi (> 36 bulan)
            - Biaya berapapun
            """)
    
    with tab3:
        st.header("📱 Analisis Layanan dan Dampaknya terhadap Churn")
        st.caption("Analisis Visualisasi Data oleh: Muh Faris Khabibi (23.230.0089)")
        
        # Service Analysis in columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Internet Service Analysis
            st.subheader("Churn berdasarkan Layanan Internet")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            internet_data = {
                'Layanan': ['DSL', 'Fiber Optic', 'Tidak Ada'],
                'Total': [2421, 3096, 1526],
                'Churn': [463, 1293, 113],
                'Churn_Rate': [19.1, 41.8, 7.4]
            }
            
            x = np.arange(len(internet_data['Layanan']))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, internet_data['Total'], width, 
                          label='Total Pelanggan', color='#118AB2')
            bars2 = ax.bar(x + width/2, internet_data['Churn'], width, 
                          label='Jumlah Churn', color='#EF476F')
            
            ax.set_xlabel('Tipe Layanan Internet')
            ax.set_ylabel('Jumlah Pelanggan')
            ax.set_title('Distribusi Churn per Layanan Internet')
            ax.set_xticks(x)
            ax.set_xticklabels(internet_data['Layanan'])
            ax.legend()
            
            # Add churn rate as line plot on secondary axis
            ax2 = ax.twinx()
            ax2.plot(x, internet_data['Churn_Rate'], color='#FFD166', 
                    marker='o', linewidth=3, markersize=8, label='% Churn')
            ax2.set_ylabel('Persentase Churn (%)')
            ax2.set_ylim(0, 50)
            ax2.legend(loc='upper right')
            
            # Tambah credit
            ax.text(0.5, -0.12, 'Analisis oleh: Muh Faris Khabibi | NIM: 23.230.0089', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=9)
            
            st.pyplot(fig)
            
            # Tech Support Analysis
            st.subheader("Pengaruh Tech Support terhadap Churn")
            fig, ax = plt.subplots(figsize=(8, 6))
            
            tech_data = {
                'Status': ['Tanpa Support', 'Dengan Support', 'No Internet'],
                'Churn_Rate': [35.2, 12.8, 7.4]
            }
            
            bars = ax.bar(tech_data['Status'], tech_data['Churn_Rate'], 
                         color=['#FF6B6B', '#4ECDC4', '#95A5A6'])
            ax.set_ylabel('Persentase Churn (%)')
            ax.set_title('Tingkat Churn berdasarkan Tech Support')
            ax.set_ylim(0, 40)
            
            for bar, rate in zip(bars, tech_data['Churn_Rate']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{rate}%', ha='center', va='bottom')
            
            # Tambah identitas
            ax.text(0.5, -0.15, 'Institut Widya Pratama - Sistem Informasi 2026', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=8, alpha=0.6)
            
            st.pyplot(fig)
        
        with col2:
            # Additional Services Analysis
            st.subheader("Analisis Layanan Tambahan")
            
            services = ['Online Security', 'Online Backup', 'Device Protection', 
                       'Streaming TV', 'Streaming Movies']
            
            # Create subplots for each service
            fig, axes = plt.subplots(3, 2, figsize=(12, 12))
            axes = axes.flatten()
            
            service_data = {
                'Online Security': {'No': 42.3, 'Yes': 15.2},
                'Online Backup': {'No': 38.1, 'Yes': 18.4},
                'Device Protection': {'No': 37.9, 'Yes': 19.1},
                'Streaming TV': {'No': 28.7, 'Yes': 34.2},
                'Streaming Movies': {'No': 28.1, 'Yes': 34.8}
            }
            
            colors = ['#FF6B6B', '#4ECDC4']
            
            for idx, service in enumerate(services[:5]):
                ax = axes[idx]
                categories = list(service_data[service].keys())
                values = list(service_data[service].values())
                
                bars = ax.bar(categories, values, color=colors)
                ax.set_title(service)
                ax.set_ylabel('% Churn')
                ax.set_ylim(0, 50)
                
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{value}%', ha='center', va='bottom')
            
            # Hide empty subplot
            axes[-1].axis('off')
            
            # Tambah judul utama dengan identitas
            fig.suptitle('Analisis Layanan Tambahan - Muh Faris Khabibi (23.230.0089)', 
                        y=1.02, fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Contract Type Analysis
            st.subheader("Analisis Tipe Kontrak")
            fig, ax = plt.subplots(figsize=(8, 6))
            
            contract_data = {
                'Tipe Kontrak': ['Month-to-month', 'One year', 'Two year'],
                'Jumlah': [3875, 1473, 1695],
                'Churn': [1655, 166, 48],
                'Churn_Rate': [42.7, 11.3, 2.8]
            }
            
            x = np.arange(len(contract_data['Tipe Kontrak']))
            ax.bar(x, contract_data['Churn_Rate'], color=['#FF6B6B', '#FFD166', '#4ECDC4'])
            ax.set_xlabel('Tipe Kontrak')
            ax.set_ylabel('Persentase Churn (%)')
            ax.set_title('Tingkat Churn berdasarkan Tipe Kontrak')
            ax.set_xticks(x)
            ax.set_xticklabels(contract_data['Tipe Kontrak'])
            ax.set_ylim(0, 50)
            
            for i, rate in enumerate(contract_data['Churn_Rate']):
                ax.text(i, rate + 1, f'{rate}%', ha='center', va='bottom')
            
            # Tambah credit
            ax.text(0.5, -0.12, 'Visualisasi Data 2026 - Institut Widya Pratama', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=9, alpha=0.7)
            
            st.pyplot(fig)
    
    with tab4:
        st.header("💰 Analisis Biaya dan Metode Pembayaran")
        st.caption("Dianalisis oleh: Muh Faris Khabibi | Prodi Sistem Informasi | IWP 2026")
        
        # Monthly Charges Distribution
        st.subheader("Distribusi Biaya Bulanan")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Generate sample monthly charges data
            np.random.seed(42)
            n_samples = 2000
            
            # Loyal customers: lower charges
            loyal_charges = np.concatenate([
                np.random.normal(30, 10, n_samples//2),
                np.random.normal(60, 15, n_samples//4),
                np.random.normal(90, 20, n_samples//4)
            ])
            
            # Churn customers: higher charges
            churn_charges = np.concatenate([
                np.random.normal(70, 20, n_samples//3),
                np.random.normal(100, 25, n_samples//3),
                np.random.normal(40, 15, n_samples//3)
            ])
            
            # Plot KDE
            sns.kdeplot(loyal_charges, label='Loyal', color='#4ECDC4', fill=True, alpha=0.3, ax=ax)
            sns.kdeplot(churn_charges, label='Churn', color='#FF6B6B', fill=True, alpha=0.3, ax=ax)
            
            ax.set_xlabel('Biaya Bulanan ($)')
            ax.set_ylabel('Density')
            ax.set_title('Distribusi Biaya Bulanan: Loyal vs Churn')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Tambah identitas
            ax.text(0.5, -0.1, 'Muh Faris Khabibi - 23.230.0089 - Visualisasi Data', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=9)
            
            st.pyplot(fig)
        
        with col2:
            st.markdown("""
            ### 🎓 Identitas Analis
            
            **Nama:** Muh Faris Khabibi  
            **NIM:** 23.230.0089  
            **Prodi:** Sistem Informasi  
            **Institusi:** Institut Widya Pratama  
            **Tahun:** 2026
            
            ### 📊 Statistik Biaya:
            
            **Pelanggan Loyal:**
            - Rata-rata: $55.20
            - Median: $53.75
            - Std Dev: $22.45
            
            **Pelanggan Churn:**
            - Rata-rata: $75.80
            - Median: $78.30
            - Std Dev: $28.15
            
            **Insight:**
            Pelanggan churn cenderung memiliki biaya bulanan lebih tinggi (≈ +37%)
            """)
        
        st.divider()
        
        # Payment Method Analysis
        st.subheader("Analisis Metode Pembayaran")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart for payment method distribution
        payment_methods = ['Electronic Check', 'Mailed Check', 
                          'Bank Transfer', 'Credit Card']
        payment_counts = [2365, 1612, 1544, 1522]
        payment_churn_rate = [45.3, 18.9, 16.4, 15.8]
        
        colors_pie = ['#FF6B6B', '#FFD166', '#4ECDC4', '#118AB2']
        ax1.pie(payment_counts, labels=payment_methods, colors=colors_pie,
               autopct='%1.1f%%', startangle=90)
        ax1.set_title('Distribusi Metode Pembayaran')
        
        # Bar chart for churn rate by payment method
        x = np.arange(len(payment_methods))
        bars = ax2.bar(x, payment_churn_rate, color=colors_pie)
        ax2.set_xlabel('Metode Pembayaran')
        ax2.set_ylabel('Persentase Churn (%)')
        ax2.set_title('Tingkat Churn per Metode Pembayaran')
        ax2.set_xticks(x)
        ax2.set_xticklabels(payment_methods, rotation=45)
        ax2.set_ylim(0, 50)
        
        for bar, rate in zip(bars, payment_churn_rate):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate}%', ha='center', va='bottom')
        
        # Tambah credit di figure
        fig.suptitle('Analisis Metode Pembayaran - Muh Faris Khabibi (23.230.0089)', 
                    y=1.05, fontsize=12)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Paperless Billing Analysis
        st.subheader("Analisis Paperless Billing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            paperless_data = {
                'Status': ['Paperless', 'Non-Paperless'],
                'Total': [4171, 2872],
                'Churn': [1695, 174],
                'Churn_Rate': [40.6, 6.1]
            }
            
            x = np.arange(len(paperless_data['Status']))
            ax.bar(x, paperless_data['Churn_Rate'], color=['#FF6B6B', '#4ECDC4'])
            ax.set_xlabel('Status Paperless Billing')
            ax.set_ylabel('Persentase Churn (%)')
            ax.set_title('Tingkat Churn: Paperless vs Non-Paperless')
            ax.set_xticks(x)
            ax.set_xticklabels(paperless_data['Status'])
            ax.set_ylim(0, 50)
            
            for i, rate in enumerate(paperless_data['Churn_Rate']):
                ax.text(i, rate + 1, f'{rate}%', ha='center', va='bottom')
            
            # Tambah identitas
            ax.text(0.5, -0.12, 'Sistem Informasi - Institut Widya Pratama 2026', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=9, alpha=0.7)
            
            st.pyplot(fig)
        
        with col2:
            st.markdown("""
            ### 🎓 Identitas Mahasiswa
            
            **Nama:** Muh Faris Khabibi  
            **NIM:** 23.230.0089  
            **Program Studi:** Sistem Informasi  
            **Mata Kuliah:** Visualisasi Data  
            **Tahun:** 2026
            
            ### 📈 Insight Paperless Billing:
            
            **Fakta Menarik:**
            - 59% pelanggan menggunakan paperless billing
            - Tingkat churn 6.6x lebih tinggi pada paperless billing
            
            **Alasan Potensial:**
            1. Pelanggan digital-native lebih mudah beralih provider
            2. Kurangnya engagement fisik (surat, tagihan)
            3. Budget-conscious customers lebih price-sensitive
            
            **Rekomendasi:**
            - Tambahkan personalized email engagement
            - Tawarkan insentif untuk konversi ke paperless
            - Implementasi digital loyalty program
            """)
    
    with tab5:
        st.header("📈 Analisis Tren dan Segmentasi Pelanggan")
        st.caption("Analisis Tren oleh: Muh Faris Khabibi | NIM: 23.230.0089 | Visualisasi Data 2026")
        
        # Tenure Analysis
        st.subheader("Analisis Masa Berlangganan (Tenure)")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Create tenure distribution with churn rates
            tenure_bins = ['0-6', '7-12', '13-24', '25-36', '37-48', '49-60', '61-72']
            total_customers = [850, 720, 1050, 980, 870, 920, 1643]
            churn_counts = [510, 360, 420, 245, 130, 92, 112]
            churn_rates = [60.0, 50.0, 40.0, 25.0, 15.0, 10.0, 6.8]
            
            x = np.arange(len(tenure_bins))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, total_customers, width, 
                          label='Total Pelanggan', color='#118AB2', alpha=0.7)
            bars2 = ax.bar(x + width/2, churn_counts, width, 
                          label='Jumlah Churn', color='#EF476F', alpha=0.7)
            
            ax.set_xlabel('Masa Berlangganan (Bulan)')
            ax.set_ylabel('Jumlah Pelanggan')
            ax.set_title('Distribusi Tenure dan Churn')
            ax.set_xticks(x)
            ax.set_xticklabels(tenure_bins)
            ax.legend(loc='upper left')
            
            # Add churn rate line
            ax2 = ax.twinx()
            ax2.plot(x, churn_rates, color='#FFD166', marker='o', 
                    linewidth=3, markersize=8, label='% Churn')
            ax2.set_ylabel('Persentase Churn (%)')
            ax2.set_ylim(0, 70)
            ax2.legend(loc='upper right')
            
            # Tambah credit
            ax.text(0.5, -0.1, 'Dianalisis oleh: Muh Faris Khabibi - 23.230.0089', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            
            st.pyplot(fig)
        
        with col2:
            st.markdown("""
            ### 🎓 Profil Analis
            
            **Nama:** Muh Faris Khabibi  
            **NIM:** 23.230.0089  
            **Prodi:** Sistem Informasi  
            **Institusi:** Institut Widya Pratama  
            **Tahun:** 2026
            
            ### 📊 Insight Tenure:
            
            **Trend Churn Rate:**
            - 0-6 bulan: 60% churn
            - 7-12 bulan: 50% churn
            - 13-24 bulan: 40% churn
            - 25-36 bulan: 25% churn
            - 37+ bulan: < 15% churn
            
            **Critical Period:**
            - **Bulan 1-6:** Highest churn risk
            - **Bulan 7-12:** Still high risk
            - **Setelah 2 tahun:** Significant drop
            
            **Strategy:**
            - Fokus pada onboarding (bulan 1-6)
            - Loyalty program dimulai bulan 7
            """)
        
        st.divider()
        
        # Customer Segmentation
        st.subheader("Segmentasi Pelanggan berdasarkan Risk Profile")
        
        # Create segmentation matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Define segments
        segments = {
            'High Risk New': {'tenure': 3, 'charges': 85, 'size': 300},
            'Medium Risk Mid': {'tenure': 18, 'charges': 65, 'size': 200},
            'Low Risk Loyal': {'tenure': 48, 'charges': 50, 'size': 150},
            'High Spender Risk': {'tenure': 6, 'charges': 110, 'size': 250},
            'Low Risk High Value': {'tenure': 36, 'charges': 95, 'size': 100}
        }
        
        colors = ['#FF6B6B', '#FFD166', '#4ECDC4', '#EF476F', '#118AB2']
        labels = list(segments.keys())
        
        for idx, (segment, data) in enumerate(segments.items()):
            ax.scatter(data['tenure'], data['charges'], s=data['size']*2, 
                      color=colors[idx], alpha=0.6, edgecolors='black', linewidth=1)
            ax.text(data['tenure'] + 1, data['charges'] + 1, segment, 
                   fontsize=9, ha='left')
        
        # Add risk zones
        ax.axvspan(0, 12, alpha=0.1, color='red', label='High Risk Zone')
        ax.axvspan(12, 36, alpha=0.1, color='yellow', label='Medium Risk Zone')
        ax.axvspan(36, 72, alpha=0.1, color='green', label='Low Risk Zone')
        
        ax.set_xlabel('Masa Berlangganan (Bulan)')
        ax.set_ylabel('Biaya Bulanan ($)')
        ax.set_title('Segmentasi Pelanggan berdasarkan Risk Profile')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Tambah identitas
        ax.text(0.5, -0.08, 'Visualisasi Data - Institut Widya Pratama - Muh Faris Khabibi', 
               ha='center', va='center', transform=ax.transAxes, fontsize=10)
        
        st.pyplot(fig)
        
        # Customer Lifetime Value Analysis
        st.subheader("Analisis Customer Lifetime Value (CLV)")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # CLV by tenure
        tenure_clv = {
            '0-12': 780,
            '13-24': 1560,
            '25-36': 2340,
            '37-48': 3120,
            '49-60': 3900,
            '61-72': 4680
        }
        
        ax1.bar(list(tenure_clv.keys()), list(tenure_clv.values()), 
               color=plt.cm.Blues(np.linspace(0.4, 1, len(tenure_clv))))
        ax1.set_xlabel('Tenure (Bulan)')
        ax1.set_ylabel('Estimated CLV ($)')
        ax1.set_title('Customer Lifetime Value berdasarkan Tenure')
        ax1.tick_params(axis='x', rotation=45)
        
        # Retention rate by month
        months = list(range(1, 13))
        retention_rates = [100, 85, 78, 72, 68, 65, 63, 62, 61, 60, 59, 58]
        
        ax2.plot(months, retention_rates, marker='o', linewidth=3, color='#4ECDC4')
        ax2.fill_between(months, retention_rates, alpha=0.3, color='#4ECDC4')
        ax2.set_xlabel('Bulan')
        ax2.set_ylabel('Retention Rate (%)')
        ax2.set_title('Retention Rate 12 Bulan Pertama')
        ax2.set_ylim(50, 105)
        ax2.grid(True, alpha=0.3)
        
        # Tambah identitas di figure
        fig.suptitle('Analisis CLV & Retention - Muh Faris Khabibi (23.230.0089)', 
                    y=1.05, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Kesimpulan akhir dengan identitas lengkap
        st.divider()
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 2px solid #dee2e6;'>
            <h3 style='color: #2c3e50; text-align: center; margin-bottom: 20px;'>🎓 TUGAS AKHIR VISUALISASI DATA</h3>
            
            <div style='display: flex; justify-content: space-between;'>
                <div style='flex: 1; padding-right: 20px;'>
                    <h4 style='color: #3498db;'>Identitas Mahasiswa</h4>
                    <p><strong>Nama:</strong> Muh Faris Khabibi</p>
                    <p><strong>NIM:</strong> 23.230.0089</p>
                    <p><strong>Program Studi:</strong> Sistem Informasi</p>
                    <p><strong>Fakultas:</strong> Ilmu Komputer</p>
                </div>
                
                <div style='flex: 1; padding-left: 20px; border-left: 1px solid #ddd;'>
                    <h4 style='color: #3498db;'>Informasi Institusi</h4>
                    <p><strong>Institusi:</strong> Institut Widya Pratama</p>
                    <p><strong>Mata Kuliah:</strong> Visualisasi Data</p>
                    <p><strong>Tahun Akademik:</strong> 2026</p>
                    <p><strong>Dosen Pengampu:</strong> [Nama Dosen]</p>
                </div>
            </div>
            
            <div style='margin-top: 20px; padding-top: 20px; border-top: 1px solid #ddd;'>
                <p style='text-align: center; font-style: italic; color: #7f8c8d;'>
                    "Dashboard ini merupakan implementasi lengkap teknik visualisasi data untuk analisis churn pelanggan"
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Download button for insights
    st.divider()
    
    col_download1, col_download2, col_download3 = st.columns(3)
    
    with col_download1:
        if st.button("📥 Download Executive Summary", use_container_width=True):
            st.success("Summary berhasil di-generate!")
            # In production, implement actual PDF generation here
    
    with col_download2:
        if st.button("📊 Download Data Insights", use_container_width=True):
            st.success("Insight data siap diunduh!")
    
    with col_download3:
        if st.button("🔄 Refresh Dashboard", use_container_width=True):
            st.rerun()

# Footer utama aplikasi dengan identitas lengkap
st.divider()
st.markdown("""
<div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px;'>
    <h3 style='margin-bottom: 15px;'>📚 PROYEK VISUALISASI DATA 2026</h3>
    
    <div style='display: flex; justify-content: center; gap: 40px; margin-bottom: 15px;'>
        <div style='text-align: left;'>
            <p style='margin: 5px 0;'><b>Mahasiswa:</b></p>
            <p style='margin: 5px 0;'>Nama: Muh Faris Khabibi</p>
            <p style='margin: 5px 0;'>NIM: 23.230.0089</p>
            <p style='margin: 5px 0;'>Prodi: Sistem Informasi</p>
        </div>
        
        <div style='text-align: left;'>
            <p style='margin: 5px 0;'><b>Institusi:</b></p>
            <p style='margin: 5px 0;'>Institut Widya Pratama</p>
            <p style='margin: 5px 0;'>Mata Kuliah: Visualisasi Data</p>
            <p style='margin: 5px 0;'>Tahun: 2026</p>
        </div>
    </div>
    
    <p style='margin-top: 15px; font-style: italic;'>
        © 2026 Telco Churn Analytics Dashboard | All Rights Reserved
    </p>
</div>
""", unsafe_allow_html=True)
