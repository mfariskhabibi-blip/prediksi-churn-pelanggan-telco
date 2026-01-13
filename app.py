import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set konfigurasi halaman agar lebih luas
st.set_page_config(page_title="Dashboard Prediksi Churn", layout="wide")

# 1. Load Model dan Kolom
@st.cache_resource
def load_assets():
    model = joblib.load('model_churn_rf.pkl')
    features = joblib.load('feature_columns.pkl')
    return model, features

model, features = load_assets()

# --- HEADER ---
st.title("📊 Telco Customer Churn Dashboard")
st.markdown("""
Aplikasi ini memprediksi risiko pelanggan berhenti berlangganan (churn) menggunakan model **Random Forest** yang telah dioptimasi dengan **SMOTE** dan **GridSearchCV**.
""")

# --- SIDEBAR INPUT ---
st.sidebar.header("Input Data Pelanggan")
def user_input_features():
    tenure = st.sidebar.slider("Masa Berlangganan (Bulan)", 0, 72, 12)
    monthly_charges = st.sidebar.number_input("Biaya Bulanan ($)", value=65.0)
    contract = st.sidebar.selectbox("Tipe Kontrak", ['Month-to-month', 'One year', 'Two year'])
    internet = st.sidebar.selectbox("Layanan Internet", ['DSL', 'Fiber optic', 'No'])
    tech_support = st.sidebar.selectbox("Dukungan Teknis", ['No', 'Yes', 'No internet service'])
    payment = st.sidebar.selectbox("Metode Pembayaran", [
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ])
    gender = st.sidebar.selectbox("Jenis Kelamin", ['Female', 'Male'])
    senior = st.sidebar.selectbox("Lansia (Senior Citizen)", [0, 1])
    paperless = st.sidebar.selectbox("Tagihan Tanpa Kertas", ['Yes', 'No'])
    partner = st.sidebar.selectbox("Memiliki Pasangan", ['Yes', 'No'])
    
    data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': tenure * monthly_charges,
        'SeniorCitizen': senior,
        'Contract': contract,
        'InternetService': internet,
        'TechSupport': tech_support,
        'PaymentMethod': payment,
        'gender': gender,
        'Partner': partner,
        'PaperlessBilling': paperless
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# --- PREPROCESSING ---
def preprocess_input(df, feat_list):
    # Mapping sesuai LabelEncoder pada file Untitled4.ipynb
    mapping = {
        'gender': {'Female': 0, 'Male': 1},
        'Partner': {'No': 0, 'Yes': 1},
        'PaperlessBilling': {'No': 0, 'Yes': 1},
        'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
        'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
        'TechSupport': {'No': 0, 'No internet service': 1, 'Yes': 2},
        'PaymentMethod': {
            'Bank transfer (automatic)': 0, 'Credit card (automatic)': 1,
            'Electronic check': 2, 'Mailed check': 3
        }
    }
    
    # Isi kolom default yang tidak ada di input sidebar
    processed_df = pd.DataFrame(0, index=[0], columns=feat_list)
    
    # Masukkan data numerik
    processed_df['tenure'] = df['tenure']
    processed_df['MonthlyCharges'] = df['MonthlyCharges']
    processed_df['TotalCharges'] = df['TotalCharges']
    processed_df['SeniorCitizen'] = df['SeniorCitizen']
    
    # Masukkan data kategorikal dengan mapping
    for col, maps in mapping.items():
        if col in df.columns:
            processed_df[col] = df[col].map(maps)
            
    return processed_df

df_final = preprocess_input(input_df, features)

# --- MAIN PAGE: HASIL PREDIKSI ---
col_pred, col_viz = st.columns([1, 2])

with col_pred:
    st.subheader("Hasil Analisis")
    prediction = model.predict(df_final)[0]
    probability = model.predict_proba(df_final)[0][1]

    if prediction == 1:
        st.error(f"### ⚠️ RISIKO CHURN: {probability*100:.1f}%")
        st.write("**Rekomendasi:** Berikan penawaran khusus atau diskon retensi segera.")
    else:
        st.success(f"### ✅ PELANGGAN LOYAL: {(1-probability)*100:.1f}%")
        st.write("**Rekomendasi:** Pertahankan layanan dan tawarkan program loyalitas.")

    st.write("---")
    st.write("**Detail Data Input:**")
    st.dataframe(input_df.T, height=300)

with col_viz:
    st.subheader("Visualisasi Faktor Utama")
    
    # Grafik 1: Distribusi Monthly Charges (Simulasi berdasarkan input)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(x=[20, 40, 60, 80, 100], y=[0.1, 0.5, 0.8, 0.4, 0.2], label="Database Umum", ax=ax, fill=True)
    ax.axvline(input_df['MonthlyCharges'][0], color='red', linestyle='--', label='Posisi Pelanggan')
    ax.set_title("Posisi Biaya Bulanan Pelanggan")
    ax.legend()
    st.pyplot(fig)

    # Grafik 2: Tingkat Risiko Berdasarkan Tenure
    st.write("Dampak Masa Berlangganan terhadap Loyalitas:")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    # Contoh visualisasi dampak tenure yang ada di file analisis
    tenure_val = input_df['tenure'][0]
    risk_level = "Tinggi" if tenure_val < 12 else "Rendah"
    sns.barplot(x=['Masa Berlangganan Saat Ini'], y=[tenure_val], palette='viridis', ax=ax2)
    ax2.set_ylim(0, 72)
    ax2.set_ylabel("Bulan")
    st.pyplot(fig2)

st.divider()
st.caption("Aplikasi dibuat menggunakan Streamlit | Model: RandomForestClassifier + SMOTE")
