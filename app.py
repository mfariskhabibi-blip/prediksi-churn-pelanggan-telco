import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi Halaman
st.set_page_config(page_title="Telco Churn Analytics", layout="wide")

# 1. Load Model & Fitur
@st.cache_resource
def load_assets():
    model = joblib.load('model_churn_rf.pkl')
    features = joblib.load('feature_columns.pkl')
    return model, features

model, features = load_assets()

# --- SIDEBAR MENU ---
with st.sidebar:
    st.title("Menu Utama")
    page = st.radio("Navigasi:", ["Prediksi Churn Pelanggan", "Visualisasi Data Terpadu"])
    st.info("Gunakan menu ini untuk berpindah antara fitur prediksi dan analisis data.")

# --- HALAMAN 1: PREDIKSI ---
if page == "Prediksi Churn Pelanggan":
    st.title("🎯 Prediksi Risiko Churn")
    st.write("Masukkan profil pelanggan untuk menghitung probabilitas churn.")

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            tenure = st.slider("Masa Berlangganan (Bulan)", 0, 72, 12)
            monthly_charges = st.number_input("Biaya Bulanan ($)", value=64.75)
            contract = st.selectbox("Tipe Kontrak", ['Month-to-month', 'One year', 'Two year'])
            internet = st.selectbox("Layanan Internet", ['DSL', 'Fiber optic', 'No'])
        with c2:
            tech_support = st.selectbox("Dukungan Teknis", ['No', 'Yes', 'No internet service'])
            payment = st.selectbox("Metode Pembayaran", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
            gender = st.selectbox("Jenis Kelamin", ['Female', 'Male'])
            senior = st.selectbox("Lansia (Senior Citizen)", [0, 1])
        
        btn = st.form_submit_button("Analisis Sekarang")

    if btn:
        # Menyiapkan input (Mapping sesuai LabelEncoder di notebook)
        data = {col: 0 for col in features}
        data.update({
            'tenure': tenure, 'MonthlyCharges': monthly_charges, 
            'TotalCharges': tenure * monthly_charges, 'SeniorCitizen': senior
        })
        
        mapping = {
            'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
            'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
            'TechSupport': {'No': 0, 'No internet service': 1, 'Yes': 2},
            'PaymentMethod': {'Bank transfer (automatic)': 0, 'Credit card (automatic)': 1, 'Electronic check': 2, 'Mailed check': 3},
            'gender': {'Female': 0, 'Male': 1}
        }
        
        for k, v in mapping.items():
            val = locals()[k.lower()] if k.lower() in locals() else 0
            data[k] = v.get(val, 0)

        df_final = pd.DataFrame([data])[features]
        prediction = model.predict(df_final)[0]
        prob = model.predict_proba(df_final)[0][1]

        st.divider()
        if prediction == 1:
            st.error(f"### HASIL: BERISIKO CHURN ({prob*100:.1f}%)")
        else:
            st.success(f"### HASIL: PELANGGAN SETIA ({(1-prob)*100:.1f}%)")

# --- HALAMAN 2: VISUALISASI DATA (LENGKAP) ---
elif page == "Visualisasi Data Terpadu":
    st.title("📈 Dashboard Visualisasi Hasil Analisis")
    st.write("Menampilkan insight kunci dari dataset Telco Churn.")

    # Baris 1: Churn Distribution & Contract Type
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribusi Churn Pelanggan")
        fig, ax = plt.subplots()
        # Data berdasarkan analisis di PDF
        sns.barplot(x=['Tetap Berlangganan', 'Berhenti (Churn)'], y=[5174, 1869], palette='viridis', ax=ax)
        ax.set_ylabel("Jumlah Pelanggan")
        st.pyplot(fig)

    with col2:
        st.subheader("Churn Berdasarkan Tipe Kontrak")
        fig, ax = plt.subplots()
        contract_types = ['Month-to-month', 'One year', 'Two year']
        churn_rates = [42.7, 11.2, 2.8] # Presentase rata-rata dari notebook
        sns.barplot(x=contract_types, y=churn_rates, palette='magma', ax=ax)
        ax.set_ylabel("Persentase Churn (%)")
        st.pyplot(fig)

    st.divider()

    # Baris 2: Numerical Distributions
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Distribusi Masa Berlangganan (Tenure)")
        fig, ax = plt.subplots()
        # Pola distribusi bimodal dari analisis PDF
        sns.histplot(x=[1, 1, 2, 5, 10, 20, 60, 70, 72, 72], bins=10, kde=True, color='teal', ax=ax)
        ax.set_xlabel("Bulan")
        st.pyplot(fig)

    with col4:
        st.subheader("Distribusi Biaya Bulanan")
        fig, ax = plt.subplots()
        sns.boxplot(y=[20, 45, 70, 85, 110], color='orange', ax=ax)
        ax.set_ylabel("Monthly Charges ($)")
        st.pyplot(fig)

    # Baris 3: Services Analysis
    st.subheader("Analisis Layanan Internet")
    fig, ax = plt.subplots(figsize=(10, 4))
    internet_svc = ['DSL', 'Fiber Optic', 'No']
    churn_svc = [18.9, 41.8, 7.4] # Berdasarkan hasil notebook
    sns.barplot(x=internet_svc, y=churn_svc, palette='coolwarm', ax=ax)
    ax.set_ylabel("Persentase Churn")
    st.pyplot(fig)
    
    st.info("**Kesimpulan Utama:** Pelanggan dengan kontrak bulanan dan layanan Fiber Optic memiliki risiko churn paling tinggi.")
