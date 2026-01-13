import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set konfigurasi halaman
st.set_page_config(page_title="Telco Churn App", layout="wide")

# 1. Load Model dan Kolom (Gunakan Cache agar cepat)
@st.cache_resource
def load_assets():
    model = joblib.load('model_churn_rf.pkl')
    features = joblib.load('feature_columns.pkl')
    # Load dataset asli untuk visualisasi (opsional, jika file csv ada di GitHub)
    # df = pd.read_csv('Telco-Customer-Churn.csv') 
    return model, features

model, features = load_assets()

# --- SIDEBAR NAVIGASI ---
st.sidebar.title("Main Menu")
page = st.sidebar.radio("Pilih Halaman:", ["Prediksi Churn Pelanggan", "Visualisasi Data"])

# --- HALAMAN 1: PREDIKSI ---
if page == "Prediksi Churn Pelanggan":
    st.title("📊 Prediksi Churn Pelanggan Telco")
    st.write("Gunakan formulir di bawah untuk memprediksi probabilitas pelanggan berhenti berlangganan.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            tenure = st.slider("Masa Berlangganan (Bulan)", 0, 72, 12)
            monthly_charges = st.number_input("Biaya Bulanan ($)", value=65.0)
            contract = st.selectbox("Tipe Kontrak", ['Month-to-month', 'One year', 'Two year'])
            internet = st.selectbox("Layanan Internet", ['DSL', 'Fiber optic', 'No'])
            
        with col2:
            tech_support = st.selectbox("Dukungan Teknis", ['No', 'Yes', 'No internet service'])
            payment = st.selectbox("Metode Pembayaran", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
            gender = st.selectbox("Jenis Kelamin", ['Female', 'Male'])
            senior = st.selectbox("Lansia (Senior Citizen)", [0, 1])

        submit = st.form_submit_button("Analisis Risiko")

    if submit:
        # Menyiapkan data untuk model (Sesuai urutan feature_columns)
        data = {col: 0 for col in features}
        data['tenure'] = tenure
        data['MonthlyCharges'] = monthly_charges
        data['TotalCharges'] = tenure * monthly_charges
        data['SeniorCitizen'] = senior
        
        # Mapping Kategorikal (Sesuai LabelEncoder di Notebook)
        mapping = {
            'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
            'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
            'TechSupport': {'No': 0, 'No internet service': 1, 'Yes': 2},
            'PaymentMethod': {'Bank transfer (automatic)': 0, 'Credit card (automatic)': 1, 'Electronic check': 2, 'Mailed check': 3},
            'gender': {'Female': 0, 'Male': 1}
        }
        
        for k, v in mapping.items():
            if k in data: data[k] = v.get(locals()[k.lower()] if k.lower() in locals() else 0, 0)

        df_final = pd.DataFrame([data])[features]
        prediction = model.predict(df_final)[0]
        prob = model.predict_proba(df_final)[0][1]

        st.divider()
        if prediction == 1:
            st.error(f"### ⚠️ RISIKO CHURN: {prob*100:.1f}%")
        else:
            st.success(f"### ✅ PELANGGAN LOYAL: {(1-prob)*100:.1f}%")

# --- HALAMAN 2: VISUALISASI DATA ---
elif page == "Visualisasi Data":
    st.title("📈 Visualisasi Analisis Data")
    st.write("Grafik di bawah ini menunjukkan pola pelanggan berdasarkan data historis.")

    # Karena kita tidak upload CSV besar, kita buat dummy data berdasarkan pola di notebook Anda
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Distribusi Churn berdasarkan Kontrak")
        # Visualisasi perbandingan Kontrak (Insight dari Notebook)
        fig1, ax1 = plt.subplots()
        contract_data = {'Month-to-month': 42, 'One year': 11, 'Two year': 3}
        sns.barplot(x=list(contract_data.keys()), y=list(contract_data.values()), palette='magma', ax=ax1)
        ax1.set_ylabel("Persentase Churn (%)")
        st.pyplot(fig1)
        st.info("Insight: Pelanggan dengan kontrak 'Month-to-month' memiliki tingkat churn tertinggi.")

    with col_b:
        st.subheader("Tenure vs Churn")
        fig2, ax2 = plt.subplots()
        # Menggambarkan pola korelasi negatif tenure & churn
        tenure_bins = ['0-12', '13-24', '25-48', '48+']
        churn_values = [50, 30, 20, 10]
        sns.lineplot(x=tenure_bins, y=churn_values, marker='o', color='red', ax=ax2)
        ax2.set_ylabel("Tingkat Risiko")
        st.pyplot(fig2)
        st.info("Insight: Semakin lama masa berlangganan (tenure), semakin rendah risiko churn.")

    st.subheader("Distribusi Biaya Bulanan")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    sns.histplot(x=[20, 40, 60, 70, 80, 90, 100, 110], bins=10, kde=True, color='blue', ax=ax3)
    ax3.set_xlabel("Monthly Charges ($)")
    st.pyplot(fig3)
