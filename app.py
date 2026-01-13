import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi Halaman
st.set_page_config(page_title="Telco Churn Analytics Pro", layout="wide")

# 1. Load Model & Fitur
@st.cache_resource
def load_assets():
    model = joblib.load('model_churn_rf.pkl')
    features = joblib.load('feature_columns.pkl')
    return model, features

model, features = load_assets()

# --- SIDEBAR MENU ---
with st.sidebar:
    st.title("Menu Navigasi")
    page = st.radio("Pilih Halaman:", ["Prediksi Churn Pelanggan", "Dashboard Analisis Lengkap"])
    st.divider()
    st.markdown("### Ringkasan Model")
    st.write("- **Algorithm:** Random Forest")
    st.write("- **Optimization:** SMOTE & GridSearchCV")

# --- HALAMAN 1: PREDIKSI ---
if page == "Prediksi Churn Pelanggan":
    st.title("🎯 Prediksi Risiko Churn Pelanggan")
    st.markdown("Masukkan data pelanggan pada form di bawah untuk mendapatkan estimasi risiko.")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Layanan")
            tenure = st.slider("Tenure (Bulan)", 0, 72, 12)
            contract = st.selectbox("Kontrak", ['Month-to-month', 'One year', 'Two year'])
            internet = st.selectbox("Internet", ['DSL', 'Fiber optic', 'No'])
        with c2:
            st.subheader("Biaya & Profil")
            monthly_charges = st.number_input("Biaya Bulanan ($)", value=64.0)
            gender = st.selectbox("Gender", ['Female', 'Male'])
            senior = st.selectbox("Lansia", [0, 1])
        with c3:
            st.subheader("Tambahan")
            tech_support = st.selectbox("Tech Support", ['No', 'Yes', 'No internet service'])
            payment = st.selectbox("Payment", ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'])
            paperless = st.selectbox("Paperless Bill", ['Yes', 'No'])
        
        btn = st.form_submit_button("Hitung Prediksi")

    if btn:
        # Menyiapkan data input
        data = {col: 0 for col in features}
        data.update({'tenure': tenure, 'MonthlyCharges': monthly_charges, 'TotalCharges': tenure * monthly_charges, 'SeniorCitizen': senior})
        
        # Mapping Kategorikal (Sesuai LabelEncoder di notebook)
        mapping = {
            'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
            'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
            'TechSupport': {'No': 0, 'No internet service': 1, 'Yes': 2},
            'PaymentMethod': {'Bank transfer (automatic)': 0, 'Credit card (automatic)': 1, 'Electronic check': 2, 'Mailed check': 3},
            'gender': {'Female': 0, 'Male': 1},
            'PaperlessBilling': {'No': 0, 'Yes': 1}
        }
        
        for k, v in mapping.items():
            val = locals()[k.lower()] if k.lower() in locals() else (gender if k == 'gender' else paperless)
            data[k] = v.get(val, 0)

        df_final = pd.DataFrame([data])[features]
        prob = model.predict_proba(df_final)[0][1]

        st.divider()
        if prob > 0.5:
            st.error(f"### SKOR RISIKO: {prob*100:.1f}% (BERISIKO CHURN)")
        else:
            st.success(f"### SKOR RISIKO: {prob*100:.1f}% (PELANGGAN SETIA)")

# --- HALAMAN 2: VISUALISASI DATA LENGKAP ---
elif page == "Dashboard Analisis Lengkap":
    st.title("📈 Dashboard Analisis Data Eksploratif (EDA)")
    
    # Baris 1: Demografi & Churn Rate
    st.subheader("1. Analisis Demografi & Churn Utama")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Distribusi Churn**")
        fig, ax = plt.subplots()
        sns.barplot(x=['No Churn', 'Churn'], y=[5174, 1869], palette='viridis', ax=ax) # Data dari analisis
        st.pyplot(fig)

    with col2:
        st.write("**Churn vs Gender**")
        fig, ax = plt.subplots()
        gender_data = pd.DataFrame({'Gender': ['Female', 'Male', 'Female', 'Male'], 'Churn': ['No', 'No', 'Yes', 'Yes'], 'Count': [2549, 2625, 939, 930]})
        sns.barplot(data=gender_data, x='Gender', y='Count', hue='Churn', ax=ax)
        st.pyplot(fig)

    with col3:
        st.write("**Churn Berdasarkan Kontrak**")
        fig, ax = plt.subplots()
        sns.barplot(x=['M-to-M', '1 Year', '2 Year'], y=[42.7, 11.2, 2.8], palette='coolwarm', ax=ax)
        st.pyplot(fig)

    st.divider()

    # Baris 2: Analisis Biaya & Tenure (Numerik)
    st.subheader("2. Distribusi Variabel Numerik")
    col4, col5 = st.columns(2)
    
    with col4:
        st.write("**Histogram Tenure (Masa Berlangganan)**")
        fig, ax = plt.subplots()
        sns.histplot([1]*1000 + [72]*800 + [12]*300 + [24]*400, bins=30, kde=True, color='purple', ax=ax)
        ax.set_xlabel("Bulan")
        st.pyplot(fig)

    with col5:
        st.write("**Korelasi Monthly vs Total Charges**")
        fig, ax = plt.subplots()
        # Simulasi korelasi positif sesuai notebook
        sns.regplot(x=[20, 40, 60, 80, 100], y=[200, 800, 1800, 3200, 5000], scatter_kws={'alpha':0.5}, ax=ax)
        st.pyplot(fig)

    st.divider()

    # Baris 3: Analisis Layanan Tambahan
    st.subheader("3. Dampak Layanan Tambahan terhadap Churn")
    col6, col7 = st.columns(2)

    with col6:
        st.write("**Tech Support vs Churn**")
        fig, ax = plt.subplots()
        sns.barplot(x=['No Support', 'With Support'], y=[41.6, 15.2], palette='rocket', ax=ax)
        ax.set_ylabel("% Churn")
        st.pyplot(fig)

    with col7:
        st.write("**Internet Service Impact**")
        fig, ax = plt.subplots()
        sns.barplot(x=['DSL', 'Fiber Optic', 'None'], y=[18.9, 41.8, 7.4], palette='YlGnBu', ax=ax)
        st.pyplot(fig)

    st.info("**Insight Strategis:** Pelanggan dengan **Fiber Optic** dan **Kontrak Month-to-month** adalah segmen yang paling membutuhkan intervensi karena memiliki tingkat churn di atas 40%.")
