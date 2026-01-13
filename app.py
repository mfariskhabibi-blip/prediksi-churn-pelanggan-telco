import streamlit as st
import pandas as pd
import joblib

# Load Model dan Kolom
model = joblib.load('model_churn_rf.pkl')
features = joblib.load('feature_columns.pkl')

st.set_page_config(page_title="Prediksi Churn Pelanggan", layout="wide")

st.title("📊 Prediksi Churn Pelanggan Telco")
st.write("Aplikasi ini memprediksi kemungkinan pelanggan berhenti berlangganan menggunakan Machine Learning.")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Informasi Layanan")
        tenure = st.slider("Masa Berlangganan (Bulan)", 0, 72, 1)
        contract = st.selectbox("Tipe Kontrak", ['Month-to-month', 'One year', 'Two year'])
        internet = st.selectbox("Layanan Internet", ['DSL', 'Fiber optic', 'No'])
        tech_support = st.selectbox("Dukungan Teknis", ['No', 'Yes', 'No internet service'])
        monthly_charges = st.number_input("Biaya Bulanan ($)", min_value=0.0, value=50.0)

    with col2:
        st.subheader("Profil & Pembayaran")
        gender = st.selectbox("Jenis Kelamin", ['Female', 'Male'])
        senior = st.selectbox("Lansia (Senior Citizen)", [0, 1])
        payment = st.selectbox("Metode Pembayaran", [
            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
        ])
        paperless = st.selectbox("Tagihan Tanpa Kertas", ['Yes', 'No'])
        # Field lain diisi default agar tidak terlalu panjang di form
        partner = st.selectbox("Memiliki Pasangan", ['Yes', 'No'])

    submit = st.form_submit_button("Analisis Risiko")

if submit:
    # 1. Persiapkan Data (Mapping Kategorikal sesuai LabelEncoder)
    data = {col: 0 for col in features} # Inisialisasi semua kolom dengan 0
    
    # Mapping Manual (Sesuaikan dengan urutan alfabet LabelEncoder)
    data['tenure'] = tenure
    data['MonthlyCharges'] = monthly_charges
    data['TotalCharges'] = tenure * monthly_charges
    data['SeniorCitizen'] = senior
    
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
    
    # Update data berdasarkan input
    data['gender'] = mapping['gender'][gender]
    data['Partner'] = mapping['Partner'][partner]
    data['Contract'] = mapping['Contract'][contract]
    data['InternetService'] = mapping['InternetService'][internet]
    data['TechSupport'] = mapping['TechSupport'][tech_support]
    data['PaymentMethod'] = mapping['PaymentMethod'][payment]
    data['PaperlessBilling'] = mapping['PaperlessBilling'][paperless]

    # 2. Prediksi
    df_input = pd.DataFrame([data])[features]
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]

    # 3. Tampilan Hasil
    st.divider()
    if prediction == 1:
        st.error(f"### ⚠️ Hasil: Berisiko Churn ({probability*100:.1f}%)")
        st.warning("Rekomendasi: Kirimkan penawaran khusus atau diskon untuk mempertahankan pelanggan ini.")
    else:
        st.success(f"### ✅ Hasil: Pelanggan Loyal ({ (1-probability)*100:.1f}%)")
        st.info("Rekomendasi: Berikan apresiasi atau tawarkan program loyalitas tambahan.")
