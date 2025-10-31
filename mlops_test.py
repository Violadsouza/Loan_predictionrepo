import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# Load Model
# ----------------------------
model = joblib.load("best_loan_model.pkl")

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(page_title="🏦 Loan Approval Prediction", layout="wide")

# ----------------------------
# Custom CSS for Styling
# ----------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #E3F2FD, #E1F5FE);
    font-family: "Segoe UI", sans-serif;
}
div[data-testid="stSidebar"] {
    background-color: #1565C0;
    color: white;
}
h1 {
    color: #0D47A1;
    text-align: center;
    font-size: 36px !important;
}
.stButton>button {
    background-color: #0D47A1;
    color: white;
    border-radius: 10px;
    font-size: 18px;
    height: 3em;
    width: 100%;
}
.stButton>button:hover {
    background-color: #1565C0;
}
.result-card {
    background-color: #E3F2FD;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    border: 2px solid #0D47A1;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Title
# ----------------------------
st.title("🏦 Loan Approval Prediction App")

st.markdown("### Fill in the details below to check your loan eligibility 👇")

# ----------------------------
# Layout: Two Columns
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    Gender = st.selectbox("👨 Gender", ["Male", "Female"])
    Married = st.selectbox("💍 Married", ["Yes", "No"])
    Dependents = st.selectbox("👶 Dependents", ["0", "1", "2", "3+"])
    Education = st.selectbox("🎓 Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("💼 Self Employed", ["Yes", "No"])
    Property_Area = st.selectbox("🏠 Property Area", ["Urban", "Semiurban", "Rural"])

with col2:
    ApplicantIncome = st.number_input("💰 Applicant Income", min_value=0)
    CoapplicantIncome = st.number_input("🤝 Coapplicant Income", min_value=0)
    LoanAmount = st.number_input("🏦 Loan Amount", min_value=0)
    Loan_Amount_Term = st.number_input("📆 Loan Term (in months)", min_value=0)
    Credit_History = st.selectbox("💳 Credit History", [1.0, 0.0])

# ----------------------------
# Feature Engineering
# ----------------------------
Total_Income = ApplicantIncome + CoapplicantIncome
Log_Income = np.log(Total_Income + 1)

# ----------------------------
# Encoding Maps
# ----------------------------
gender_map = {"Male": 1, "Female": 0}
married_map = {"Yes": 1, "No": 0}
dependents_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
education_map = {"Graduate": 1, "Not Graduate": 0}
self_employed_map = {"Yes": 1, "No": 0}
property_area_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}

# ----------------------------
# Prediction Button
# ----------------------------
if st.button("🔍 Predict Loan Approval"):
    input_df = pd.DataFrame({
        "Gender": [gender_map[Gender]],
        "Married": [married_map[Married]],
        "Dependents": [dependents_map[Dependents]],
        "Education": [education_map[Education]],
        "Self_Employed": [self_employed_map[Self_Employed]],
        "ApplicantIncome": [ApplicantIncome],
        "CoapplicantIncome": [CoapplicantIncome],
        "LoanAmount": [LoanAmount],
        "Loan_Amount_Term": [Loan_Amount_Term],
        "Credit_History": [Credit_History],
        "Property_Area": [property_area_map[Property_Area]],
        "Total_Income": [Total_Income],
        "Log_Income": [Log_Income]
    })

    prediction = model.predict(input_df)

    st.markdown("---")
    st.subheader("📊 Prediction Result:")

    if prediction[0] == 1:
        st.markdown(
            '<div class="result-card" style="background-color:#C8E6C9;">'
            '<h3>✅ Loan is Likely to be APPROVED!</h3>'
            '<p>Congratulations! You meet the eligibility criteria.</p>'
            '</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="result-card" style="background-color:#FFCDD2;">'
            '<h3>❌ Loan is Likely to be REJECTED</h3>'
            '<p>Based on current information, you may not qualify.</p>'
            '</div>', unsafe_allow_html=True)

# ----------------------------
# Optional Sidebar Info
# ----------------------------
with st.sidebar:
    st.header("ℹ️ About This App")
    st.write("""
    This interactive app uses a machine learning model
    to predict whether a loan will be approved or not.
    Provide your details, and the model evaluates
    your eligibility based on historical data.
    """)
    st.write("Built with ❤️ using **Streamlit and Python**.")
