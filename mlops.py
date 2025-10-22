import streamlit as st
import joblib
import pandas as pd

# Load your trained model
model = joblib.load("best_loan_model.pkl")

# Title
st.title("Loan Approval Prediction")

# Input fields for each feature your model needs
ApplicantIncome = st.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
LoanAmount = st.number_input("Loan Amount", min_value=0)
Loan_Amount_Term = st.number_input("Loan Amount Term (in days)", min_value=0)
Credit_History = st.selectbox("Credit History (1=good, 0=bad)", [1, 0])

# When user clicks predict button
if st.button("Predict Loan Approval"):
    # Prepare input data in a DataFrame with exact feature names
    input_df = pd.DataFrame({
        'ApplicantIncome': [ApplicantIncome],
        'CoapplicantIncome': [CoapplicantIncome],
        'LoanAmount': [LoanAmount],
        'Loan_Amount_Term': [Loan_Amount_Term],
        'Credit_History': [Credit_History]
    })

    # Make prediction using the loaded model
    prediction = model.predict(input_df)

    # Display result
    if prediction[0] == 1:
        st.success("Loan is likely to be APPROVED.")
    else:
        st.error("Loan is likely to be REJECTED.")
