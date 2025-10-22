import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your trained model
model = joblib.load("best_loan_model.pkl")

st.title("Loan Approval Prediction")

# Input widgets for all required features:
LoanAmount = st.number_input("Loan Amount", min_value=0)
Loan_Amount_Term = st.number_input("Loan Amount Term (in days)", min_value=0)
Credit_History = st.selectbox("Credit History (1=good, 0=bad)", [1.0, 0.0])

# Property_Area: Choose as string, then map to number for model input
Property_Area_choice = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
property_area_map = {"Urban": 0, "Semiurban": 1, "Rural": 2}
Property_Area = property_area_map[Property_Area_choice]

Total_Income = st.number_input("Total Income", min_value=0)
# Calculate Log_Income
Log_Income = np.log(Total_Income + 1)  # Add 1 to avoid log(0) error

if st.button("Predict Loan Approval"):
    # Prepare DataFrame with all required columns in training order
    input_df = pd.DataFrame({
        'LoanAmount': [LoanAmount],
        'Loan_Amount_Term': [Loan_Amount_Term],
        'Credit_History': [Credit_History],
        'Property_Area': [Property_Area],
        'Total_Income': [Total_Income],
        'Log_Income': [Log_Income]
    })
    
    st.write("Input sent to model:", input_df)
    
    prediction = model.predict(input_df)
    if prediction[0] == 1:
        st.success("Loan is likely to be APPROVED.")
    else:
        st.error("Loan is likely to be REJECTED.")
