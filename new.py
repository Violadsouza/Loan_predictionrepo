import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("best_loan_model.pkl")
st.title("Loan Approval Prediction")

# Categorical input fields (matching your column names)
Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self_Employed", ["Yes", "No"])
Property_Area = st.selectbox("Property_Area", ["Urban", "Semiurban", "Rural"])

# Numeric input fields
ApplicantIncome = st.number_input("ApplicantIncome", min_value=0)
CoapplicantIncome = st.number_input("CoapplicantIncome", min_value=0)
LoanAmount = st.number_input("LoanAmount", min_value=0)
Loan_Amount_Term = st.number_input("Loan_Amount_Term", min_value=0)
Credit_History = st.selectbox("Credit_History", [1.0, 0.0])

# Calculated features for total and log income
Total_Income = ApplicantIncome + CoapplicantIncome
Log_Income = np.log(Total_Income + 1)

# Mapping categorical features to numeric values (change if your encoders differ)
gender_map = {"Male": 1, "Female": 0}
married_map = {"Yes": 1, "No": 0}
dependents_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
education_map = {"Graduate": 1, "Not Graduate": 0}
self_employed_map = {"Yes": 1, "No": 0}
property_area_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}

if st.button("Predict Loan Approval"):
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

    st.write("Input to model:", input_df)

    prediction = model.predict(input_df)
    if prediction[0] == 1:
        st.success("Loan is likely to be APPROVED.")
    else:
        st.error("Loan is likely to be REJECTED.")
