# app.py

import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("model/churn_model.pkl", "rb"))

st.set_page_config(page_title="Customer Churn Predictor")
st.title("📞 Telecom Customer Churn Prediction")

st.markdown("Fill in the customer details below to predict churn.")

# User Input
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Has Partner", ["Yes", "No"])
dependents = st.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (in months)", 0, 72)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No"])
online_backup = st.selectbox("Online Backup", ["Yes", "No"])
device_protection = st.selectbox("Device Protection", ["Yes", "No"])
tech_support = st.selectbox("Tech Support", ["Yes", "No"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

# Mapping
def preprocess_inputs():
    return pd.DataFrame([{
        'gender': 1 if gender == "Male" else 0,
        'SeniorCitizen': senior_citizen,
        'Partner': 1 if partner == "Yes" else 0,
        'Dependents': 1 if dependents == "Yes" else 0,
        'tenure': tenure,
        'PhoneService': 1 if phone_service == "Yes" else 0,
        'MultipleLines': 1 if multiple_lines == "Yes" else 0,
        'InternetService': {"DSL": 0, "Fiber optic": 1, "No": 2}[internet_service],
        'OnlineSecurity': 1 if online_security == "Yes" else 0,
        'OnlineBackup': 1 if online_backup == "Yes" else 0,
        'DeviceProtection': 1 if device_protection == "Yes" else 0,
        'TechSupport': 1 if tech_support == "Yes" else 0,
        'StreamingTV': 1 if streaming_tv == "Yes" else 0,
        'StreamingMovies': 1 if streaming_movies == "Yes" else 0,
        'Contract': {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract],
        'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
        'PaymentMethod': {
            "Electronic check": 0,
            "Mailed check": 1,
            "Bank transfer (automatic)": 2,
            "Credit card (automatic)": 3
        }[payment_method],
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }])

if st.button("Predict Churn"):
    input_df = preprocess_inputs()
    result = model.predict(input_df)[0]
    if result == 1:
        st.error("🔴 This customer is likely to CHURN.")
    else:
        st.success("🟢 This customer is NOT likely to churn.")
