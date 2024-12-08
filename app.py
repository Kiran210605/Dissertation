# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load Models, Imputer, and Scaler
models = joblib.load("best_models.pkl")
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")

# Title and Description
st.title("Disease Prediction App")
st.write("""
This app predicts the likelihood of **Chronic Kidney Disease (CKD)** and **Diabetes** 
based on your medical details.
""")

# Sidebar Inputs
st.sidebar.header("Enter Your Medical Details")

def get_user_input():
    age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)
    bp = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=200, value=80)
    bgr = st.sidebar.number_input("Glucose Level", min_value=0, max_value=300, value=120)
    bu = st.sidebar.number_input("Blood Urea", min_value=0.0, max_value=300.0, value=40.0)
    sc = st.sidebar.number_input("Serum Creatinine", min_value=0.0, max_value=20.0, value=1.2)
    hemo = st.sidebar.number_input("Hemoglobin", min_value=0.0, max_value=20.0, value=13.0)

    data = pd.DataFrame({
        'age': [age], 
        'bp': [bp], 
        'bgr': [bgr], 
        'bu': [bu], 
        'sc': [sc], 
        'hemo': [hemo]
    })
    return data

# Predict Function
def predict_diseases(models, user_input):
    user_input_imputed = imputer.transform(user_input)
    user_input_scaled = scaler.transform(user_input_imputed)

    ckd_rf_prediction = models['Random Forest'].predict(user_input_scaled)
    ckd_ann_prediction = models['Keras ANN'].predict(user_input_scaled)[0][0]
    diabetes_rf_prediction = models['Random Forest'].predict(user_input_scaled)

    st.subheader("Prediction Results")
    if ckd_rf_prediction[0] == 1 or ckd_ann_prediction >= 0.5:
        st.error("🔴 You may have Chronic Kidney Disease (CKD).")
    else:
        st.success("🟢 You are less likely to have Chronic Kidney Disease (CKD).")

    if diabetes_rf_prediction[0] == 1:
        st.error("🔴 You may have Diabetes.")
    else:
        st.success("🟢 You are less likely to have Diabetes.")

# Main Execution
user_input = get_user_input()

if st.sidebar.button("Predict"):
    predict_diseases(models, user_input)
