import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load Models
models = joblib.load("models/best_models.pkl")
imputer = joblib.load("models/imputer.pkl")
scaler = joblib.load("models/scaler.pkl")

# App Title
st.title("Chronic Kidney Disease & Diabetes Prediction")

# User Input
st.header("Enter Your Medical Information")

age = st.number_input("Age", min_value=1, max_value=120, step=1)
bp = st.number_input("Blood Pressure (BP)", min_value=50, max_value=200, step=1)
bgr = st.number_input("Blood Glucose Random (BGR)", min_value=50, max_value=300, step=1)
bu = st.number_input("Blood Urea (BU)", min_value=1.0, max_value=200.0, step=0.1)
sc = st.number_input("Serum Creatinine (SC)", min_value=0.1, max_value=15.0, step=0.1)
hemo = st.number_input("Hemoglobin (Hemo)", min_value=3.0, max_value=20.0, step=0.1)

# Prediction Button
if st.button("Predict"):
    user_data = pd.DataFrame([[age, bp, bgr, bu, sc, hemo]], 
                             columns=['age', 'bp', 'bgr', 'bu', 'sc', 'hemo'])

    # Preprocess Input
    user_data_imputed = imputer.transform(user_data)
    user_data_scaled = scaler.transform(user_data_imputed)

    # Predictions
    ckd_prediction = models['Random Forest'].predict(user_data_scaled)
    diabetes_prediction = models['Random Forest'].predict(user_data_scaled)
    ann_prediction = models['Keras ANN'].predict(user_data_scaled)[0][0]

    st.header("Prediction Results")
    
    if ckd_prediction[0] == 1 or ann_prediction >= 0.5:
        st.error("You may have Chronic Kidney Disease (CKD).")
    else:
        st.success("You are less likely to have CKD.")

    if diabetes_prediction[0] == 1:
        st.error("You may have Diabetes.")
    else:
        st.success("You are less likely to have Diabetes.")
