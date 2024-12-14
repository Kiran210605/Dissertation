import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load Models and Preprocessors
models = joblib.load("best_models.pkl")
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("Chronic Kidney Disease and Diabetes Prediction")

# User Input
age = st.number_input("Enter your age:", min_value=0, max_value=120, value=30, step=1)
bp = st.number_input("Enter your blood pressure:", min_value=0, max_value=300, value=120, step=1)
bgr = st.number_input("Enter your glucose level:", min_value=0, max_value=500, value=100, step=1)
bu = st.number_input("Enter your blood urea:", min_value=0.0, max_value=200.0, value=25.0, step=0.1)
sc = st.number_input("Enter your serum creatinine:", min_value=0.0, max_value=20.0, value=1.2, step=0.1)
hemo = st.number_input("Enter your hemoglobin:", min_value=0.0, max_value=20.0, value=13.5, step=0.1)

# Prediction Button
if st.button("Predict"):
    user_input = pd.DataFrame([[age, bp, bgr, bu, sc, hemo]],
                              columns=['age', 'bp', 'bgr', 'bu', 'sc', 'hemo'])
    user_input_imputed = imputer.transform(user_input)
    user_input_scaled = scaler.transform(user_input_imputed)

    ckd_prediction = models['Random Forest'].predict(user_input_scaled)[0]
    diabetes_prediction = models['Random Forest'].predict(user_input_scaled)[0]
    ann_prediction = models['Keras ANN'].predict(user_input_scaled)[0][0]

    st.subheader("Predictions Based on Your Symptoms")
    if ckd_prediction == 1 or ann_prediction >= 0.5:
        st.error("You may have Chronic Kidney Disease (CKD).")
    else:
        st.success("You are less likely to have Chronic Kidney Disease (CKD).")

    if diabetes_prediction == 1:
        st.error("You may have Diabetes.")
    else:
        st.success("You are less likely to have Diabetes.")

    if ckd_prediction == 1:
        st.error("You may have Hypertension (High Blood Pressure).")
    else:
        st.success("You are less likely to have Hypertension.")

    if ckd_prediction == 1:
        st.error("You may have Anemia.")
    else:
        st.success("You are less likely to have Anemia.")

# Deploying Instructions
# Save this file as `app.py`
# Run the following command in the terminal to deploy the application:
# streamlit run app.py
