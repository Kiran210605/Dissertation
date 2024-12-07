# Streamlit App for Disease Prediction

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load Saved Models
models = joblib.load("best_models.pkl")

# Load Preprocessing Tools
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

# Streamlit App
st.title("Disease Prediction App")
st.write("Predict chronic kidney disease, diabetes, hypertension, and anemia based on your health parameters.")

# Collect User Inputs
age = st.number_input("Enter your age:", min_value=0, max_value=120, value=30)
bp = st.number_input("Enter your blood pressure (mmHg):", min_value=0, max_value=200, value=80)
bgr = st.number_input("Enter your glucose level (mg/dL):", min_value=0, max_value=300, value=100)
bu = st.number_input("Enter your blood urea (mg/dL):", min_value=0, max_value=300, value=40)
sc = st.number_input("Enter your serum creatinine (mg/dL):", min_value=0.0, max_value=20.0, value=1.2)
hemo = st.number_input("Enter your hemoglobin level (g/dL):", min_value=0.0, max_value=20.0, value=13.5)

# Prediction Button
if st.button("Predict Diseases"):
    # Create DataFrame from User Input
    user_input = pd.DataFrame([[age, bp, bgr, bu, sc, hemo]], 
                              columns=['age', 'bp', 'bgr', 'bu', 'sc', 'hemo'])
    
    # Preprocess Input Data
    user_input_imputed = imputer.fit_transform(user_input)
    user_input_scaled = scaler.fit_transform(user_input_imputed)
    
    # Perform Predictions
    ckd_rf_prediction = models['Random Forest'].predict(user_input_scaled)[0]
    diabetes_rf_prediction = models['Random Forest'].predict(user_input_scaled)[0]
    htn_rf_prediction = models['Random Forest'].predict(user_input_scaled)[0]
    ane_rf_prediction = models['Random Forest'].predict(user_input_scaled)[0]
    ckd_ann_prediction = models['Keras ANN'].predict(user_input_scaled)[0][0]

    # Display Results
    st.subheader("Prediction Results:")
    if ckd_rf_prediction == 1 or ckd_ann_prediction >= 0.5:
        st.error("You may have Chronic Kidney Disease (CKD).")
    else:
        st.success("You are less likely to have CKD.")

    if diabetes_rf_prediction == 1:
        st.error("You may have Diabetes.")
    else:
        st.success("You are less likely to have Diabetes.")

    if htn_rf_prediction == 1:
        st.error("You may have Hypertension (High Blood Pressure).")
    else:
        st.success("You are less likely to have Hypertension.")

    if ane_rf_prediction == 1:
        st.error("You may have Anemia.")
    else:
        st.success("You are less likely to have Anemia.")

# Run Streamlit App using: streamlit run app.py
