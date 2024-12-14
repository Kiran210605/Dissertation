# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# Load Models and Scalers
models = joblib.load("best_models.pkl")
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

# Streamlit App
st.title("Medical Disease Prediction App")

st.sidebar.header("Enter Your Medical Details")
age = st.sidebar.number_input("Enter your age", min_value=0, max_value=120, value=25, step=1)
bp = st.sidebar.number_input("Enter your blood pressure", min_value=0, max_value=250, value=120, step=1)
bgr = st.sidebar.number_input("Enter your glucose level", min_value=0, max_value=500, value=100, step=1)
bu = st.sidebar.number_input("Enter your blood urea", min_value=0, max_value=300, value=40, step=1)
sc = st.sidebar.number_input("Enter your serum creatinine", min_value=0.0, max_value=10.0, value=1.2, step=0.1)
hemo = st.sidebar.number_input("Enter your hemoglobin", min_value=0.0, max_value=20.0, value=13.5, step=0.1)

# Make Predictions
user_input = pd.DataFrame([[age, bp, bgr, bu, sc, hemo]], 
                          columns=['age', 'bp', 'bgr', 'bu', 'sc', 'hemo'])
user_input_imputed = imputer.fit_transform(user_input)
user_input_scaled = scaler.fit_transform(user_input_imputed)

st.header("Predictions Based on Your Symptoms")

# Predictions
ckd_rf_prediction = models['Random Forest'].predict(user_input_scaled)
diabetes_rf_prediction = models['Random Forest'].predict(user_input_scaled)
ckd_ann_prediction = models['Keras ANN'].predict(user_input_scaled)[0][0]

if ckd_rf_prediction[0] == 1 or ckd_ann_prediction >= 0.5:
    st.warning("You may have Chronic Kidney Disease (CKD).")
else:
    st.success("You are less likely to have Chronic Kidney Disease (CKD).")

if diabetes_rf_prediction[0] == 1:
    st.warning("You may have Diabetes.")
else:
    st.success("You are less likely to have Diabetes.")

if ckd_rf_prediction[0] == 1:
    st.warning("You may have Hypertension (High Blood Pressure).")
else:
    st.success("You are less likely to have Hypertension (High Blood Pressure).")

if ckd_rf_prediction[0] == 1:
    st.warning("You may have Anemia.")
else:
    st.success("You are less likely to have Anemia.")
