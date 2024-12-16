import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load Models
models = joblib.load("best_models.pkl")

# Load Data
kidney_disease_df = pd.read_csv('kidney_disease.csv')
diabetes_df = pd.read_csv('diabetes.csv')

# Preprocessing Function
def preprocess_input(age, bp, bgr, bu, sc, hemo):
    user_input = pd.DataFrame([[age, bp, bgr, bu, sc, hemo]],
                              columns=['age', 'bp', 'bgr', 'bu', 'sc', 'hemo'])
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    imputer.fit(kidney_disease_df[['age', 'bp', 'bgr', 'bu', 'sc', 'hemo']])
    X_imputed = imputer.transform(user_input)
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled

# Streamlit App
st.title("Medical Disease Prediction App")
st.sidebar.header("Input Medical Details")

# User Inputs
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)
bp = st.sidebar.number_input("Blood Pressure (BP)", min_value=60, max_value=200, value=120)
bgr = st.sidebar.number_input("Blood Glucose Level (BGR)", min_value=50, max_value=300, value=100)
bu = st.sidebar.number_input("Blood Urea (BU)", min_value=1, max_value=300, value=50)
sc = st.sidebar.number_input("Serum Creatinine (SC)", min_value=0.1, max_value=15.0, value=1.2)
hemo = st.sidebar.number_input("Hemoglobin (Hemo)", min_value=5.0, max_value=20.0, value=12.0)

# Prediction Button
if st.sidebar.button("Predict"):
    X_processed = preprocess_input(age, bp, bgr, bu, sc, hemo)

    ckd_prediction = models['Random Forest'].predict(X_processed)[0]
    diabetes_prediction = models['Random Forest'].predict(X_processed)[0]
    ann_prediction = models['Keras ANN'].predict(X_processed)[0][0]

    st.subheader("Predictions")

    if ckd_prediction == 1 or ann_prediction >= 0.5:
        st.error("You may have Chronic Kidney Disease (CKD).")
    else:
        st.success("You are less likely to have Chronic Kidney Disease (CKD).")

    if diabetes_prediction == 1:
        st.error("You may have Diabetes.")
    else:
        st.success("You are less likely to have Diabetes.")
