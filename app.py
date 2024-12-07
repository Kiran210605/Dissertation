import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the models
models = joblib.load("models.pkl")
scaler = StandardScaler()
imputer = SimpleImputer(strategy='mean')

# Streamlit Web App
st.title("Disease Prediction System")
st.write("Predict Chronic Diseases like CKD, Diabetes, Hypertension, and Anemia")

# Input Fields
age = st.number_input("Enter your age:", min_value=1, max_value=120, value=30)
bp = st.number_input("Enter your blood pressure:", min_value=50, max_value=200, value=120)
bgr = st.number_input("Enter your glucose level:", min_value=50, max_value=300, value=100)
bu = st.number_input("Enter your blood urea:", min_value=10, max_value=200, value=40)
sc = st.number_input("Enter your serum creatinine:", min_value=0.1, max_value=20.0, value=1.2)
hemo = st.number_input("Enter your hemoglobin:", min_value=5.0, max_value=20.0, value=13.5)

# Prediction
if st.button("Predict"):
    user_input = pd.DataFrame([[age, bp, bgr, bu, sc, hemo]], 
                              columns=['age', 'bp', 'bgr', 'bu', 'sc', 'hemo'])
    user_input_imputed = imputer.fit_transform(user_input)
    user_input_scaled = scaler.fit_transform(user_input_imputed)

    # Predictions
    ckd_prediction = models['Random Forest'].predict(user_input_scaled)
    diabetes_prediction = models['Random Forest'].predict(user_input_scaled)
    htn_prediction = models['Random Forest'].predict(user_input_scaled)
    ane_prediction = models['Random Forest'].predict(user_input_scaled)
    ann_prediction = models['Keras ANN'].predict(user_input_scaled)[0][0]

    st.subheader("Prediction Results:")
    if ckd_prediction[0] == 1 or ann_prediction >= 0.5:
        st.error("You may have Chronic Kidney Disease (CKD).")
    else:
        st.success("You are less likely to have Chronic Kidney Disease (CKD).")

    if diabetes_prediction[0] == 1:
        st.error("You may have Diabetes.")
    else:
        st.success("You are less likely to have Diabetes.")

    if htn_prediction[0] == 1:
        st.error("You may have Hypertension.")
    else:
        st.success("You are less likely to have Hypertension.")

    if ane_prediction[0] == 1:
        st.error("You may have Anemia.")
    else:
        st.success("You are less likely to have Anemia.")
