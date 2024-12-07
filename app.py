import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import logging

# Configure Logging
logging.basicConfig(
    filename="app_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load Models
try:
    models = joblib.load("models.pkl")
    logging.info("Models loaded successfully.")
except FileNotFoundError as e:
    logging.error(f"Model file not found: {e}")
    st.error("Model file not found. Please upload the trained models.")
    st.stop()

# Initialize Scaler and Imputer
scaler = StandardScaler()
imputer = SimpleImputer(strategy='mean')

# Streamlit App Header
st.title("Disease Prediction System")
st.write("Predict Chronic Diseases like CKD, Diabetes, Hypertension, and Anemia")

# User Input Fields
try:
    age = st.number_input("Enter your age:", min_value=1, max_value=120, value=30)
    bp = st.number_input("Enter your blood pressure:", min_value=50, max_value=200, value=120)
    bgr = st.number_input("Enter your glucose level:", min_value=50, max_value=300, value=100)
    bu = st.number_input("Enter your blood urea:", min_value=10, max_value=200, value=40)
    sc = st.number_input("Enter your serum creatinine:", min_value=0.1, max_value=20.0, value=1.2)
    hemo = st.number_input("Enter your hemoglobin:", min_value=5.0, max_value=20.0, value=13.5)
except Exception as e:
    logging.error(f"Error in input fields: {e}")
    st.error("Invalid input detected. Please try again.")

# Prediction Logic
if st.button("Predict"):
    try:
        # Create DataFrame for User Input
        user_input = pd.DataFrame(
            [[age, bp, bgr, bu, sc, hemo]], 
            columns=['age', 'bp', 'bgr', 'bu', 'sc', 'hemo']
        )
        logging.info(f"User input: {user_input}")

        # Preprocess Data
        user_input_imputed = imputer.fit_transform(user_input)
        user_input_scaled = scaler.fit_transform(user_input_imputed)

        # Predictions
        ckd_prediction = models['Random Forest'].predict(user_input_scaled)
        diabetes_prediction = models['Random Forest'].predict(user_input_scaled)
        htn_prediction = models['Random Forest'].predict(user_input_scaled)
        ane_prediction = models['Random Forest'].predict(user_input_scaled)
        ann_prediction = models['Keras ANN'].predict(user_input_scaled)[0][0]

        # Print Predictions in Logs
        logging.info(f"CKD Prediction: {ckd_prediction[0]}")
        logging.info(f"Diabetes Prediction: {diabetes_prediction[0]}")
        logging.info(f"Hypertension Prediction: {htn_prediction[0]}")
        logging.info(f"Anemia Prediction: {ane_prediction[0]}")
        logging.info(f"Keras ANN CKD Prediction: {ann_prediction}")

        # Display Results on Streamlit
        st.subheader("Prediction Results:")

        if ckd_prediction[0] == 1 or ann_prediction >= 0.5:
            st.error("You may have Chronic Kidney Disease (CKD).")
            st.info(f"CKD Model Prediction Value: {ckd_prediction[0]} | Keras ANN: {ann_prediction:.4f}")
        else:
            st.success("You are less likely to have Chronic Kidney Disease (CKD).")
            st.info(f"CKD Model Prediction Value: {ckd_prediction[0]} | Keras ANN: {ann_prediction:.4f}")

        if diabetes_prediction[0] == 1:
            st.error("You may have Diabetes.")
            st.info(f"Diabetes Model Prediction Value: {diabetes_prediction[0]}")
        else:
            st.success("You are less likely to have Diabetes.")
            st.info(f"Diabetes Model Prediction Value: {diabetes_prediction[0]}")

        if htn_prediction[0] == 1:
            st.error("You may have Hypertension.")
            st.info(f"Hypertension Model Prediction Value: {htn_prediction[0]}")
        else:
            st.success("You are less likely to have Hypertension.")
            st.info(f"Hypertension Model Prediction Value: {htn_prediction[0]}")

        if ane_prediction[0] == 1:
            st.error("You may have Anemia.")
            st.info(f"Anemia Model Prediction Value: {ane_prediction[0]}")
        else:
            st.success("You are less likely to have Anemia.")
            st.info(f"Anemia Model Prediction Value: {ane_prediction[0]}")

        logging.info("Prediction completed successfully.")
    
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        st.error("An error occurred while making predictions. Please check your inputs and try again.")
