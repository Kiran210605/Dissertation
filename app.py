import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load Models
models = joblib.load("models.pkl")

# Load the datasets
kidney_disease_df = pd.read_csv('kidney_disease.csv')
diabetes_df = pd.read_csv('diabetes.csv')

# Preprocess Data
def preprocess_data():
    kidney_disease_df.fillna({
        'age': kidney_disease_df['age'].median(),
        'bp': kidney_disease_df['bp'].median(),
        'bgr': kidney_disease_df['bgr'].median(),
        'bu': kidney_disease_df['bu'].median(),
        'sc': kidney_disease_df['sc'].median(),
        'hemo': kidney_disease_df['hemo'].median(),
        'htn': kidney_disease_df['htn'].mode()[0],
        'ane': kidney_disease_df['ane'].mode()[0],
    }, inplace=True)

    kidney_disease_df['htn'] = kidney_disease_df['bp'].apply(lambda x: 1 if x > 130 else 0)
    kidney_disease_df['ane'] = kidney_disease_df['ane'].apply(lambda x: 1 if x == 'yes' else 0)
    kidney_disease_df['classification'] = kidney_disease_df['classification'].apply(lambda x: 1 if x == 'ckd' else 0)

    combined_df = pd.concat([
        kidney_disease_df[['age', 'bp', 'bgr', 'bu', 'sc', 'hemo', 'htn', 'ane', 'classification']].rename(columns={'classification': 'ckd'}),
        diabetes_df[['Age', 'BloodPressure', 'Glucose', 'BMI', 'Outcome']].rename(columns={'Age': 'age', 'BloodPressure': 'bp', 'Glucose': 'bgr', 'Outcome': 'diabetes'})
    ], axis=0, ignore_index=True)

    return combined_df

def split_data(df):
    X = df[['age', 'bp', 'bgr', 'bu', 'sc', 'hemo']]
    return X

# Preprocess and fit imputer and scaler
combined_df = preprocess_data()
X = split_data(combined_df)

imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)

# Streamlit Interface
st.title("Disease Prediction System")
st.header("Enter Your Medical Details")

age = st.number_input("Enter your age:", min_value=0, max_value=120, value=30)
bp = st.number_input("Enter your blood pressure:", min_value=0, max_value=200, value=120)
bgr = st.number_input("Enter your glucose level:", min_value=0, max_value=300, value=100)
bu = st.number_input("Enter your blood urea:", min_value=0, max_value=200, value=20)
sc = st.number_input("Enter your serum creatinine:", min_value=0.0, max_value=20.0, value=1.2)
hemo = st.number_input("Enter your hemoglobin:", min_value=0.0, max_value=20.0, value=13.5)

if st.button("Predict"):
    user_input = pd.DataFrame([[age, bp, bgr, bu, sc, hemo]],
                              columns=['age', 'bp', 'bgr', 'bu', 'sc', 'hemo'])

    user_input_imputed = imputer.transform(user_input)
    user_input_scaled = scaler.transform(user_input_imputed)

    # Predictions
    ckd_prediction = models['Random Forest'].predict(user_input_scaled)
    diabetes_prediction = models['Random Forest'].predict(user_input_scaled)
    ann_prediction = models['Keras ANN'].predict(user_input_scaled)[0][0]

    if ckd_prediction[0] == 1 or ann_prediction >= 0.5:
        st.error("Based on your symptoms, you may have Chronic Kidney Disease (CKD).")
    else:
        st.success("You are less likely to have Chronic Kidney Disease (CKD).")

    if diabetes_prediction[0] == 1:
        st.error("Based on your symptoms, you may have Diabetes.")
    else:
        st.success("You are less likely to have Diabetes.")

    if ckd_prediction[0] == 1:
        st.error("Based on your symptoms, you may have Hypertension (High Blood Pressure).")
    else:
        st.success("You are less likely to have Hypertension (High Blood Pressure).")

    if ckd_prediction[0] == 1:
        st.error("Based on your symptoms, you may have Anemia.")
    else:
        st.success("You are less likely to have Anemia.")
