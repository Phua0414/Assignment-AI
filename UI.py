import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

def download_file(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception("Download failed.")

model_url = 'https://github.com/Phua0414/Assignment-AI/releases/download/Tag-1/all_models.pkl'
scaler_url = 'https://github.com/Phua0414/Assignment-AI/releases/download/Tag-1/scaler.pkl'

# 📥 Load model and scaler
models = pickle.loads(download_file(model_url))
scaler = pickle.loads(download_file(scaler_url))

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")
st.title("🩺 Diabetes Prediction System")
st.markdown("This intelligent system helps predict the risk of diabetes based on basic patient health information.")

st.subheader("🎛️ Select Classification Model")
model_names = list(models.keys())
model_choice = st.selectbox("Choose a model below:", model_names)

# Accuracy display (dummy accuracies for now, replace with real values if needed)
model_accuracy = {
    'K-Nearest Neighbors': 0.94,
    'Random Forest': 0.95,
    'Logistic Regression': 0.87
}

if model_choice in model_accuracy:
    st.info(f"**Model Accuracy:** {model_accuracy[model_choice] * 100:.2f}%")

st.subheader("Enter Patient Information")
with st.form("patient_form"):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.radio("Gender", ["Male", "Female"])
        hypertension = st.radio("Hypertension", ["No", "Yes"])
        heart_disease = st.radio("Heart Disease", ["No", "Yes"])
        smoking_history = st.selectbox("Smoking History", ["never", "No Info", "current", "former", "ever", "not current"])
    with col2:
        age = st.number_input("Age", 1, 100, value=30)
        bmi = st.number_input("BMI", 10.0, 50.0, value=24.0)
        hba1c = st.slider("HbA1c Level", 4.0, 10.0, 5.5)
        glucose = st.slider("Blood Glucose Level", 50, 300, 100)

    submitted = st.form_submit_button("🔮 Predict")


if submitted:
    # Encode inputs
    gender = 0 if gender == "Male" else 1
    hypertension = 0 if hypertension == "No" else 1
    heart_disease = 0 if heart_disease == "No" else 1
    smoking_mapping = {'never': 0, 'No Info': 1, 'current': 2, 'former': 3, 'ever': 4, 'not current': 5}
    smoking_history = smoking_mapping[smoking_history]

    age_group = 0 if age < 18 else 1 if age < 25 else 2 if age < 45 else 3 if age < 60 else 4
    bmi_category = 0 if bmi < 18.5 else 1 if bmi < 25 else 2 if bmi < 30 else 3

    input_data = pd.DataFrame([[gender, hypertension, heart_disease, smoking_history, hba1c, glucose, age_group, bmi_category]],
                               columns=['gender', 'hypertension', 'heart_disease', 'smoking_history',
                                        'HbA1c_level', 'blood_glucose_level', 'age_group', 'bmi_category'])

    scaled_input = scaler.transform(input_data)

    # Predict
    model = models[model_choice]
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    # --------------------------
    # 🎯 Display Results
    # --------------------------
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"**Prediction: Diabetic** (Risk: {probability*100:.2f}%)")
        st.warning("⚠️ High chance of diabetes. Seek medical advice.")
    else:
        st.success(f"**Prediction: Not Diabetic** (Risk: {probability*100:.2f}%)")
        st.info("✅ Low risk. Maintain healthy habits!")

    st.subheader("Risk Interpretation")
    if probability < 0.3:
        st.success("Low Risk (0% - 30%)")
    elif probability < 0.7:
        st.warning("Moderate Risk (30% - 70%)")
    else:
        st.error("High Risk (70% - 100%)")

    st.caption("This prediction is for informational purposes only and does not replace medical advice.")
