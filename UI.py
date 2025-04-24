import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Function to download files from GitHub (if needed)
def download_file(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception("Error downloading file")

# GitHub URLs for the model and scaler files (optional, adjust URLs if needed)
model_url = 'https://github.com/Phua0414/Assignment-AI/releases/download/Tag1/all_models.pkl'
scaler_url = 'https://github.com/Phua0414/Assignment-AI/releases/download/Tag1/scaler.pkl'

# Download the model and scaler files (if you use them from GitHub)
model_data = download_file(model_url)
scaler_data = download_file(scaler_url)

# Load the models and scaler from the downloaded data
models = pickle.loads(model_data)
scaler = pickle.loads(scaler_data)

# List of available models
model_names = list(models.keys())

# Streamlit UI Layout
st.title("Diabetes Prediction System")
st.write("Select a model and enter the features to predict diabetes")

# Form to select model and input features
with st.form(key='prediction_form'):
    # Dropdown for selecting model
    selected_model_name = st.selectbox("Select a Model", model_names)
    selected_model = models[selected_model_name]

    # Feature inputs
    age = st.number_input('Age', min_value=0, max_value=120)
    bmi = st.number_input('BMI', min_value=10.0, max_value=50.0)
    hypertension = st.selectbox('Hypertension', ['Yes', 'No'])
    heart_disease = st.selectbox('Heart Disease', ['Yes', 'No'])
    smoking_history = st.selectbox('Smoking History', ['never', 'current', 'former', 'not current', 'No Info'])
    hbA1c_level = st.number_input('HbA1c Level', min_value=0.0, max_value=15.0)
    blood_glucose_level = st.number_input('Blood Glucose Level', min_value=50, max_value=300)
    gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])

    # Submit button inside the form
    submit_button = st.form_submit_button("Make Prediction")

    if submit_button:
        # Preprocess input features (encode categorical features)
        gender_mapping = {'Male': 0, 'Female': 1, 'Other': 2}
        smoking_history_mapping = {'never': 0, 'No Info': 1, 'current': 2, 'former': 3, 'not current': 4}
        hypertension_mapping = {'Yes': 1, 'No': 0}
        heart_disease_mapping = {'Yes': 1, 'No': 0}

        input_data = [
            age, 
            bmi, 
            hypertension_mapping[hypertension], 
            heart_disease_mapping[heart_disease], 
            smoking_history_mapping[smoking_history], 
            hbA1c_level, 
            blood_glucose_level, 
            gender_mapping[gender]
        ]
        
        input_data = np.array(input_data).reshape(1, -1)

        # Scale the input data using the same scaler fitted during training
        input_features_scaled = scaler.transform(input_data)

        # Prediction
        with st.spinner("Predicting..."):
            prediction = selected_model.predict(input_features_scaled)

        # Display prediction result
        st.subheader("Prediction Result")
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        st.write(f"Prediction: {result}")

        # Display Evaluation Metrics (Accuracy, Confusion Matrix)
        st.subheader(f"{selected_model_name} Evaluation Metrics")

        # If feature importance is available (for tree-based models like XGBoost, RandomForest)
        if hasattr(selected_model, 'feature_importances_'):
            feature_importance = selected_model.feature_importances_
            sorted_idx = np.argsort(feature_importance)

            feature_names = ['Age', 'BMI', 'Hypertension', 'Heart Disease', 
                             'Smoking History', 'HbA1c Level', 'Blood Glucose Level', 'Gender']

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(feature_names[sorted_idx], feature_importance[sorted_idx], color='skyblue')
            ax.set_xlabel('Feature Importance')
            st.pyplot(fig)

# Optional: Add a "Reset" button if needed to clear inputs
if st.button('Reset'):
    st.experimental_rerun()
