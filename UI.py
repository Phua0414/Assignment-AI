import streamlit as st
import pickle
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Function to download files from GitHub
def download_file(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception("Error downloading file")

# GitHub URLs for the model and scaler files
model_url = 'https://github.com/Phua0414/Assignment-AI/releases/download/Tag1/all_models.pkl'
scaler_url = 'https://github.com/Phua0414/Assignment-AI/releases/download/Tag1/scaler.pkl'

# Download the model and scaler files
model_data = download_file(model_url)
scaler_data = download_file(scaler_url)

# Load the models and scaler from the downloaded data
models = pickle.loads(model_data)
scaler = pickle.loads(scaler_data)

# List of available models
model_names = list(models.keys())

# Center the content using custom CSS
st.markdown(
    """
    <style>
        .main {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            text-align: center;
        }
        .block-container {
            width: 80%;
        }
    </style>
    """, unsafe_allow_html=True)

# Streamlit UI setup
st.title("Diabetes Prediction System")
st.sidebar.header("Choose Your Model")

selected_model_name = st.sidebar.selectbox("Select a Model", model_names)
selected_model = models[selected_model_name]

# Display selected model and its accuracy
accuracies = {
    "Random Forest": 90.60,
    "XGBoost": 94.71,
    "Logistic Regression": 80.34,
    "K-Nearest Neighbors": 92.91
}

st.sidebar.write(f"You selected: {selected_model_name}")
st.sidebar.write(f"Accuracy: {accuracies[selected_model_name]:.2f}%")

# Input Form for Features
st.sidebar.header("Enter the Features")

# Input validation for age
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=25, step=1)

# Convert age to Age Group
age_group = None
if age < 18:
    age_group = "Minor"
elif age < 25:
    age_group = "Young"
elif age < 45:
    age_group = "Adult"
elif age < 60:
    age_group = "Middle-Aged"
else:
    age_group = "Senior"

# Input validation for BMI
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)

# Convert BMI to BMI Category
bmi_category = None
if bmi < 18.5:
    bmi_category = "Underweight"
elif bmi < 24.9:
    bmi_category = "Normal"
elif bmi < 29.9:
    bmi_category = "Overweight"
else:
    bmi_category = "Obesity"

# Other features (you can adjust these according to your dataset)
hypertension = st.sidebar.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
heart_disease = st.sidebar.selectbox("Heart Disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
smoking_history = st.sidebar.selectbox("Smoking History", ["never", "No Info", "current", "former", "ever", "not current"])

# HbA1c level and Blood glucose level validation
hbA1c_level = st.sidebar.slider("HbA1c Level", min_value=4.0, max_value=10.0, value=6.0, step=0.1)
blood_glucose_level = st.sidebar.slider("Blood Glucose Level", min_value=50, max_value=300, value=100, step=1)

# Gender input (0 for Male, 1 for Female)
gender = st.sidebar.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")

# Mappings for categorical variables
age_group_mapping = {'Minor': 0, 'Young': 1, 'Adult': 2, 'Middle-Aged': 3, 'Senior': 4}
bmi_category_mapping = {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obesity': 3}
smoking_history_mapping = {
    'never': 0, 'No Info': 1, 'current': 2, 'former': 3, 'ever': 4, 'not current': 5
}

# Convert categorical inputs to numeric
input_features = np.array([[
    age_group_mapping[age_group],
    bmi_category_mapping[bmi_category],
    hypertension,
    heart_disease,
    smoking_history_mapping[smoking_history],
    hbA1c_level,
    blood_glucose_level,
    gender
]])

# Apply the same scaler used during training to the input features
input_features_scaled = scaler.transform(input_features)

# Prediction
if st.sidebar.button("Make Prediction"):
    with st.spinner("Predicting..."):
        prediction = selected_model.predict(input_features_scaled)
    
    # Display the prediction result
    st.subheader("Prediction Result")
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    st.write(f"Prediction: {result}")

    # Display Feature Importance (if applicable)
    if hasattr(selected_model, 'feature_importances_'):
        feature_importance = selected_model.feature_importances_
        sorted_idx = np.argsort(feature_importance)  # Get indices of sorted features
        
        # Define feature names
        feature_names = np.array(['Age Group', 'BMI Category', 'Hypertension', 'Heart Disease', 'Smoking History', 'HbA1c Level', 'Blood Glucose Level', 'Gender'])
        
        st.subheader("Feature Importance")
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names[sorted_idx], feature_importance[sorted_idx], color='skyblue')
        plt.xlabel('Feature Importance')
        st.pyplot()
