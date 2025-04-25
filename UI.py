import streamlit as st
import pickle
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

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

# Feature names
feature_names = np.array([
    'Age Group', 'BMI Category', 'Hypertension', 'Heart Disease', 
    'Smoking History', 'HbA1c Level', 'Blood Glucose Level', 'Gender'
])

# Function to get feature importance based on model type
def get_feature_importance(model, model_name, X_sample):
    """Get feature importance based on model type"""
    if model_name in ["Random Forest", "XGBoost"]:
        # These models have feature_importances_ attribute
        importance = model.feature_importances_
        return importance, "Feature Importance"
    
    elif model_name == "Logistic Regression":
        # For logistic regression, use coefficients (absolute values)
        importance = np.abs(model.coef_[0])
        return importance, "Coefficient Magnitude"
    
    elif model_name == "K-Nearest Neighbors":
        # For KNN, we need to calculate permutation importance
        # This is a simplified version - in production, you'd want to use a cached or precomputed value
        with st.spinner("Calculating feature importance..."):
            # Create a small sample to quickly calculate permutation importance
            # This is just for demonstration - in practice, you'd want to precompute this
            dummy_y = np.zeros(X_sample.shape[0])
            perm_importance = permutation_importance(
                model, X_sample, dummy_y, 
                n_repeats=3, random_state=42, n_jobs=-1
            )
            importance = perm_importance.importances_mean
        return importance, "Permutation Importance"
    
    return None, None

# Center the content using custom CSS for the title and inputs
st.markdown(
    """
    <style>
        .main {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            text-align: center;
            flex-direction: column;
        }
        .block-container {
            width: 80%;
            padding: 2rem;
            border-radius: 10px;
            background-color: #f4f7fc;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        .stSlider, .stButton {
            font-size: 16px;
        }
        .stTextInput {
            margin-bottom: 1rem;
        }
        .stTitle {
            font-size: 32px;
            font-weight: bold;
            color: #3a3a3a;
            margin-bottom: 20px;
        }
        .stSidebar {
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)

# Centered Title
st.title("Diabetes Prediction System")

# Display selected model and accuracy
with st.container():
    st.header("Choose Your Model")
    selected_model_name = st.selectbox("Select a Model", model_names)
    selected_model = models[selected_model_name]

    accuracies = {
        "Random Forest": 90.60,
        "XGBoost": 94.71,
        "Logistic Regression": 80.34,
        "K-Nearest Neighbors": 92.91
    }

    st.write(f"You selected: {selected_model_name}")
    st.write(f"Accuracy: {accuracies[selected_model_name]:.2f}%")

    st.subheader("Enter the Features")

    # Form to enter input features
    with st.form(key="input_form"):
        # Input fields
        age = st.number_input("Age", min_value=18, max_value=100, value=25, step=1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        smoking_history = st.selectbox("Smoking History", ["never", "No Info", "current", "former", "ever", "not current"])
        hbA1c_level = st.slider("HbA1c Level", 4.0, 10.0, 6.0)
        blood_glucose_level = st.slider("Blood Glucose Level", 50, 300, 100)
        gender = st.selectbox("Gender", ["Male", "Female"])

        # Submit button for the form
        submit_button = st.form_submit_button(label="Make Prediction")

    # Handling prediction after form submission
    if submit_button:
        # Mapping input values to numerical
        # Determine age group based on age value
        if age < 18:
            age_group = 0  # Minor
        elif age < 25:
            age_group = 1  # Young
        elif age < 45:
            age_group = 2  # Adult
        elif age < 60:
            age_group = 3  # Middle-Aged
        else:
            age_group = 4  # Senior
            
        # Determine BMI category based on BMI value
        if bmi < 18.5:
            bmi_category = 0  # Underweight
        elif bmi < 25:
            bmi_category = 1  # Normal
        elif bmi < 30:
            bmi_category = 2  # Overweight
        else:
            bmi_category = 3  # Obesity
            
        smoking_history_mapping = { 'never': 0, 'No Info': 1, 'current': 2, 'former': 3, 'ever': 4, 'not current': 5 }
        hypertension = 0 if hypertension == "No" else 1
        heart_disease = 0 if heart_disease == "No" else 1
        smoking_history = smoking_history_mapping[smoking_history]
        gender = 0 if gender == "Male" else 1

        # Input features array
        input_features = np.array([[
            gender, hypertension, heart_disease,
            smoking_history, hbA1c_level, blood_glucose_level, 
            age_group, bmi_category
        ]])

        # Apply scaling to input features
        input_features_scaled = scaler.transform(input_features)

        # Prediction
        with st.spinner("Predicting..."):
            prediction = selected_model.predict(input_features_scaled)

        # Display the prediction result
        st.subheader("Prediction Result")
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        st.write(f"Prediction: {result}")

        # Display Feature Importance for all models
        importance, importance_type = get_feature_importance(selected_model, selected_model_name, input_features_scaled)
        
        if importance is not None:
            st.subheader(f"{importance_type}")
            sorted_idx = np.argsort(importance)
            
            # Create the visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Use different colors based on model
            colors = {
                "Random Forest": "skyblue",
                "XGBoost": "lightgreen",
                "Logistic Regression": "orange",
                "K-Nearest Neighbors": "salmon"
            }
            
            ax.barh(feature_names[sorted_idx], importance[sorted_idx], color=colors[selected_model_name])
            ax.set_xlabel(importance_type)
            ax.set_title(f"{selected_model_name} - {importance_type}")
            st.pyplot(fig)
