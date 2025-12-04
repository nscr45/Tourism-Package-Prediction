import streamlit as st
import pandas as pd
import joblib
import numpy as np
from huggingface_hub import hf_hub_download

# Configuration
HF_USERNAME = "Sricharan451706"
HF_MODEL_REPO = f"{HF_USERNAME}/Tourism-Package-Predictor"

st.title("Wellness Tourism Package Prediction")
st.write("Enter customer details to predict if they will purchase the package.")

@st.cache_resource
def load_model():
    try:
        # Try loading locally first for development
        model = joblib.load("models/model.joblib")
        return model
    except:
        # Load from Hugging Face
        try:
            model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="model.joblib")
            model = joblib.load(model_path)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

model = load_model()

if model:
    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
            city_tier = st.selectbox("City Tier", [1, 2, 3])
            occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            number_of_person_visiting = st.number_input("Number of Person Visiting", min_value=1, value=2)
            preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
            marital_status = st.selectbox("Marital Status", ["Married", "Divorced", "Single", "Unmarried"])
            number_of_trips = st.number_input("Number of Trips", min_value=0, value=1)
            
        with col2:
            passport = st.selectbox("Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            own_car = st.selectbox("Own Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, value=0)
            designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])
            monthly_income = st.number_input("Monthly Income", min_value=0, value=20000)
            pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
            product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
            number_of_followups = st.number_input("Number of Followups", min_value=0, value=3)
            duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, value=10)
        
        submitted = st.form_submit_button("Predict")
        
    if submitted:
        # Create dataframe from inputs
        input_data = pd.DataFrame({
            'Age': [age],
            'TypeofContact': [type_of_contact],
            'CityTier': [city_tier],
            'Occupation': [occupation],
            'Gender': [gender],
            'NumberOfPersonVisiting': [number_of_person_visiting],
            'PreferredPropertyStar': [preferred_property_star],
            'MaritalStatus': [marital_status],
            'NumberOfTrips': [number_of_trips],
            'Passport': [passport],
            'OwnCar': [own_car],
            'NumberOfChildrenVisiting': [number_of_children_visiting],
            'Designation': [designation],
            'MonthlyIncome': [monthly_income],
            'PitchSatisfactionScore': [pitch_satisfaction_score],
            'ProductPitched': [product_pitched],
            'NumberOfFollowups': [number_of_followups],
            'DurationOfPitch': [duration_of_pitch]
        })
        
        # Make prediction using the pipeline
        # The pipeline handles encoding and scaling automatically
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            
            if prediction == 1:
                st.success(f"Prediction: Will Purchase (Probability: {probability:.2f})")
            else:
                st.warning(f"Prediction: Will Not Purchase (Probability: {probability:.2f})")
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.info("Ensure all columns used in training are provided in the input dataframe.")
