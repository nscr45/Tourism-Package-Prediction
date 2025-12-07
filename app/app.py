import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from huggingface_hub import hf_hub_download, HfApi

# Configuration
HF_USERNAME = "Sricharan451706"
HF_MODEL_REPO = f"{HF_USERNAME}/Tourism-Package-Predictor"
HF_DATASET_REPO = f"{HF_USERNAME}/Tourism" 
LOG_FILENAME = "prediction_logs.csv"


st.title("Wellness Tourism Package Prediction")
st.write("Enter customer details to predict if they will purchase the package.")

@st.cache_resource
def load_model():
    try:
        # Try loading locally first for development
        model = joblib.load("models/model.joblib")
        print("Loaded model locally.")
        return model
    except Exception as e:
        print(f"Local load failed: {e}. Trying Hugging Face...")
        # Load from Hugging Face
        try:
            # Try getting token from secrets (Streamlit Cloud) or env var
            token = os.environ.get("HF_TOKEN")
            
            if not token:
                try:
                    # Check if secrets are available (this raises error if no secrets.toml exists)
                    if "HF_TOKEN" in st.secrets:
                        token = st.secrets["HF_TOKEN"]
                except FileNotFoundError:
                    # Secrets file doesn't exist, which is fine for local dev if env var is set or repo is public
                    pass
                except Exception:
                    pass
                
            model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="model.joblib", token=token)
            model = joblib.load(model_path)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.info("If the model repo is Private, make sure to add 'HF_TOKEN' to your Space Secrets (or local .streamlit/secrets.toml).")
            return None

def save_prediction(input_data, prediction, probability):
    """
    Saves the input data and prediction result to a CSV file on Hugging Face.
    """
    # 1. Get Token
    token = os.environ.get("HF_TOKEN")
    if not token and "HF_TOKEN" in st.secrets:
        token = st.secrets["HF_TOKEN"]
        
    if not token:
        print("No token found. Skipping log save.")
        return

    try:
        api = HfApi(token=token)
        
        # 2. Add prediction metadata
        log_entry = input_data.copy()
        log_entry['Prediction'] = prediction
        log_entry['Probability'] = probability
        log_entry['Timestamp'] = pd.Timestamp.now()

        # 3. Download existing log file (if exists)
        updated_log = None
        try:
            log_path = hf_hub_download(repo_id=HF_DATASET_REPO, filename=LOG_FILENAME, repo_type="dataset", token=token)
            existing_log = pd.read_csv(log_path)
            updated_log = pd.concat([existing_log, log_entry], ignore_index=True)
        except Exception:
            # File likely doesn't exist yet
            updated_log = log_entry

        # 4. Save and Upload
        updated_log.to_csv(LOG_FILENAME, index=False)
        
        api.upload_file(
            path_or_fileobj=LOG_FILENAME,
            path_in_repo=LOG_FILENAME,
            repo_id=HF_DATASET_REPO,
            repo_type="dataset"
        )
        print("Prediction logged successfully.")
        
    except Exception as e:
        print(f"Failed to log prediction: {e}")

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
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            
            if prediction == 1:
                st.success(f"Prediction: Will Purchase (Probability: {probability:.2f})")
            else:
                st.warning(f"Prediction: Will Not Purchase (Probability: {probability:.2f})")
                
            # Log the prediction
            save_prediction(input_data, prediction, probability)
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.info("Ensure all columns used in training are provided in the input dataframe.")
