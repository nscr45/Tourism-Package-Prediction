import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
import joblib
import os
import json
import sklearn
from huggingface_hub import HfApi, HfFolder, hf_hub_download

# Configuration
HF_USERNAME = "Sricharan451706"
HF_DATASET_REPO = f"{HF_USERNAME}/Tourism"
HF_MODEL_REPO = f"{HF_USERNAME}/Tourism-Package-Predictor"

def train_model():
    print(f"Training with scikit-learn {sklearn.__version__}")
    if sklearn.__version__ != "1.6.1":
        raise RuntimeError("Install the pinned dependencies first: scikit-learn==1.6.1. The saved pipeline must be trained with the same version used by the Space.")
    print("Loading training and test data...")
    local_train = "data/processed/train.csv"
    local_test = "data/processed/test.csv"
    try:
        if os.path.isfile(local_train) and os.path.isfile(local_test):
            train_df = pd.read_csv(local_train)
            test_df = pd.read_csv(local_test)
            print("Using the processed data artifacts from data/processed/.")
        else:
            token = get_token()
            train_path = hf_hub_download(
                repo_id=HF_DATASET_REPO, filename="train.csv",
                repo_type="dataset", token=token
            )
            test_path = hf_hub_download(
                repo_id=HF_DATASET_REPO, filename="test.csv",
                repo_type="dataset", token=token
            )
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            print("Loaded train/test data from Hugging Face.")
    except Exception as e:
        raise RuntimeError(
            "Could not load data. Run src/data_prep.py first or check the Hugging Face dataset files."
        ) from e

    if "ProdTaken" not in train_df.columns or "ProdTaken" not in test_df.columns:
        raise ValueError("Both train.csv and test.csv must contain the ProdTaken target column.")
    if train_df["ProdTaken"].nunique() < 2:
        raise ValueError("Training data must contain both target classes (0 and 1).")

    X_train = train_df.drop('ProdTaken', axis=1)
    y_train = train_df['ProdTaken']
    X_test = test_df.drop('ProdTaken', axis=1)
    y_test = test_df['ProdTaken']
    
    # Identify column types
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    
    # Preprocessing for numerical features
    numerical_transformer = SimpleImputer(strategy='mean')
    
    # Preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Create a preprocessor object using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Calculate scale_pos_weight for imbalance handling
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / max(pos_count, 1)
    print(f"Class Imbalance Detected. Using scale_pos_weight: {scale_pos_weight:.2f}")

    # Model parameters
    print(f"Defining XGBoost Model...")
    xgb_model = XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
    
    # Create pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', xgb_model)])
    
    # Define Parameter Grid for Tuning
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [3, 5, 10],
        'model__learning_rate': [0.01, 0.1, 0.2]
    }
    
    print("Starting Hyperparameter Tuning (GridSearchCV)...")
    # Experimentation Tracking: Log that tuning started
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Log tuned parameters
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print("\n------------------------------------------------")
    print(f"Best Parameters Found: {best_params}")
    print(f"Best Cross-Validation Accuracy: {best_score:.4f}")
    print("------------------------------------------------\n")
    
    # Save parameters to a log file (Experiment Tracking)
    os.makedirs("models", exist_ok=True)
    with open("models/experiment_log.json", "w") as f:
        log_data = {
            "model": "XGBoost",
            "best_params": best_params,
            "best_cv_score": best_score
        }
        json.dump(log_data, f, indent=4)
    print("Experiment parameters logged to models/experiment_log.json")

    # Use best model for final evaluation
    best_pipeline = grid_search.best_estimator_
    
    # --- Save Processed Datasets ---
    preprocessor_step = best_pipeline.named_steps['preprocessor']
    
    # Transform data
    X_train_processed = preprocessor_step.transform(X_train)
    X_test_processed = preprocessor_step.transform(X_test)
    
    # Get feature names
    feature_names = preprocessor_step.get_feature_names_out()
    
    # Convert to DataFrame
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
        X_test_processed = X_test_processed.toarray()
        
    X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names)
    
    # Add target variable
    X_train_processed_df['ProdTaken'] = y_train.values
    X_test_processed_df['ProdTaken'] = y_test.values
    
    # Save to CSV
    os.makedirs("data/processed_features", exist_ok=True)
    X_train_processed_df.to_csv("data/processed_features/train_processed.csv", index=False)
    X_test_processed_df.to_csv("data/processed_features/test_processed.csv", index=False)
    print("Processed datasets saved to data/processed_features/")
    # ------------------------------------
    
    predictions = best_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Test Set Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    
    # Save the BEST pipeline
    joblib.dump(best_pipeline, "models/model.joblib")
    print("Best model pipeline saved locally.")
    
    return best_pipeline, accuracy

def get_token():
    # 1. Try environment variable
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    
    # 2. Try local secrets.toml (relative to this script)
    try:
        # Go up one level from src/ to root, then into .streamlit/
        secrets_path = os.path.join(os.path.dirname(__file__), "../.streamlit/secrets.toml")
        if os.path.exists(secrets_path):
            with open(secrets_path, "r") as f:
                for line in f:
                    if "HF_TOKEN" in line:
                        # Parse: HF_TOKEN = "hf_..."
                        parts = line.split("=")
                        if len(parts) == 2:
                            return parts[1].strip().strip('"').strip("'")
    except Exception as e:
        print(f"Error reading secrets: {e}")
        pass
    
    # 3. Try Hugging Face CLI login
    return HfFolder.get_token()

def upload_model_to_hf():
    print("Uploading model to Hugging Face...")
    token = get_token()
    if not token:
        raise RuntimeError("HF_TOKEN not found. Set it before uploading the model.")
    api = HfApi(token=token)
    
    if token:
        try:
            api.create_repo(repo_id=HF_MODEL_REPO, exist_ok=True, token=token)
            
            # Upload Model
            api.upload_file(
                path_or_fileobj="models/model.joblib",
                path_in_repo="model.joblib",
                repo_id=HF_MODEL_REPO,
                token=token
            )
            
            # Upload Experiment Log
            if os.path.exists("models/experiment_log.json"):
                api.upload_file(
                    path_or_fileobj="models/experiment_log.json",
                    path_in_repo="experiment_log.json",
                    repo_id=HF_MODEL_REPO,
                    token=token
                )
                
            print(f"Model and logs uploaded/updated in {HF_MODEL_REPO}")
        except Exception as e:
            print(f"Failed to upload model: {e}")
    else:
        print("Hugging Face token not found. Please login or add HF_TOKEN to .streamlit/secrets.toml")

if __name__ == "__main__":
    train_model()
    upload_model_to_hf()


