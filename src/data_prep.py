import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, HfFolder, hf_hub_download
import os
import numpy as np

# Configuration
HF_USERNAME = "Sricharan451706"
HF_DATASET_REPO = f"{HF_USERNAME}/Tourism"
RAW_DATA_FILENAME = "cleaned_tourism.csv" 

def process_and_update_data():
    print("Starting Data Preparation Pipeline...")
    
    # 1. Download Raw Data from Hugging Face
    try:
        print(f"Downloading {RAW_DATA_FILENAME} from {HF_DATASET_REPO}...")
        raw_data_path = hf_hub_download(repo_id=HF_DATASET_REPO, filename=RAW_DATA_FILENAME, repo_type="dataset")
        df = pd.read_csv(raw_data_path)
        print("Raw data downloaded successfully.")
    except Exception as e:
        print(f"Error downloading raw data: {e}")
        print("Please ensure 'Tourism.csv' exists in your Hugging Face Dataset repository.")
        return

    # 2. Data Cleaning
    print("Cleaning data...")
    # Drop CustomerID as it's not predictive
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])
    
    # Handle missing values
    if df.isna().any().any():
    # Numeric columns: fill with median (only if needed)
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in num_cols:
            if df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"  {col}: Filled {df[col].isna().sum()} NaNs with median {median_val}")
        
        # Categorical columns: fill with mode (only if needed)
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df[col].isna().any():
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
                    print(f"  {col}: Filled {df[col].isna().sum()} NaNs with mode '{mode_val[0]}'")
                else:
                    df[col] = df[col].fillna('Unknown')
                    print(f"  {col}: Filled with 'Unknown' (no mode found)")
        
        print("Data cleaning completed.")

    else:
        print("Data cleaning completed.")

    # 3. Train-Test Split
    print("Splitting data...")
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save locally temporarily
    os.makedirs("data/processed", exist_ok=True)
    train.to_csv("data/processed/train.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)
    
    # Save full cleaned data
    cleaned_full_path = f"data/processed/{RAW_DATA_FILENAME}"
    df.to_csv(cleaned_full_path, index=False)
    print(f"Full cleaned dataset saved locally as {cleaned_full_path}")

    print("Train/Test split saved locally.")

    # 4. Upload Updated Splits to Hugging Face
    print("Uploading new splits and cleaned data to Hugging Face...")
    api = HfApi()
    token = HfFolder.get_token()
    
    if token:
        try:
            # Upload Train
            api.upload_file(
                path_or_fileobj="data/processed/train.csv",
                path_in_repo="train.csv",
                repo_id=HF_DATASET_REPO,
                repo_type="dataset"
            )
            # Upload Test
            api.upload_file(
                path_or_fileobj="data/processed/test.csv",
                path_in_repo="test.csv",
                repo_id=HF_DATASET_REPO,
                repo_type="dataset"
            )
            # Upload Full Cleaned Data
            api.upload_file(
                path_or_fileobj=cleaned_full_path,
                path_in_repo=RAW_DATA_FILENAME,
                repo_id=HF_DATASET_REPO,
                repo_type="dataset"
            )
            print(f"Successfully updated {RAW_DATA_FILENAME}, train.csv, and test.csv in {HF_DATASET_REPO}")
        except Exception as e:
            print(f"Failed to upload to Hugging Face: {e}")
    else:
        print("Hugging Face token not found. Please login.")

if __name__ == "__main__":
    process_and_update_data()


