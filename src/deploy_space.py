from huggingface_hub import HfApi
import os

HF_USERNAME = "Sricharan451706"
SPACE_NAME = "tourism-package-prediction" # Ensure this space exists or create it
REPO_ID = f"{HF_USERNAME}/{SPACE_NAME}"

def deploy():
    api = HfApi()
    
    print(f"Deploying to {REPO_ID}...")
    
    # List of files to upload
    # We map local paths to remote paths (root of Space)
    files_to_upload = {
        "../app/app.py": "app.py",
        "../app/requirements.txt": "requirements.txt",
        "../app/Dockerfile": "Dockerfile",
        "../models/model.joblib": "model.joblib" 
    }
    
    try:
        # Create repo if it doesn't exist (optional, usually done manually or once)
        api.create_repo(repo_id=REPO_ID, repo_type="space", space_sdk="streamlit", exist_ok=True)
        
        for local_path, remote_path in files_to_upload.items():
            if os.path.exists(local_path):
                print(f"Uploading {local_path} as {remote_path}...")
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=remote_path,
                    repo_id=REPO_ID,
                    repo_type="space"
                )
            else:
                print(f"Warning: {local_path} not found.")
                
        print("Deployment completed successfully!")
        
    except Exception as e:
        print(f"Deployment failed: {e}")

if __name__ == "__main__":
    deploy()
