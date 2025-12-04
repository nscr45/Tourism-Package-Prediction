# Wellness Tourism Package Prediction - MLOps Project

This project implements an end-to-end MLOps pipeline to predict whether a customer will purchase a Wellness Tourism Package. It includes data preparation, automated model training (XGBoost), and deployment using Streamlit and Docker.

## Project Structure

```
├── .github/
│   └── workflows/
│       └── pipeline.yml        # GitHub Actions CI/CD workflow
├── app/
│   ├── app.py                  # Streamlit application
│   ├── Dockerfile              # Docker configuration
│   └── requirements.txt        # App dependencies
├── data/                       # Data folder (ensure your dataset is here)
│   ├── train.csv               # Training dataset
│   └── test.csv                # Testing dataset
├── models/                     # Trained models and logs (gitignored)
│   ├── model.joblib            # Saved XGBoost pipeline
│   └── experiment_log.json     # Hyperparameter tuning logs
├── notebook/
│   └── tourism_mlops.ipynb     # Exploratory notebook
├── src/
│   ├── data_prep.py            # Data upload script
│   ├── train.py                # Model training (XGBoost + Tuning)
│   ├── compare_models.py       # Script to compare multiple algorithms
│   └── deploy_space.py         # Script to deploy to Hugging Face Spaces
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Deployment Instructions

### 1. Hugging Face Dataset (Manual Step)
You must manually upload your **raw dataset** to Hugging Face **once** to start the cycle.
*   **Where**: Hugging Face Dataset Repository (`Sricharan451706/tourism-package-prediction`)
*   **File**: Upload `cleaned_tourism.csv` (Your original raw data file).

### 2. GitHub Repository (Code)
Upload all the files in this folder to your GitHub Repository.
*   `.github/workflows/pipeline.yml`
*   `app/` (`app.py`, `Dockerfile`, `requirements.txt`)
*   `src/` (`data_prep.py`, `train.py`, `compare_models.py`, `deploy_space.py`)
*   `notebook/` (`tourism_mlops.ipynb`)
*   `requirements.txt`
*   `README.md`
*   `.gitignore`

### 3. Automation (Magic Happens Here)
Once you push the code to GitHub, the **GitHub Actions Pipeline** will automatically:
1.  **Data Prep**: Download `cleaned_tourism.csv` from HF, clean it, split it, and upload `train.csv` & `test.csv` back to HF.
2.  **Train**: Download `train.csv` & `test.csv` from HF, train XGBoost, and upload the model to HF Model Hub.
3.  **Deploy**: Deploy `app.py` to your Hugging Face Space.

## Setup Instructions

1.  **Prerequisites**:
    *   Python 3.9+
    *   Hugging Face Account (and Token)
    *   GitHub Account

2.  **Installation**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Running Locally**:
    *   **Train Model**: `python src/train.py`
    *   **Run App**: `streamlit run app.py`

4.  **Automation**:
    *   Push this repository to GitHub.
    *   Add `HF_TOKEN` to your GitHub Repository Secrets.
    *   The pipeline will automatically run on every push to `main`.

## Model Details
*   **Algorithm**: XGBoost Classifier
*   **Preprocessing**: Automated Pipeline (Imputation + OneHotEncoding)
*   **Tuning**: GridSearchCV for optimal parameters.
