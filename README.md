# Wellness Tourism Package Prediction

I built this project to predict whether a customer is likely to purchase a wellness tourism package. It uses customer, travel and sales-pitch information to make a binary prediction, with an XGBoost model at the centre of the workflow.

I also wanted to take the project beyond a model running in a notebook. The repository includes data preparation and training scripts, a Streamlit app for trying predictions, and integration with Hugging Face for storing the dataset, model and prediction logs.

## What the project does

- Prepares the tourism customer data and creates training and test splits.
- Handles missing numerical and categorical values.
- One-hot encodes categorical features and imputes missing values through a scikit-learn preprocessing pipeline.
- Trains an XGBoost classifier to predict the `ProdTaken` target.
- Uses GridSearchCV to tune the number of estimators, tree depth and learning rate.
- Evaluates the selected pipeline on the test data and saves the model with joblib.
- Provides a Streamlit form where customer details can be entered to get a prediction.
- Can upload the trained model and experiment details to Hugging Face.
- Can record submitted predictions and their probabilities in a Hugging Face dataset repository.

## Tech stack

| Area | Tools |
|---|---|
| Language | Python |
| Data handling | Pandas, NumPy |
| Machine learning | Scikit-learn, XGBoost |
| Preprocessing | SimpleImputer, OneHotEncoder, ColumnTransformer |
| Model tuning | GridSearchCV |
| Model storage | Joblib |
| Application | Streamlit |
| Model and dataset hosting | Hugging Face Hub |
| Automation and deployment | GitHub Actions, Docker |

## Project structure

```text
.
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── pipeline.yml
├── app/
│   ├── app.py
│   ├── tourism_hero.svg
│   ├── Dockerfile
│   └── requirements.txt
├── docs/
│   └── MODEL_CARD.md
├── src/
│   ├── data_prep.py
│   ├── train.py
│   ├── compare_models.py
│   └── deploy_space.py
├── .gitignore
├── requirements.txt
└── README.md
```

Generated datasets, trained model files and experiment logs are created at runtime and are not committed to this repository.

I have also documented the model's intended use, training approach, inputs and limitations in [the model card](docs/MODEL_CARD.md).

## How the model is trained

The training script reads the training and test CSV files from the configured Hugging Face dataset repository. It separates the `ProdTaken` target from the input features and identifies numerical and categorical columns.

The preprocessing pipeline:
1. Fills missing numerical values using the mean.
2. Fills missing categorical values using the most frequent value.
3. Applies one-hot encoding to categorical columns, while allowing categories that were not seen during training.

The preprocessor and XGBoost classifier are combined into one scikit-learn pipeline. GridSearchCV runs three-fold cross-validation over the configured values for `n_estimators`, `max_depth` and `learning_rate`. The best pipeline is evaluated on the test split and saved as `models/model.joblib`. The selected parameters and cross-validation score are written to `models/experiment_log.json`.

I have not added an accuracy figure here because I want the reported result to come from a reproducible run of the current training code.

## Running the Streamlit app

1. Clone the repository.
2. Install the application dependencies.
3. Make the trained model available locally at `models/model.joblib`, or configure access to the Hugging Face model repository used by the app.
4. Start Streamlit from the repository root.

```bash
git clone https://github.com/nscr45/Tourism-Package-Prediction.git
cd Tourism-Package-Prediction
pip install -r app/requirements.txt
streamlit run app/app.py
```

The app first looks for a local model at `models/model.joblib`. If it is not found, it attempts to download `model.joblib` from the configured Hugging Face model repository.

## Hugging Face configuration

The scripts currently refer to these repositories:

- Dataset: `Sricharan451706/Tourism`
- Model: `Sricharan451706/Tourism-Package-Predictor`
- Streamlit Space: `Sricharan451706/Tourism-Package-Predictor-Space`

The training and deployment scripts need a Hugging Face access token with the permissions required for the repositories being updated. For GitHub Actions, add it as a repository secret named `HF_TOKEN`. Do not commit access tokens or local secrets files. The deployment script uploads only the app, its requirements, the scenic SVG background and the trained model when present. It does not delete or replace unrelated files already in the Space.

The app can also log predictions to the configured Hugging Face dataset repository. This requires write access. Prediction logging is skipped when a token is not available.

## Current implementation notes

The repository includes two GitHub Actions workflows. `ci.yml` runs Python syntax checks on pushes and pull requests. `pipeline.yml` is a manually triggered MLOps workflow: it downloads the configured source CSV from the Hugging Face dataset, prepares and publishes train/test splits, trains and uploads the model, then updates the Hugging Face Space. Add `HF_TOKEN` to the GitHub repository secrets before running it. The source dataset file must exist as `cleaned_tourism.csv` in the configured dataset repository.

The Streamlit interface uses an animated, dark glass-style design with a coastal SVG background stored in `app/tourism_hero.svg`. The app embeds this asset at runtime, so the deployed Space does not depend on a separate image-hosting service. Run the deployment script from the repository root with `HF_TOKEN` set, or trigger the manual GitHub Actions workflow.

## Next improvements

- Make the GitHub Actions jobs pass processed data and model artifacts consistently.
- Add a reproducible evaluation report with accuracy, precision, recall, F1-score and a confusion matrix.
- Add tests for preprocessing, model loading and prediction inputs.
- Track model versions and training runs.
- Document the dataset source and its usage terms.

## Disclaimer

This is a machine learning portfolio project. Its predictions are estimates based on the data and model used for training. They should not be treated as guaranteed customer behaviour or used as the sole basis for business decisions.
