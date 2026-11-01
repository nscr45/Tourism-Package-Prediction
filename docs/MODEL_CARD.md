# Model card: Wellness Tourism Package Prediction

## Overview

This project uses an XGBoost classifier to predict whether a customer will purchase a wellness tourism package. The target column used by the training script is `ProdTaken`.

I built the model as part of a machine learning project and connected it to a Streamlit app so that customer details can be entered and a prediction can be generated.

## Intended use

The model is intended for learning, experimentation and portfolio demonstration. It can be used to explore how customer and sales-pitch attributes relate to the purchase outcome represented in the training data.

It is not a guarantee of an individual customer's future behaviour and should not be the only input to a business decision.

## Model and training approach

- **Estimator:** XGBoost classifier (`XGBClassifier`)
- **Pipeline:** scikit-learn `Pipeline` with a `ColumnTransformer`
- **Numerical preprocessing:** mean imputation
- **Categorical preprocessing:** most-frequent imputation followed by one-hot encoding
- **Hyperparameter search:** three-fold `GridSearchCV`
- **Search parameters:** `n_estimators`, `max_depth`, and `learning_rate`
- **Class imbalance:** the training script calculates `scale_pos_weight` from the negative and positive class counts
- **Saved format:** joblib pipeline at `models/model.joblib`

The complete preprocessing and prediction pipeline is saved together so the same transformations are applied when the app receives new customer details.

## Input features

The Streamlit app collects customer and interaction details including age, contact type, city tier, occupation, gender, party size, preferred property rating, marital status, travel history, passport and car ownership, designation, monthly income, pitch satisfaction, product pitched, follow-up count and pitch duration. The app uses an animated coastal SVG background stored in `app/tourism_hero.svg`; this visual asset does not affect model predictions.

The exact feature columns expected by the model are determined by the training dataset used for that run.

## Evaluation

The training script calculates test accuracy and prints a classification report after fitting the selected pipeline. It also records the best cross-validation parameters and score in `models/experiment_log.json`.

No fixed performance metrics are included in this card because a verified output from a current, reproducible training run has not been added to the repository.

## Data

The training script first uses `data/processed/train.csv` and `data/processed/test.csv` when those files are present. Otherwise, it downloads the same filenames from the configured Hugging Face dataset repository. The data preparation script reads `cleaned_tourism.csv` from that repository and publishes the train/test splits. `ProdTaken` is the target column. Dataset provenance, collection details, licensing and redistribution terms should be documented here once confirmed.

## Limitations

- Model performance depends on the quality, representativeness and distribution of the training data.
- The app's predictions may be unreliable for customer profiles that differ from the data used during training.
- Accuracy alone does not describe all classification errors; precision, recall, F1-score and a confusion matrix should also be reviewed.
- The app's input options are manually specified and should be kept consistent with the categories and ranges supported by the training data.
- A model prediction is an estimate, not a confirmed purchase decision.

## Maintenance

Retrain and evaluate the model when the underlying data changes. The manually triggered GitHub Actions workflow prepares the data, trains and uploads the model, then updates the Hugging Face Space. Keep the training data, preprocessing, model artifact and evaluation results versioned so that predictions can be traced to the model that generated them.
