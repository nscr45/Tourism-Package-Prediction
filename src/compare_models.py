"""Compare baseline classifiers on the prepared tourism train/test split.

Run from the repository root after running src/data_prep.py.
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

DATA_DIR = Path("data/processed")


def compare_models() -> None:
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            "Prepared train/test data not found. Run python src/data_prep.py first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    if "ProdTaken" not in train_df or "ProdTaken" not in test_df:
        raise ValueError("Both datasets must contain the ProdTaken target column.")

    X_train, y_train = train_df.drop(columns="ProdTaken"), train_df["ProdTaken"]
    X_test, y_test = test_df.drop(columns="ProdTaken"), test_df["ProdTaken"]

    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
    numerical_cols = X_train.select_dtypes(include=["number"]).columns
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Bagging": BaggingClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(
            random_state=42, eval_metric="logloss", n_jobs=-1
        ),
    }

    results = []
    for name, estimator in models.items():
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="mean"), numerical_cols),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_cols,
                ),
            ]
        )
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", estimator)]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, zero_division=0)
        results.append({"model": name, "accuracy": accuracy, "f1_score": f1})
        print(f"\n{name}: accuracy={accuracy:.4f}, F1={f1:.4f}")
        print(classification_report(y_test, predictions, zero_division=0))

    results_df = pd.DataFrame(results).sort_values(
        ["f1_score", "accuracy"], ascending=False
    )
    print("\nModel comparison (sorted by F1, then accuracy):")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    compare_models()
