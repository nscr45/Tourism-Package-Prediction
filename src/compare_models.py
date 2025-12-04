import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
import os

def compare_models():
    try:
        # Update paths to match user's structure
        base_path = ".github/workflows/tourism_project/data"
        train_df = pd.read_csv(f"{base_path}/train.csv")
        test_df = pd.read_csv(f"{base_path}/test.csv")
    except FileNotFoundError:
        print("Processed data not found. Please ensure files are in .github/workflows/tourism_project/data/")
        return

    X_train = train_df.drop('ProdTaken', axis=1)
    y_train = train_df['ProdTaken']
    X_test = test_df.drop('ProdTaken', axis=1)
    y_test = test_df['ProdTaken']
    
    # Identify column types
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    
    # Define Preprocessing Pipeline
    numerical_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Define models to compare
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Bagging": BaggingClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    best_model_name = None
    best_accuracy = 0.0
    
    print("Starting model comparison...")
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', model)])
        
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            
    print(f"\n------------------------------------------------")
    print(f"Best Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")
    print(f"------------------------------------------------")
    print(f"Please update src/train.py to use {best_model_name} if you wish to use it for production.")

if __name__ == "__main__":
    compare_models()
