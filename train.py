"""
Train script for Sales Forecast Model
Automatically trains and saves the model
"""

import os
import pandas as pd
import numpy as np
import joblib
import datetime
from datasets import load_dataset

# Create output directories if they don't exist
os.makedirs("data/processed", exist_ok=True)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("=" * 60)
print("MODEL TRAINING PIPELINE")
print("=" * 60)

# Step 1: Load Dataset
print("\n[1/5] Loading dataset from Hugging Face...")
try:
    dataset = load_dataset("AngadSi/sales-forecast-dataset")
    df = dataset["train"].to_pandas()
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    print(f"Available columns: {df.columns.tolist()}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Step 2: Data Preprocessing
print("\n[2/5] Preprocessing data...")
try:
    # Remove unnecessary columns that may not always be present
    df_cleaned = df.drop(
        columns=["Product_Id", "Store_Id"],
        errors="ignore"
    )
    
    # Calculate Store Age if the year column exists
    current_year = datetime.datetime.now().year
    if "Store_Establishment_Year" in df_cleaned.columns:
        df_cleaned["Store_Age"] = current_year - df_cleaned["Store_Establishment_Year"]
        df_cleaned.drop("Store_Establishment_Year", axis=1, inplace=True)
        print("Store age feature created from establishment year")
    else:
        print("Store establishment year column not found - skipping store age feature")
    
    # Separate features and target variable
    X = df_cleaned.drop("Product_Store_Sales_Total", axis=1)
    y = df_cleaned["Product_Store_Sales_Total"]
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Save processed data for reference
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    
    print(f"Data preprocessing complete. Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
except Exception as e:
    print(f"Error during preprocessing: {e}")
    exit(1)

# Step 3: Prepare Data Transformation Pipeline
print("\n[3/5] Setting up preprocessing pipeline...")
try:
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )
    
    print(f"Pipeline configured with {len(numerical_cols)} numerical and {len(categorical_cols)} categorical features")
except Exception as e:
    print(f"Error setting up preprocessor: {e}")
    exit(1)

# Step 4: Model Training
print("\n[4/5] Training model with hyperparameter tuning...")
try:
    # Create machine learning pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=42))
    ])
    
    # Define hyperparameters to test
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5]
    }
    
    # Use n_jobs from environment for stability in CI/CD
    n_jobs = int(os.getenv("N_JOBS", "1"))
    
    # Perform grid search cross-validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=n_jobs
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    print(f"Model training complete. Best parameters: {grid_search.best_params_}")
except Exception as e:
    print(f"Error during model training: {e}")
    exit(1)

# Step 5: Model Evaluation
print("\n[5/5] Evaluating model...")
try:
    y_pred = best_model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model evaluation complete:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R-squared: {r2:.4f}")
    
    # Save the trained model
    joblib.dump(best_model, "sales_forecast_model.pkl")
    print(f"Model saved successfully as 'sales_forecast_model.pkl'")
    
except Exception as e:
    print(f"Error during model evaluation: {e}")
    exit(1)

print("\n" + "=" * 60)
print("MODEL TRAINING PIPELINE COMPLETED")
print("=" * 60)
