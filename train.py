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
print("STARTING MODEL TRAINING PIPELINE")
print("=" * 60)

# ============ Step 1: Load Dataset ============
print("\n[1/5] Loading dataset from Hugging Face...")
try:
    dataset = load_dataset("AngadSi/sales-forecast-dataset")
    df = dataset["train"].to_pandas()
    print(f"✓ Dataset loaded successfully! Shape: {df.shape}")
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    exit(1)

# ============ Step 2: Data Preprocessing ============
print("\n[2/5] Preprocessing data...")
try:
    # Remove unnecessary columns
    df_cleaned = df.drop(
        columns=["Product_Id", "Store_Id"],
        errors="ignore"  # ← Prevents crash if columns missing
    )
    
    # Feature engineering: Calculate Store Age
    current_year = datetime.datetime.now().year
    df_cleaned["Store_Age"] = current_year - df_cleaned["Store_Establishment_Year"]
    df_cleaned.drop("Store_Establishment_Year", axis=1, inplace=True)
    
    # Split features and target
    X = df_cleaned.drop("Product_Store_Sales_Total", axis=1)
    y = df_cleaned["Product_Store_Sales_Total"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Save processed data
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    
    print(f"✓ Data preprocessed successfully!")
    print(f"  Train shape: {X_train.shape}")
    print(f"  Test shape: {X_test.shape}")
except Exception as e:
    print(f"✗ Error preprocessing data: {e}")
    exit(1)

# ============ Step 3: Feature Engineering & Preprocessing ============
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
    
    print(f"✓ Preprocessing pipeline created!")
    print(f"  Categorical features: {categorical_cols}")
    print(f"  Numerical features: {numerical_cols}")
except Exception as e:
    print(f"✗ Error setting up preprocessor: {e}")
    exit(1)

# ============ Step 4: Model Training ============
print("\n[4/5] Training model with GridSearchCV...")
try:
    # Create pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=42))
    ])
    
    # Define hyperparameter grid
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5]
    }
    
    # Use n_jobs from environment (1 for CI, -1 for local speed)
    n_jobs = int(os.getenv("N_JOBS", "1"))
    
    # Grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=n_jobs
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    print(f"✓ Model training completed!")
    print(f"  Best parameters: {grid_search.best_params_}")
except Exception as e:
    print(f"✗ Error training model: {e}")
    exit(1)

# ============ Step 5: Model Evaluation ============
print("\n[5/5] Evaluating model...")
try:
    y_pred = best_model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"✓ Model evaluation completed!")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R² Score: {r2:.4f}")
    
    # Save model
    joblib.dump(best_model, "sales_forecast_model.pkl")
    print(f"\n✓ Model saved as 'sales_forecast_model.pkl'")
    
except Exception as e:
    print(f"✗ Error evaluating model: {e}")
    exit(1)

print("\n" + "=" * 60)
print("MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 60)
