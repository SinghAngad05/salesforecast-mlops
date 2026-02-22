"""
Upload Model to Hugging Face Hub
This script automatically uploads the trained model to Hugging Face Model Hub
"""

import os
from huggingface_hub import HfApi, create_repo

print("=" * 60)
print("STARTING MODEL UPLOAD TO HUGGING FACE")
print("=" * 60)

# Get HF token from environment variable
try:
    HF_TOKEN = os.environ["HF_TOKEN"]
except KeyError:
    print("✗ Error: HF_TOKEN environment variable not set!")
    print("  Please set HF_TOKEN as a secret in GitHub Actions")
    exit(1)

# ============ Step 1: Initialize API Client ============
print("\n[1/3] Initializing Hugging Face API...")
try:
    api = HfApi(token=HF_TOKEN)
    print("✓ API client initialized!")
except Exception as e:
    print(f"✗ Failed to initialize API: {e}")
    exit(1)

# ============ Step 2: Create Repo (if doesn't exist) ============
print("\n[2/3] Ensuring model repository exists...")
try:
    create_repo(
        "AngadSi/sales-forecast-model",
        repo_type="model",
        exist_ok=True,
        private=False,
        token=HF_TOKEN
    )
    print("✓ Repository ready!")
except Exception as e:
    print(f"✗ Error creating/accessing repository: {e}")
    exit(1)

# ============ Step 3: Upload Model ============
print("\n[3/3] Uploading model to Hugging Face Hub...")
try:
    # Check if model file exists
    if not os.path.exists("sales_forecast_model.pkl"):
        print("✗ Error: sales_forecast_model.pkl not found!")
        print("  Make sure train.py has been executed first")
        exit(1)
    
    # Upload the model
    api.upload_file(
        path_or_fileobj="sales_forecast_model.pkl",
        path_in_repo="sales_forecast_model.pkl",
        repo_id="AngadSi/sales-forecast-model",
        repo_type="model",
        commit_message="Updated model from GitHub Actions"
    )
    
    print("✓ Model uploaded successfully!")
    print(f"  Repository: https://huggingface.co/AngadSi/sales-forecast-model")
    
except Exception as e:
    print(f"✗ Error uploading model: {e}")
    exit(1)

print("\n" + "=" * 60)
print("MODEL UPLOAD COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\n📌 Your Streamlit Space will automatically use the updated model!")
