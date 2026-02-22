"""
Upload Model to Hugging Face Hub
This script automatically uploads the trained model to Hugging Face Model Hub
"""

import os
from huggingface_hub import HfApi, create_repo

print("=" * 60)
print("UPLOADING MODEL TO HUGGING FACE HUB")
print("=" * 60)

# Retrieve HF token from environment
try:
    HF_TOKEN = os.environ["HF_TOKEN"]
except KeyError:
    print("Error: HF_TOKEN environment variable not set")
    print("Please set HF_TOKEN as a secret in GitHub Actions")
    exit(1)

# Initialize API client
print("\nInitializing Hugging Face API...")
try:
    api = HfApi(token=HF_TOKEN)
    print("API client initialized")
except Exception as e:
    print(f"Failed to initialize API: {e}")
    exit(1)

# Create model repository
print("\nEnsuring model repository exists...")
try:
    create_repo(
        "AngadSi/sales-forecast-model",
        repo_type="model",
        exist_ok=True,
        private=False,
        token=HF_TOKEN
    )
    print("Repository ready")
except Exception as e:
    print(f"Error creating/accessing repository: {e}")
    exit(1)

# Upload model file
print("\nUploading model to Hugging Face Hub...")
try:
    # Verify model file exists
    if not os.path.exists("sales_forecast_model.pkl"):
        print("Error: sales_forecast_model.pkl not found")
        print("Make sure train.py has been executed first")
        exit(1)
    
    # Upload the model
    api.upload_file(
        path_or_fileobj="sales_forecast_model.pkl",
        path_in_repo="sales_forecast_model.pkl",
        repo_id="AngadSi/sales-forecast-model",
        repo_type="model",
        commit_message="Updated model from GitHub Actions"
    )
    
    print("Model uploaded successfully")
    print(f"Repository: https://huggingface.co/AngadSi/sales-forecast-model")
    
except Exception as e:
    print(f"Error uploading model: {e}")
    exit(1)

print("\n" + "=" * 60)
print("MODEL UPLOAD COMPLETED")
print("=" * 60)
print("\nStreamlit Space will automatically use the updated model")
