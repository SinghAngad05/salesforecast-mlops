# 📊 Sales Forecast ML Ops Pipeline

A production-ready Machine Learning Operations (MLOps) project that automatically trains and deploys a sales forecasting model using GitHub Actions, Hugging Face Model Hub, and Streamlit.

## 🎯 Overview

This project demonstrates an end-to-end automated ML pipeline:

```
GitHub Push → GitHub Actions → Train Model → Upload to HF Hub → Update Streamlit Space
```

### Key Features
- ✅ **Automated Training**: GitHub Actions triggers model training on every push
- ✅ **Model Registry**: Models stored in Hugging Face Model Hub
- ✅ **Continuous Deployment**: Streamlit app automatically uses the latest model
- ✅ **MLOps Best Practices**: Modular code, version control, and CI/CD integration
- ✅ **Data Pipeline**: Automated data processing and feature engineering

## 📁 Project Structure

```
SALESFORECAST-MLOPS/
│
├── .github/
│   └── workflows/
│       └── pipeline.yml              # GitHub Actions workflow
│
├── data/
│   ├── raw/
│   │   └── SuperKart.csv             # Raw dataset
│   └── processed/
│       ├── train.csv                 # Processed training data
│       └── test.csv                  # Processed test data
│
├── .gitignore
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── salesforecast.ipynb               # Development notebook
├── train.py                          # Main training script
└── upload_model.py                   # Model upload script
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/salesforecast-mlops.git
cd salesforecast-mlops
```

### 2. Setup Environment
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Train Model Locally
```bash
python train.py
```

### 4. Upload Model to Hugging Face
```bash
# Set your Hugging Face token
export HF_TOKEN="your_hf_token_here"

# Run upload script
python upload_model.py
```

## 🔄 GitHub Actions Workflow

The pipeline (`pipeline.yml`) automates the entire ML lifecycle:

### Trigger Events
- **On Push**: Automatically runs when you push code changes
- **Manual Dispatch**: Manually trigger from GitHub UI
- **Scheduled** (Optional): Can be configured for periodic retraining

### Pipeline Steps

| Step | Action | Purpose |
|------|--------|---------|
| 1️⃣ | Checkout Code | Pull latest repository code |
| 2️⃣ | Setup Python | Configure Python 3.11 environment |
| 3️⃣ | Install Dependencies | Install required packages |
| 4️⃣ | Train Model | Execute `train.py` |
| 5️⃣ | Verify Model | Check if model file exists |
| 6️⃣ | Upload to HF | Execute `upload_model.py` |
| 7️⃣ | Commit Data | Push processed data to repository |
| 8️⃣ | Summary | Display pipeline results |

## 📝 Configuration

### GitHub Secrets Setup

You need to set one secret in GitHub for the pipeline to work:

1. Go to your repository → **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Add:
   - **Name**: `HF_TOKEN`
   - **Value**: Your Hugging Face API token (get from https://huggingface.co/settings/tokens)

```bash
# Generate HF token at: https://huggingface.co/settings/tokens
# Then add it as a GitHub Secret
```

## 📊 Data

The project uses the **Sales Forecast Dataset** from Hugging Face:
- **Dataset ID**: `AngadSi/sales-forecast-dataset`
- **Features**: Product attributes, Store information
- **Target**: Product Store Sales Total

### Data Processing Steps
1. Load data from Hugging Face
2. Remove unnecessary columns (Product_Id, Store_Id)
3. Feature engineering (Store Age calculation)
4. Train-test split (80-20)
5. Save processed data to `data/processed/`

## 🤖 Model

**Algorithm**: Random Forest Regressor with GridSearchCV hyperparameter tuning

### Model Performance
- Metrics tracked: RMSE, MAE, R² Score
- Hyperparameters tuned:
  - `n_estimators`: [100, 200]
  - `max_depth`: [None, 10, 20]
  - `min_samples_split`: [2, 5]

### Model Storage
- Local: `sales_forecast_model.pkl`
- Cloud: Hugging Face Model Hub (`AngadSi/sales-forecast-model`)

## 🎯 Script Details

### `train.py`
Fully automated training pipeline with 5 steps:
1. Load dataset from HF
2. Preprocess data (cleaning, feature engineering)
3. Setup preprocessing pipeline
4. Train model with GridSearchCV
5. Evaluate and save model

**Usage**:
```bash
python train.py
```

**Output**: `sales_forecast_model.pkl`

### `upload_model.py`
Automatically uploads trained model to Hugging Face Hub with 3 steps:
1. Authenticate with HF_TOKEN
2. Create/ensure model repository exists
3. Upload model file

**Usage**:
```bash
export HF_TOKEN="your_token"
python upload_model.py
```

**Requirements**: HF_TOKEN environment variable must be set

## 🌐 Streamlit Integration

The Streamlit app bundles the model locally for fast, reliable startup:

**Streamli Repository**: `SALESFORECAST-PROJECT-FORECAST-APP`

```python
# Streamlit loads model from local bundled file
import joblib
model = joblib.load("sales_forecast_model.pkl")
```

### How Updates Work:
1. GitHub Actions trains new model
2. Model uploaded to HF Hub (registry/backup)
3. You manually download latest model to Space folder
4. Streamlit Space uses updated local model on next deploy

**Benefits of Local Loading:**
- ✅ Fast startup (no network delays)
- ✅ Works offline
- ✅ Reliable (no HF Hub download issues)
- ✅ No authentication needed

## 📦 Dependencies

See `requirements.txt` for all required packages:
- `streamlit` - Web app framework
- `pandas` - Data manipulation
- `scikit-learn` - ML algorithms
- `joblib` - Model serialization
- `huggingface_hub` - HF integration
- `datasets` - Dataset loading

## ✅ Testing the Workflow

### Method 1: Push Changes
```bash
# Make a small change
echo "# Updated" >> README.md

# Commit and push
git add .
git commit -m "Test pipeline"
git push origin main
```

### Method 2: Manual Trigger
1. Go to GitHub repository
2. Click **Actions** tab
3. Select **Sales Forecast ML Pipeline**
4. Click **Run workflow**

## 📊 Monitoring

Check pipeline status:
1. Go to repository **Actions** tab
2. Click on the workflow run
3. View detailed logs for each step

## 🔧 Customization

### Change Trigger Conditions
Edit `.github/workflows/pipeline.yml` trigger section:
```yaml
on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday
```

### Modify Model Parameters
Edit `train.py` `param_grid`:
```python
param_grid = {
    "model__n_estimators": [50, 100, 200, 300],
    "model__max_depth": [5, 10, 15, 20, None],
}
```

### Change HF Repository
Update these files with your HF username:
- `train.py`: Dataset repo ID
- `upload_model.py`: Model repo ID
- Streamlit app: Model loading repo ID

## 🐛 Troubleshooting

### Pipeline Fails in GitHub Actions
1. **Check logs**: Click on failed workflow run
2. **Verify secrets**: Ensure `HF_TOKEN` is set correctly
3. **Check dependencies**: Verify `requirements.txt` is complete
4. **Test locally**: Run `python train.py` locally first

### Model Upload Fails
- Verify HF_TOKEN is correct and has write permissions
- Check repository name matches your HF username
- Ensure model file exists: `sales_forecast_model.pkl`

### Streamlit Not Using Updated Model
- Wait for HF Hub caching to refresh (usually 5 mins)
- Manual refresh: Clear Streamlit cache in sidebar
- Check model exists at: https://huggingface.co/AngadSi/sales-forecast-model

## 📚 Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Hugging Face Hub Guide](https://huggingface.co/docs/hub/index)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-Learn Docs](https://scikit-learn.org/stable/)

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Support

For issues or questions:
- Create a GitHub Issue
- Check existing issues first
- Provide clear description and steps to reproduce

---

**Happy MLOps! 🚀**

*Last Updated: February 2026*
