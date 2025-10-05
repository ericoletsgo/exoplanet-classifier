# Exoplanet Classifier 🪐

Machine learning system for classifying exoplanet candidates from NASA's Kepler mission data.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
cd frontend
npm install
cd ..
```

### 2. Start the Application
**Terminal 1 (API Server):**
```bash
.\start_api.bat
```

**Terminal 2 (React Frontend):**
```bash
.\start_frontend.bat
```

Then open **http://localhost:5173** (or the port shown in terminal) in your browser.

---

## Features

- **Smart Feature Selection**: Uses only the 19 most important features for accurate exoplanet detection
- **Modern Web Interface**: React + TypeScript frontend with interactive visualizations
- **REST API**: FastAPI backend for predictions and metrics
- **Real Performance**: ~83% accuracy on unseen data using ensemble machine learning

---

## Project Structure

```
exoplanet-classifier/
├── api/                          # FastAPI backend
│   └── main.py                   # API endpoints
├── frontend/                     # React frontend
│   └── src/                      # Source code
├── data/                         # NASA datasets (KOI, K2, TOI)
├── models/                       # Saved model versions & metadata
├── properly_trained_model.joblib # Main trained model
├── requirements.txt              # Python dependencies
├── start_api.bat                 # API server startup
└── start_frontend.bat            # Frontend startup
```

---

## Most Important Features for Detection

The model uses 19 key features across 4 categories:

### Signal Quality (5 features)
- **koi_dikco_msky** - Difference Image KIC-Centroid Offset (most important!)
- **koi_dicco_msky** - Difference Image Centroid Offset
- **koi_max_mult_ev** - Maximum Multiple Event Statistic
- **koi_model_snr** - Transit Signal-to-Noise Ratio
- **koi_dikco_mra** - KIC-Centroid Offset (RA)

### Flux Centroid (5 features)
- Flux-weighted centroid offsets and errors

### Orbital Parameters (5 features)
- Period, depth, duration, radius, impact parameter

### Stellar Parameters (4 features)
- Temperature, radius, surface gravity, magnitude

---

## Features

### React Frontend Pages
- **Home** - Dashboard with model info and feature overview
- **Predict** - Individual exoplanet classification with random examples
- **Batch Upload** - CSV file processing for multiple candidates
- **Model Retraining** - Train new models and manage existing ones
- **Metrics** - Model performance analysis and feature importance
- **Datasets** - Browse KOI, K2, and TOI datasets

### API Endpoints
- `GET /` - Health check
- `GET /features` - List all features with descriptions
- `POST /predict` - Make predictions
- `POST /predict-raw` - Predict using raw dataset rows
- `GET /metrics` - Model performance metrics
- `GET /datasets/{name}` - Browse datasets with pagination
- `GET /random-example/{dataset}` - Get random examples for testing
- `GET /models` - List all trained models

API docs: **http://localhost:8000/docs**

---

## Model Performance (on Held-Out Test Set)

- **Accuracy**: 83%
- **Precision**: 82%
- **Recall**: 83%
- **F1 Score**: 82%

**Per-Class Performance:**
- False Positive: 87% precision, 89% recall
- Candidate: 65% precision, 56% recall
- Confirmed Planet: 86% precision, 92% recall

Classes: Confirmed Planet, Candidate, False Positive

**Note**: These metrics are calculated on a held-out test set (20% of data) that the model has never seen during training. This represents the true performance on unseen data.

---

## License

MIT
