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

### 2. Train the Model (First Time Only)
```bash
python fast_proper_training.py
```

### 3. Start the Application
```bash
.\start.ps1
```

Then open **http://localhost:5173** in your browser.

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
├── models/                       # Saved model versions
├── scripts/                      # Utility scripts
├── docs/                         # Documentation
├── properly_trained_model.joblib # Main trained model
├── fast_proper_training.py       # Training script
├── enhanced_exoplanet_classifier.py  # Streamlit app
├── requirements.txt              # Python dependencies
└── start.ps1                     # Startup script
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

## API Endpoints

- `GET /` - Health check
- `GET /features` - List all features
- `POST /predict` - Make predictions
- `GET /metrics` - Model performance metrics
- `GET /datasets/{name}` - Browse datasets

API docs: **http://localhost:8000/docs**

---

## Alternative: Streamlit Interface

```bash
streamlit run enhanced_exoplanet_classifier.py
```

---

## Documentation

See the `docs/` folder for detailed guides:
- **QUICKSTART.md** - Quick setup guide
- **FRONTEND_SETUP.md** - Frontend configuration
- **RETRAINING_GUIDE.md** - Model training instructions
- **APPLICATION_GUIDE.md** - Full application guide

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
