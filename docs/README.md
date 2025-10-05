# Exoplanet Classifier

Machine learning system for classifying exoplanet candidates from NASA's Kepler mission data using ensemble models.

## Features

- **Advanced ML Models**: Ensemble of Gradient Boosting, Random Forest, XGBoost, and LightGBM
- **Modern Web Interface**: React + TypeScript frontend with interactive visualizations
- **REST API**: FastAPI backend for predictions, metrics, and dataset access
- **Streamlit Dashboard**: Alternative UI for model training and analysis
- **Multiple Datasets**: Support for KOI, K2, and TOI datasets

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 18+
- pip and npm

### Installation

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

2. **Install frontend dependencies:**
```bash
cd frontend
npm install
cd ..
```

### Running the Application

**Option 1: Use the startup script (Windows)**
```bash
.\start.ps1
```

**Option 2: Manual start**

Terminal 1 - Backend:
```bash
cd api
python main.py
```

Terminal 2 - Frontend:
```bash
cd frontend
npm run dev
```

Then open `http://localhost:5173` in your browser.

### Alternative: Streamlit Interface

```bash
streamlit run enhanced_exoplanet_classifier.py
```

## Project Structure

```
exoplanet-classifier/
├── api/                              # FastAPI backend
│   ├── main.py                       # API endpoints
│   └── __init__.py
├── frontend/                         # React frontend
│   ├── src/
│   │   ├── pages/                    # Page components
│   │   ├── lib/                      # API client & utils
│   │   ├── App.tsx                   # Main app
│   │   └── main.tsx                  # Entry point
│   ├── package.json
│   └── vite.config.ts
├── models/                           # Saved model versions
├── enhanced_exoplanet_classifier.py  # Streamlit app
├── properly_trained_model.joblib     # Trained model
├── koi.csv, k2.csv, toi.csv         # Datasets
├── requirements.txt                  # Python dependencies
├── start.ps1                         # Startup script
└── FRONTEND_SETUP.md                 # Detailed setup guide

```

## API Endpoints

- `GET /` - Health check
- `GET /features` - List all features
- `POST /predict` - Make predictions
- `GET /metrics` - Model performance metrics
- `GET /datasets/{name}` - Browse datasets (koi, k2, toi)
- `GET /models` - List saved models

API documentation: `http://localhost:8000/docs`

## Features Used

The model analyzes 105 features across three categories:

- **Stellar Parameters** (26): Temperature, radius, mass, metallicity, magnitudes
- **Orbital Parameters** (39): Period, duration, depth, planet radius, eccentricity
- **Signal Quality** (40): SNR, centroid offsets, limb darkening coefficients

## Model Performance

- **Accuracy**: ~99%
- **Algorithms**: Gradient Boosting, Random Forest, XGBoost, LightGBM
- **Classes**: Confirmed, Candidate, False Positive

## Documentation

- **[FRONTEND_SETUP.md](FRONTEND_SETUP.md)** - Complete setup guide
- **[frontend/README.md](frontend/README.md)** - Frontend documentation
- **[RETRAINING_GUIDE.md](RETRAINING_GUIDE.md)** - Model retraining instructions

## Development

### Backend Development

```bash
# Run with auto-reload
uvicorn api.main:app --reload
```

### Frontend Development

```bash
cd frontend
npm run dev
```

Hot reload is enabled for both services.

## License

MIT
