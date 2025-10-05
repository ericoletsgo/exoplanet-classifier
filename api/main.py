"""
FastAPI Backend for Exoplanet Classifier
Provides REST API endpoints for prediction, training, metrics, and dataset access
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import joblib
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

app = FastAPI(
    title="Exoplanet Classifier API",
    description="REST API for exoplanet classification using machine learning",
    version="1.0.1"  # Bumped to force new deployment
)

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Must be False when allow_origins=["*"]
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# CORS middleware handles OPTIONS requests automatically

# Constants
# Get the parent directory (project root) to find model files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "properly_trained_model.joblib")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODELS_METADATA_FILE = os.path.join(BASE_DIR, "models", "models_metadata.json")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Most Important Features for Exoplanet Detection
# Based on feature importance analysis from trained model
# These are the top features that actually determine if something is an exoplanet
RELEVANT_FEATURES = {
    'signal_quality': [
        'koi_dikco_msky',      # Difference Image KIC-Centroid Offset (sky) - Most important
        'koi_dicco_msky',      # Difference Image Centroid Offset (sky)
        'koi_max_mult_ev',     # Maximum Multiple Event Statistic
        'koi_model_snr',       # Transit Signal-to-Noise Ratio
        'koi_dikco_mra',       # Difference Image KIC-Centroid Offset (RA)
    ],
    'flux_centroid': [
        'koi_fwm_srao',        # Flux-Weighted Centroid Offset (RA out-of-transit)
        'koi_fwm_sdeco',       # Flux-Weighted Centroid Offset (Dec out-of-transit)
        'koi_fwm_sra_err',     # Flux-Weighted Centroid RA error
        'koi_fwm_sdec_err',    # Flux-Weighted Centroid Dec error
        'koi_fwm_srao_err',    # Flux-Weighted Centroid RA out-of-transit error
    ],
    'orbital_params': [
        'koi_period',          # Orbital Period (days)
        'koi_depth',           # Transit Depth (ppm)
        'koi_duration',        # Transit Duration (hours)
        'koi_prad',            # Planetary Radius (Earth radii)
        'koi_impact',          # Impact Parameter
    ],
    'stellar_params': [
        'koi_steff',           # Stellar Effective Temperature (K)
        'koi_srad',            # Stellar Radius (solar radii)
        'koi_slogg',           # Stellar Surface Gravity (log10(cm/s^2))
        'koi_kepmag',          # Kepler Magnitude
    ],
    'error_params': [
        'koi_period_err1',     # Period Uncertainty (days)
        'koi_duration_err1',   # Duration Uncertainty (hours)
        'koi_depth_err1',      # Depth Uncertainty (ppm)
    ]
}

# Human-readable labels for features
FEATURE_LABELS = {
    # Signal Quality
    'koi_dikco_msky': 'Centroid Offset (Sky) - KIC',
    'koi_dicco_msky': 'Centroid Offset (Sky)',
    'koi_max_mult_ev': 'Multiple Event Statistic',
    'koi_model_snr': 'Signal-to-Noise Ratio',
    'koi_dikco_mra': 'Centroid Offset (RA) - KIC',
    
    # Flux Centroid
    'koi_fwm_srao': 'Flux Centroid RA (out-of-transit)',
    'koi_fwm_sdeco': 'Flux Centroid Dec (out-of-transit)',
    'koi_fwm_sra_err': 'Flux Centroid RA Error',
    'koi_fwm_sdec_err': 'Flux Centroid Dec Error',
    'koi_fwm_srao_err': 'Flux Centroid RA Error (out-of-transit)',
    
    # Orbital Parameters
    'koi_period': 'Orbital Period (days)',
    'koi_depth': 'Transit Depth (ppm)',
    'koi_duration': 'Transit Duration (hours)',
    'koi_prad': 'Planet Radius (Earth radii)',
    'koi_impact': 'Impact Parameter',
    
    # Stellar Parameters
    'koi_steff': 'Star Temperature (Kelvin)',
    'koi_srad': 'Star Radius (solar radii)',
    'koi_slogg': 'Star Surface Gravity (log g)',
    'koi_kepmag': 'Kepler Magnitude (brightness)',
    
    # Error Parameters
    'koi_period_err1': 'Period Uncertainty (days)',
    'koi_duration_err1': 'Duration Uncertainty (hours)',
    'koi_depth_err1': 'Depth Uncertainty (ppm)',
}

# Feature descriptions for tooltips
FEATURE_DESCRIPTIONS = {
    'koi_dikco_msky': 'Measures how much the light source shifts position during transit - helps detect false positives from background stars',
    'koi_dicco_msky': 'Similar to above, measures positional shift of the light source',
    'koi_max_mult_ev': 'Statistical measure of how likely multiple transits are real events',
    'koi_model_snr': 'How strong the transit signal is compared to background noise',
    'koi_dikco_mra': 'Positional shift in right ascension coordinate',
    
    'koi_fwm_srao': 'Where the center of light is when the planet is NOT transiting (RA)',
    'koi_fwm_sdeco': 'Where the center of light is when the planet is NOT transiting (Dec)',
    'koi_fwm_sra_err': 'Uncertainty in the RA centroid measurement',
    'koi_fwm_sdec_err': 'Uncertainty in the Dec centroid measurement',
    'koi_fwm_srao_err': 'Uncertainty in the out-of-transit RA measurement',
    
    'koi_period': 'How long it takes the planet to orbit its star',
    'koi_depth': 'How much the star dims during transit (parts per million)',
    'koi_duration': 'How long the transit lasts',
    'koi_prad': 'Size of the planet compared to Earth',
    'koi_impact': 'How close the planet passes to the center of the star (0=center, 1=edge)',
    
    'koi_steff': 'Surface temperature of the host star',
    'koi_srad': 'Size of the star compared to our Sun',
    'koi_slogg': 'Surface gravity of the star (higher = denser star)',
    'koi_kepmag': 'How bright the star appears (lower number = brighter)',
    
    'koi_period_err1': 'Measurement uncertainty in the orbital period',
    'koi_duration_err1': 'Measurement uncertainty in the transit duration',
    'koi_depth_err1': 'Measurement uncertainty in the transit depth',
}

def get_all_relevant_features():
    """Get all relevant features as a flat list"""
    all_features = []
    for category in RELEVANT_FEATURES.values():
        all_features.extend(category)
    return all_features

# Pydantic models
class PredictionRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="Feature values for prediction")

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    prediction_class: int

class MetricsResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]
    roc_data: Optional[Dict[str, Any]] = None
    feature_importances: Optional[List[Dict[str, Any]]] = None
    model_info: Dict[str, Any]

class TrainingRequest(BaseModel):
    dataset: str = Field(default="koi.csv", description="Dataset to train on (koi.csv, k2.csv, toi.csv)")
    model_name: str = Field(default="New Model", description="Name for the trained model")
    description: str = Field(default="", description="Description of the model")

class TrainingResponse(BaseModel):
    status: str
    message: str
    model_id: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

class DatasetResponse(BaseModel):
    total_rows: int
    columns: List[str]
    data: List[Dict[str, Any]]
    page: int
    page_size: int
    total_pages: int

# Global model cache
_model_cache = None

def load_model():
    """Load the trained model"""
    global _model_cache
    if _model_cache is None:
        print(f"[DEBUG] Model path: {MODEL_PATH}")
        print(f"[DEBUG] Model exists: {os.path.exists(MODEL_PATH)}")
        print(f"[DEBUG] Current working directory: {os.getcwd()}")
        print(f"[DEBUG] Files in current dir: {os.listdir('.')}")
        
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=404, detail=f"Model file not found: {MODEL_PATH}")
        
        try:
            print(f"[INFO] Loading model from {MODEL_PATH}")
            _model_cache = joblib.load(MODEL_PATH)
            print(f"[INFO] Model loaded successfully: {type(_model_cache).__name__}")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
    
    return _model_cache

def get_feature_names(model):
    """Extract feature names from model"""
    if hasattr(model, 'feature_names'):
        return model.feature_names
    elif hasattr(model, 'feature_names_in_'):
        return model.feature_names_in_.tolist()
    else:
        return get_all_relevant_features()

# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    model_status = "unknown"
    try:
        load_model()
        model_status = "loaded"
    except Exception as e:
        model_status = f"error: {str(e)}"
    
    return {
        "status": "online",
        "service": "Exoplanet Classifier API",
        "version": "1.0.0",
        "model_status": model_status,
        "endpoints": ["/predict", "/metrics", "/train", "/datasets", "/features"]
    }

@app.options("/features")
@app.get("/features")
async def get_features():
    """Get list of all features required for prediction with human-readable labels"""
    return {
        "features": RELEVANT_FEATURES,
        "labels": FEATURE_LABELS,
        "descriptions": FEATURE_DESCRIPTIONS,
        "total_features": len(get_all_relevant_features()),
        "categories": list(RELEVANT_FEATURES.keys())
    }

@app.options("/predict")
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction using the trained model"""
    try:
        model = load_model()
        feature_names = get_feature_names(model)
        
        # Create feature vector in correct order
        feature_vector = []
        missing_features = []
        
        for feature in feature_names:
            if feature in request.features:
                feature_vector.append(request.features[feature])
            else:
                feature_vector.append(0.0)  # Default to 0 for missing features
                missing_features.append(feature)
        
        # Make prediction
        X = np.array([feature_vector])
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # Map prediction to label
        label_map = {0: "FALSE POSITIVE", 1: "CANDIDATE", 2: "CONFIRMED"}
        prediction_label = label_map.get(int(prediction), "UNKNOWN")
        
        # Create probability dict
        prob_dict = {
            "FALSE POSITIVE": float(probabilities[0]),
            "CANDIDATE": float(probabilities[1]),
            "CONFIRMED": float(probabilities[2])
        }
        
        return PredictionResponse(
            prediction=prediction_label,
            confidence=float(max(probabilities)),
            probabilities=prob_dict,
            prediction_class=int(prediction)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.options("/predict-raw")
@app.post("/predict-raw", response_model=PredictionResponse)
async def predict_raw(request: dict):
    """Make a prediction using raw dataset row data"""
    try:
        model = load_model()
        feature_names = get_feature_names(model)
        
        # Extract features from raw row data
        feature_vector = []
        for feature in feature_names:
            if feature in request and pd.notna(request[feature]):
                feature_vector.append(float(request[feature]))
            else:
                feature_vector.append(0.0)
        
        # Make prediction
        X = np.array([feature_vector])
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # Map prediction to label
        label_map = {0: "FALSE POSITIVE", 1: "CANDIDATE", 2: "CONFIRMED"}
        prediction_label = label_map.get(int(prediction), "UNKNOWN")
        
        # Create probability dict
        prob_dict = {
            "FALSE POSITIVE": float(probabilities[0]),
            "CANDIDATE": float(probabilities[1]),
            "CONFIRMED": float(probabilities[2])
        }
        
        return PredictionResponse(
            prediction=prediction_label,
            confidence=float(max(probabilities)),
            probabilities=prob_dict,
            prediction_class=int(prediction)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Raw prediction failed: {str(e)}")

@app.options("/metrics")
@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get model performance metrics on held-out test set"""
    try:
        model = load_model()
        
        # Get feature names from model
        feature_names = get_feature_names(model)
        
        # Try to load the held-out test set first
        test_set_path = os.path.join(DATA_DIR, "test_set.joblib")
        if os.path.exists(test_set_path):
            # Use the proper held-out test set
            print(f"[INFO] Loading held-out test set from {test_set_path}")
            test_data = joblib.load(test_set_path)
            X = test_data['X_test']
            y = test_data['y_test']
            print(f"[INFO] Test set loaded: {len(X)} samples")
        else:
            # Fallback: Load full dataset (will show warning)
            # This is NOT ideal - metrics will be inflated
            print("[WARNING] Test set not found, using full dataset (metrics will be inflated!)")
            koi_path = os.path.join(DATA_DIR, "koi.csv")
            if not os.path.exists(koi_path):
                raise HTTPException(status_code=404, detail="Dataset not found")
            
            df = pd.read_csv(koi_path, comment='#')
            df['target'] = df['koi_disposition'].map({'CONFIRMED': 2, 'CANDIDATE': 1, 'FALSE POSITIVE': 0})
            df = df[df['target'].notna()]
            
            # Get features
            available_features = [f for f in feature_names if f in df.columns]
            
            X = df[available_features].fillna(0)
            y = df['target'].astype(int)
        
        # Make predictions
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y, y_pred)
        
        # ROC curve data (for multi-class)
        y_bin = label_binarize(y, classes=[0, 1, 2])
        roc_data = {}
        
        for i, label in enumerate(["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"]):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            roc_data[label] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": float(roc_auc)
            }
        
        # Feature importances
        feature_importances = None
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importances = [
                {"feature": name, "importance": float(imp)}
                for name, imp in zip(feature_names, importances)
            ]
            feature_importances.sort(key=lambda x: x['importance'], reverse=True)
        elif hasattr(model, 'estimators_'):
            # For ensemble models, try to get from first estimator
            try:
                first_estimator = model.estimators_[0][1] if hasattr(model.estimators_[0], '__getitem__') else model.estimators_[0]
                if hasattr(first_estimator, 'feature_importances_'):
                    importances = first_estimator.feature_importances_
                    feature_importances = [
                        {"feature": name, "importance": float(imp)}
                        for name, imp in zip(feature_names, importances)
                    ]
                    feature_importances.sort(key=lambda x: x['importance'], reverse=True)
            except:
                pass
        
        # Model info
        model_info = {
            "model_type": type(model).__name__,
            "n_features": len(feature_names),
            "n_samples": len(X),
            "classes": ["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"]
        }
        
        if hasattr(model, 'metadata'):
            model_info.update(model.metadata)
        
        return MetricsResponse(
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
            confusion_matrix=cm.tolist(),
            roc_data=roc_data,
            feature_importances=feature_importances,
            model_info=model_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@app.options("/train")
@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Trigger model training (runs in background)"""
    try:
        # For now, return a message that training is not implemented in API
        # In production, you'd use Celery or similar for background tasks
        return TrainingResponse(
            status="not_implemented",
            message="Training endpoint is not yet implemented. Please use the Streamlit interface or training scripts.",
            model_id=None,
            metrics=None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.options("/datasets/{dataset_name}")
@app.get("/datasets/{dataset_name}", response_model=DatasetResponse)
async def get_dataset(
    dataset_name: str,
    page: int = 1,
    page_size: int = 50,
    filter_disposition: Optional[str] = None
):
    """Get dataset with pagination and filtering"""
    try:
        # Validate dataset name
        valid_datasets = ["koi", "k2", "toi"]
        if dataset_name not in valid_datasets:
            raise HTTPException(status_code=400, detail=f"Invalid dataset. Must be one of: {valid_datasets}")
        
        file_path = os.path.join(DATA_DIR, f"{dataset_name}.csv")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Dataset file not found: {file_path}")
        
        # Load dataset
        df = pd.read_csv(file_path, comment='#')
        
        # Apply filter if specified
        if filter_disposition and 'koi_disposition' in df.columns:
            df = df[df['koi_disposition'] == filter_disposition.upper()]
        
        total_rows = len(df)
        total_pages = (total_rows + page_size - 1) // page_size
        
        # Paginate
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_df = df.iloc[start_idx:end_idx]
        
        # Convert to dict, handling NaN values
        data = page_df.replace({np.nan: None}).to_dict('records')
        
        return DatasetResponse(
            total_rows=total_rows,
            columns=df.columns.tolist(),
            data=data,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {str(e)}")

@app.options("/random-example/{dataset_name}")
@app.get("/random-example/{dataset_name}")
async def get_random_example(dataset_name: str, disposition: Optional[str] = None):
    """Get a random example from the dataset for testing predictions"""
    try:
        # Validate dataset name
        valid_datasets = ["koi", "k2", "toi"]
        if dataset_name not in valid_datasets:
            raise HTTPException(status_code=400, detail=f"Invalid dataset. Must be one of: {valid_datasets}")
        
        file_path = os.path.join(DATA_DIR, f"{dataset_name}.csv")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Dataset file not found: {file_path}")
        
        # Load dataset
        df = pd.read_csv(file_path, comment='#')
        
        # Filter by disposition if specified
        if disposition and 'koi_disposition' in df.columns:
            df = df[df['koi_disposition'] == disposition.upper()]
        
        if len(df) == 0:
            raise HTTPException(status_code=404, detail=f"No examples found for disposition: {disposition}")
        
        # Get random row
        random_row = df.sample(n=1).iloc[0]
        
        # Extract key features for the frontend
        # We'll use the actual row data for accurate predictions
        feature_names = get_all_relevant_features()
        features = {}
        
        for feature in feature_names:
            if feature in random_row and pd.notna(random_row[feature]):
                features[feature] = float(random_row[feature])
            else:
                features[feature] = 0.0
        
        # Get additional metadata
        metadata = {
            'row_index': int(random_row.name),
            'koi_name': str(random_row.get('kepoi_name', 'Unknown')),
            'expected_disposition': str(random_row.get('koi_disposition', 'Unknown')),
            'dataset': dataset_name
        }
        
        return {
            'features': features,
            'metadata': metadata,
            'raw_row': random_row.replace({np.nan: None}).to_dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get random example: {str(e)}")

@app.options("/models")
@app.get("/models")
async def list_models():
    """List all available trained models"""
    try:
        if not os.path.exists(MODELS_METADATA_FILE):
            return {"models": []}
        
        with open(MODELS_METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        
        return {"models": metadata}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

# Add static file serving for production
from fastapi.staticfiles import StaticFiles
import os

# Serve React build files in production
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    # Serve React app for frontend routes only
    from fastapi.responses import FileResponse
    
    # Serve React app for non-conflicting frontend routes
    @app.get("/batch")
    async def serve_batch_page():
        if os.path.exists("static/index.html"):
            return FileResponse("static/index.html")
        else:
            raise HTTPException(status_code=404, detail="Frontend not built")
    
    @app.get("/retrain")
    async def serve_retrain_page():
        if os.path.exists("static/index.html"):
            return FileResponse("static/index.html")
        else:
            raise HTTPException(status_code=404, detail="Frontend not built")
    
    # Catch-all for other frontend routes (React Router will handle routing)
    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        # Don't serve React for API routes or system routes
        if (full_path.startswith("api/") or 
            full_path.startswith("docs") or 
            full_path.startswith("redoc") or
            full_path.startswith("static/") or
            full_path == "openapi.json" or
            full_path in ["features", "metrics", "predict", "train", "datasets", "models", "random-example"] or
            full_path.startswith("datasets/") or
            full_path.startswith("random-example/")):
            raise HTTPException(status_code=404, detail="Not found")
        
        # Serve React app for all other routes
        if os.path.exists("static/index.html"):
            return FileResponse("static/index.html")
        else:
            raise HTTPException(status_code=404, detail="Frontend not built")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
