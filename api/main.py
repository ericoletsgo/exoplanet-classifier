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
    version="1.1.2"  # Bumped to force deployment with timeout and error handling improvements
)

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Must be False when allow_origins=["*"]
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Add timeout middleware for better error handling (increased timeout for heavy operations)
@app.middleware("http")
async def timeout_middleware(request, call_next):
    import asyncio
    # Longer timeout for heavy operations like metrics and correlations
    timeout = 60.0 if request.url.path in ['/metrics', '/feature-correlations', '/train'] else 30.0
    try:
        response = await asyncio.wait_for(call_next(request), timeout=timeout)
        return response
    except asyncio.TimeoutError:
        return {"error": "Request timeout", "detail": f"The request took too long to process (>{timeout}s)"}

# CORS middleware handles OPTIONS requests automatically

# Constants
# Get the parent directory (project root) to find model files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "balanced_model_20251005_115605.joblib")
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
    model_id: Optional[str] = Field(default=None, description="Specific model ID to use for prediction")

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    prediction_class: int

class BatchPredictionRequest(BaseModel):
    records: List[Dict[str, float]] = Field(..., description="List of records for batch prediction")

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

class MetricsResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]
    roc_data: Optional[Dict[str, Any]] = None
    feature_importances: Optional[List[Dict[str, Any]]] = None
    model_info: Dict[str, Any]

class Hyperparameters(BaseModel):
    # Gradient Boosting parameters
    gb_n_estimators: int = Field(default=100, ge=50, le=500, description="Number of boosting stages")
    gb_learning_rate: float = Field(default=0.1, ge=0.01, le=0.3, description="Learning rate shrinks contribution of each tree")
    gb_max_depth: int = Field(default=3, ge=1, le=10, description="Maximum depth of individual regression estimators")
    gb_min_samples_split: int = Field(default=2, ge=2, le=20, description="Minimum number of samples required to split an internal node")
    
    # Random Forest parameters
    rf_n_estimators: int = Field(default=100, ge=50, le=500, description="Number of trees in the forest")
    rf_max_depth: int = Field(default=10, ge=5, le=20, description="Maximum depth of the tree")
    rf_min_samples_split: int = Field(default=2, ge=2, le=20, description="Minimum number of samples required to split an internal node")
    rf_max_features: str = Field(default="sqrt", description="Number of features to consider when looking for the best split")
    
    # XGBoost parameters
    xgb_n_estimators: int = Field(default=100, ge=50, le=500, description="Number of boosting rounds")
    xgb_learning_rate: float = Field(default=0.05, ge=0.01, le=0.3, description="Boosting learning rate")
    xgb_max_depth: int = Field(default=6, ge=3, le=10, description="Maximum tree depth for base learners")
    xgb_subsample: float = Field(default=1.0, ge=0.6, le=1.0, description="Subsample ratio of the training instances")
    
    # LightGBM parameters
    lgb_n_estimators: int = Field(default=100, ge=50, le=500, description="Number of boosting iterations")
    lgb_learning_rate: float = Field(default=0.05, ge=0.01, le=0.3, description="Boosting learning rate")
    lgb_max_depth: int = Field(default=-1, ge=-1, le=10, description="Maximum tree depth (-1 means no limit)")
    lgb_num_leaves: int = Field(default=31, ge=10, le=100, description="Maximum number of leaves in one tree")

class TrainingRequest(BaseModel):
    dataset: str = Field(default="koi.csv", description="Dataset to train on (koi.csv, k2.csv, or 'combined' for KOI+K2)")
    model_name: str = Field(default="New Model", description="Name for the trained model")
    description: str = Field(default="", description="Description of the model")
    test_size: float = Field(default=0.2, description="Test set size (0.1-0.5)")
    algorithms: List[str] = Field(default=["gradient_boosting", "random_forest", "xgboost", "lightgbm"], description="Algorithms to include in ensemble")
    hyperparameters: Optional[Hyperparameters] = Field(default=None, description="Algorithm-specific hyperparameters")
    use_hyperparameter_tuning: bool = Field(default=False, description="Enable hyperparameter tuning with grid search")
    include_k2: bool = Field(default=False, description="Include K2 dataset in combined training")
    target_column: Optional[str] = Field(default=None, description="Target column name (auto-detected if not specified)")
    target_mapping: Optional[Dict[str, int]] = Field(default=None, description="Custom mapping of target values to classes (0, 1, 2)")
    csv_data: Optional[str] = Field(default=None, description="CSV file content as string (for uploaded files)")

class TrainingResponse(BaseModel):
    status: str
    message: str
    model_id: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    algorithms_used: Optional[List[str]] = None
    cv_accuracy: Optional[float] = None
    dataset_summary: Optional[Dict[str, Any]] = None

class DatasetResponse(BaseModel):
    total_rows: int
    columns: List[str]
    data: List[Dict[str, Any]]
    page: int
    page_size: int
    total_pages: int

# Global caches for performance
model = None
metrics_cache = None
correlations_cache = None
models_metadata_cache = None

@app.on_event("startup")
def load_model_on_startup():
    """Load the trained model at application startup"""
    global model
    print(f"[DEBUG] Model path: {MODEL_PATH}")
    print(f"[DEBUG] Model exists: {os.path.exists(MODEL_PATH)}")
    print(f"[DEBUG] Current working directory: {os.getcwd()}")
    print(f"[DEBUG] Files in current dir: {os.listdir('.')}")
    print(f"[DEBUG] BASE_DIR: {BASE_DIR}")
    print(f"[DEBUG] Files in BASE_DIR: {os.listdir(BASE_DIR) if os.path.exists(BASE_DIR) else 'BASE_DIR not found'}")
    
    # Try multiple possible model locations
    possible_paths = [
        MODEL_PATH,
        os.path.join(BASE_DIR, "balanced_model_20251005_115605.joblib"),
        os.path.join(os.getcwd(), "properly_trained_model.joblib"),
        os.path.join(os.getcwd(), "balanced_model_20251005_115605.joblib"),
        "/app/properly_trained_model.joblib",
        "/app/balanced_model_20251005_115605.joblib"
    ]
    
    model_path_to_use = None
    for path in possible_paths:
        if os.path.exists(path):
            print(f"[INFO] Found model at: {path}")
            model_path_to_use = path
            break
    
    if not model_path_to_use:
        print(f"[ERROR] Model file not found in any of these locations:")
        for path in possible_paths:
            print(f"  - {path}")
        print(f"[ERROR] Model will not be available for predictions")
        model = None
        return
    
    try:
        print(f"[INFO] Loading model from {model_path_to_use}")
        model = joblib.load(model_path_to_use)
        print(f"[INFO] Model loaded successfully: {type(model).__name__}")
        print("[INFO] Model loaded successfully at startup")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {str(e)}")
        print(f"[ERROR] Model will not be available for predictions")
        model = None

def get_model():
    """Get the loaded model, raising an error if it's not available"""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check server logs for errors."
        )
    return model


def get_feature_names(model):
    """Extract feature names from model"""
    # For Pipeline objects, use the model's feature_names_in_ (input features)
    if hasattr(model, 'feature_names_in_'):
        return model.feature_names_in_.tolist()
    
    # Handle Pipeline objects with steps
    if hasattr(model, 'steps') and len(model.steps) > 0:
        # For Pipeline, get features from the last step (usually the classifier)
        last_step = model.steps[-1][1]
        if hasattr(last_step, 'feature_names_in_'):
            return last_step.feature_names_in_.tolist()
        elif hasattr(last_step, 'feature_names'):
            return last_step.feature_names
    
    # Handle regular models
    elif hasattr(model, 'feature_names'):
        return model.feature_names
    else:
        # Fallback to our predefined features
        return get_all_relevant_features()

# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    model_status = "unknown"
    try:
        get_model()
        model_status = "loaded"
    except Exception as e:
        model_status = f"error: {str(e)}"
    
    return {
        "status": "online",
        "service": "Exoplanet Classifier API",
        "version": "1.1.2",
        "model_status": model_status,
        "endpoints": ["/predict", "/metrics", "/train", "/datasets", "/features", "/models", "/batch-predict"],
        "timestamp": datetime.now().isoformat()
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
        # Load specific model if model_id provided, otherwise use default
        if request.model_id:
            model_path = os.path.join(MODELS_DIR, f"{request.model_id}.joblib")
            if not os.path.exists(model_path):
                raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
            model = joblib.load(model_path)
        else:
            model = get_model()
        
        feature_names = get_feature_names(model)
        
        # Create feature vector with all required features
        feature_vector = []
        
        for feature in feature_names:
            # Use provided value or default to 0
            feature_vector.append(request.features.get(feature, 0.0))
        
        # Make prediction
        import pandas as pd
        X = pd.DataFrame([feature_vector], columns=feature_names)
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

@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """Make predictions for a batch of records - MUCH faster than individual predictions"""
    try:
        model = get_model()
        feature_names = get_feature_names(model)

        # Create feature matrix for the entire batch
        feature_matrix = []
        for record in request.records:
            # Create feature vector with all required features
            feature_vector = []
            for feature in feature_names:
                feature_vector.append(record.get(feature, 0.0))
            feature_matrix.append(feature_vector)

        if not feature_matrix:
            return BatchPredictionResponse(predictions=[])

        # Make predictions in a single batch (vectorized operation)
        # Create DataFrame with proper feature names to avoid sklearn warnings
        import pandas as pd
        X = pd.DataFrame(feature_matrix, columns=feature_names)
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        # Map predictions to labels
        label_map = {0: "FALSE POSITIVE", 1: "CANDIDATE", 2: "CONFIRMED"}
        results = []
        for i in range(len(predictions)):
            pred_class = int(predictions[i])
            pred_label = label_map.get(pred_class, "UNKNOWN")
            probs = probabilities[i]
            prob_dict = {
                "FALSE POSITIVE": float(probs[0]),
                "CANDIDATE": float(probs[1]),
                "CONFIRMED": float(probs[2])
            }
            
            results.append(PredictionResponse(
                prediction=pred_label,
                confidence=float(max(probs)),
                probabilities=prob_dict,
                prediction_class=pred_class
            ))

        return BatchPredictionResponse(predictions=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/predict-raw", response_model=PredictionResponse)
async def predict_raw(request: dict):
    """Make a prediction using raw dataset row data"""
    try:
        model = get_model()
        feature_names = get_feature_names(model)
        
        # Create feature vector with all required features
        feature_vector = []
        
        import pandas as pd
        for feature in feature_names:
            if feature in request and pd.notna(request[feature]):
                feature_vector.append(float(request[feature]))
            else:
                feature_vector.append(0.0)
        
        # Make prediction
        import pandas as pd
        X = pd.DataFrame([feature_vector], columns=feature_names)
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
    """Get model performance metrics on held-out test set - OPTIMIZED with caching"""
    global metrics_cache
    
    # Return cached metrics if available (valid for 1 hour)
    if metrics_cache and metrics_cache.get('timestamp'):
        from datetime import timedelta
        cache_time = datetime.fromisoformat(metrics_cache['timestamp'])
        if datetime.now() - cache_time < timedelta(hours=1):
            print("[INFO] Returning cached metrics")
            return MetricsResponse(**metrics_cache['data'])
    
    try:
        model = get_model()
        
        # Get feature names from model
        feature_names = get_feature_names(model)
        
        # Use a smaller sample for faster computation (1000 samples max)
        koi_path = os.path.join(DATA_DIR, "koi.csv")
        if not os.path.exists(koi_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        print("[INFO] Loading sample dataset for metrics (optimized)")
        df = pd.read_csv(koi_path, comment='#')
        df['target'] = df['koi_disposition'].map({'CONFIRMED': 2, 'CANDIDATE': 1, 'FALSE POSITIVE': 0})
        df = df[df['target'].notna()]
        
        # Use stratified sample for representative metrics
        sample_size = min(1000, len(df))
        df_sample = df.groupby('target', group_keys=False).apply(
            lambda x: x.sample(min(len(x), sample_size // 3), random_state=42)
        )
        
        # Get features
        available_features = [f for f in feature_names if f in df_sample.columns]
        X = df_sample[available_features].fillna(0)
        y = df_sample['target'].astype(int)
        
        print(f"[INFO] Computing metrics on {len(X)} samples")
        
        # Make predictions
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y, y_pred)
        
        # Simplified ROC data (only AUC, not full curves)
        y_bin = label_binarize(y, classes=[0, 1, 2])
        roc_data = {}
        
        for i, label in enumerate(["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"]):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            roc_data[label] = {
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
        
        result = MetricsResponse(
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
            confusion_matrix=cm.tolist(),
            roc_data=roc_data,
            feature_importances=feature_importances,
            model_info=model_info
        )
        
        # Cache the result
        metrics_cache = {
            'data': result.dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@app.options("/train")
@app.post("/train", response_model=TrainingResponse)
async def train_advanced_ensemble(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Advanced ensemble training matching Streamlit functionality"""
    try:
        # Load the dataset(s)
        if request.csv_data:
            # Handle uploaded CSV data
            print(f"[INFO] Loading uploaded CSV data")
            from io import StringIO
            df = pd.read_csv(StringIO(request.csv_data), comment='#')
            print(f"[INFO] CSV loaded: {len(df)} samples")
            
            # Determine target column
            if request.target_column:
                disposition_col = request.target_column
                if disposition_col not in df.columns:
                    raise HTTPException(status_code=400, detail=f"Target column '{disposition_col}' not found in CSV")
            else:
                # Auto-detect disposition column
                disposition_col = None
                if 'koi_disposition' in df.columns:
                    disposition_col = 'koi_disposition'
                elif 'disposition' in df.columns:
                    disposition_col = 'disposition'
                else:
                    raise HTTPException(status_code=400, detail="No disposition column found in CSV. Please specify target_column.")
            
            print(f"[INFO] Using target column: {disposition_col}")
            print(f"[INFO] Target distribution: {df[disposition_col].value_counts().to_dict()}")
            
            # Use custom mapping if provided, otherwise use default
            if request.target_mapping:
                df['target'] = df[disposition_col].map(request.target_mapping)
            else:
                df['target'] = df[disposition_col].map({'CONFIRMED': 2, 'CANDIDATE': 1, 'FALSE POSITIVE': 0})
            
            df = df[df['target'].notna()]
            df['dataset_source'] = 'uploaded_csv'
            
            print(f"[INFO] Valid samples after filtering: {len(df)}")
            
        elif request.dataset == "combined":
            # Load multiple datasets and combine them
            datasets = []
            dataset_info = []
            
            # Always include KOI as base
            koi_path = os.path.join(DATA_DIR, "koi.csv")
            if os.path.exists(koi_path):
                koi_df = pd.read_csv(koi_path, comment='#')
                koi_df['target'] = koi_df['koi_disposition'].map({'CONFIRMED': 2, 'CANDIDATE': 1, 'FALSE POSITIVE': 0})
                koi_df = koi_df[koi_df['target'].notna()]
                koi_df['dataset_source'] = 'koi'
                datasets.append(koi_df)
                dataset_info.append(f"KOI: {len(koi_df)} samples")
            
            # Include K2 if requested
            if request.include_k2:
                k2_path = os.path.join(DATA_DIR, "k2_converted.csv")
                if os.path.exists(k2_path):
                    k2_df = pd.read_csv(k2_path, comment='#')
                    k2_df['target'] = k2_df['koi_disposition'].map({'CONFIRMED': 2, 'CANDIDATE': 1, 'FALSE POSITIVE': 0})
                    k2_df = k2_df[k2_df['target'].notna()]
                    k2_df['dataset_source'] = 'k2'
                    datasets.append(k2_df)
                    dataset_info.append(f"K2: {len(k2_df)} samples")
            
            if len(datasets) == 0:
                raise HTTPException(status_code=404, detail="No datasets found for combined training")
            
            # Combine all datasets
            df = pd.concat(datasets, ignore_index=True, sort=False)
            print(f"[INFO] Combined dataset created with {len(df)} samples from: {', '.join(dataset_info)}")
            
        else:
            # Single dataset training (original logic)
            dataset_path = os.path.join(DATA_DIR, request.dataset)
            if not os.path.exists(dataset_path):
                raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset}")
            
            print(f"[INFO] Loading dataset: {request.dataset}")
            df = pd.read_csv(dataset_path, comment='#')
            
            # Determine which disposition column to use
            if request.target_column:
                disposition_col = request.target_column
                if disposition_col not in df.columns:
                    raise HTTPException(status_code=400, detail=f"Target column '{disposition_col}' not found in dataset")
            else:
                # Auto-detect disposition column
                disposition_col = None
                if 'koi_disposition' in df.columns:
                    disposition_col = 'koi_disposition'
                elif 'disposition' in df.columns:
                    disposition_col = 'disposition'
                else:
                    raise HTTPException(status_code=400, detail="No disposition column found in dataset. Please specify target_column.")
            
            print(f"[INFO] Dataset loaded: {len(df)} samples with target distribution:")
            print(df[disposition_col].value_counts().to_dict())
            
            # Use custom mapping if provided, otherwise use default
            if request.target_mapping:
                df['target'] = df[disposition_col].map(request.target_mapping)
            else:
                df['target'] = df[disposition_col].map({'CONFIRMED': 2, 'CANDIDATE': 1, 'FALSE POSITIVE': 0})
            
            df = df[df['target'].notna()]
            df['dataset_source'] = request.dataset.replace('.csv', '')
            
            print(f"[INFO] Valid samples after filtering: {len(df)}")
        
        # Get numeric features (same approach as Streamlit)
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_features:
            numeric_features.remove('target')
        
        print(f"[INFO] Using {len(numeric_features)} numeric features")
        
        # Prepare features and target (same NaN handling as Streamlit)
        X = df[numeric_features].copy()
        
        # Fill NaN values properly (matching Streamlit logic)
        for col in X.columns:
            if X[col].isna().any():
                median_val = X[col].median()
                if pd.isna(median_val):
                    X[col] = X[col].fillna(0)
                else:
                    X[col] = X[col].fillna(median_val)
        
        # Final check (matching Streamlit)
        if X.isna().any().any():
            X = X.fillna(0)
        
        y = df['target']
        
        # Split data with configurable test size
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=request.test_size, random_state=42, stratify=y
        )
        
        # Create models based on request with hyperparameters
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
        models = []
        algorithms_used = []
        
        # Use default hyperparameters if not provided
        hyperparams = request.hyperparameters or Hyperparameters()
        
        # Always include basic models
        if 'gradient_boosting' in request.algorithms:
            gb_model = GradientBoostingClassifier(
                n_estimators=hyperparams.gb_n_estimators,
                learning_rate=hyperparams.gb_learning_rate,
                max_depth=hyperparams.gb_max_depth,
                min_samples_split=hyperparams.gb_min_samples_split,
                random_state=42
            )
            models.append(('gradient_boosting', gb_model))
            algorithms_used.append('gradient_boosting')
        
        if 'random_forest' in request.algorithms:
            rf_model = RandomForestClassifier(
                n_estimators=hyperparams.rf_n_estimators,
                max_depth=hyperparams.rf_max_depth,
                min_samples_split=hyperparams.rf_min_samples_split,
                max_features=hyperparams.rf_max_features,
                random_state=42
            )
            models.append(('random_forest', rf_model))
            algorithms_used.append('random_forest')
        
        # Try XGBoost
        if 'xgboost' in request.algorithms:
            try:
                import xgboost as xgb
                xgb_model = xgb.XGBClassifier(
                    n_estimators=hyperparams.xgb_n_estimators,
                    learning_rate=hyperparams.xgb_learning_rate,
                    max_depth=hyperparams.xgb_max_depth,
                    subsample=hyperparams.xgb_subsample,
                    random_state=42,
                    eval_metric='logloss'
                )
                models.append(('xgboost', xgb_model))
                algorithms_used.append('xgboost')
            except ImportError:
                pass  # XGBoost not available
                
        # Try LightGBM
        if 'lightgbm' in request.algorithms:
            try:
                import lightgbm as lgb
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=hyperparams.lgb_n_estimators,
                    learning_rate=hyperparams.lgb_learning_rate,
                    max_depth=hyperparams.lgb_max_depth,
                    num_leaves=hyperparams.lgb_num_leaves,
                    random_state=42,
                    verbose=-1
                )
                models.append(('lightgbm', lgb_model))
                algorithms_used.append('lightgbm')
            except ImportError:
                pass  # LightGBM not available
        
        if len(models) == 0:
            raise HTTPException(status_code=400, detail="No algorithms could be loaded")
        
        # Create ensemble (same as Streamlit)
        ensemble = VotingClassifier(models, voting='soft')
        
        print(f"[INFO] Training ensemble with {len(models)} algorithms: {algorithms_used}")
        
        # Hyperparameter tuning with grid search if enabled
        if request.use_hyperparameter_tuning:
            print("[INFO] Starting hyperparameter tuning with grid search...")
            from sklearn.model_selection import GridSearchCV
            
            # Define parameter grid for ensemble tuning
            param_grid = {
                'voting': ['soft', 'hard']
            }
            
            # Add individual algorithm tuning if only one algorithm is selected
            if len(request.algorithms) == 1:
                algorithm = request.algorithms[0]
                if algorithm == 'gradient_boosting':
                    param_grid.update({
                        'gradient_boosting__learning_rate': [0.05, 0.1, 0.15],
                        'gradient_boosting__max_depth': [3, 5, 7],
                        'gradient_boosting__n_estimators': [100, 200]
                    })
                elif algorithm == 'random_forest':
                    param_grid.update({
                        'random_forest__max_depth': [5, 10, 15],
                        'random_forest__n_estimators': [100, 200],
                        'random_forest__max_features': ['sqrt', 'log2']
                    })
                elif algorithm == 'xgboost':
                    param_grid.update({
                        'xgboost__learning_rate': [0.05, 0.1, 0.15],
                        'xgboost__max_depth': [4, 6, 8],
                        'xgboost__n_estimators': [100, 200]
                    })
                elif algorithm == 'lightgbm':
                    param_grid.update({
                        'lightgbm__learning_rate': [0.05, 0.1, 0.15],
                        'lightgbm__max_depth': [3, 5, 7],
                        'lightgbm__num_leaves': [20, 31, 50]
                    })
            
            # Perform grid search with cross-validation
            grid_search = GridSearchCV(
                ensemble, 
                param_grid, 
                cv=3, 
                scoring='accuracy', 
                n_jobs=-1, 
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            ensemble = grid_search.best_estimator_
            
            print(f"[INFO] Best parameters: {grid_search.best_params_}")
            print(f"[INFO] Best cross-validation score: {grid_search.best_score_:.4f}")
        else:
            # Train the model normally
            ensemble.fit(X_train, y_train)
        
        # Evaluate on train and test sets (matching Streamlit)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        y_pred_train = ensemble.predict(X_train)
        y_pred_test = ensemble.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Calculate additional metrics (matching Streamlit)
        precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
        
        # Get confusion matrix (matching Streamlit)
        cm = confusion_matrix(y_test, y_pred_test)
        
        print(f"[INFO] Training complete - Train Accuracy: {train_accuracy:.2%}, Test Accuracy: {test_accuracy:.2%}")
        
        metrics = {
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'accuracy': float(test_accuracy),  # For backward compatibility
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': len(numeric_features)
        }
        
        # Generate model ID
        model_id = f"advanced_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save model (in production, you'd save to a proper model store)
        model_path = os.path.join(MODELS_DIR, f"{model_id}.joblib")
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Store metadata in model (matching Streamlit exactly)
        ensemble.feature_names = numeric_features
        ensemble.metadata = {
            'id': model_id,
            'name': request.model_name,
            'description': request.description,
            'created_at': datetime.now().isoformat(),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'features': numeric_features,
            'n_features': len(numeric_features),
            'algorithms': algorithms_used
        }
        
        joblib.dump(ensemble, model_path)
        
        # Update models metadata
        metadata_file = os.path.join(MODELS_DIR, "models_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = []
        
        # Create metadata for JSON file (matching Streamlit structure)
        model_metadata = {
            'id': model_id,
            'name': request.model_name,
            'description': request.description,
            'created_at': ensemble.metadata['created_at'],
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'features': numeric_features,
            'n_features': len(numeric_features),
            'algorithms': algorithms_used,
            'model_path': model_path
        }
        
        metadata.append(model_metadata)
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create dataset summary with source information
        dataset_summary = {
            'total_samples': len(X), 
            'features': len(numeric_features),
            'dataset_sources': df['dataset_source'].value_counts().to_dict() if 'dataset_source' in df.columns else {request.dataset.replace('.csv', ''): len(X)}
        }
        
        return TrainingResponse(
            status="completed",
            message=f"✅ Model trained successfully! Train Accuracy: {train_accuracy:.1%}, Test Accuracy: {test_accuracy:.1%}",
            model_id=model_id,
            metrics=metrics,
            algorithms_used=algorithms_used,
            cv_accuracy=float(train_accuracy),  # Use train accuracy for consistency
            dataset_summary=dataset_summary
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
        
        # Use k2_converted for K2 dataset
        if dataset_name == "k2":
            file_path = os.path.join(DATA_DIR, "k2_converted.csv")
        else:
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
            'koi_name': str(random_row.get('kepoi_name', random_row.get('toi_id', 'Unknown'))),
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

@app.options("/random-example")
@app.get("/random-example")
async def get_random_example_from_all_datasets(disposition: Optional[str] = None):
    """Get a random example from all available datasets"""
    try:
        # Get all available datasets
        available_datasets = []
        
        # Check KOI
        koi_path = os.path.join(DATA_DIR, "koi.csv")
        if os.path.exists(koi_path):
            koi_df = pd.read_csv(koi_path, comment='#')
            if disposition and 'koi_disposition' in koi_df.columns:
                koi_df = koi_df[koi_df['koi_disposition'] == disposition.upper()]
            if len(koi_df) > 0:
                available_datasets.append(('koi', koi_df, koi_path))
        
        # Check K2
        k2_path = os.path.join(DATA_DIR, "k2_converted.csv")
        if os.path.exists(k2_path):
            k2_df = pd.read_csv(k2_path, comment='#')
            if disposition and 'koi_disposition' in k2_df.columns:
                k2_df = k2_df[k2_df['koi_disposition'] == disposition.upper()]
            if len(k2_df) > 0:
                available_datasets.append(('k2', k2_df, k2_path))
        
        # Check TOI
        toi_path = os.path.join(DATA_DIR, "toi.csv")
        if os.path.exists(toi_path):
            toi_df = pd.read_csv(toi_path, comment='#')
            if disposition and 'koi_disposition' in toi_df.columns:
                toi_df = toi_df[toi_df['koi_disposition'] == disposition.upper()]
            if len(toi_df) > 0:
                available_datasets.append(('toi', toi_df, toi_path))
        
        if len(available_datasets) == 0:
            raise HTTPException(status_code=404, detail=f"No examples found for disposition: {disposition}")
        
        # Randomly select a dataset
        selected_index = np.random.choice(len(available_datasets))
        dataset_name, df, _ = available_datasets[selected_index]
        
        # Get random row from selected dataset
        random_row = df.sample(n=1).iloc[0]
        
        # Extract key features for the frontend
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
            'koi_name': str(random_row.get('kepoi_name', random_row.get('toi_id', 'Unknown'))),
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

@app.get("/health")
async def health_check():
    """Dedicated health check endpoint for monitoring"""
    try:
        model = get_model()
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_type": type(model).__name__,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.options("/models")
@app.get("/models")
async def list_models():
    """List all available trained models - OPTIMIZED with caching"""
    global models_metadata_cache
    
    # Return cached models if available (valid for 30 minutes)
    if models_metadata_cache and models_metadata_cache.get('timestamp'):
        from datetime import timedelta
        cache_time = datetime.fromisoformat(models_metadata_cache['timestamp'])
        if datetime.now() - cache_time < timedelta(minutes=30):
            print("[INFO] Returning cached models metadata")
            return {"models": models_metadata_cache['data']}
    
    try:
        if not os.path.exists(MODELS_METADATA_FILE):
            return {"models": []}
        
        with open(MODELS_METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        
        # Cache the result
        models_metadata_cache = {
            'data': metadata,
            'timestamp': datetime.now().isoformat()
        }
        
        return {"models": metadata}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@app.options("/models/{model_id}/evaluate")
@app.get("/models/{model_id}/evaluate")
async def evaluate_model(model_id: str):
    """Evaluate a specific model and return detailed metrics"""
    try:
        # First try to load the model file
        model_path = os.path.join(MODELS_DIR, f"{model_id}.joblib")
        metadata = None
        
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                if hasattr(model, 'metadata'):
                    metadata = model.metadata
            except Exception as e:
                print(f"[WARNING] Failed to load model {model_id}: {e}")
        
        # If no model file or metadata, try to get from models_metadata.json
        if metadata is None:
            if os.path.exists(MODELS_METADATA_FILE):
                with open(MODELS_METADATA_FILE, 'r') as f:
                    models_metadata = json.load(f)
                
                # Find the model in metadata
                model_metadata = next((m for m in models_metadata if m.get('id') == model_id), None)
                if model_metadata:
                    metadata = model_metadata
                else:
                    raise HTTPException(status_code=404, detail=f"Model {model_id} not found in metadata")
            else:
                raise HTTPException(status_code=404, detail=f"Model {model_id} not found and no metadata file available")
        
        # Return comprehensive evaluation
        return {
            "model_id": model_id,
            "model_info": {
                "name": metadata.get('name', 'Unknown'),
                "description": metadata.get('description', ''),
                "created_at": metadata.get('created_at', ''),
                "algorithms": metadata.get('algorithms', []),
                "n_features": metadata.get('n_features', 0)
            },
            "metrics": {
                "train_accuracy": metadata.get('train_accuracy', 0),
                "test_accuracy": metadata.get('test_accuracy', 0),
                "precision": metadata.get('precision', 0),
                "recall": metadata.get('recall', 0),
                "f1_score": metadata.get('f1_score', 0)
            },
            "confusion_matrix": metadata.get('confusion_matrix', []),
            "dataset_info": {
                "train_samples": metadata.get('train_samples', 0),
                "test_samples": metadata.get('test_samples', 0)
            },
            "features": metadata.get('features', [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to evaluate model: {str(e)}")

@app.options("/datasets/{dataset_name}/columns")
@app.get("/datasets/{dataset_name}/columns")
async def get_dataset_columns(dataset_name: str):
    """Get column names and sample values from a dataset"""
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
        
        # Get columns with their types and sample unique values
        columns_info = []
        for col in df.columns:
            col_info = {
                "name": col,
                "type": str(df[col].dtype),
                "non_null_count": int(df[col].notna().sum()),
                "null_count": int(df[col].isna().sum())
            }
            
            # Add unique values for categorical columns
            if df[col].dtype == 'object' or df[col].nunique() < 20:
                unique_vals = df[col].dropna().unique().tolist()[:10]
                col_info["sample_values"] = [str(v) for v in unique_vals]
                col_info["unique_count"] = int(df[col].nunique())
            
            columns_info.append(col_info)
        
        return {
            "dataset": dataset_name,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": columns_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset columns: {str(e)}")

@app.options("/feature-correlations")
@app.get("/feature-correlations")
async def get_feature_correlations():
    """Get feature correlation matrix for visualization - OPTIMIZED with caching"""
    global correlations_cache
    
    # Return cached correlations if available (valid for 2 hours)
    if correlations_cache and correlations_cache.get('timestamp'):
        from datetime import timedelta
        cache_time = datetime.fromisoformat(correlations_cache['timestamp'])
        if datetime.now() - cache_time < timedelta(hours=2):
            print("[INFO] Returning cached correlations")
            return correlations_cache['data']
    
    try:
        # Load a sample of the dataset for correlation analysis
        koi_path = os.path.join(DATA_DIR, "koi.csv")
        if not os.path.exists(koi_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Load a smaller sample for faster computation
        df = pd.read_csv(koi_path, comment='#')
        
        # Get the relevant features used by the model
        model = get_model()
        feature_names = get_feature_names(model)
        
        # Filter to only features that exist in the dataset
        available_features = [f for f in feature_names if f in df.columns]
        
        # Take a smaller sample for performance (1000 samples max)
        sample_size = min(1000, len(df))
        df_sample = df[available_features].sample(n=sample_size, random_state=42)
        
        # Fill NaN values with 0 to prevent correlation calculation errors
        df_sample = df_sample.fillna(0)
        
        print(f"[INFO] Computing correlations on {sample_size} samples")
        
        # Calculate correlation matrix
        correlation_matrix = df_sample.corr()
        
        # Replace NaN values with 0 for JSON serialization
        correlation_matrix = correlation_matrix.fillna(0)
        
        # Convert to format suitable for frontend
        correlations = {
            "features": list(correlation_matrix.columns),
            "matrix": correlation_matrix.values.tolist(),
            "sample_size": sample_size,
            "total_features": len(available_features)
        }
        
        # Cache the result
        correlations_cache = {
            'data': correlations,
            'timestamp': datetime.now().isoformat()
        }
        
        return correlations
        
    except Exception as e:
        print(f"[ERROR] Correlation calculation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to calculate correlations: {str(e)}")

@app.options("/algorithms")
@app.get("/algorithms")
async def get_available_algorithms():
    """Check which algorithms are available for training"""
    algorithms = {
        "gradient_boosting": True,  # Always available
        "random_forest": True,      # Always available
        "xgboost": False,
        "lightgbm": False
    }
    
    # Check XGBoost availability
    try:
        import xgboost as xgb
        algorithms["xgboost"] = True
    except ImportError:
        pass
    
    # Check LightGBM availability
    try:
        import lightgbm as lgb
        algorithms["lightgbm"] = True
    except ImportError:
        pass
    
    return {
        "algorithms": algorithms,
        "available_count": sum(algorithms.values()),
        "total_count": len(algorithms)
    }

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
            full_path in ["features", "metrics", "train", "datasets", "models", "random-example", "predict", "batch-predict", "predict-raw", "feature-correlations", "algorithms"] or
            full_path.startswith("datasets/") or
            full_path.startswith("random-example/") or
            full_path.startswith("models/")):
            raise HTTPException(status_code=404, detail="Not found")
        
        # Serve React app for all other routes
        if os.path.exists("static/index.html"):
            return FileResponse("static/index.html")
        else:
            raise HTTPException(status_code=404, detail="Frontend not built")

# Vercel compatibility
handler = app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
