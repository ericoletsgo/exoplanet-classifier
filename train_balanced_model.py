#!/usr/bin/env python3
"""
Train a balanced exoplanet classification model
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from datetime import datetime

# Features to use (same as in the API)
RELEVANT_FEATURES = [
    # Signal Quality (most important)
    'koi_dikco_msky', 'koi_dicco_msky', 'koi_max_mult_ev', 'koi_model_snr', 'koi_dikco_mra',
    # Flux Centroid
    'koi_fwm_srao', 'koi_fwm_sdeco', 'koi_fwm_sra_err', 'koi_fwm_sdec_err', 'koi_fwm_srao_err',
    # Orbital Parameters
    'koi_period', 'koi_depth', 'koi_duration', 'koi_prad', 'koi_impact',
    # Stellar Parameters
    'koi_steff', 'koi_srad', 'koi_slogg', 'koi_kepmag',
    # Error Parameters
    'koi_period_err1', 'koi_duration_err1', 'koi_depth_err1'
]

def load_and_prepare_data():
    """Load and prepare the KOI dataset"""
    print("Loading KOI dataset...")
    
    # Load the dataset
    df = pd.read_csv('data/koi.csv', comment='#')
    print(f"Loaded {len(df)} total records")
    
    # Create target variable
    df['target'] = df['koi_disposition'].map({
        'CONFIRMED': 2, 
        'CANDIDATE': 1, 
        'FALSE POSITIVE': 0
    })
    
    # Remove rows with missing target
    df = df[df['target'].notna()]
    print(f"Records with valid disposition: {len(df)}")
    
    # Print class distribution
    class_counts = df['target'].value_counts().sort_index()
    print("\nClass distribution:")
    for target, count in class_counts.items():
        label = {0: 'FALSE POSITIVE', 1: 'CANDIDATE', 2: 'CONFIRMED'}[target]
        print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
    
    # Get features that exist in the dataset
    available_features = [f for f in RELEVANT_FEATURES if f in df.columns]
    missing_features = [f for f in RELEVANT_FEATURES if f not in df.columns]
    
    print(f"\nUsing {len(available_features)} of {len(RELEVANT_FEATURES)} features")
    if missing_features:
        print(f"Missing features: {missing_features}")
    
    # Extract features and target
    X = df[available_features]
    y = df['target']
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    return X, y, available_features

def train_model(X, y, feature_names):
    """Train a balanced Random Forest model"""
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create a pipeline with preprocessing and model
    print("\nCreating model pipeline...")
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
        ('scaler', StandardScaler()),  # Normalize features
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',  # Handle class imbalance
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Train the model
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate on test set
    print("\nEvaluating model...")
    y_pred = pipeline.predict(X_test)
    
    # Print classification report
    target_names = ['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED']
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print("Predicted ->")
    print("Actual ↓   FP   CAND  CONF")
    for i, (actual_label, row) in enumerate(zip(target_names, cm)):
        print(f"{actual_label[:4]:>6} {row[0]:>4} {row[1]:>5} {row[2]:>5}")
    
    # Cross-validation score
    print("\nCross-validation scores:")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1_weighted')
    print(f"CV F1-weighted: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Store feature names in the pipeline for later use
    pipeline.feature_names = feature_names
    
    return pipeline, X_test, y_test

def save_model(pipeline, X_test, y_test):
    """Save the trained model and test set"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save the model
    model_path = f"balanced_model_{timestamp}.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save test set for evaluation
    test_set = {'X_test': X_test, 'y_test': y_test}
    test_path = "data/test_set.joblib"
    joblib.dump(test_set, test_path)
    print(f"Test set saved to: {test_path}")
    
    return model_path

def main():
    """Main training function"""
    print("=" * 60)
    print("TRAINING BALANCED EXOPLANET CLASSIFICATION MODEL")
    print("=" * 60)
    
    try:
        # Load and prepare data
        X, y, feature_names = load_and_prepare_data()
        
        # Train model
        pipeline, X_test, y_test = train_model(X, y, feature_names)
        
        # Save model
        model_path = save_model(pipeline, X_test, y_test)
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"New model saved as: {model_path}")
        print("\nTo use this model, update the MODEL_PATH in api/main.py")
        print("Then restart the API server.")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
