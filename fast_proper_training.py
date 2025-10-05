#!/usr/bin/env python3
"""
FAST Proper ML Training - Quick Results with Real Learning
Removes data leakage and creates a proper model in under 2 minutes
"""
import pandas as pd
import numpy as np
import joblib
import gc
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data():
    """Load data and remove ALL data leakage - FAST"""
    print("=" * 60)
    print("STEP 1: Loading and Cleaning Data")
    print("=" * 60)
    
    print("Loading KOI dataset...")
    df = pd.read_csv("data/koi.csv", comment='#')
    print(f"[OK] Data loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Clean target
    df['target'] = df['koi_disposition'].map({'CONFIRMED': 2, 'CANDIDATE': 1, 'FALSE POSITIVE': 0})
    df = df[df['target'].notna()]
    print(f"[OK] Target cleaned: {len(df)} samples")
    
    # Get numeric features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features.remove('target')
    print(f"[OK] Found {len(numeric_features)} numeric features")
    
    # REMOVE ALL DATA LEAKAGE - FAST VERSION
    data_leakage_features = ['koi_score', 'rowid', 'kepid', 'koi_pdisposition']
    non_physics_features = ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'koi_count', 'koi_num_transits']
    
    # Remove high missing features (>50% missing)
    high_missing = df[numeric_features].isnull().mean() > 0.5
    high_missing_features = high_missing[high_missing].index.tolist()
    
    features_to_remove = set(data_leakage_features + non_physics_features + high_missing_features)
    clean_features = [f for f in numeric_features if f not in features_to_remove]
    
    print(f"[OK] Removed {len(features_to_remove)} problematic features")
    print(f"[OK] Clean features remaining: {len(clean_features)}")
    
    # Prepare data
    X = df[clean_features].fillna(0)
    y = df['target']
    
    del df
    gc.collect()
    
    print(f"[OK] Final dataset: {X.shape}")
    print(f"[OK] Class distribution: {dict(y.value_counts())}")
    
    return X, y, clean_features

def create_fast_ensemble(X, y, feature_names):
    """Create ensemble quickly with good parameters"""
    print("\n" + "=" * 60)
    print("STEP 2: Creating Fast Ensemble")
    print("=" * 60)
    
    # Use fewer features for speed (top 30 most important)
    print("Selecting top 30 most important features...")
    feature_selector = SelectKBest(score_func=mutual_info_classif, k=min(30, len(feature_names)))
    
    # Create preprocessing pipeline
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    
    # Create models with good but fast parameters
    print("Creating models...")
    models = [
        ('gradient_boosting', GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )),
        ('random_forest', RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42
        ))
    ]
    
    # Try to add XGBoost if available
    try:
        import xgboost as xgb
        models.append(('xgboost', xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )))
        print("[OK] Added XGBoost")
    except ImportError:
        print("[INFO] XGBoost not available")
    
    # Try to add LightGBM if available
    try:
        import lightgbm as lgb
        models.append(('lightgbm', lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            verbose=-1
        )))
        print("[OK] Added LightGBM")
    except ImportError:
        print("[INFO] LightGBM not available")
    
    # Create ensemble pipeline
    print(f"[OK] Creating ensemble with {len(models)} models...")
    ensemble = Pipeline([
        ("preprocess", preprocessor),
        ("feature_selection", feature_selector),
        ("ensemble", VotingClassifier(models, voting='soft'))
    ])
    
    # Train ensemble
    print("Training ensemble...")
    ensemble.fit(X, y)
    
    # Evaluate with cross-validation
    print("Evaluating ensemble...")
    cv_scores = cross_val_score(ensemble, X, y, cv=3, scoring='accuracy')
    print(f"[OK] Ensemble accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    
    # Store metadata
    ensemble.cv_accuracy = cv_scores.mean()
    ensemble.cv_std = cv_scores.std()
    ensemble.feature_names = feature_names
    ensemble.n_features_selected = min(30, len(feature_names))
    ensemble.models_used = len(models)
    ensemble.dataset_summary = {
        'total_samples': len(X),
        'total_features': len(feature_names),
        'features_selected': min(30, len(feature_names)),
        'removed_data_leakage': True,
        'removed_high_missing': True,
        'removed_non_physics': True
    }
    
    return ensemble

def evaluate_model(model, X, y):
    """Quick model evaluation"""
    print("\n" + "=" * 60)
    print("STEP 3: Model Evaluation")
    print("=" * 60)
    
    # Get predictions
    print("Generating predictions...")
    y_pred = model.predict(X)
    
    print("\nClassification Report:")
    print(classification_report(y, y_pred, 
                              target_names=['False Positive', 'Candidate', 'Confirmed Planet']))
    
    # Feature importance
    try:
        feature_selector = model.named_steps['feature_selection']
        selected_features = X.columns[feature_selector.get_support()].tolist()
        importances = feature_selector.scores_[feature_selector.get_support()]
        
        importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['Feature']:<25} {row['Importance']:.3f}")
        
    except:
        print("[INFO] Feature importance not available")

def main():
    """Main training pipeline - FAST VERSION"""
    print("=" * 60)
    print("FAST PROPER MACHINE LEARNING TRAINING")
    print("Real learning from orbital physics - Under 2 minutes!")
    print("=" * 60)
    
    try:
        # Load and clean data
        X, y, feature_names = load_and_clean_data()
        
        # PROPER TRAIN/TEST SPLIT - Prevent evaluation data leakage
        print("\n" + "=" * 60)
        print("Creating Train/Test Split (80/20)")
        print("=" * 60)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"[OK] Training set: {len(X_train)} samples")
        print(f"[OK] Test set: {len(X_test)} samples (held out for evaluation)")
        
        # Create fast ensemble - TRAIN ONLY ON TRAINING DATA
        optimized_model = create_fast_ensemble(X_train, y_train, feature_names)
        
        # Evaluate model ON TEST SET ONLY
        print("\n" + "=" * 60)
        print("STEP 3: Model Evaluation on HELD-OUT Test Set")
        print("=" * 60)
        print("Generating predictions on unseen test data...")
        y_pred = optimized_model.predict(X_test)
        
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, y_pred, 
                                  target_names=['False Positive', 'Candidate', 'Confirmed Planet']))
        
        # Calculate test accuracy
        from sklearn.metrics import accuracy_score
        test_accuracy = accuracy_score(y_test, y_pred)
        optimized_model.test_accuracy = test_accuracy
        
        # Save model AND test set
        print(f"\nSaving model and test set...")
        joblib.dump(optimized_model, 'properly_trained_model.joblib')
        joblib.dump({'X_test': X_test, 'y_test': y_test}, 'data/test_set.joblib')
        
        print(f"\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        print(f"Cross-Validation Accuracy (Training): {optimized_model.cv_accuracy:.3f} +/- {optimized_model.cv_std:.3f}")
        print(f"Test Set Accuracy (Unseen Data): {test_accuracy:.3f}")
        print(f"Features Used: {optimized_model.n_features_selected}/{len(feature_names)}")
        print(f"Models in Ensemble: {optimized_model.models_used}")
        print(f"Data Leakage Removed: {optimized_model.dataset_summary['removed_data_leakage']}")
        print(f"\nModel saved to 'properly_trained_model.joblib'")
        print(f"Test set saved to 'data/test_set.joblib'")
        print("\nWARNING: The test accuracy is the TRUE performance on unseen data!")
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
