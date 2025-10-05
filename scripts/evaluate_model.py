#!/usr/bin/env python3
"""
Model Evaluation Script
Test the actual performance of the trained model
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the same data used for training"""
    print("Loading KOI dataset...")
    df = pd.read_csv("koi.csv", comment='#')
    
    # Clean target (same as training)
    df['target'] = df['koi_disposition'].map({'CONFIRMED': 2, 'CANDIDATE': 1, 'FALSE POSITIVE': 0})
    df = df[df['target'].notna()]
    
    # Get numeric features (same as training)
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features.remove('target')
    
    # Remove data leakage features (same as training)
    data_leakage_features = ['koi_score', 'rowid', 'kepid', 'koi_pdisposition']
    non_physics_features = ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'koi_count', 'koi_num_transits']
    high_missing = df[numeric_features].isnull().mean() > 0.5
    high_missing_features = high_missing[high_missing].index.tolist()
    
    features_to_remove = set(data_leakage_features + non_physics_features + high_missing_features)
    clean_features = [f for f in numeric_features if f not in features_to_remove]
    
    X = df[clean_features].fillna(0)
    y = df['target']
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {dict(y.value_counts())}")
    
    return X, y, clean_features

def evaluate_model():
    """Evaluate the trained model"""
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    try:
        # Load the trained model
        print("Loading trained model...")
        model = joblib.load('properly_trained_model.joblib')
        print(f"[OK] Model loaded: {type(model)}")
        
        # Load data
        X, y, feature_names = load_data()
        
        # Check model metadata
        print(f"\nModel Metadata:")
        print(f"  - CV Accuracy: {model.cv_accuracy:.3f}")
        print(f"  - CV Std: {model.cv_std:.3f}")
        print(f"  - Features Selected: {model.n_features_selected}")
        print(f"  - Models Used: {model.models_used}")
        
        # Test on full dataset (same as training)
        print(f"\nTesting on full dataset ({len(X)} samples)...")
        y_pred = model.predict(X)
        full_accuracy = accuracy_score(y, y_pred)
        print(f"[OK] Full dataset accuracy: {full_accuracy:.3f}")
        
        # Cross-validation evaluation
        print(f"\nCross-validation evaluation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        print(f"[OK] CV Accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
        print(f"CV Scores: {[f'{score:.3f}' for score in cv_scores]}")
        
        # Train/test split evaluation
        print(f"\nTrain/Test split evaluation...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        # Retrain on train set and test on test set
        model_test = joblib.load('properly_trained_model.joblib')  # Fresh copy
        model_test.fit(X_train, y_train)
        y_test_pred = model_test.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f"[OK] Test set accuracy: {test_accuracy:.3f}")
        
        # Detailed classification report
        print(f"\nDetailed Classification Report (Test Set):")
        print(classification_report(y_test, y_test_pred, 
                                  target_names=['False Positive', 'Candidate', 'Confirmed Planet']))
        
        # Confusion matrix
        print(f"\nConfusion Matrix (Test Set):")
        cm = confusion_matrix(y_test, y_test_pred)
        print(cm)
        
        # Feature importance
        try:
            feature_selector = model.named_steps['feature_selection']
            selected_features = X.columns[feature_selector.get_support()].tolist()
            importances = feature_selector.scores_[feature_selector.get_support()]
            
            importance_df = pd.DataFrame({
                'Feature': selected_features,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            print(f"\nTop 10 Most Important Features:")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
                print(f"{i:2d}. {row['Feature']:<25} {row['Importance']:.3f}")
                
        except:
            print("[INFO] Feature importance not available")
        
        # Summary
        print(f"\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Model Type: {type(model)}")
        print(f"Stored CV Accuracy: {model.cv_accuracy:.3f}")
        print(f"Recomputed CV Accuracy: {cv_scores.mean():.3f}")
        print(f"Test Set Accuracy: {test_accuracy:.3f}")
        print(f"Full Dataset Accuracy: {full_accuracy:.3f}")
        print(f"Features Used: {model.n_features_selected}/{len(feature_names)}")
        print(f"Models in Ensemble: {model.models_used}")
        
        # Verify the stored accuracy matches
        stored_vs_computed = abs(model.cv_accuracy - cv_scores.mean())
        if stored_vs_computed < 0.01:
            print(f"[OK] Stored accuracy matches computed accuracy (diff: {stored_vs_computed:.3f})")
        else:
            print(f"[WARNING] Stored accuracy doesn't match computed accuracy (diff: {stored_vs_computed:.3f})")
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    evaluate_model()
