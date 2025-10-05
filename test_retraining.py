"""
Test script to verify the retraining functionality
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import functions from the main app
from enhanced_exoplanet_classifier import (
    train_new_model, 
    load_models_metadata, 
    get_all_relevant_features,
    load_model,
    get_model_path
)

def test_retraining():
    """Test the retraining pipeline"""
    print("=" * 60)
    print("Testing Exoplanet Classifier Retraining Pipeline")
    print("=" * 60)
    
    # Load KOI data
    print("\n1. Loading KOI dataset...")
    df = pd.read_csv("koi.csv", comment='#')
    print(f"   ✓ Loaded {len(df)} samples")
    
    # Prepare target
    print("\n2. Preparing target variable...")
    label_map = {'CONFIRMED': 2, 'CANDIDATE': 1, 'FALSE POSITIVE': 0}
    df['target'] = df['koi_disposition'].map(label_map)
    df = df[df['target'].notna()]
    print(f"   ✓ Valid samples: {len(df)}")
    
    # Get relevant features
    print("\n3. Selecting relevant features...")
    all_relevant = get_all_relevant_features()
    available_features = [f for f in all_relevant if f in df.columns]
    print(f"   ✓ Available features: {len(available_features)}/{len(all_relevant)}")
    print(f"   Features: {', '.join(available_features[:10])}...")
    
    # Prepare data
    print("\n4. Preparing training data...")
    X = df[available_features].copy()
    
    # Fill NaN values with median for each column
    for col in X.columns:
        if X[col].isna().any():
            median_val = X[col].median()
            if pd.isna(median_val):  # If median is also NaN, use 0
                X[col].fillna(0, inplace=True)
            else:
                X[col].fillna(median_val, inplace=True)
    
    # Double check for any remaining NaNs
    if X.isna().any().any():
        print(f"   Warning: Still have NaN values, filling with 0")
        X.fillna(0, inplace=True)
    
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   ✓ Train samples: {len(X_train)}")
    print(f"   ✓ Test samples: {len(X_test)}")
    
    # Train model
    print("\n5. Training model (this may take a few minutes)...")
    try:
        model_id, metadata = train_new_model(
            X_train, y_train, X_test, y_test,
            model_name="Test_Model",
            description="Test model for verification"
        )
        
        if model_id:
            print(f"   ✓ Model trained successfully!")
            print(f"   Model ID: {model_id}")
            print(f"   Test Accuracy: {metadata['test_accuracy']:.2%}")
            print(f"   Precision: {metadata['precision']:.2%}")
            print(f"   Recall: {metadata['recall']:.2%}")
            print(f"   F1 Score: {metadata['f1_score']:.2%}")
        else:
            print("   ✗ Model training failed!")
            return False
    except Exception as e:
        print(f"   ✗ Model training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify model was saved
    print("\n6. Verifying model was saved...")
    model_path = get_model_path(model_id)
    if os.path.exists(model_path):
        print(f"   ✓ Model file exists: {model_path}")
    else:
        print(f"   ✗ Model file not found!")
        return False
    
    # Load metadata
    print("\n7. Verifying metadata was saved...")
    metadata_list = load_models_metadata()
    if metadata_list and any(m['id'] == model_id for m in metadata_list):
        print(f"   ✓ Metadata saved successfully")
        print(f"   Total models: {len(metadata_list)}")
    else:
        print(f"   ✗ Metadata not found!")
        return False
    
    # Load and test model
    print("\n8. Loading and testing model...")
    loaded_model = load_model(model_id)
    if loaded_model:
        print(f"   ✓ Model loaded successfully")
        
        # Make a test prediction
        test_sample = X_test.iloc[:1]
        prediction = loaded_model.predict(test_sample)
        print(f"   ✓ Test prediction: {prediction[0]}")
    else:
        print(f"   ✗ Failed to load model!")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Retraining pipeline is working correctly.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    try:
        success = test_retraining()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
