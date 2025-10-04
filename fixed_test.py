import joblib
import pandas as pd
import numpy as np

# Test loading the model and making a prediction
try:
    model = joblib.load("model.joblib")
    print("Model loaded successfully")
    
    # Get the feature names from the original training
    preprocessor = model.named_steps['preprocess']
    feature_names = preprocessor.transformers_[0][2]
    
    # Create test data as a DataFrame with proper column names
    test_data = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
    
    # Make prediction
    pred = model.predict(test_data)[0]
    proba = model.predict_proba(test_data)[0]
    
    inv_label_map = {2: "Confirmed Planet", 1: "Candidate", 0: "False Positive"}
    
    print(f"Test prediction: {inv_label_map[pred]}")
    print(f"Confidence scores:")
    print(f"   Confirmed Planet: {proba[2]:.3f}")
    print(f"   Candidate: {proba[1]:.3f}")
    print(f"   False Positive: {proba[0]:.3f}")
    
    print(f"\nModel expects {len(feature_names)} features")
    print("First 10 features:", feature_names[:10])
    
    print("\nModel test completed successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
