"""
Model builder script that trains and saves the model during deployment.
This avoids compatibility issues with pre-trained model files.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
import joblib
import os

def train_model():
    """Train the exoplanet classification model."""
    
    # Load the KOI dataset
    df = pd.read_csv("koi.csv", comment='#')
    
    # Remove identifier and textual columns that have no predictive value
    drop_cols = [
        "rowid", "kepid", "kepoi_name", "kepler_name", "koi_pdisposition", "koi_score"
    ]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=col)
    
    # Define the label and map dispositions to integers
    label_map = {"CONFIRMED": 2, "CANDIDATE": 1, "FALSE POSITIVE": 0}
    df = df[df["koi_disposition"].isin(label_map)]
    df["label"] = df["koi_disposition"].map(label_map)
    df = df.drop(columns=["koi_disposition"])
    
    # Identify numeric features
    numeric_features = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    numeric_features.remove("label")
    
    # Build preprocessing pipeline
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)]
    )
    
    # Assemble full pipeline with classifier
    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                random_state=42,
                n_jobs=-1,
            )),
        ]
    )
    
    # Train/test split
    X = df[numeric_features]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Fit the model
    clf.fit(X_train, y_train)
    
    # Save the trained model
    joblib.dump(clf, "model.joblib")
    
    # Print performance stats (optional, for debugging)
    y_pred = clf.predict(X_test)
    print(f"Model trained successfully with {len(numeric_features)} features")
    
    return clf

def get_or_create_model():
    """Get existing model or create new one if needed."""
    if os.path.exists("model.joblib"):
        try:
            model = joblib.load("model.joblib")
            print("Loaded existing model")
            return model
        except (AttributeError, ImportError, ValueError):
            print("Existing model incompatible, training new model...")
            return train_model()
    else:
        print("No existing model found, training new model...")
        return train_model()

if __name__ == "__main__":
    model = get_or_create_model()
