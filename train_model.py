import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
import joblib

# 1. Load the KOI dataset
# The CSV file has comment lines starting with # that we need to skip
df = pd.read_csv("koi.csv", comment='#')

# 2. Remove identifier and textual columns that have no predictive value
drop_cols = [
    "rowid", "kepid", "kepoi_name", "kepler_name", "koi_pdisposition", "koi_score"
]
for col in drop_cols:
    if col in df.columns:
        df = df.drop(columns=col)

# 3. Define the label and map dispositions to integers
# Possible values: CONFIRMED, CANDIDATE, FALSE POSITIVE
label_map = {"CONFIRMED": 2, "CANDIDATE": 1, "FALSE POSITIVE": 0}
df = df[df["koi_disposition"].isin(label_map)]  # keep only rows with valid labels
df["label"] = df["koi_disposition"].map(label_map)
df = df.drop(columns=["koi_disposition"])

# 4. Identify numeric and categorical features
numeric_features = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
numeric_features.remove("label")

# 5. Build preprocessing pipeline
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[("num", numeric_transformer, numeric_features)]
)

# 6. Assemble full pipeline with classifier
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

# 7. Train/test split
X = df[numeric_features]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 8. Fit the model
clf.fit(X_train, y_train)

# 9. Evaluate on the hold‑out set
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# 10. Save the trained model to disk
joblib.dump(clf, "model.joblib")
print("Model saved to model.joblib")
