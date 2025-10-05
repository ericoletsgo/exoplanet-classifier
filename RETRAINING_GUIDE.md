# Exoplanet Classifier - Model Retraining Guide

## Overview

The enhanced exoplanet classifier now includes a comprehensive model retraining and versioning system. This allows you to:

1. **Train new models** with custom datasets
2. **Compare model performance** across different versions
3. **Select specific models** for predictions
4. **Manage model versions** with full metadata tracking

## Features

### 1. Model Retraining

Navigate to **Model Retraining** → **Train New Model** tab.

#### Steps to Train a New Model:

1. **Configure Model Settings:**
   - Enter a model name (e.g., "Production_Model_v2")
   - Set test set size (default: 20%)
   - Add an optional description

2. **Upload Training Data:**
   - Upload a CSV file with exoplanet data
   - The system will automatically filter for relevant features

3. **Configure Target Column:**
   - **Select Target Column:** Choose which column contains your classification labels
   - **Map Values:** Assign values to three categories:
     - 🌍 **Confirmed Planet** - Verified exoplanets
     - 🔍 **Candidate** - Potential exoplanets
     - ❌ **False Positive** - Non-planetary signals
   - The system auto-detects common values (e.g., "CONFIRMED", "CANDIDATE", "FALSE POSITIVE")
   - You can map any custom values to these three categories

4. **Feature Selection:**
   The system uses only relevant features based on three categories:
   
   - **Stellar Parameters:** Position (RA, Dec), magnitude, temperature, radius, mass, surface gravity
   - **Exoplanet Parameters:** Orbital period, duration, depth, radius, impact parameter, eccentricity
   - **Signal Quality:** SNR, transit count, and other light curve/radial velocity metrics

5. **Train:**
   - Click "🚀 Train Model"
   - Training uses an ensemble of algorithms (Gradient Boosting, Random Forest, XGBoost, LightGBM)
   - View real-time metrics: accuracy, precision, recall, F1 score
   - See confusion matrix visualization

### 2. Model Evaluations

Navigate to **Model Retraining** → **Model Evaluations** tab.

#### Features:

- **Comparison Table:** View all models side-by-side with key metrics
- **Detailed View:** Select any model to see:
  - Full metadata (creation date, description, algorithms used)
  - Performance metrics
  - Confusion matrix
  - List of features used

### 3. Model Selection for Predictions

On the **Classification** page:

1. Use the **Model Selection** dropdown in the sidebar
2. Choose from:
   - Default models (if available)
   - Any trained versioned models (with accuracy shown)
3. Selected model info is displayed in the sidebar

### 4. Model Management

Navigate to **Model Retraining** → **Model Management** tab.

#### Features:

- View all saved models
- See quick stats for each model
- Delete models you no longer need

## Model Versioning System

### Storage Structure:

```
models/
├── models_metadata.json          # Metadata for all models
├── model_20251004_193916.joblib  # Model files
├── model_20251004_194523.joblib
└── ...
```

### Metadata Tracked:

- Model ID (timestamp-based)
- Model name and description
- Creation timestamp
- Training/test sample counts
- Performance metrics (accuracy, precision, recall, F1)
- Confusion matrix
- Feature list
- Algorithms used

## Example: Training a New Model

### Example 1: Standard KOI Dataset

```
1. Go to "Model Retraining" page
2. Click "Train New Model" tab
3. Enter name: "High_Precision_Model"
4. Upload: koi.csv
5. Select target column: "koi_disposition"
6. Confirm mapping:
   - CONFIRMED → Confirmed Planet ✓
   - CANDIDATE → Candidate ✓
   - FALSE POSITIVE → False Positive ✓
7. Review: 23 relevant features detected
8. Click "🚀 Train Model"
9. Wait 2-3 minutes
10. View results: ~82% accuracy
11. Go to "Classification" page
12. Select your new model from dropdown
13. Make predictions!
```

### Example 2: Custom Target Column

If your CSV has a different column name or values:

```
Your CSV has column "status" with values: "planet", "maybe", "noise"

1. Upload your CSV
2. Select target column: "status"
3. Map values:
   - "planet" → Confirmed Planet
   - "maybe" → Candidate
   - "noise" → False Positive
4. Continue with training
```

### Example 3: Multiple Values Per Category

```
Your CSV has "classification" with: "confirmed", "validated", "candidate", 
"false_positive", "not_transit"

1. Select target column: "classification"
2. Map values:
   - Confirmed Planet: ["confirmed", "validated"]
   - Candidate: ["candidate"]
   - False Positive: ["false_positive", "not_transit"]
3. System shows: "Unmapped values: none"
4. Continue with training
```

## Relevant Features Used

The system automatically selects these feature categories:

### Stellar Parameters (7 features):
- `koi_steff` - Stellar effective temperature
- `koi_srad` - Stellar radius
- `koi_smass` - Stellar mass
- `koi_slogg` - Stellar surface gravity
- `ra` - Right ascension
- `dec` - Declination
- `koi_kepmag` - Kepler magnitude

### Orbital Parameters (10 features):
- `koi_period` - Orbital period
- `koi_duration` - Transit duration
- `koi_depth` - Transit depth
- `koi_prad` - Planetary radius
- `koi_impact` - Impact parameter
- `koi_eccen` - Orbital eccentricity
- `koi_longp` - Longitude of periastron
- `koi_time0` - Transit epoch
- `koi_ror` - Planet-star radius ratio
- `koi_srho` - Fitted stellar density

### Signal Quality (6 features):
- `koi_model_snr` - Transit signal-to-noise ratio
- `koi_count` - Number of transits
- `koi_num_transits` - Number of transits observed
- `koi_max_mult_ev` - Maximum multiple event statistic
- `koi_max_sngle_ev` - Maximum single event statistic
- `koi_tce_plnt_num` - TCE planet number

## Best Practices

1. **Use Balanced Datasets:** Ensure your training data has reasonable representation of all three classes
2. **Monitor Metrics:** Don't just look at accuracy - check precision, recall, and F1 score
3. **Test Before Production:** Train and evaluate models before using them for critical predictions
4. **Keep Descriptions:** Add meaningful descriptions to track what makes each model unique
5. **Regular Retraining:** Retrain models as new data becomes available

## Troubleshooting

### Issue: "Less than 5 relevant features found"
**Solution:** Ensure your CSV contains the standard Kepler/TESS column names

### Issue: Training fails with NaN errors
**Solution:** The system automatically handles NaN values, but ensure your data quality is good

### Issue: Low accuracy (<70%)
**Solution:** Check class balance, ensure sufficient training samples, verify feature quality

## API Reference

For programmatic access, see `test_retraining.py` for examples of:
- `train_new_model(X_train, y_train, X_test, y_test, model_name, description)`
- `load_model(model_id)`
- `load_models_metadata()`
- `get_all_relevant_features()`
