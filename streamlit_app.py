import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Exoplanet Classifier", layout="centered")
st.title("Exoplanet Classification Tool")

st.markdown(
    "Enter the orbital and transit parameters below. The model will predict whether the object is a confirmed planet, a candidate, or a false positive."
)

# Load the trained model (rebuild if needed for compatibility)
with st.spinner("Loading exoplanet classification model..."):
    try:
        from model_builder import get_or_create_model
        model = get_or_create_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Get actual feature names from the model
preprocessor = model.named_steps['preprocess']
actual_features = preprocessor.transformers_[0][2]

# Define key features to show to the user with friendly labels
key_features = [
    ("koi_fpflag_nt", "Not Transit-Like Flag", 0),
    ("koi_fpflag_ss", "Stellar Eclipse Flag", 0),
    ("koi_fpflag_co", "Centroid Offset Flag", 0),
    ("koi_fpflag_ec", "Ephemeris Match Flag", 0),
    ("koi_period", "Orbital Period (days)", 0.0),
    ("koi_period_err1", "Period Error (+) (days)", 0.0),
    ("koi_period_err2", "Period Error (-) (days)", 0.0),
    ("koi_time0bk", "Transit Epoch (BK JD)", 0.0),
    ("koi_time0bk_err1", "Epoch Error (+) (BK JD)", 0.0),
    ("koi_time0bk_err2", "Epoch Error (-) (BK JD)", 0.0),
    ("koi_impact", "Impact Parameter", 0.0),
    ("koi_impact_err1", "Impact Parameter Error (+)", 0.0),
    ("koi_impact_err2", "Impact Parameter Error (-)", 0.0),
    ("koi_duration", "Transit Duration (hours)", 0.0),
    ("koi_duration_err1", "Duration Error (+) (hours)", 0.0),
    ("koi_duration_err2", "Duration Error (-) (hours)", 0.0),
    ("koi_depth", "Transit Depth (ppm)", 0.0),
    ("koi_depth_err1", "Depth Error (+) (ppm)", 0.0),
    ("koi_depth_err2", "Depth Error (-) (ppm)", 0.0),
    ("koi_ror", "Radius Ratio (planet/star)", 0.0),
    ("koi_ror_err1", "Radius Ratio Error (+)", 0.0),
    ("koi_ror_err2", "Radius Ratio Error (-)", 0.0),
    ("koi_model_snr", "Signal-to-Noise Ratio", 0.0),
    ("koi_count", "Number of Transits", 0),
    ("koi_num_transits", "Total Transits Observed", 0),
]

# Create input fields for key features
user_inputs = {}
st.subheader("Kepler Object Properties")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**False Positive Flags**")
    for feature, label, default_val in key_features[0:4]:
        user_inputs[feature] = st.number_input(
            label=label,
            value=int(default_val),
            min_value=0,
            max_value=1,
            format="%d",
        )

with col2:
    st.markdown("**Detection Quality**")
    for feature, label, default_val in key_features[4:12]:
        if feature in actual_features:
            user_inputs[feature] = st.number_input(
                label=label,
                value=float(default_val),
                format="%.6f",
            )

st.subheader("Orbital and Transit Parameters")
col3, col4 = st.columns(2)

with col3:
    for feature, label, default_val in key_features[12:20]:
        if feature in actual_features:
            user_inputs[feature] = st.number_input(
                label=label,
                value=float(default_val),
                format="%.6f",
            )

with col4:
    for feature, label, default_val in key_features[20:]:
        if feature in actual_features:
            user_inputs[feature] = st.number_input(
                label=label,
                value=int(default_val) if feature.startswith("koi_count") or feature.startswith("koi_num_transits") else float(default_val),
                format="%d" if feature.startswith("koi_count") or feature.startswith("koi_num_transits") else "%.6f",
            )

# Fill all features with user inputs or defaults
feature_values = []
for feature in actual_features:
    if feature in user_inputs:
        feature_values.append(user_inputs[feature])
    else:
        # Default values for unspecified features
        feature_values.append(0.0)

if st.button("Predict Exoplanet Classification"):
    # Convert to pandas DataFrame with proper column names for prediction
    X_user = pd.DataFrame(np.array(feature_values).reshape(1, -1), columns=actual_features)
    
    try:
        # Predict class and probabilities
        pred_label = model.predict(X_user)[0]
        pred_proba = model.predict_proba(X_user)[0]
        
        # Map numeric labels back to strings
        inv_label_map = {2: "Confirmed Planet", 1: "Candidate", 0: "False Positive"}
        
        st.success(f"**Prediction: {inv_label_map[pred_label]}**")
        
        st.subheader("Prediction Confidence")
        # Create a nice visualization of probabilities
        prob_data = {
            "Confirmed Planet": pred_proba[2],
            "Candidate": pred_proba[1], 
            "False Positive": pred_proba[0]
        }
        
        # Display probabilities with bars
        for category, prob in prob_data.items():
            st.write(f"{category}: {prob:.3f}")
            st.progress(prob)
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.write("Please ensure all input fields are filled with valid numbers.")

# Add some information about the model
st.markdown("---")
st.markdown("**Model Information:**")
st.write("- **Algorithm:** Random Forest Classifier")
st.write("- **Training Data:** Kepler Object of Interest (KOI) catalog")
st.write("- **Accuracy:** ~93% on test dataset")
st.write("- **Classes:** Confirmed Planet, Candidate, False Positive")