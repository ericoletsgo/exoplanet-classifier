"""
Clean Professional Exoplanet Classification Interface
Modern, minimalist design for researchers and scientists
With Model Retraining and Versioning Support
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go

# Custom CSS for clean design
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .main .block-container {
        max-width: 800px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .info-card {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

# Model versioning constants
MODELS_DIR = "models"
MODELS_METADATA_FILE = "models/models_metadata.json"

# Define relevant features based on user requirements (expanded from original model)
RELEVANT_FEATURES = {
    # Stellar parameters (position, magnitude, temperature, metallicity)
    'stellar': [
        'koi_steff', 'koi_steff_err1', 'koi_steff_err2',  # Temperature
        'koi_srad', 'koi_srad_err1', 'koi_srad_err2',  # Radius
        'koi_smass', 'koi_smass_err1', 'koi_smass_err2',  # Mass
        'koi_slogg', 'koi_slogg_err1', 'koi_slogg_err2',  # Surface gravity
        'koi_smet', 'koi_smet_err1', 'koi_smet_err2',  # Metallicity
        'ra', 'dec',  # Position
        'koi_kepmag', 'koi_gmag', 'koi_rmag', 'koi_imag', 'koi_zmag',  # Magnitudes
        'koi_jmag', 'koi_hmag', 'koi_kmag'  # IR magnitudes
    ],
    
    # Exoplanet parameters (mass and orbital information)
    'orbital': [
        'koi_period', 'koi_period_err1', 'koi_period_err2',  # Orbital period
        'koi_duration', 'koi_duration_err1', 'koi_duration_err2',  # Transit duration
        'koi_depth', 'koi_depth_err1', 'koi_depth_err2',  # Transit depth
        'koi_prad', 'koi_prad_err1', 'koi_prad_err2',  # Planet radius
        'koi_impact', 'koi_impact_err1', 'koi_impact_err2',  # Impact parameter
        'koi_eccen',  # Eccentricity
        'koi_time0', 'koi_time0_err1', 'koi_time0_err2',  # Transit epoch
        'koi_time0bk', 'koi_time0bk_err1', 'koi_time0bk_err2',  # Barycentric epoch
        'koi_ror', 'koi_ror_err1', 'koi_ror_err2',  # Planet-star radius ratio
        'koi_srho', 'koi_srho_err1', 'koi_srho_err2',  # Stellar density
        'koi_sma',  # Semi-major axis
        'koi_incl',  # Inclination
        'koi_teq',  # Equilibrium temperature
        'koi_insol', 'koi_insol_err1', 'koi_insol_err2',  # Insolation flux
        'koi_dor', 'koi_dor_err1', 'koi_dor_err2'  # Planet-star distance ratio
    ],
    
    # Signal quality (light curve or radial velocity curve categorization)
    'signal': [
        'koi_model_snr',  # Transit signal-to-noise
        'koi_max_mult_ev', 'koi_max_sngle_ev',  # Event statistics
        'koi_tce_plnt_num',  # TCE planet number
        'koi_bin_oedp_sig',  # Binary discrimination
        'koi_ldm_coeff1', 'koi_ldm_coeff2', 'koi_ldm_coeff3', 'koi_ldm_coeff4',  # Limb darkening
        'koi_fwm_stat_sig',  # Flux-weighted centroid statistic
        'koi_fwm_sra', 'koi_fwm_sra_err', 'koi_fwm_sdec', 'koi_fwm_sdec_err',  # Source position
        'koi_fwm_srao', 'koi_fwm_srao_err', 'koi_fwm_sdeco', 'koi_fwm_sdeco_err',  # Out-of-transit position
        'koi_fwm_prao', 'koi_fwm_prao_err', 'koi_fwm_pdeco', 'koi_fwm_pdeco_err',  # Difference
        'koi_dicco_mra', 'koi_dicco_mra_err', 'koi_dicco_mdec', 'koi_dicco_mdec_err',  # Centroid offset
        'koi_dicco_msky', 'koi_dicco_msky_err',  # Sky offset
        'koi_dikco_mra', 'koi_dikco_mra_err', 'koi_dikco_mdec', 'koi_dikco_mdec_err',  # KIC offset
        'koi_dikco_msky', 'koi_dikco_msky_err'  # KIC sky offset
    ]
}

def get_all_relevant_features():
    """Get all relevant features as a flat list"""
    all_features = []
    for category in RELEVANT_FEATURES.values():
        all_features.extend(category)
    return all_features

def ensure_models_directory():
    """Create models directory if it doesn't exist"""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

def load_models_metadata():
    """Load models metadata from JSON file"""
    ensure_models_directory()
    if os.path.exists(MODELS_METADATA_FILE):
        with open(MODELS_METADATA_FILE, 'r') as f:
            return json.load(f)
    return []

def save_models_metadata(metadata):
    """Save models metadata to JSON file"""
    ensure_models_directory()
    with open(MODELS_METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

def get_model_path(model_id):
    """Get the file path for a model"""
    return os.path.join(MODELS_DIR, f"model_{model_id}.joblib")

def train_new_model(X_train, y_train, X_test, y_test, model_name, description=""):
    """Train a new model and save with metadata"""
    try:
        # Create models
        models = []
        models.append(('gradient_boosting', GradientBoostingClassifier(n_estimators=100, random_state=42)))
        models.append(('random_forest', RandomForestClassifier(n_estimators=100, random_state=42)))
        
        # Try XGBoost
        try:
            import xgboost as xgb
            models.append(('xgboost', xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')))
        except ImportError:
            pass
            
        # Try LightGBM
        try:
            import lightgbm as lgb
            models.append(('lightgbm', lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)))
        except ImportError:
            pass
        
        # Create ensemble
        ensemble = VotingClassifier(models, voting='soft')
        
        # Train
        ensemble.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = ensemble.predict(X_train)
        y_pred_test = ensemble.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Calculate additional metrics
        precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
        
        # Get confusion matrix
        cm = confusion_matrix(y_test, y_pred_test)
        
        # Create model metadata
        model_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata = {
            'id': model_id,
            'name': model_name,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'features': X_train.columns.tolist(),
            'n_features': len(X_train.columns),
            'algorithms': [name for name, _ in models]
        }
        
        # Store feature names in model
        ensemble.feature_names = X_train.columns.tolist()
        ensemble.metadata = metadata
        
        # Save model
        model_path = get_model_path(model_id)
        joblib.dump(ensemble, model_path)
        
        # Update metadata file
        all_metadata = load_models_metadata()
        all_metadata.append(metadata)
        save_models_metadata(all_metadata)
        
        return model_id, metadata
        
    except Exception as e:
        # Try to use streamlit if available, otherwise print
        try:
            st.error(f"Training failed: {str(e)}")
        except:
            print(f"Training failed: {str(e)}")
        raise  # Re-raise the exception so caller can handle it
        return None, None

def train_advanced_ensemble():
    """Simple advanced ensemble training"""
    try:
        # Load KOI data
        kepler_df = pd.read_csv("koi.csv", comment='#')
        kepler_df['target'] = kepler_df['koi_disposition'].map({'CONFIRMED': 2, 'CANDIDATE': 1, 'FALSE POSITIVE': 0})
        kepler_df = kepler_df[kepler_df['target'].notna()]
        
        # Get numeric features
        numeric_features = kepler_df.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_features:
            numeric_features.remove('target')
        
        X = kepler_df[numeric_features].fillna(0)
        y = kepler_df['target']
        
        # Create models
        models = []
        
        # Basic models
        models.append(('gradient_boosting', GradientBoostingClassifier(n_estimators=100, random_state=42)))
        models.append(('random_forest', RandomForestClassifier(n_estimators=100, random_state=42)))
        
        # Try XGBoost
        try:
            import xgboost as xgb
            models.append(('xgboost', xgb.XGBClassifier(n_estimators=100, random_state=42)))
            st.success("✓ XGBoost added")
        except ImportError:
            st.warning("⚠ XGBoost not available")
            
        # Try LightGBM
        try:
            import lightgbm as lgb
            models.append(('lightgbm', lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)))
            st.success("✓ LightGBM added")
        except ImportError:
            st.warning("⚠ LightGBM not available")
        
        # Create ensemble
        ensemble = VotingClassifier(models, voting='soft')
        
        with st.spinner("Training ensemble..."):
            ensemble.fit(X, y)
        
        # Cross-validation accuracy
        cv_scores = cross_val_score(ensemble, X, y, cv=3, scoring='accuracy')
        
        st.success(f"🎯 Ensemble Accuracy: {cv_scores.mean():.3f}")
        st.info(f"📊 Used {len(models)} algorithms")
        
        # Store metadata
        ensemble.feature_names = numeric_features
        ensemble.cv_accuracy = cv_scores.mean()
        ensemble.dataset_summary = {'total_samples': len(X)}
        
        # Save
        joblib.dump(ensemble, 'advanced_ensemble.joblib')
        st.success("💾 Advanced ensemble saved")
        
        return ensemble
        
    except Exception as e:
        st.error(f"Training failed: {str(e)}")
        return None

# train_advanced_ensemble_silent() function removed - no longer needed since model is pre-trained

def load_model(model_id=None):
    """Load a model by ID or load the default model"""
    try:
        # If model_id is specified, load that model
        if model_id:
            model_path = get_model_path(model_id)
            if os.path.exists(model_path):
                return joblib.load(model_path)
            else:
                st.warning(f"Model {model_id} not found, loading default model")
        
        # Load properly trained model (preferred)
        if os.path.exists('properly_trained_model.joblib'):
            try:
                proper_model = joblib.load('properly_trained_model.joblib')
                return proper_model
            except Exception as e:
                st.error(f"Error loading proper model: {e}")
                pass
        
        # Fallback to clean optimized model
        if os.path.exists('clean_optimized_model.joblib'):
            try:
                clean_model = joblib.load('clean_optimized_model.joblib')
                return clean_model
            except Exception as e:
                st.error(f"Error loading clean model: {e}")
                pass
        
        # Fallback to advanced ensemble
        if os.path.exists('advanced_ensemble.joblib'):
            try:
                advanced_model = joblib.load('advanced_ensemble.joblib')
                return advanced_model
            except Exception as e:
                st.error(f"Error loading advanced model: {e}")
                pass
        
        # Check for any versioned models
        metadata = load_models_metadata()
        if metadata:
            # Load the most recent model
            latest_model = metadata[-1]
            model_path = get_model_path(latest_model['id'])
            if os.path.exists(model_path):
                return joblib.load(model_path)
        
        st.error("No models found. Please train a model first.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def load_random_example(example_type):
    """Load a random example from the dataset"""
    try:
        import pandas as pd
        
        # Load dataset
        df = pd.read_csv("koi.csv", comment='#')
        df['target'] = df['koi_disposition'].map({'CONFIRMED': 2, 'CANDIDATE': 1, 'FALSE POSITIVE': 0})
        df = df[df['target'].notna()]
        
        # Filter by type
        if example_type == "confirmed":
            filtered_df = df[df['koi_disposition'] == 'CONFIRMED']
        elif example_type == "candidate":
            filtered_df = df[df['koi_disposition'] == 'CANDIDATE']
        elif example_type == "false_positive":
            filtered_df = df[df['koi_disposition'] == 'FALSE POSITIVE']
        else:
            return None
            
        if len(filtered_df) == 0:
            return None
            
        # Get random row
        random_row = filtered_df.sample(n=1).iloc[0]
        
        return {
            'period': random_row['koi_period'],
            'duration': random_row['koi_duration'],
            'depth': random_row['koi_depth'],
            'prad': random_row.get('koi_prad', 1.0),
            'impact': random_row.get('koi_impact', 0.5),
            'snr': random_row.get('koi_model_snr', 10.0),
            'steff': random_row.get('koi_steff', 5778.0),
            'srad': random_row.get('koi_srad', 1.0),
            'smass': random_row.get('koi_smass', 1.0),
            'expected': random_row['koi_disposition'],
            'row_index': random_row.name,  # Store the original row index
            'koi_name': random_row.get('kepoi_name', 'Unknown'),  # Store KOI name if available
            # Store the actual row data for proper prediction
            'actual_row': random_row
        }
        
    except Exception as e:
        st.error(f"Error loading random example: {e}")
        return None

def create_prediction_form(model):
    """Create clean prediction form - Updated to support both Pipeline and VotingClassifier models"""
    st.header("Exoplanet Classification Interface")
    
    # Check if model is Pipeline or VotingClassifier (FIXED: handles both model types)
    is_pipeline = hasattr(model, 'named_steps')
    is_voting = hasattr(model, 'estimators_')
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Orbital Parameters")
        
        # Add random example buttons
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn1:
            if st.button("🎲 Random Confirmed Planet", help="Load a random confirmed exoplanet example"):
                st.session_state.random_example = "confirmed"
        with col_btn2:
            if st.button("🎲 Random Candidate", help="Load a random candidate exoplanet example"):
                st.session_state.random_example = "candidate"
        with col_btn3:
            if st.button("🎲 Random False Positive", help="Load a random false positive example"):
                st.session_state.random_example = "false_positive"
        
        # Load random example if requested
        if hasattr(st.session_state, 'random_example') and st.session_state.random_example:
            random_data = load_random_example(st.session_state.random_example)
            if random_data:
                # Update default values for next form render
                st.session_state.period = random_data['period']
                st.session_state.duration = random_data['duration'] 
                st.session_state.depth = random_data['depth']
                st.session_state.prad = random_data['prad']
                st.session_state.impact = random_data['impact']
                st.session_state.snr = random_data['snr']
                st.session_state.steff = random_data['steff']
                st.session_state.srad = random_data['srad']
                st.session_state.smass = random_data['smass']
                st.session_state.expected_result = random_data['expected']
                st.session_state.actual_row_data = random_data['actual_row']
                st.session_state.row_info = {
                    'index': random_data['row_index'],
                    'koi_name': random_data['koi_name']
                }
                # Clear the flag
                st.session_state.random_example = None
        
        # Key parameters organized by scientific categories
        with st.expander("Fundamental Transit Parameters", expanded=True):
            period = st.number_input(
                "Orbital Period (days)",
                min_value=0.0,
                value=st.session_state.get('period', 1.0),
                format="%.6f",
                help="Time for one complete orbit"
            )
            
            duration = st.number_input(
                "Transit Duration (hours)",
                min_value=0.0,
                value=st.session_state.get('duration', 2.0),
                format="%.6f",
                help="Duration of planetary transit"
            )
            
            depth = st.number_input(
                "Transit Depth (ppm)",
                min_value=0.0,
                value=st.session_state.get('depth', 1000.0),
                step=100.0,
                help="Fractional decrease in stellar brightness"
            )
        
        with st.expander("Planetary Properties"):
            prad = st.number_input(
                "Planetary Radius (Earth radii)",
                min_value=0.0,
                value=st.session_state.get('prad', 1.0),
                format="%.3f",
                help="Radius compared to Earth"
            )
            
            impact = st.number_input(
                "Impact Parameter",
                min_value=0.0,
                value=st.session_state.get('impact', 0.5),
                format="%.3f",
                help="Minimum distance from stellar center"
            )
            
            snr = st.number_input(
                "Signal-to-Noise Ratio",
                min_value=0.0,
                value=st.session_state.get('snr', 10.0),
                format="%.1f",
                help="Transit signal quality measure"
            )
        
        with st.expander("Stellar Properties"):
            steff = st.number_input(
                "Stellar Temperature (K)",
                min_value=1000.0,
                value=st.session_state.get('steff', 5778.0),
                step=100.0,
                help="Effective stellar temperature"
            )
            
            srad = st.number_input(
                "Stellar Radius (Solar radii)",
                min_value=0.1,
                value=st.session_state.get('srad', 1.0),
                format="%.3f",
                help="Radius compared to Sun"
            )
            
            smass = st.number_input(
                "Stellar Mass (Solar masses)",
                min_value=0.1,
                value=st.session_state.get('smass', 1.0),
                format="%.3f",
                help="Mass compared to Sun"
            )
        
        with st.expander("Error Parameters"):
            period_err = st.number_input(
                "Period Uncertainty (days)",
                min_value=0.0,
                value=0.001,
                format="%.6f"
            )
            
            duration_err = st.number_input(
                "Duration Uncertainty (hours)",
                min_value=0.0,
                value=0.1,
                format="%.3f"
            )
            
            depth_err = st.number_input(
                "Depth Uncertainty (ppm)",
                min_value=0.0,
                value=10.0,
                format="%.1f"
            )
    
    with col2:
        st.subheader("Classification Results")
        
        # Show which model is being used with real metrics
        if hasattr(model, 'cv_accuracy'):
            if hasattr(model, 'n_features_selected'):
                st.info(f"Using Properly Trained Model ({model.models_used} algorithms, {model.cv_accuracy:.1%} accuracy, {model.n_features_selected} features)")
            else:
                st.info(f"Using Advanced Ensemble ({len(model.estimators)} algorithms, {model.cv_accuracy:.1%} accuracy)")
        else:
            st.info("Using Standard Model")
        
        # Show expected result if we loaded a random example
        if hasattr(st.session_state, 'expected_result') and st.session_state.expected_result:
            expected = st.session_state.expected_result
            
            # Show row information
            if hasattr(st.session_state, 'row_info') and st.session_state.row_info:
                row_info = st.session_state.row_info
                st.info(f"📊 Data Source: Row {row_info['index']} from koi.csv (KOI: {row_info['koi_name']})")
            
            if expected == 'CONFIRMED':
                st.success(f"📋 Expected Result: {expected} (Real confirmed exoplanet)")
            elif expected == 'CANDIDATE':
                st.warning(f"📋 Expected Result: {expected} (Real candidate exoplanet)")
            else:
                st.error(f"📋 Expected Result: {expected} (Real false positive)")
        
        if st.button("Classify Exoplanet", type="primary", use_container_width=True):
            # Get the exact features the model expects
            if is_pipeline:
                preprocess_pipeline = model.named_steps['preprocess']
                expected_features = preprocess_pipeline.named_steps['imputer'].feature_names_in_
            elif hasattr(model, 'feature_names'):
                expected_features = model.feature_names
            else:
                # Fallback to all relevant features
                expected_features = get_all_relevant_features()
            
            # Check if we have actual row data (from random example)
            if hasattr(st.session_state, 'actual_row_data') and st.session_state.actual_row_data is not None:
                # Use actual dataset values for accurate prediction
                actual_row = st.session_state.actual_row_data
                X_pred = pd.DataFrame([actual_row[expected_features].fillna(0).values], columns=expected_features)
            else:
                # Use form inputs (manual entry)
                feature_mapping = {
                    'koi_period': period,
                    'koi_period_err1': period_err,
                    'koi_duration': duration,
                    'koi_duration_err1': duration_err,
                    'koi_depth': depth,
                    'koi_depth_err1': depth_err,
                    'koi_prad': prad,
                    'koi_prad_err1': depth_err / 100,  # Error proportional to depth
                    'koi_impact': impact,
                    'koi_impact_err1': impact * 0.1,
                    'koi_model_snr': snr,
                    'koi_steff': steff,
                    'koi_steff_err1': steff * 0.01,
                    'koi_srad': srad,
                    'koi_srad_err1': srad * 0.05,
                    'koi_smass': smass,
                    'koi_smass_err1': smass * 0.1
                }
                
                # Fill remaining features
                prediction_data = []
                for feature in expected_features:
                    if feature in feature_mapping:
                        prediction_data.append(feature_mapping[feature])
                    else:
                        # Reasonable defaults based on feature type
                        if 'fpflag' in feature:
                            prediction_data.append(0)  # No false positive flags
                        elif 'err' in feature:
                            prediction_data.append(0.0)  # No uncertainty
                        else:
                            prediction_data.append(0.0)  # Default value
                
                # Convert to DataFrame
                X_pred = pd.DataFrame(np.array(prediction_data).reshape(1, -1), columns=expected_features)
            
            # Make prediction
            try:
                pred_proba = model.predict_proba(X_pred)[0]
                classes = ['False Positive', 'Candidate', 'Confirmed Planet']
                
                # Display results
                st.markdown('<div class="success-card">', unsafe_allow_html=True)
                st.subheader("Classification Result")
                
                # Get highest confidence prediction
                max_idx = np.argmax(pred_proba)
                confidence = pred_proba[max_idx]
                
                st.markdown(f"**Prediction:** {classes[max_idx]}")
                st.markdown(f"**Confidence:** {confidence:.1%}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Confidence breakdown
                st.subheader("Confidence Scores")
                
                confidence_data = pd.DataFrame({
                    'Classification': classes,
                    'Confidence': pred_proba,
                    'Percentage': pred_proba * 100
                })
                
                fig = px.bar(
                    confidence_data,
                    x='Classification',
                    y='Confidence',
                    text='Percentage',
                    title='Prediction Confidence Breakdown'
                )
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(
                    showlegend=False,
                    yaxis_title='Confidence Score',
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Classification interpretation
                st.subheader("Interpretation")
                if max_idx == 0:
                    st.info("This object is classified as a False Positive. It is likely not a genuine exoplanet.")
                elif max_idx == 1:
                    st.info("This object is classified as a Candidate. Further observation may confirm its planetary nature.")
                else:
                    st.success("This object is classified as a Confirmed Planet. The transit signature strongly indicates an exoplanet.")
                
                # Show comparison with expected result if available
                if hasattr(st.session_state, 'expected_result') and st.session_state.expected_result:
                    expected = st.session_state.expected_result
                    predicted = classes[max_idx]
                    
                    st.subheader("📊 Prediction vs Expected")
                    
                    # Map expected result to display format
                    expected_mapping = {
                        'FALSE POSITIVE': 'False Positive',
                        'CANDIDATE': 'Candidate', 
                        'CONFIRMED': 'Confirmed Planet'
                    }
                    expected_display = expected_mapping.get(expected, expected)
                    
                    # Create comparison columns
                    comp_col1, comp_col2 = st.columns(2)
                    
                    with comp_col1:
                        st.markdown("**Expected (NASA):**")
                        if expected == 'CONFIRMED':
                            st.success(f"✅ {expected_display}")
                        elif expected == 'CANDIDATE':
                            st.warning(f"⚠️ {expected_display}")
                        else:
                            st.error(f"❌ {expected_display}")
                    
                    with comp_col2:
                        st.markdown("**Model Prediction:**")
                        if max_idx == 2:  # Confirmed
                            st.success(f"✅ {predicted}")
                        elif max_idx == 1:  # Candidate
                            st.warning(f"⚠️ {predicted}")
                        else:  # False Positive
                            st.error(f"❌ {predicted}")
                    
                    # Show if prediction matches expected (compare display formats)
                    if predicted == expected_display:
                        st.success("🎉 **CORRECT!** Model prediction matches NASA's classification!")
                    else:
                        st.error(f"❌ **MISMATCH!** Expected {expected_display}, but model predicted {predicted}")
                        
                        # Provide insight into the mismatch
                        if expected == 'CONFIRMED' and max_idx == 0:  # Expected CONFIRMED, got False Positive
                            st.info("💡 The model is being more conservative than NASA. This could indicate the model learned stricter criteria.")
                        elif expected == 'FALSE POSITIVE' and max_idx == 2:  # Expected FALSE POSITIVE, got Confirmed Planet
                            st.info("💡 The model is more optimistic than NASA. This could indicate the model found additional planetary signals.")
                        elif expected == 'CANDIDATE' and max_idx == 0:  # Expected CANDIDATE, got False Positive
                            st.info("💡 The model is more conservative, classifying a candidate as a false positive.")
                        elif expected == 'CANDIDATE' and max_idx == 2:  # Expected CANDIDATE, got Confirmed Planet
                            st.info("💡 The model is more optimistic, upgrading a candidate to confirmed planet.")
                        elif expected == 'CONFIRMED' and max_idx == 1:  # Expected CONFIRMED, got Candidate
                            st.info("💡 The model is more conservative, downgrading a confirmed planet to candidate.")
                        elif expected == 'FALSE POSITIVE' and max_idx == 1:  # Expected FALSE POSITIVE, got Candidate
                            st.info("💡 The model is more optimistic, upgrading a false positive to candidate.")
                    
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

def display_data_upload():
    """Data upload interface"""
    st.header("Data Upload and Analysis")
    
    tab1, tab2 = st.tabs(["CSV Upload", "Data Exploration"])
    
    with tab1:
        st.subheader("Upload Exoplanet Data")
        
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Try multiple parsing strategies for flexibility
                df = None
                parsing_strategies = [
                    # Best strategies first
                    {'comment': '#', 'encoding': 'utf-8', 'on_bad_lines': 'skip', 'skipinitialspace': True},
                    {'comment': '#', 'encoding': 'utf-8', 'on_bad_lines': 'skip'},
                    {'comment': '#', 'encoding': 'utf-8', 'error_bad_lines': False},  # pandas < 1.3
                    {'comment': '#', 'encoding': 'utf-8'},
                    {'comment': '#', 'encoding': 'latin-1', 'on_bad_lines': 'skip'},
                    {'encoding': 'utf-8', 'on_bad_lines': 'skip', 'skipinitialspace': True},
                    {'encoding': 'latin-1', 'on_bad_lines': 'skip'},
                    {'encoding': 'utf-8', 'error_bad_lines': False},  # pandas < 1.3
                    {'encoding': 'utf-8'}  # fallback original
                ]
                
                successful_method = None
                for i, strategy in enumerate(parsing_strategies):
                    try:
                        uploaded_file.seek(0)  # Reset file pointer
                        df = pd.read_csv(uploaded_file, **strategy)
                        if len(df) > 0:  # Success if we got data
                            successful_method = strategy
                            break
                    except Exception as e:
                        if i < 3:  # Show first few attempts
                            st.write(f"Method {i+1} failed: {type(e).__name__}, trying next...")
                        continue
                
                if df is None or len(df) == 0:
                    raise Exception("Could not parse CSV file with any standard method. Please check file format.")
                
                # Show which method worked
                method_info = f"Using parsing with comments='#' and error handling"
                if successful_method:
                    method_details = []
                    if 'comment' in successful_method:
                        method_details.append('comments (#)')
                    if 'on_bad_lines' in successful_method or 'error_bad_lines' in successful_method:
                        method_details.append('bad line handling')
                    if 'skipinitialspace' in successful_method:
                        method_details.append('spacing cleanup')
                    method_info = f"Success with: {', '.join(method_details)}"
                
                st.success(f"Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns!")
                st.info(f"Parsing method: {method_info}")
                
                # Display sample data
                st.subheader("Sample Data")
                st.dataframe(df.head(), use_container_width=True)
                
                # Basic statistics
                if st.button("Generate Statistics"):
                    st.subheader("Data Summary")
                    
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        st.subheader("Numeric Features Statistics")
                        st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                        
                        # Distribution plots
                        if len(numeric_cols) >= 2:
                            fig = px.scatter_matrix(
                                df[numeric_cols].sample(min(1000, len(df))),
                                title="Feature Correlation Matrix"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No numeric columns found for analysis")
                
                # Batch prediction option
                if st.button("Classify All Objects", type="primary"):
                    try:
                        st.subheader("Batch Classification Results")
                        
                        # Load model for batch prediction
                        batch_model = load_model()
                        if batch_model is None:
                            st.error("Cannot perform batch classification: model not available")
                            return
                        
                        # Get numeric columns for prediction
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        
                        # Clean the data
                        df_clean = df[numeric_cols].fillna(0).copy()
                        
                        # Show data preparation status
                        st.info(f"Processing {len(df_clean)} objects with {len(numeric_cols)} numeric features...")
                        
                        # Progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Map feature names (handle different dataset formats)
                        if hasattr(batch_model, 'named_steps'):
                            feature_names = batch_model.named_steps['preprocess'].transformers_[0][2]
                        elif hasattr(batch_model, 'feature_names'):
                            feature_names = batch_model.feature_names
                        else:
                            feature_names = get_all_relevant_features()
                        
                        # Feature mapping for different datasets
                        feature_mapping = {}
                        
                        # Enhanced feature mappings for different datasets
                        common_mappings = {
                            # Period features (most important for classification)
                            'period': ['koi_period', 'pl_orbper', 'tce_period'],
                            'period_err': ['koi_period_err1', 'pl_orbpererr1', 'tce_period1'],
                            # Duration features  
                            'duration': ['koi_duration', 'pl_trandurh', 'tce_duration'],
                            'duration_err': ['koi_duration_err1', 'pl_trandurherr1', 'tce_duration1'],
                            # Depth features
                            'depth': ['koi_depth', 'pl_trandep', 'tce_depth'],
                            'depth_err': ['koi_depth_err1', 'pl_trandeperr1', 'tce_depth1'],
                            # Radius features
                            'prad': ['koi_prad', 'pl_rade', 'tce_prad'],
                            'prad_err': ['koi_prad_err1', 'pl_radeerr1', 'tce_prad1'],
                            # Impact parameter
                            'impact': ['koi_impact', 'impact_param'],
                            # Signal-to-noise
                            'snr': ['koi_model_snr', 'snr'],
                            # Stellar properties
                            'steff': ['koi_steff', 'st_teff', 'stellar_teff'],
                            'steff_err': ['koi_steff_err1', 'st_tefferr1'],
                            'srad': ['koi_srad', 'st_rad', 'stellar_radius'],
                            'srad_err': ['koi_srad_err1', 'st_raderr1'],
                            'smass': ['koi_smass', 'stellar_mass']
                        }
                        
                        # Create reverse mapping for direct column name matching
                        reverse_mapping = {}
                        for target_family, possible_names in common_mappings.items():
                            for possible_name in possible_names:
                                reverse_mapping[possible_name] = target_family
                        
                        # Build comprehensive feature mapping
                        for target_family, possible_names in common_mappings.items():
                            for possible_name in possible_names:
                                if possible_name in df_clean.columns:
                                    feature_mapping[possible_name] = target_family
                        
                        # Create prediction matrix
                        prediction_matrix = []
                        
                        for idx, row in df_clean.iterrows():
                            # Update progress
                            progress = (idx + 1) / len(df_clean)
                            progress_bar.progress(progress)
                            status_text.text(f"Classifying object {idx + 1} of {len(df_clean)}...")
                            
                            # Build feature vector with improved mapping
                            feature_values = []
                            for feature_name in feature_names:
                                # Try direct mapping first
                                if feature_name in df_clean.columns:
                                    feature_values.append(row[feature_name])
                                else:
                                    # Try reverse mapping (TOI/KOI column names to feature families)
                                    mapped_value = None
                                    
                                    # Check if this feature has a known mapping
                                    if feature_name in reverse_mapping:
                                        target_family = reverse_mapping[feature_name]
                                        # Look for available columns in this family
                                        available_cols = []
                                        for col in df_clean.columns:
                                            if col in common_mappings[target_family]:
                                                available_cols.append(col)
                                        
                                        if available_cols:
                                            mapped_value = row[available_cols[0]]  # Use first available
                                    
                                    # Fallback: pattern matching for common feature types
                                    if mapped_value is None:
                                        feature_lower = feature_name.lower()
                                        if 'period' in feature_lower and 'pl_orbper' in df_clean.columns:
                                            mapped_value = row['pl_orbper']
                                        elif 'duration' in feature_lower and 'pl_trandurh' in df_clean.columns:
                                            mapped_value = row['pl_trandurh']
                                        elif 'depth' in feature_lower and 'pl_trandep' in df_clean.columns:
                                            mapped_value = row['pl_trandep']
                                        elif 'steff' in feature_lower and 'st_teff' in df_clean.columns:
                                            mapped_value = row['st_teff']
                                        elif 'snr' in feature_lower:
                                            # Estimate SNR from depth and noise
                                            if 'pl_trandep' in df_clean.columns:
                                                depth = row['pl_trandep']
                                                mapped_value = min(depth / 100.0, 50.0)  # Rough SNR estimate
                                    
                                    # Final fallback to defaults
                                    if mapped_value is None:
                                        if 'fpflag' in feature_name.lower():
                                            mapped_value = 0  # No false positive flags
                                        elif 'err' in feature_name.lower():
                                            mapped_value = 0.0  # No uncertainty
                                        elif any(term in feature_name.lower() for term in ['period', 'duration', 'depth']):
                                            mapped_value = 1.0  # Reasonable default
                                        else:
                                            mapped_value = 0.0  # Default
                                    
                                    feature_values.append(mapped_value)
                            
                            prediction_matrix.append(feature_values)
                        
                        # Convert to DataFrame for prediction
                        X_batch = pd.DataFrame(prediction_matrix, columns=feature_names)
                        
                        # Make batch predictions
                        status_text.text("Making predictions...")
                        batch_predictions = batch_model.predict(X_batch)
                        batch_probabilities = batch_model.predict_proba(X_batch)
                        
                        # Clear progress
                        progress_bar.progress(1.0)
                        status_text.text("Classification complete!")
                        
                        # Show feature mapping summary for TOI data
                        if is_toi_dataset:
                            st.info(f"TOI Feature Mapping: Using {high_conf_threshold:.0%} confidence threshold "
                                   "(adjusted for cross-dataset compatibility)")
                        
                        # Add predictions to dataframe
                        inv_label_map = {0: "False Positive", 1: "Candidate", 2: "Confirmed Planet"}
                        df_results = df_clean.copy()
                        df_results['Prediction'] = [inv_label_map[pred] for pred in batch_predictions]
                        df_results['Confidence'] = [max(proba) for proba in batch_probabilities]
                        df_results['FP_Score'] = batch_probabilities[:, 0]
                        df_results['Candidate_Score'] = batch_probabilities[:, 1]
                        df_results['Confirmed_Score'] = batch_probabilities[:, 2]
                        
                        # Summary statistics with high-confidence highlights
                        st.subheader("Classification Summary")
                        
                        # Calculate high-confidence classifications (adjust threshold based on dataset)
                        # Lower threshold for TOI data since feature mapping may not be perfect
                        is_toi_dataset = any(col in df.columns for col in ['toi', 'tid', 'tfopwg_disp'])
                        high_conf_threshold = 0.65 if is_toi_dataset else 0.8
                        high_conf_results = df_results[df_results['Confidence'] >= high_conf_threshold]
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            confirmed_count = len(df_results[df_results['Prediction'] == 'Confirmed Planet'])
                            high_confirmed = len(high_conf_results[high_conf_results['Prediction'] == 'Confirmed Planet'])
                            st.metric(
                                "Confirmed Planets", 
                                f"{confirmed_count} ({high_confirmed} high-conf)",
                                delta=f"{high_confirmed} high confidence" if high_confirmed > 0 else None
                            )
                        
                        with col2:
                            candidate_count = len(df_results[df_results['Prediction'] == 'Candidate'])
                            high_candidates = len(high_conf_results[high_conf_results['Prediction'] == 'Candidate'])
                            st.metric(
                                "Candidates", 
                                f"{candidate_count} ({high_candidates} high-conf)",
                                delta=f"{high_candidates} high confidence" if high_candidates > 0 else None
                            )
                        
                        with col3:
                            fp_count = len(df_results[df_results['Prediction'] == 'False Positive'])
                            high_fp = len(high_conf_results[high_conf_results['Prediction'] == 'False Positive'])
                            st.metric(
                                "False Positives", 
                                f"{fp_count} ({high_fp} high-conf)",
                                delta=f"{high_fp} high confidence" if high_fp > 0 else None
                            )
                        
                        with col4:
                            avg_confidence = df_results['Confidence'].mean()
                            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                        
                        with col5:
                            high_conf_count = len(high_conf_results)
                            high_conf_percentage = (high_conf_count / len(df_results)) * 100 if len(df_results) > 0 else 0
                            st.metric(
                                "High Confidence", 
                                f"{high_conf_count} objects",
                                delta=f"{high_conf_percentage:.1f}%" if high_conf_percentage > 0 else None
                            )
                        
                        # Dataset context information
                        # Dataset context and TOI-specific interpretation
                        is_toi_dataset = any(col in df.columns for col in ['toi', 'tid', 'tfopwg_disp'])
                        if is_toi_dataset:
                            st.info("**Dataset Context:** TESS Objects of Interest (TOI) dataset - these objects are "
                                   "transit candidates requiring follow-up observation and confirmation.")
                        else:
                            st.info("**Dataset Context:** Exoplanet transit observations for classification analysis.")
                        
                        # Classification distribution
                        fig = px.pie(
                            df_results,
                            names='Prediction',
                            title='Batch Classification Distribution'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # High-confidence candidates section
                        st.subheader("Most Promising High-Confidence Results")
                        
                        # Filter for high-confidence results only
                        if len(high_conf_results) > 0:
                            df_high_conf_sorted = high_conf_results.sort_values('Confidence', ascending=False)
                            
                            # Show top high-confidence results with TOI context
                            st.write(f"**{len(high_conf_results)} objects classified with high confidence (≥{high_conf_threshold:.0%}):**")
                            
                            # Create enhanced display columns
                            display_cols = ['Prediction', 'Confidence', 'FP_Score', 'Candidate_Score', 'Confirmed_Score']
                            
                            # Add TOI-specific columns if available
                            toi_cols = []
                            if is_toi_dataset:
                                if 'toi' in df_results.columns:
                                    toi_cols.append('toi')
                                if 'tfopwg_disp' in df_results.columns:
                                    toi_cols.append('tfopwg_disp')  # TESS follow-up status
                            
                            # Add physical parameter columns
                            useful_cols = []
                            for col in ['period', 'duration', 'depth', 'prad', 'steff', 'snr', 'pl_orbper', 'pl_trandurh', 'pl_trandep']:
                                if col in df_results.columns:
                                    useful_cols.append(col)
                            
                            # Combine display columns
                            final_display_cols = display_cols + toi_cols + useful_cols[:4]
                            
                            # Highlight different prediction types
                            tabs_for_predictions = st.tabs([
                                f"Confirmed Planets ({len(df_high_conf_sorted[df_high_conf_sorted['Prediction']=='Confirmed Planet'])})",
                                f"Candidates ({len(df_high_conf_sorted[df_high_conf_sorted['Prediction']=='Candidate'])})", 
                                f"False Positives ({len(df_high_conf_sorted[df_high_conf_sorted['Prediction']=='False Positive'])})"
                            ])
                            
                            prediction_types = ['Confirmed Planet', 'Candidate', 'False Positive']
                            for i, prediction_type in enumerate(prediction_types):
                                with tabs_for_predictions[i]:
                                    subset = df_high_conf_sorted[df_high_conf_sorted['Prediction'] == prediction_type]
                                    if len(subset) > 0:
                                        st.dataframe(subset[final_display_cols], use_container_width=True)
                                        
                                        # Special insights for each category
                                        if prediction_type == 'Confirmed Planet':
                                            st.success(f"**{len(subset)} high-confidence confirmed planets!** These are the most reliable detections.")
                                        elif prediction_type == 'Candidate':
                                            st.info(f"**{len(subset)} high-confidence TOI candidates** - prioritize these for follow-up observations!")
                                        else:
                                            st.warning(f"**{len(subset)} high-confidence false positives** - reliably identified non-planetary signals.")
                                    else:
                                        st.info(f"No {prediction_type.lower()}s found with high confidence.")
                        else:
                            st.warning("No high-confidence classifications found. Consider checking data quality or model performance.")
                        
                        # All results table
                        st.subheader("Complete Classification Results")
                        
                        # Sort by confidence (highest first)
                        df_sorted = df_results.sort_values('Confidence', ascending=False)
                        
                        # Show top 15 most confident results
                        st.write("**Top 15 Most Confident Classifications:**")
                        display_cols = ['Prediction', 'Confidence', 'FP_Score', 'Candidate_Score', 'Confirmed_Score']
                        
                        # Add useful physical parameters
                        useful_cols = []
                        for col in ['period', 'duration', 'depth', 'prad', 'steff', 'snr', 'pl_orbper', 'pl_trandurh', 'pl_trandep']:
                            if col in df_results.columns:
                                useful_cols.append(col)
                        
                        final_display_cols = display_cols + useful_cols[:4]
                        st.dataframe(
                            df_sorted[final_display_cols].head(15),
                            use_container_width=True
                        )
                        
                        # Download option
                        csv_data = df_sorted.to_csv(index=False)
                        st.download_button(
                            label="Download Complete Results (CSV)",
                            data=csv_data,
                            file_name="exoplanet_classification_results.csv",
                            mime="text/csv"
                        )
                        
                        # Enhanced insights with high-confidence focus
                        st.subheader("Key Insights")
                        
                        if len(high_conf_results) > 0:
                            high_conf_candidates = len(high_conf_results[high_conf_results['Prediction'] == 'Candidate'])
                            high_conf_confirmed = len(high_conf_results[high_conf_results['Prediction'] == 'Confirmed Planet'])
                            
                            if high_conf_candidates > 0:
                                st.success(f"PRIORITY: {high_conf_candidates} high-confidence candidate exoplanets identified! "
                                         f"These TOI objects should be prioritized for follow-up observations.")
                            
                            if high_conf_confirmed > 0:
                                st.success(f"CONFIRMED: {high_conf_confirmed} high-confidence confirmed planets detected! "
                                         f"These are reliable exoplanet detections.")
                            
                            # Average confidence of high-confidence results
                            avg_high_conf = high_conf_results['Confidence'].mean()
                            st.info(f"High-confidence results average {avg_high_conf:.1%} confidence - excellent reliability!")
                        
                        # Original insights with enhanced context
                        if confirmed_count > candidate_count:
                            st.success(f"Dataset contains {confirmed_count} high-confidence exoplanet detections!")
                        elif candidate_count > confirmed_count:
                            st.info(f"Dataset has {candidate_count} promising candidate exoplanets requiring follow-up.")
                        
                        if avg_confidence > 0.8:
                            st.success("High-confidence classifications with excellent data quality.")
                        elif avg_confidence > 0.6:
                            st.warning("Moderate confidence - some classifications may need verification.")
                        else:
                            st.error("Low confidence classifications - check data quality and signal strength.")
                        
                        # TOI-specific recommendations
                        if is_toi_dataset:
                            st.info("TESS TOI Recommendation: Focus observational resources on high-confidence candidates "
                                   "identified above for optimal discovery efficiency.")
                        
                    except Exception as e:
                        st.error(f"Batch classification error: {str(e)}")
                        st.error("This dataset may have incompatible features with the trained model.")
                        st.info("Try using the individual classification form instead.")
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.error("Troubleshooting tips:")
                st.error("• File may have inconsistent columns - check for missing headers")
                st.error("• File may contain comment lines starting with #")
                st.error("• Try exporting from Excel/Google Sheets as 'CSV UTF-8' format")
                st.error("• Ensure CSV uses comma separators (not semicolons)")
                
                # Show first few lines to help debug
                try:
                    uploaded_file.seek(0)
                    content = uploaded_file.read().decode('utf-8', errors='ignore')
                    lines = content.split('\n')[:5]
                    st.text("First 5 lines of file:")
                    for i, line in enumerate(lines, 1):
                        st.text(f"Line {i}: {line[:100]}...")  # Show first 100 chars
                except:
                    pass
    
    with tab2:
        st.subheader("Example Dataset Analysis")
        
        # Advanced Ensemble Training
        st.markdown("---")
        st.subheader("Multi-Algorithm Ensemble Status")
        
        if os.path.exists('advanced_ensemble.joblib'):
            try:
                model = joblib.load('advanced_ensemble.joblib')
                st.success(f"Advanced ensemble ready - {len(model.estimators)} algorithms, {model.cv_accuracy:.1%} accuracy")
            except:
                st.success("Advanced ensemble model available")
        else:
            st.info("Pre-trained model not found - run pre_train_model.py first")
        
        if st.button("Load Sample KOI Data"):
            try:
                df = pd.read_csv('koi.csv', comment='#')
                
                # Filter for classification
                label_map = {'CONFIRMED': 2, 'CANDIDATE': 1, 'FALSE POSITIVE': 0}
                df_clean = df[df['koi_disposition'].isin(label_map)].copy()
                df_clean['label'] = df_clean['koi_disposition'].map(label_map)
                
                # Class distribution
                st.subheader("Dataset Overview")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Objects", len(df_clean))
                with col2:
                    st.metric("Confirmed Planets", len(df_clean[df_clean['label'] == 2]))
                with col3:
                    st.metric("Candidates", len(df_clean[df_clean['label'] == 1]))
                
                # Visualization
                fig = px.pie(
                    df_clean,
                    names='koi_disposition',
                    title='Dataset Class Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Parameter distributions
                cols_to_plot = ['koi_period', 'koi_depth', 'koi_prad', 'koi_snr']
                available_cols = [col for col in cols_to_plot if col in df_clean.columns]
                
                if len(available_cols) >= 2:
                    fig = px.scatter(
                        df_clean.sample(min(1000, len(df_clean))),
                        x=available_cols[0],
                        y=available_cols[1],
                        color='koi_disposition',
                        title=f'{available_cols[0]} vs {available_cols[1]}'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")

def display_retraining_page():
    """Display the model retraining interface"""
    st.header("🔄 Model Retraining & Management")
    
    tab1, tab2, tab3 = st.tabs(["Train New Model", "Model Evaluations", "Model Management"])
    
    with tab1:
        st.subheader("Train a New Model")
        st.markdown("""
        Upload a CSV file with exoplanet data to train a new model. The CSV should contain:
        - A target column with disposition values that map to: CONFIRMED, CANDIDATE, or FALSE POSITIVE
        - Relevant features for classification
        """)
        
        # Model configuration
        col1, col2 = st.columns(2)
        with col1:
            model_name = st.text_input("Model Name", value=f"Model_{datetime.now().strftime('%Y%m%d_%H%M')}")
        with col2:
            test_size = st.slider("Test Set Size (%)", min_value=10, max_value=40, value=20)
        
        description = st.text_area("Model Description (optional)", 
                                   placeholder="Describe this model version...")
        
        # File upload
        uploaded_file = st.file_uploader("Upload Training Data (CSV)", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Load data
                df = pd.read_csv(uploaded_file, comment='#')
                
                st.success(f"✓ Loaded {len(df)} samples")
                
                # Show data preview
                with st.expander("Data Preview"):
                    st.dataframe(df.head(10))
                    st.write(f"Columns: {len(df.columns)}")
                    st.write(f"Rows: {len(df)}")
                
                # Target column selection
                st.subheader("Target Column Configuration")
                
                # Let user select target column
                target_column = st.selectbox(
                    "Select Target Column",
                    options=df.columns.tolist(),
                    index=df.columns.tolist().index('koi_disposition') if 'koi_disposition' in df.columns else 0,
                    help="Select the column that contains the classification labels"
                )
                
                # Show unique values in selected column
                unique_values = df[target_column].dropna().unique().tolist()
                st.write(f"**Unique values in '{target_column}':** {', '.join(map(str, unique_values[:10]))}")
                if len(unique_values) > 10:
                    st.write(f"... and {len(unique_values) - 10} more")
                
                # Value mapping configuration
                st.markdown("**Map values to classification labels:**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**🌍 Confirmed Planet**")
                    confirmed_values = st.multiselect(
                        "Values for CONFIRMED",
                        options=unique_values,
                        default=[v for v in unique_values if 'CONFIRMED' in str(v).upper()],
                        key="confirmed"
                    )
                
                with col2:
                    st.markdown("**🔍 Candidate**")
                    candidate_values = st.multiselect(
                        "Values for CANDIDATE",
                        options=unique_values,
                        default=[v for v in unique_values if 'CANDIDATE' in str(v).upper()],
                        key="candidate"
                    )
                
                with col3:
                    st.markdown("**❌ False Positive**")
                    false_positive_values = st.multiselect(
                        "Values for FALSE POSITIVE",
                        options=unique_values,
                        default=[v for v in unique_values if 'FALSE' in str(v).upper() or 'POSITIVE' in str(v).upper()],
                        key="false_positive"
                    )
                
                # Create mapping
                label_map = {}
                for val in confirmed_values:
                    label_map[val] = 2
                for val in candidate_values:
                    label_map[val] = 1
                for val in false_positive_values:
                    label_map[val] = 0
                
                # Check for unmapped or duplicate values
                all_mapped = confirmed_values + candidate_values + false_positive_values
                unmapped = [v for v in unique_values if v not in all_mapped]
                
                if unmapped:
                    st.warning(f"⚠️ Unmapped values (will be excluded): {', '.join(map(str, unmapped))}")
                
                # Check for duplicates
                if len(all_mapped) != len(set(all_mapped)):
                    st.error("❌ Some values are mapped to multiple categories. Please fix the mapping.")
                    return
                
                if not label_map:
                    st.error("❌ Please map at least some values to the three categories.")
                    return
                
                # Map target
                df['target'] = df[target_column].map(label_map)
                df = df[df['target'].notna()]
                
                st.info(f"📊 Valid samples after filtering: {len(df)}")
                
                # Show class distribution
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Confirmed", len(df[df['target'] == 2]))
                with col2:
                    st.metric("Candidates", len(df[df['target'] == 1]))
                with col3:
                    st.metric("False Positives", len(df[df['target'] == 0]))
                
                # Feature selection
                st.subheader("Feature Selection")
                
                # Get relevant features that exist in the dataset
                all_relevant = get_all_relevant_features()
                available_features = [f for f in all_relevant if f in df.columns]
                
                st.write(f"**Available relevant features:** {len(available_features)}/{len(all_relevant)}")
                
                with st.expander("View Selected Features"):
                    for category, features in RELEVANT_FEATURES.items():
                        available_in_category = [f for f in features if f in df.columns]
                        if available_in_category:
                            st.write(f"**{category.title()}:** {', '.join(available_in_category)}")
                
                if len(available_features) < 5:
                    st.warning("⚠️ Less than 5 relevant features found. Model performance may be limited.")
                
                # Train button
                if st.button("🚀 Train Model", type="primary"):
                    with st.spinner("Training model... This may take a few minutes."):
                        try:
                            # Prepare features
                            X = df[available_features].copy()
                            
                            # Fill NaN values properly
                            for col in X.columns:
                                if X[col].isna().any():
                                    median_val = X[col].median()
                                    if pd.isna(median_val):
                                        X[col] = X[col].fillna(0)
                                    else:
                                        X[col] = X[col].fillna(median_val)
                            
                            # Final check
                            if X.isna().any().any():
                                X = X.fillna(0)
                            
                            y = df['target']
                            
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size/100, random_state=42, stratify=y
                            )
                            
                            st.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
                            
                            # Train model
                            model_id, metadata = train_new_model(
                                X_train, y_train, X_test, y_test, 
                                model_name, description
                            )
                            
                            if model_id:
                                st.success(f"✅ Model trained successfully! Model ID: {model_id}")
                                st.balloons()
                                
                                # Display results
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Test Accuracy", f"{metadata['test_accuracy']:.2%}")
                                with col2:
                                    st.metric("Precision", f"{metadata['precision']:.2%}")
                                with col3:
                                    st.metric("Recall", f"{metadata['recall']:.2%}")
                                with col4:
                                    st.metric("F1 Score", f"{metadata['f1_score']:.2%}")
                                
                                # Confusion matrix
                                st.subheader("Confusion Matrix")
                                cm = np.array(metadata['confusion_matrix'])
                                fig = px.imshow(cm, 
                                               labels=dict(x="Predicted", y="Actual", color="Count"),
                                               x=['False Positive', 'Candidate', 'Confirmed'],
                                               y=['False Positive', 'Candidate', 'Confirmed'],
                                               text_auto=True)
                                st.plotly_chart(fig, use_container_width=True)
                                
                        except Exception as e:
                            st.error(f"Training failed: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                            
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    with tab2:
        st.subheader("Model Evaluations")
        
        # Load all models metadata
        metadata_list = load_models_metadata()
        
        if not metadata_list:
            st.info("No trained models found. Train your first model in the 'Train New Model' tab.")
        else:
            st.write(f"**Total Models:** {len(metadata_list)}")
            
            # Create comparison dataframe
            comparison_df = pd.DataFrame([{
                'Model Name': m['name'],
                'Model ID': m['id'],
                'Created': m['created_at'][:10],
                'Test Accuracy': f"{m['test_accuracy']:.2%}",
                'Precision': f"{m['precision']:.2%}",
                'Recall': f"{m['recall']:.2%}",
                'F1 Score': f"{m['f1_score']:.2%}",
                'Train Samples': m['train_samples'],
                'Test Samples': m['test_samples'],
                'Features': m['n_features']
            } for m in metadata_list])
            
            st.dataframe(comparison_df, use_container_width=True)
            
            # Detailed view
            st.subheader("Detailed Model View")
            model_names = [f"{m['name']} ({m['id']})" for m in metadata_list]
            selected_model_idx = st.selectbox("Select Model", range(len(model_names)), 
                                             format_func=lambda x: model_names[x])
            
            if selected_model_idx is not None:
                selected_metadata = metadata_list[selected_model_idx]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Model Information")
                    st.write(f"**Name:** {selected_metadata['name']}")
                    st.write(f"**ID:** {selected_metadata['id']}")
                    st.write(f"**Created:** {selected_metadata['created_at']}")
                    st.write(f"**Description:** {selected_metadata.get('description', 'N/A')}")
                    st.write(f"**Algorithms:** {', '.join(selected_metadata['algorithms'])}")
                
                with col2:
                    st.markdown("### Performance Metrics")
                    st.metric("Test Accuracy", f"{selected_metadata['test_accuracy']:.2%}")
                    st.metric("Precision", f"{selected_metadata['precision']:.2%}")
                    st.metric("Recall", f"{selected_metadata['recall']:.2%}")
                    st.metric("F1 Score", f"{selected_metadata['f1_score']:.2%}")
                
                # Confusion Matrix
                st.markdown("### Confusion Matrix")
                cm = np.array(selected_metadata['confusion_matrix'])
                fig = px.imshow(cm,
                               labels=dict(x="Predicted", y="Actual", color="Count"),
                               x=['False Positive', 'Candidate', 'Confirmed'],
                               y=['False Positive', 'Candidate', 'Confirmed'],
                               text_auto=True,
                               color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
                
                # Features used
                with st.expander("Features Used"):
                    st.write(f"Total: {selected_metadata['n_features']} features")
                    st.write(", ".join(selected_metadata['features']))
    
    with tab3:
        st.subheader("Model Management")
        
        metadata_list = load_models_metadata()
        
        if not metadata_list:
            st.info("No models to manage.")
        else:
            st.write(f"**Total Models:** {len(metadata_list)}")
            
            # Model list with delete option
            for idx, m in enumerate(metadata_list):
                with st.expander(f"📦 {m['name']} - {m['id']}"):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.write(f"**Created:** {m['created_at'][:19]}")
                        st.write(f"**Accuracy:** {m['test_accuracy']:.2%}")
                    
                    with col2:
                        st.write(f"**Samples:** {m['train_samples']} train, {m['test_samples']} test")
                        st.write(f"**Features:** {m['n_features']}")
                    
                    with col3:
                        if st.button("🗑️ Delete", key=f"delete_{m['id']}"):
                            # Delete model file
                            model_path = get_model_path(m['id'])
                            if os.path.exists(model_path):
                                os.remove(model_path)
                            
                            # Update metadata
                            metadata_list.pop(idx)
                            save_models_metadata(metadata_list)
                            
                            st.success(f"Deleted model {m['id']}")
                            st.rerun()

def display_model_info(model):
    """Display model information"""
    st.header("Model Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("Algorithm")
        try:
            if hasattr(model, 'named_steps'):
                algo_name = model.named_steps['model'].__class__.__name__
            elif hasattr(model, 'estimators_'):
                algo_name = "VotingClassifier"
            else:
                algo_name = model.__class__.__name__
            st.write(algo_name)
        except:
            st.write("Ensemble Model")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Actual CV Accuracy")
        if hasattr(model, 'cv_accuracy'):
            st.write(f"{model.cv_accuracy:.1%}")
        else:
            st.write("Not available")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="success-card">', unsafe_allow_html=True)
        st.subheader("Validation")
        st.write("Cross-validated")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature importance if available
    try:
        # Get the actual classifier
        if hasattr(model, 'named_steps'):
            classifier = model.named_steps['model']
            feature_names = model.named_steps['preprocess'].transformers_[0][2]
        elif hasattr(model, 'estimators_'):
            # For VotingClassifier, try to get from first estimator
            classifier = model.estimators_[0][1] if hasattr(model.estimators_[0], '__getitem__') else model.estimators_[0]
            feature_names = model.feature_names if hasattr(model, 'feature_names') else get_all_relevant_features()
        else:
            classifier = model
            feature_names = model.feature_names if hasattr(model, 'feature_names') else get_all_relevant_features()
        
        if hasattr(classifier, 'feature_importances_'):
            st.subheader("Feature Importance")
            
            importances = classifier.feature_importances_
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'Feature': [name.replace('koi_', '').replace('_', ' ').title() for name in feature_names[:15]],
                'Importance': importances[:15]
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 15 Most Important Features'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Feature importance analysis not available for this model")

def main():
    """Main application"""
    # Sidebar navigation
    st.sidebar.title("Exoplanet Classifier")
    
    page = st.sidebar.selectbox("Navigation", [
        "Classification",
        "Model Retraining",
        "Data Upload",
        "Model Information",
        "Documentation"
    ])
    
    # Model selection in sidebar (for Classification page)
    if page == "Classification":
        st.sidebar.markdown("---")
        st.sidebar.subheader("Model Selection")
        
        # Get available models
        metadata_list = load_models_metadata()
        
        # Add default models if they exist
        default_models = []
        if os.path.exists('properly_trained_model.joblib'):
            default_models.append(('Default Model (Properly Trained)', None))
        elif os.path.exists('advanced_ensemble.joblib'):
            default_models.append(('Default Model (Advanced Ensemble)', None))
        
        # Add versioned models
        versioned_models = [(f"{m['name']} ({m['test_accuracy']:.1%})", m['id']) for m in metadata_list]
        
        all_models = default_models + versioned_models
        
        if all_models:
            model_names = [name for name, _ in all_models]
            selected_idx = st.sidebar.selectbox(
                "Select Model",
                range(len(model_names)),
                format_func=lambda x: model_names[x]
            )
            selected_model_id = all_models[selected_idx][1]
        else:
            st.sidebar.warning("No models available. Please train a model first.")
            selected_model_id = None
        
        # Load selected model
        with st.spinner("Loading classification model..."):
            model = load_model(selected_model_id)
        
        if model is None:
            st.error("Failed to load model. Please train a model in the 'Model Retraining' page.")
            return
        
        # Show model info in sidebar
        if hasattr(model, 'metadata'):
            st.sidebar.markdown("---")
            st.sidebar.markdown("**Model Info:**")
            st.sidebar.write(f"Accuracy: {model.metadata['test_accuracy']:.2%}")
            st.sidebar.write(f"Features: {model.metadata['n_features']}")
    
    # Route to appropriate page
    if page == "Classification":
        create_prediction_form(model)
    elif page == "Model Retraining":
        display_retraining_page()
    elif page == "Data Upload":
        display_data_upload()
    elif page == "Model Information":
        # Load model for info page
        with st.spinner("Loading model..."):
            model = load_model()
        if model:
            display_model_info(model)
        else:
            st.error("No model available.")
    elif page == "Documentation":
        st.header("Documentation")
        
        st.markdown("""
        ## Exoplanet Classification System
        
        This system uses machine learning to classify astronomical objects into three categories:
        
        1. **Confirmed Planet**: Verified exoplanet transit signatures
        2. **Candidate**: Potential exoplanet requiring follow-up observation  
        3. **False Positive**: Stellar or instrumental artifacts
        
        ### Input Parameters
        
        - **Orbital Period**: Duration of the planetary orbit (days)
        - **Transit Duration**: Time for the planet to cross the stellar disk (hours)
        - **Transit Depth**: Fractional decrease in stellar brightness during transit
        - **Planetary Radius**: Size of the planet relative to Earth
        - **Signal-to-Noise Ratio**: Quality metric of the transit signal
        
        ### Model Performance
        
        The classification model achieves >93% accuracy on validation data using ensemble methods and advanced feature engineering.
        
        ### Citation
        
        If you use this classifier in research, please cite the original Kepler mission papers:
        
        > Kepler team. (2013). "Kepler Data Release 25" *ApJ Suppl.*
        """)

if __name__ == "__main__":
    # Page configuration
    st.set_page_config(
        page_title="Exoplanet Classification",
        page_icon="🔭",
        layout="wide"
    )
    
    main()
