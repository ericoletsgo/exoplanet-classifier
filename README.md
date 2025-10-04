# Exoplanet Classification Tool

A machine learning application that classifies Kepler Object of Interest (KOI) candidates as Confirmed Planets, Candidates, or False Positives.

## 🚀 Features

- **Random Forest Classifier** with ~93% accuracy
- **Interactive Web Interface** built with Streamlit
- **Real-time Predictions** with confidence scores
- **Comprehensive Input Fields** for orbital and transit parameters

## 📊 Model Performance

- **Overall Accuracy:** 93%
- **Precision/Recall/F1-Scores:**
  - Confirmed Planets: 91% precision, 92% recall
  - Candidates: 88% precision, 80% recall  
  - False Positives: 96% precision, 99% recall

## 🛠️ Setup

### Prerequisites

- Python 3.7+
- Required packages: pandas, scikit-learn, joblib, streamlit

### Installation

1. Install required packages:
```bash
pip install pandas scikit-learn joblib streamlit
```

2. Clone or download the project files:
   - `train_model.py` - Model training script
   - `streamlit_app.py` - Web application
   - `koi.csv` - Kepler dataset
   - `model.joblib` - Trained model (generated after training)

## 🎯 Usage

### Training the Model

Run the training script to train the model:

```bash
python train_model.py
```

This will:
1. Load and preprocess the KOI dataset
2. Train a Random Forest Classifier
3. Evaluate performance on test data
4. Save the trained model as `model.joblib`

### Running the Web App

Launch the Streamlit web application:

```bash
streamlit run streamlit_app.py
```

Then open your browser to the URL provided (typically `http://localhost:8501`).

### Using the Interface

1. **Fill in the parameters** in the web interface:
   - **False Positive Flags:** Binary flags (0 or 1)
   - **Detection Quality:** Orbital period, epoch, errors
   - **Orbital Parameters:** Impact parameter, duration, depth
   - **Physical Properties:** Radius ratio, signal-to-noise

2. **Click "Predict Exoplanet Classification"** to get:
   - Prediction (Confirmed Planet/Candidate/False Positive)
   - Confidence scores for all three classes
   - Visual progress bars

## 📁 File Structure

```
exoplanet-classifier/
├── README.md           # This file
├── train_model.py      # Model training script
├── streamlit_app.py    # Web application
├── koi.csv            # Kepler dataset
├── model.joblib       # Trained model (after training)
└── fixed_test.py      # Test script for model validation
```

## 🔬 Data Features

The model uses 121 features from the KOI dataset, including:

- **Transit Parameters:** Period, duration, depth, ingress
- **Error Values:** Upper and lower uncertainties
- **Physical Properties:** Planet/stellar radii, temperature
- **Quality Flags:** False positive detection flags
- **Signal Quality:** S/N ratio, transit counts

## 🔧 Model Details

- **Algorithm:** Random Forest Classifier
- **Preprocessing:** StandardScaler + MedianImputer
- **Cross-validation:** Train/test split (80/20)
- **Random State:** 42 (for reproducibility)

## 🐛 Troubleshooting

### Common Issues

1. **CSV Parsing Error:** The dataset has comment lines starting with `#` - this is handled automatically
2. **Feature Mismatch:** Ensure the model and app use the same feature set
3. **Import Errors:** Make sure all required packages are installed

### Test the Model

Run the test script to verify everything works:

```bash
python fixed_test.py
```

This should output:
- Model loads successfully
- Prediction: Candidate (for default zeros input)
- Confidence scores
- Feature count: 121

## 📈 Model Interpretation

The model predicts three classes:
- **Class 0 (False Positive):** Objects that are not planets
- **Class 1 (Candidate):** Potential planets requiring verification  
- **Class 2 (Confirmed Planet):** Verified exoplanets

The confidence scores show the model's probability for each class.

## 🤝 Contributing

Feel free to improve the model or interface:
- Add more features from the dataset
- Experiment with different algorithms
- Enhance the web interface
- Add validation and error handling

## 📋 License

This project uses the Kepler Object of Interest dataset from NASA's Exoplanet Archive.

---

*Built with ❤️ for exoplanet discovery and classification*
