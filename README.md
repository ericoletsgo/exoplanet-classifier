## **Only the Scaffold and Random Forest Algo for idea**

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

## Usage

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

The model uses 121 features from the KOI dataset, including:

- **Transit Parameters:** Period, duration, depth, ingress
- **Error Values:** Upper and lower uncertainties
- **Physical Properties:** Planet/stellar radii, temperature
- **Quality Flags:** False positive detection flags
- **Signal Quality:** S/N ratio, transit counts
