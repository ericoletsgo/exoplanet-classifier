# Quick Start Guide - Model Retraining

## 🚀 Start the App

```bash
cd /Users/yaoruixu/dev/nasahackathon-2025/exoplanet-classifier
streamlit run enhanced_exoplanet_classifier.py
```

## 📝 Train Your First Model (5 Steps)

### Step 1: Navigate to Model Retraining
- Click **"Model Retraining"** in the sidebar

### Step 2: Configure Model
- **Tab:** "Train New Model"
- **Model Name:** `My_First_Model`
- **Test Size:** 20% (default)
- **Description:** "Testing the retraining pipeline"

### Step 3: Upload Data
- Click **"Upload Training Data (CSV)"**
- Select `koi.csv` from your project directory
- Wait for data preview to appear

### Step 4: Configure Target Column
- **Select Target Column:** Choose `koi_disposition`
- **Verify Mapping:**
  - CONFIRMED → Confirmed Planet ✓
  - CANDIDATE → Candidate ✓
  - FALSE POSITIVE → False Positive ✓
- System auto-detects these values!

### Step 5: Review & Train
- Check that 23 relevant features were detected
- Click **"🚀 Train Model"** button
- Wait 2-3 minutes for training to complete

### Step 6: View Results
- See accuracy, precision, recall, F1 score
- View confusion matrix
- Model is automatically saved!

## 🎯 Use Your Trained Model

### Step 1: Go to Classification Page
- Click **"Classification"** in sidebar

### Step 2: Select Your Model
- In sidebar, find **"Model Selection"**
- Choose your model from dropdown
- See model info displayed

### Step 3: Make Predictions
- Enter exoplanet parameters
- Click **"Classify Object"**
- Get prediction with confidence

## 📊 Compare Models

### Navigate to Model Evaluations
- **Model Retraining** → **"Model Evaluations"** tab
- View comparison table
- Select any model for detailed view
- Compare confusion matrices

## 🧪 Test the System

```bash
# Run automated tests
python test_retraining.py

# Expected output:
# ✅ All tests passed!
# Model trained with ~82% accuracy
```

## 📋 What Features Are Used?

The system automatically selects **23 features** in 3 categories:

### Stellar (7)
Position, temperature, radius, mass, magnitude

### Orbital (10)
Period, duration, depth, radius, eccentricity, etc.

### Signal (6)
SNR, transit counts, event statistics

## 💡 Tips

1. **Use the KOI dataset** (`koi.csv`) for your first model
2. **Custom target columns** - You can use any column name and map custom values to the three categories
3. **Check the comparison table** to see which model performs best
4. **Add descriptions** to remember what makes each model unique
5. **Delete old models** you don't need anymore

## 🔧 Using Custom Data

If your CSV has different column names or values:

1. **Any target column name works** - Select it from the dropdown
2. **Map your values** - Use the multiselect boxes to assign values to:
   - Confirmed Planet (e.g., "planet", "confirmed", "validated")
   - Candidate (e.g., "candidate", "maybe", "potential")
   - False Positive (e.g., "false", "noise", "artifact")
3. **Multiple values per category** - You can map several values to the same category
4. **Unmapped values excluded** - Any values not mapped will be filtered out

## 🐛 Troubleshooting

**App won't start?**
```bash
# Make sure you're in the right directory
cd /Users/yaoruixu/dev/nasahackathon-2025/exoplanet-classifier

# Activate virtual environment if needed
source venv/bin/activate
```

**Training fails?**
- Select the correct target column from the dropdown
- Map at least some values to the three categories
- Ensure you have enough samples in each category

**No models showing?**
- Train at least one model first
- Check `models/` directory exists

## 📚 More Info

- **Full Guide:** See `RETRAINING_GUIDE.md`
- **Implementation Details:** See `IMPLEMENTATION_SUMMARY.md`
- **Test Code:** See `test_retraining.py`

## ✅ Success Checklist

- [ ] App starts successfully
- [ ] Can navigate to Model Retraining page
- [ ] Can upload koi.csv
- [ ] Training completes without errors
- [ ] Can see model in evaluations tab
- [ ] Can select model in classification page
- [ ] Can make predictions with selected model

---

**Ready to go!** 🎉 Start with the 5-step training guide above.
