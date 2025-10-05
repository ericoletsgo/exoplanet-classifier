# Quick Start Guide - Exoplanet Classifier Frontend

Get the full-stack application running in 3 steps.

## Step 1: Install Dependencies

```bash
# Install Python dependencies (from project root)
pip install -r requirements.txt

# Install Node.js dependencies
cd frontend
npm install
cd ..
```

## Step 2: Start the Services

### Windows (PowerShell)
```bash
.\start.ps1
```

### Manual Start (Any OS)

**Terminal 1 - Backend:**
```bash
cd api
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

## Step 3: Access the Application

- **Frontend UI**: http://localhost:5173
- **API Docs**: http://localhost:8000/docs
- **Backend API**: http://localhost:8000

## What You Can Do

### 1. Home Page
- View system status
- Check model information
- Quick navigation

### 2. Predict Page
- Enter feature values manually
- Load example data from dataset
- Get classification with confidence scores
- View probability distribution

### 3. Metrics Page
- Model accuracy, precision, recall, F1 score
- Confusion matrix visualization
- ROC curves for each class
- Top 15 feature importances

### 4. Datasets Page
- Browse KOI, K2, TOI datasets
- Filter by disposition (Confirmed/Candidate/False Positive)
- Pagination support
- View first 10 columns of each dataset

## Troubleshooting

### Backend won't start
```bash
# Check if model file exists
ls properly_trained_model.joblib

# If missing, train a model first
python fast_proper_training.py
```

### Frontend won't start
```bash
# Clear cache and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### API connection failed
- Ensure backend is running on port 8000
- Check browser console for CORS errors
- Verify `vite.config.ts` proxy settings

### Port already in use
```bash
# Backend - change port in api/main.py
uvicorn.run(app, host="0.0.0.0", port=8001)

# Frontend - change port in vite.config.ts
server: { port: 3000 }
```

## Next Steps

- Read [FRONTEND_SETUP.md](FRONTEND_SETUP.md) for detailed configuration
- Explore API at http://localhost:8000/docs
- Check [RETRAINING_GUIDE.md](RETRAINING_GUIDE.md) for model training

## Tech Stack

- **Backend**: FastAPI + Python 3.8+
- **Frontend**: React 18 + TypeScript + Vite
- **Styling**: Tailwind CSS
- **Charts**: Chart.js + react-chartjs-2
- **Icons**: Lucide React
- **ML**: scikit-learn, XGBoost, LightGBM

## File Structure

```
exoplanet-classifier/
├── api/main.py              ← Backend API
├── frontend/
│   ├── src/
│   │   ├── pages/           ← React pages
│   │   ├── lib/api.ts       ← API client
│   │   └── App.tsx          ← Main app
│   └── package.json
├── properly_trained_model.joblib
├── koi.csv, k2.csv, toi.csv
└── start.ps1                ← Startup script
```

Enjoy exploring exoplanet data! 🚀🪐
