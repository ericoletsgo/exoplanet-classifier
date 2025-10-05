# Exoplanet Classifier - Full Stack Setup Guide

This guide will help you set up and run both the FastAPI backend and React frontend.

## Architecture

```
exoplanet-classifier/
├── api/                    # FastAPI backend
│   └── main.py
├── frontend/               # React + TypeScript frontend
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
├── enhanced_exoplanet_classifier.py
├── properly_trained_model.joblib
├── koi.csv, k2.csv, toi.csv
└── requirements.txt
```

## Prerequisites

- **Python 3.8+** with pip
- **Node.js 18+** with npm
- Virtual environment (recommended)

## Backend Setup (FastAPI)

### 1. Install Python Dependencies

```bash
# From project root
pip install -r requirements.txt
```

This installs:
- pandas, scikit-learn, joblib, numpy
- fastapi, uvicorn, pydantic
- xgboost, lightgbm
- streamlit, plotly, torch

### 2. Start the API Server

```bash
# From project root
cd api
python main.py
```

Or using uvicorn directly:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 3. Test the API

Visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI).

Key endpoints:
- `GET /` - Health check
- `GET /features` - List all features
- `POST /predict` - Make predictions
- `GET /metrics` - Model performance metrics
- `GET /datasets/{name}` - Browse datasets (koi, k2, toi)

## Frontend Setup (React)

### 1. Install Node Dependencies

```bash
# From project root
cd frontend
npm install
```

### 2. Start the Dev Server

```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

### 3. Build for Production

```bash
npm run build
npm run preview
```

## Running Both Services

### Option 1: Two Terminal Windows

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

### Option 2: Using a Process Manager

Create a `start.ps1` (PowerShell) script:

```powershell
# start.ps1
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd api; python main.py"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; npm run dev"
```

Then run:
```bash
.\start.ps1
```

## Usage

1. **Open the frontend**: Navigate to `http://localhost:5173`
2. **Check API status**: The home page shows connection status
3. **Make predictions**: Go to Predict page, enter feature values, click Predict
4. **View metrics**: See model performance, ROC curves, feature importance
5. **Browse datasets**: Explore KOI, K2, TOI datasets with filtering

## API Configuration

The frontend is configured to proxy API requests through Vite:

```typescript
// vite.config.ts
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
      rewrite: (path) => path.replace(/^\/api/, '')
    }
  }
}
```

If your backend runs on a different port, update this configuration.

## CORS Configuration

The backend allows requests from the frontend:

```python
# api/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Troubleshooting

### Backend Issues

**Model not found:**
- Ensure `properly_trained_model.joblib` exists in the project root
- Check the `MODEL_PATH` constant in `api/main.py`

**Dataset not found:**
- Ensure `koi.csv`, `k2.csv`, `toi.csv` exist in the project root

**Import errors:**
- Run `pip install -r requirements.txt` again
- Check Python version (3.8+)

### Frontend Issues

**API connection failed:**
- Ensure backend is running on port 8000
- Check browser console for CORS errors
- Verify proxy configuration in `vite.config.ts`

**Build errors:**
- Delete `node_modules` and run `npm install` again
- Clear npm cache: `npm cache clean --force`

**Port already in use:**
- Change port in `vite.config.ts`:
  ```typescript
  server: { port: 3000 }
  ```

## Development Tips

### Hot Reload

Both services support hot reload:
- **Backend**: Use `uvicorn api.main:app --reload`
- **Frontend**: Vite automatically reloads on file changes

### API Testing

Use the Swagger UI at `http://localhost:8000/docs` to test endpoints before integrating with the frontend.

### Debugging

**Backend:**
- Add print statements or use Python debugger
- Check terminal output for errors

**Frontend:**
- Use browser DevTools (F12)
- Check Network tab for API requests
- Use React DevTools extension

## Production Deployment

### Backend

Deploy to a cloud platform (e.g., Railway, Render, AWS):

```bash
# Ensure requirements.txt is up to date
pip freeze > requirements.txt

# Use gunicorn for production
pip install gunicorn
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Frontend

Build and deploy to Vercel, Netlify, or similar:

```bash
npm run build
# Deploy the 'dist' folder
```

Update API URL in production:
- Set environment variable `VITE_API_URL`
- Update `api.ts` to use `import.meta.env.VITE_API_URL`

## Next Steps

- Add authentication (JWT tokens)
- Implement model retraining endpoint
- Add more visualizations
- Create user management
- Add dataset upload functionality

## Support

For issues or questions:
1. Check the API docs at `/docs`
2. Review browser console for errors
3. Check backend logs for exceptions
