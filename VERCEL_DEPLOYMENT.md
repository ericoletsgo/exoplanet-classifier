# Deploying to Vercel

This guide explains how to deploy the Exoplanet Classifier to Vercel with the FastAPI backend.

## Prerequisites

1. A Vercel account (free tier works)
2. Your model file hosted externally (see Model Hosting section below)

## Model Hosting

Since Vercel has a 50MB deployment size limit, you need to host your model file externally. Options:

### Option 1: GitHub Releases (Recommended - Free)
1. Go to your GitHub repository
2. Click "Releases" → "Create a new release"
3. Upload your `balanced_model_20251005_115605.joblib` file
4. Publish the release
5. Copy the direct download URL (right-click the file → Copy link address)

### Option 2: Vercel Blob Storage
1. Install Vercel CLI: `npm i -g vercel`
2. Upload model: `vercel blob upload balanced_model_20251005_115605.joblib`
3. Copy the returned URL

### Option 3: AWS S3 / Google Cloud Storage
1. Upload your model file to S3/GCS
2. Make it publicly accessible (or use signed URLs)
3. Copy the public URL

## Deployment Steps

### 1. Connect to Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# Login to Vercel
vercel login

# Deploy
vercel
```

### 2. Set Environment Variables

In your Vercel project dashboard:

1. Go to **Settings** → **Environment Variables**
2. Add the following variable:
   - **Name**: `MODEL_URL`
   - **Value**: Your model file URL (from step above)
   - **Environment**: Production, Preview, Development

### 3. Deploy

```bash
# Deploy to production
vercel --prod
```

## Configuration Files

The following files configure Vercel deployment:

- **`vercel.json`**: Configures builds and routing
  - Builds the Python API as a serverless function
  - Builds the React frontend as static files
  - Routes `/api/*` to the Python backend
  
- **`.vercelignore`**: Excludes large files from deployment
  - Model files (*.joblib)
  - Data files (*.csv)
  - Development files

- **`requirements.txt`**: Python dependencies for the API

## Local Development

For local development, both servers need to run:

```bash
# Terminal 1: Start API
start_api.bat

# Terminal 2: Start Frontend
start_frontend.bat
```

The frontend will proxy API requests to `localhost:8000` in development mode.

## Troubleshooting

### "Method Not Allowed" Error
- **Cause**: API is not running or not accessible
- **Solution**: Ensure MODEL_URL environment variable is set in Vercel

### Model Loading Fails
- **Cause**: MODEL_URL is incorrect or file is not accessible
- **Solution**: 
  1. Test the URL in your browser - it should download the file
  2. Check Vercel function logs for error messages
  3. Ensure the URL is publicly accessible

### Deployment Size Too Large
- **Cause**: Large files included in deployment
- **Solution**: Check `.vercelignore` is properly excluding large files

### Function Timeout
- **Cause**: Model loading takes too long (Vercel has 10s timeout on free tier)
- **Solution**: 
  1. Upgrade to Vercel Pro for 60s timeout
  2. Use a smaller/compressed model
  3. Consider using Vercel Edge Config for model caching

## Architecture

```
┌─────────────────────────────────────────┐
│           Vercel Deployment             │
├─────────────────────────────────────────┤
│                                         │
│  Frontend (Static)                      │
│  ├─ React + TypeScript                  │
│  ├─ Vite build                          │
│  └─ Served from /                       │
│                                         │
│  Backend (Serverless Function)          │
│  ├─ FastAPI                             │
│  ├─ Python 3.9+                         │
│  ├─ Loads model from MODEL_URL          │
│  └─ Served from /api/*                  │
│                                         │
└─────────────────────────────────────────┘
         │
         ├─→ External Model Storage
         │   (GitHub Releases / S3 / Vercel Blob)
         │
         └─→ Users
```

## Performance Notes

- **Cold starts**: First request may be slow (5-10s) as Vercel spins up the function
- **Model loading**: Happens once per cold start, cached for subsequent requests
- **Timeout limits**: 
  - Free tier: 10 seconds
  - Pro tier: 60 seconds
- **Memory limits**: 1024 MB (sufficient for most models)

## Cost Estimate

- **Vercel Free Tier**: 
  - 100 GB bandwidth
  - 100 GB-hours serverless function execution
  - Sufficient for personal projects and demos

- **Vercel Pro** ($20/month):
  - 1 TB bandwidth
  - 1000 GB-hours execution
  - 60s function timeout
  - Better for production use
