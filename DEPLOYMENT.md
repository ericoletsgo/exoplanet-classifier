# 🚀 Deployment Guide

This guide covers multiple deployment options for your Exoplanet Classifier React/FastAPI application.

## 🐳 Docker Deployment (Recommended)

### Local Docker
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t exoplanet-classifier .
docker run -p 8000:8000 exoplanet-classifier
```

### Cloud Docker (Railway, Render, etc.)
1. Push your code to GitHub
2. Connect your repository to your cloud provider
3. Use the provided `Dockerfile` and `docker-compose.yml`

---

## 🌐 Railway Deployment (Easiest)

Railway automatically detects the `railway.json` configuration.

### Steps:
1. **Sign up** at [railway.app](https://railway.app)
2. **Connect GitHub** repository
3. **Deploy** - Railway will automatically:
   - Build the React frontend
   - Install Python dependencies
   - Start the FastAPI server
   - Serve both frontend and backend

### Environment Variables (if needed):
- `PORT` - Railway sets this automatically
- `PYTHONPATH=/app`

---

## ⚡ Vercel + Railway (Hybrid)

Deploy frontend to Vercel and backend to Railway for optimal performance.

### Frontend (Vercel):
1. **Connect** your GitHub repo to [vercel.com](https://vercel.com)
2. **Set root directory** to `frontend/`
3. **Update** `frontend/vercel.json` with your Railway backend URL
4. **Deploy**

### Backend (Railway):
1. Deploy backend using Railway (see above)
2. **Copy** your Railway app URL
3. **Update** `frontend/vercel.json` to point to your Railway URL

---

## 🐍 Heroku Deployment

1. **Install** Heroku CLI
2. **Login** and create app:
   ```bash
   heroku login
   heroku create your-app-name
   ```
3. **Deploy**:
   ```bash
   git push heroku main
   ```

---

## ☁️ AWS/GCP/Azure

### Option 1: Container Service
- Use the provided `Dockerfile`
- Deploy to ECS (AWS), Cloud Run (GCP), or Container Instances (Azure)

### Option 2: Serverless
- **Backend**: Deploy FastAPI to AWS Lambda, Google Cloud Functions, or Azure Functions
- **Frontend**: Deploy React build to S3 + CloudFront (AWS), Cloud Storage (GCP), or Blob Storage (Azure)

---

## 🔧 Environment Configuration

### Production Environment Variables:
```bash
# Backend
PYTHONPATH=/app
PYTHONUNBUFFERED=1
PORT=8000

# Frontend (if separate deployment)
VITE_API_URL=https://your-backend-url.railway.app
```

### Update API URL in Frontend:
If deploying separately, update `frontend/src/lib/api.ts`:
```typescript
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
```

---

## 📊 Monitoring & Health Checks

Your deployment includes:
- **Health check** endpoint at `/`
- **API documentation** at `/docs`
- **Automatic restarts** on failure
- **Static file serving** for React app

---

## 🚨 Troubleshooting

### Common Issues:

1. **Frontend not loading**:
   - Ensure React build files exist in `static/` directory
   - Check that static file serving is enabled

2. **API not responding**:
   - Verify all Python dependencies are installed
   - Check that model file exists: `properly_trained_model.joblib`

3. **CORS errors**:
   - Backend includes CORS middleware for `localhost:5173`
   - For production, update CORS origins in `api/main.py`

### Logs:
- **Railway**: Check deployment logs in dashboard
- **Heroku**: `heroku logs --tail`
- **Docker**: `docker logs <container-id>`

---

## 🎯 Recommended Deployment

**For beginners**: Use **Railway** - it's the easiest and handles everything automatically.

**For production**: Use **Railway + Vercel** - backend on Railway, frontend on Vercel for optimal performance.

**For enterprise**: Use **AWS/GCP/Azure** with container services for full control.
