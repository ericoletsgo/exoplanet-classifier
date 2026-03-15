# Vercel Full-Stack Deployment Guide

## 🚀 OPTIMIZED FOR MAXIMUM PERFORMANCE

This setup deploys everything on Vercel for:
- ✅ **Fastest cold starts** (1-2 seconds)
- ✅ **Single deployment** (no CORS issues)
- ✅ **Global CDN** (fast worldwide)
- ✅ **Automatic scaling**
- ✅ **Free tier** (generous limits)

## 📁 Project Structure
```
exoplanet-classifier/
├── api/
│   └── main.py              # FastAPI backend
├── frontend/
│   ├── src/                 # React frontend
│   ├── package.json
│   └── vite.config.ts
├── data/                    # ML datasets
├── *.joblib                 # Trained models
├── requirements.txt         # Python dependencies
├── vercel.json             # Vercel configuration
└── README.md
```

## 🚀 Deployment Steps

### 1. Install Vercel CLI
```bash
npm i -g vercel
```

### 2. Login to Vercel
```bash
vercel login
```

### 3. Deploy
```bash
vercel --prod
```

### 4. Set Environment Variables (if needed)
In Vercel dashboard:
- `VITE_API_URL=/api`

## ⚡ Performance Optimizations Applied

### Backend Optimizations:
- ✅ **Caching**: Metrics (1h), Correlations (2h), Models (30m)
- ✅ **Smaller samples**: 1000 vs 5000+ samples
- ✅ **Optimized timeouts**: 60s for heavy operations
- ✅ **Lazy loading**: Heavy data only when needed

### Frontend Optimizations:
- ✅ **Code splitting**: Vendor, charts, router chunks
- ✅ **Terser minification**: Smaller bundle size
- ✅ **Optimized API calls**: Reduced unnecessary requests
- ✅ **Lazy loading**: Correlations only when tab accessed

### Deployment Optimizations:
- ✅ **Single platform**: No cross-origin requests
- ✅ **Global CDN**: Fast worldwide delivery
- ✅ **Serverless**: Auto-scaling, pay-per-use
- ✅ **Edge functions**: Fast API responses

## 📊 Expected Performance

| Metric | Before (Render+Vercel) | After (Vercel Full-Stack) |
|--------|----------------------|---------------------------|
| **Cold Start** | 30+ seconds | 1-2 seconds |
| **First Load** | 30+ seconds | 2-5 seconds |
| **Cached Requests** | 5-10 seconds | <1 second |
| **Global Speed** | Variable | Fast worldwide |

## 🔧 Local Development

### Backend:
```bash
cd api
python main.py
```

### Frontend:
```bash
cd frontend
npm run dev
```

## 🎯 Benefits for Resume

- ✅ **Modern deployment**: Vercel is industry standard
- ✅ **Full-stack**: Shows full application skills
- ✅ **Performance**: Demonstrates optimization skills
- ✅ **Production-ready**: Real-world deployment experience
