# Multi-stage build for React + FastAPI application

# Stage 1: Build React frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
# Set environment variable for production build
ENV VITE_API_URL=/api
RUN npm run build

# Stage 2: Python backend
FROM python:3.11-slim AS backend
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY api/ ./api/
COPY data/ ./data/
COPY properly_trained_model.joblib ./

# Copy built frontend
COPY --from=frontend-builder /app/frontend/dist ./static

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check (commented out for Render compatibility)
# HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:8000/ || exit 1

# Start the application
CMD ["sh", "-c", "python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
