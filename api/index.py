"""
Vercel Serverless Function Entry Point for FastAPI
This file exports the FastAPI app for Vercel's Python runtime
"""
from api.main import app

# Vercel will use this as the ASGI application
handler = app
