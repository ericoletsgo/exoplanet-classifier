# SPA Routing Fix

## Problem
When navigating to routes like `/retrain` via the React Router (clicking navigation), the page worked fine. However, when reloading the page directly at `/retrain`, users would get a 404 error.

## Root Cause
This is a common issue with Single Page Applications (SPAs). When you reload a page at `/retrain`, the server tries to find a file at that path instead of serving the React app's `index.html` file. The React Router only works after the JavaScript has loaded.

## Solution Applied

### 1. Vercel Configuration Updates

**Root `vercel.json`:**
- Added explicit routes for each frontend page (`/predict`, `/batch`, `/retrain`, `/datasets`, `/metrics`)
- Each route now serves `/frontend/dist/index.html` to ensure React Router can handle the routing

**Frontend `vercel.json`:**
- Updated the catch-all route `/(.*)` to serve `/dist/index.html` instead of `/dist/$1`
- This ensures any unmatched route serves the React app

### 2. FastAPI Backend Updates

**`api/main.py`:**
- Added explicit route handlers for each frontend page
- Each handler serves `static/index.html` when the static directory exists
- Updated the catch-all route to exclude frontend routes and serve the React app for other paths

## Files Modified

1. `vercel.json` - Root Vercel configuration
2. `frontend/vercel.json` - Frontend-specific Vercel configuration  
3. `api/main.py` - FastAPI backend routing

## How It Works Now

1. **Client-side navigation**: React Router handles routing normally
2. **Direct URL access**: Server serves `index.html` for frontend routes, React Router takes over
3. **Page refresh**: Server serves `index.html`, React Router restores the correct page
4. **API routes**: Still work normally and are excluded from frontend routing

## Testing

To test the fix:

1. **Development**: 
   - Start the API: `cd api && python main.py`
   - Start the frontend: `cd frontend && npm run dev`
   - Navigate to `http://localhost:5173/retrain`
   - Refresh the page - should work without 404

2. **Production**:
   - Deploy to Vercel
   - Navigate to your deployed URL + `/retrain`
   - Refresh the page - should work without 404

## Additional Notes

- The fix maintains backward compatibility with existing API endpoints
- Static file serving continues to work normally
- The solution works for both development and production environments
- All frontend routes (`/predict`, `/batch`, `/retrain`, `/datasets`, `/metrics`) are now properly handled
