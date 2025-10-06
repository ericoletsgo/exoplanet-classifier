@echo off
echo ========================================
echo   Exoplanet Classifier - Vercel Deploy
echo ========================================
echo.

REM Check if Vercel CLI is installed
where vercel >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Vercel CLI is not installed.
    echo.
    echo Please install it first:
    echo   npm install -g vercel
    echo.
    pause
    exit /b 1
)

echo [INFO] Vercel CLI found!
echo.

REM Check if model file exists
if not exist "balanced_model_20251005_115605.joblib" (
    echo [WARNING] Model file not found locally.
    echo This is OK if you've already uploaded it to external storage.
    echo.
    echo If you haven't uploaded your model yet, please:
    echo   1. Upload balanced_model_20251005_115605.joblib to GitHub Releases, S3, or Vercel Blob
    echo   2. Set MODEL_URL environment variable in Vercel dashboard
    echo.
    echo See VERCEL_DEPLOYMENT.md for detailed instructions.
    echo.
)

echo [INFO] Starting Vercel deployment...
echo.
echo IMPORTANT: After deployment, remember to set the MODEL_URL environment variable!
echo.
pause

REM Deploy to Vercel
vercel --prod

echo.
echo ========================================
echo   Deployment Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Go to your Vercel dashboard
echo   2. Navigate to Settings ^> Environment Variables
echo   3. Add MODEL_URL with your model file URL
echo   4. Redeploy to apply the environment variable
echo.
echo See VERCEL_DEPLOYMENT.md for more details.
echo.
pause
