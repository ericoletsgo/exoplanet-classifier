# PowerShell script to start both backend and frontend services
# Run this from the project root directory

Write-Host "Starting Exoplanet Classifier Services..." -ForegroundColor Cyan

# Start Backend API
Write-Host "`nStarting FastAPI Backend on http://localhost:8000..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot\api'; Write-Host 'Backend API Server' -ForegroundColor Yellow; python main.py"

# Wait a moment for backend to start
Start-Sleep -Seconds 2

# Start Frontend
Write-Host "Starting React Frontend on http://localhost:5173..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot\frontend'; Write-Host 'Frontend Dev Server' -ForegroundColor Yellow; npm run dev"

Write-Host "`nServices starting in separate windows..." -ForegroundColor Cyan
Write-Host "Backend API: http://localhost:8000" -ForegroundColor White
Write-Host "Frontend UI: http://localhost:5173" -ForegroundColor White
Write-Host "API Docs: http://localhost:8000/docs" -ForegroundColor White
