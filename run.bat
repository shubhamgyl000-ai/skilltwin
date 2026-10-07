@echo off
start cmd /k "cd backend && python -m pip install -r requirements.txt && python -m uvicorn main:app --reload"
timeout /t 3 >nul
start frontend/index.html
