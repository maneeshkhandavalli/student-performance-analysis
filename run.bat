@echo off
cd /d %~dp0
echo Installing requirements (first time only takes a minute)...
python -m pip install -q -r requirements.txt
echo.
echo Starting Student Performance Prediction System at http://127.0.0.1:5000
echo Keep this window open. Press Ctrl+C to stop.
start "" cmd /c "timeout /t 6 >nul & start http://127.0.0.1:5000"
python app.py
pause
