@echo off
title SuctionSailLab - Dashboard

REM Navigate to the project root directory
cd /d "%~dp0"

REM Activate the Python virtual environment
call "dashboard_matlab\Scripts\activate.bat"

REM Check whether the virtual environment was activated
if errorlevel 1 (
    echo ERROR: Could not activate the Python virtual environment.
    pause
    exit /b 1
)

REM Navigate to the dashboard directory
cd /d "%~dp0Dashboard"

REM Start the Streamlit dashboard
streamlit run app.py

pause
