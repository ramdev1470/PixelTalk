@echo off
REM One-time setup for PixelTalk on Windows: creates a virtual environment, installs everything,
REM downloads the models (about 6 GB, needs internet) and runs a real-model check.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" python -m venv .venv
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 goto :fail
python download_models.py
if errorlevel 1 goto :fail
python smoke_test.py
echo.
echo Setup finished. Start the app with run_pixeltalk.bat
pause
exit /b 0
:fail
echo.
echo Setup stopped because a step failed. Read the messages above, fix the problem and run this file again.
pause
exit /b 1
