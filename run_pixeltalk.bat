@echo off
REM Starts PixelTalk on Windows. Double-click or run from a terminal in this folder.
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
streamlit run app.py
pause
