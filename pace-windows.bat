@echo off
cd /d "%~dp0"

if not exist ".venv" (
    echo Setting up environment for the first time...
    py -3.12 -m venv .venv
)

call .venv\Scripts\activate.bat

python -m pip install --upgrade pip setuptools wheel websockets -q

:: Open the GUI in the default browser (non-blocking)
start "" "%~dp0index.html"

:: Start the PACE backend
python dev-pace.py

pause
