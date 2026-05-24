#!/bin/bash
set -e

# Go to the directory this script lives in (like %~dp0)
cd "$(cd "$(dirname "$0")" && pwd)"

if [ ! -d ".venv" ]; then
  echo "Setting up environment for the first time..."
  # Prefer python3.12 if available, else fall back to python3
  if command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv .venv
  else
    python3 -m venv .venv
  fi
fi

# Activate venv
source ".venv/bin/activate"

python -m pip install --upgrade pip setuptools wheel websockets -q

# Open the GUI in the default browser (non-blocking)
open "index.html"

# Start the PACE backend (blocking, like the .bat)
python "dev-pace.py"

read -r -p "Press Enter to exit..."
