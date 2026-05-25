# Pace-Lite-1.0
Lite version of pace designed to have a GUI<br>
Added internet search, allowing for more relevant answers.
## Main Pace Repo
https://github.com/Katsugachi/Pace-1.0/tree/main
## Start
Basically just download entire thing and unzip. <br><br>Run `pace-windows.bat` for Windows or `pace-mac.command` for macOS. Those launchers create the virtual environment, install the WebSocket dependency, open the GUI, and start the backend.

## Model setup note
The GUI can now connect even if the local model is unavailable, but full responses still require `llama-cpp-python` to be installed successfully. If the backend starts in degraded mode, follow the setup instructions printed in the terminal, then restart `dev-pace.py`.
