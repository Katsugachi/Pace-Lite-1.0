# Pace-Lite-1.0
Highly capable open source local AI model <br>
Lite version of pace designed to have a GUI<br>
Added internet search, allowing for more relevant answers. <br><br>
Debugging Script:
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python pace.py
```
## Main Pace Repo
https://github.com/Katsugachi/Pace-1.0/tree/main
## Start
Basically just download entire thing and unzip. <br><br>Run `pace-windows.bat` for Windows or `pace-mac.command` for macOS. Those launchers create the virtual environment, install the WebSocket dependency, open the GUI, and start the backend. <br>
Additionally, if huggingface is blocked, you can download gemma from the releases section of this repo and drop it straight into .pace-agent
## Model setup note
The GUI can now connect even if the local model is unavailable, but full responses still require `llama-cpp-python` to be installed successfully. If the backend starts in degraded mode, follow the setup instructions printed in the terminal, then restart `dev-pace.py`.

## Internet mode toggle
PACE can now run in two internet modes:
- **Web On**: uses live internet research (tutorials, coding docs, CDNs, and other web sources).
- **Web Off**: uses only local model knowledge, better from creative writing and tasks not requiring web sources

You can toggle this in both places:
- **GUI**: use the `Web On` / `Web Off` button in the header.
- **Terminal**: use `/internet on`, `/internet off`, `/internet toggle`, or `/internet status`.

When Web mode is on, Pace now only uses internet research when the prompt appears to need external/current references (for example docs, tutorials, API/version/current-event lookups). Simple greetings and creative/narrative writing stay local.
