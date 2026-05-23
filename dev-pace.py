# PACE 1.0: Local Agentic Terminal + GUI WebSocket Server
import os
import sys
import subprocess
import urllib.request
import re
import threading
import json
import asyncio
from pathlib import Path

try:
    import llama_cpp
    has_llama_cpp = True
    llama_import_error = None
except Exception as e:
    has_llama_cpp = False
    llama_import_error = e

try:
    import websockets
    has_ws = True
except ImportError:
    has_ws = False

try:
    import zipfile
    import xml.etree.ElementTree as ET
    has_pdf_deps = True
except Exception:
    has_pdf_deps = False

# Constants
MODEL_NAME = "gemma-3-1b-it-Q4_K_M.gguf"
MODEL_URL = "https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf"
PACE_DIR_NAME = ".pace_agent"

# Shared state for WebSocket server
_ws_state = {
    "llm": None,
    "history": [],
    "system_prompt": "",
    "lock": threading.Lock(),
}

# Colors for terminal
class Colors:
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    WHITE = "\033[37m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"

ASCII_ART = f"""{Colors.WHITE}{Colors.BOLD}
 ██████╗  █████╗  ██████╗███████╗   
 ██╔══██╗██╔══██╗██╔════╝██╔════╝  
 ██████╔╝███████║██║     █████╗    
 ██╔═══╝ ██╔══██║██║     ██╔══╝    
 ██║     ██║  ██║╚██████╗███████╗  
 ╚═╝     ╚═╝  ╚═╝ ╚═════╝╚══════╝   
{Colors.WHITE}Local Lite AI Model {Colors.YELLOW}[Gemma 3 1B]{Colors.RESET}
"""

def install_dependencies():
    global has_llama_cpp

    if has_llama_cpp:
        return

    print(f"{Colors.RED}llama-cpp-python is not installed or failed to import in this Python environment.{Colors.RESET}")

    if llama_import_error is not None:
        print()
        print(f"{Colors.YELLOW}Import error:{Colors.RESET} {llama_import_error}")

    print()
    print(f"{Colors.YELLOW}PACE will not auto-install it because your system needs a manual/custom install.{Colors.RESET}")
    print()
    print("Run these commands in PowerShell while your .venv is active:")
    print()
    print(f"{Colors.CYAN}.\\.venv\\Scripts\\Activate.ps1{Colors.RESET}")
    print()
    print(f"{Colors.CYAN}mkdir C:\\t -Force{Colors.RESET}")
    print(f"{Colors.CYAN}$env:TEMP=\"C:\\t\"{Colors.RESET}")
    print(f"{Colors.CYAN}$env:TMP=\"C:\\t\"{Colors.RESET}")
    print()
    print(f"{Colors.CYAN}python -m pip install cmake ninja{Colors.RESET}")
    print()
    print("If you are on Windows ARM64, try:")
    print()
    print(f"{Colors.CYAN}$env:CMAKE_ARGS=\"-DGGML_NATIVE=OFF\"{Colors.RESET}")
    print(f"{Colors.CYAN}$env:FORCE_CMAKE=\"1\"{Colors.RESET}")
    print(f"{Colors.CYAN}python -m pip install --no-cache-dir --force-reinstall --no-binary llama-cpp-python llama-cpp-python{Colors.RESET}")
    print()
    print("If you get compiler errors, install Visual Studio Build Tools 2022 with:")
    print("- Desktop development with C++")
    print("- C++ CMake tools for Windows")
    print("- Windows SDK")
    print("- ARM64 build tools if you are on Windows ARM64")
    print()
    print("After installation succeeds, test with:")
    print()
    print(f"{Colors.CYAN}python -c \"from llama_cpp import Llama; print('llama-cpp-python works')\"{Colors.RESET}")
    print()
    print("Then run:")
    print()
    print(f"{Colors.CYAN}python pace.py{Colors.RESET}")
    print()

    sys.exit(1)

def get_pace_dir():
    base_dir = Path(__file__).resolve().parent
    pace_dir = base_dir / PACE_DIR_NAME
    pace_dir.mkdir(exist_ok=True)
    return pace_dir

def download_model(pace_dir):
    model_path = pace_dir / MODEL_NAME

    if model_path.exists():
        return model_path

    print(f"\n{Colors.YELLOW}Downloading Gemma 3 1B model ({MODEL_NAME})...{Colors.RESET}")
    print(f"{Colors.BLUE}Source: {MODEL_URL}{Colors.RESET}")
    print(f"{Colors.BLUE}Size: about 750 MB. This is a one-time small download. You must be on Home Wifi{Colors.RESET}\n")

    def progress_hook(count, block_size, total_size):
        progress = count * block_size

        if total_size <= 0:
            sys.stdout.write(f"\r{Colors.CYAN}Downloaded {progress / (1024 * 1024):.1f} MB{Colors.RESET}")
            sys.stdout.flush()
            return

        percent = min(100, int(progress * 100 / total_size))
        bar_length = 40
        filled_length = int(bar_length * percent / 100)
        bar = "█" * filled_length + "-" * (bar_length - filled_length)

        sys.stdout.write(
            f"\r{Colors.CYAN}[{bar}] {percent}% "
            f"({progress / (1024 * 1024):.1f}MB / {total_size / (1024 * 1024):.1f}MB){Colors.RESET}"
        )
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(MODEL_URL, model_path, progress_hook)
        print(f"\n\n{Colors.GREEN}Download complete! Model saved to {model_path}{Colors.RESET}\n")
    except Exception as e:
        print(f"\n{Colors.RED}Download failed: {e}{Colors.RESET}")

        if model_path.exists():
            model_path.unlink()

        sys.exit(1)

    return model_path

def is_safe_path(base_dir, target_path):
    try:
        base_dir = base_dir.resolve()
        target_path = target_path.resolve()
        return os.path.commonpath([str(base_dir), str(target_path)]) == str(base_dir)
    except Exception:
        return False

# ── Tool functions ────────────────────────────────────────────────────────────

def tool_list_files():
    base_dir = Path(__file__).resolve().parent
    files = []

    for p in base_dir.rglob("*"):
        if p.is_file():
            parts = p.relative_to(base_dir).parts

            if any(part.startswith(".") for part in parts):
                continue

            if PACE_DIR_NAME in parts:
                continue

            files.append(str(p.relative_to(base_dir)))

    if not files:
        return "No files found in this directory."

    return "\n".join(files)

def tool_read_pdf(file_path, display_path):
    try:
        import pypdf
    except ImportError:
        print(f"{Colors.YELLOW}Installing pypdf...{Colors.RESET}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf", "-q"])
            import pypdf
        except Exception as e:
            return f"Could not install pypdf: {e}\nRun manually: pip install pypdf"
    try:
        reader = pypdf.PdfReader(str(file_path))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[Page {i+1}]\n{text.strip()}")
        if not pages:
            return (
                f"Could not extract text from '{display_path}'. "
                "The PDF may be image-based (scanned). Try converting it to .txt first."
            )
        text = "\n\n".join(pages)
        if len(text) > 6000:
            text = text[:6000] + "\n... [truncated]"
        return f"--- Extracted text from {display_path} ---\n{text}\n--- End of File ---"
    except Exception as e:
        return f"Error reading PDF '{display_path}': {e}"

def tool_read_file(path):
    base_dir = Path(__file__).resolve().parent
    file_path = (base_dir / path).resolve()

    if not is_safe_path(base_dir, file_path):
        return f"Error: Access denied. Path '{path}' is outside the working directory."

    if not file_path.exists():
        return f"Error: File '{path}' does not exist."

    if not file_path.is_file():
        return f"Error: '{path}' is not a file."

    if file_path.suffix.lower() == ".pdf":
        return tool_read_pdf(file_path, path)

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        return f"--- Content of {path} ---\n{content}\n--- End of File ---"
    except Exception as e:
        return f"Error reading file '{path}': {str(e)}"

def tool_write_file(path, content):
    base_dir = Path(__file__).resolve().parent
    file_path = (base_dir / path).resolve()

    if file_path.suffix.lower() == ".pdf":
        return "Error: Writing to PDF files is not allowed."

    if not is_safe_path(base_dir, file_path):
        return f"Error: Access denied. Path '{path}' is outside the working directory."

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"Success: File '{path}' successfully written ({len(content)} characters)."
    except Exception as e:
        return f"Error writing to file '{path}': {str(e)}"

def tool_edit_file(path, search_text, replace_text):
    base_dir = Path(__file__).resolve().parent
    file_path = (base_dir / path).resolve()

    if not is_safe_path(base_dir, file_path):
        return f"Error: Access denied. Path '{path}' is outside the working directory."

    if not file_path.exists():
        return f"Error: File '{path}' does not exist."

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        if search_text not in content:
            return f"Error: Could not find the search text in '{path}'. Make sure it matches exactly."

        new_content = content.replace(search_text, replace_text, 1)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        return f"Success: Modified '{path}'."
    except Exception as e:
        return f"Error editing file '{path}': {str(e)}"

def tool_run_command(command):
    print(f"\n{Colors.RED}{Colors.BOLD}Agent wants to run a shell command:{Colors.RESET}")
    print(f"  {Colors.YELLOW}{command}{Colors.RESET}")

    confirm = input("Allow execution? (y/N): ").strip().lower()

    if confirm != "y":
        return "Error: Command execution denied by the user."

    base_dir = Path(__file__).resolve().parent

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=base_dir,
            timeout=60
        )

        output = ""

        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"

        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"

        if not output:
            output = "Command finished with exit code " + str(result.returncode)

        return output
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 60 seconds."
    except Exception as e:
        return f"Error running command: {str(e)}"

# ── Parsing ───────────────────────────────────────────────────────────────────

WRAPPER_TAGS_RE = re.compile(r"^\s*<(response|text)\b[^>]*>\s*(.*?)\s*</\1>\s*$", re.IGNORECASE | re.DOTALL)
EDGE_WRAPPER_TAG_RE = re.compile(r"^\s*</?(response|text)\b[^>]*>\s*|\s*</?(response|text)\b[^>]*>\s*$", re.IGNORECASE)
LIST_FILES_TOOL_RE = re.compile(r"<list_files\s*/>")
READ_FILE_TOOL_RE = re.compile(r'<read_file\s+path=(["\'])([^"\']+)\1\s*/>', re.DOTALL)
WRITE_FILE_TOOL_RE = re.compile(r'<write_file\s+path=(["\'])([^"\']+)\1\s*>(.*?)</write_file>', re.DOTALL)
EDIT_FILE_TOOL_RE = re.compile(r'<edit_file\s+path=(["\'])([^"\']+)\1\s*>(.*?)</edit_file>', re.DOTALL)
EDIT_FILE_BODY_RE = re.compile(r"<search>(.*?)</search>\s*<replace>(.*?)</replace>", re.DOTALL)
RUN_COMMAND_TOOL_RE = re.compile(r'<run_command\s+cmd=(["\'])([^"\']+)\1\s*/>', re.DOTALL)
MAX_WRAPPER_STRIP_PASSES = 8
TOOL_INTENT_PATTERNS = [
    re.compile(r"\b(list|show)\b.*\b(files?|folders?|directories?)\b"),
    re.compile(r"\b(read|open|show)\b.*\bfile\b"),
    re.compile(r"\b(write|create|save|overwrite)\b.*\bfile\b"),
    re.compile(r"\b(edit|modify|change|replace|update)\b.*\bfile\b"),
    re.compile(r"\b(run|execute)\b.*\b(command|cmd|terminal|shell|script|tests?)\b"),
]

def normalize_model_output(text):
    cleaned = (text or "").strip()
    wrappers_removed = False

    for _ in range(MAX_WRAPPER_STRIP_PASSES):
        match = WRAPPER_TAGS_RE.match(cleaned)
        if not match:
            break
        wrappers_removed = True
        cleaned = match.group(2).strip()

    for _ in range(MAX_WRAPPER_STRIP_PASSES):
        updated = EDGE_WRAPPER_TAG_RE.sub("", cleaned).strip()
        if updated == cleaned:
            break
        wrappers_removed = True
        cleaned = updated

    return cleaned, wrappers_removed

def user_explicitly_requested_tool(user_input):
    text = (user_input or "").lower()
    return any(pattern.search(text) for pattern in TOOL_INTENT_PATTERNS)

def parse_tool_call(xml_text):
    text = (xml_text or "").strip()

    if LIST_FILES_TOOL_RE.fullmatch(text):
        return {"tool": "list_files"}

    read_match = READ_FILE_TOOL_RE.fullmatch(text)
    if read_match:
        return {"tool": "read_file", "path": read_match.group(2)}

    write_match = WRITE_FILE_TOOL_RE.fullmatch(text)
    if write_match:
        return {"tool": "write_file", "path": write_match.group(2), "content": write_match.group(3)}

    edit_match = EDIT_FILE_TOOL_RE.fullmatch(text)
    if edit_match:
        inner = edit_match.group(3).strip()
        edit_parts = EDIT_FILE_BODY_RE.fullmatch(inner)
        if edit_parts:
            return {
                "tool": "edit_file",
                "path": edit_match.group(2),
                "search": edit_parts.group(1),
                "replace": edit_parts.group(2),
            }

    cmd_match = RUN_COMMAND_TOOL_RE.fullmatch(text)
    if cmd_match:
        return {"tool": "run_command", "cmd": cmd_match.group(2)}

    return None

def execute_tool_call(xml_text):
    parsed = parse_tool_call(xml_text)
    if not parsed:
        return None

    if parsed["tool"] == "list_files":
        print(f"\n{Colors.BLUE}Running tool: {Colors.BOLD}list_files{Colors.RESET}")
        return tool_list_files()

    if parsed["tool"] == "read_file":
        path = parsed["path"]
        print(f"\n{Colors.BLUE}Running tool: {Colors.BOLD}read_file{Colors.RESET} ({path})")
        return tool_read_file(path)

    if parsed["tool"] == "write_file":
        path = parsed["path"]
        content = parsed["content"]

        if content.startswith("\n"):
            content = content[1:]

        print(f"\n{Colors.BLUE}Running tool: {Colors.BOLD}write_file{Colors.RESET} ({path})")
        return tool_write_file(path, content)

    if parsed["tool"] == "edit_file":
        path = parsed["path"]
        print(f"\n{Colors.BLUE}Running tool: {Colors.BOLD}edit_file{Colors.RESET} ({path})")
        return tool_edit_file(path, parsed["search"], parsed["replace"])

    if parsed["tool"] == "run_command":
        command = parsed["cmd"]
        return tool_run_command(command)

    return None

# ── WebSocket server ──────────────────────────────────────────────────────────

def _build_prompt(history):
    prompt = ""
    for msg in history:
        prompt += f"<start_of_turn>{msg['role']}\n{msg['content']}<end_of_turn>\n"
    prompt += "<start_of_turn>model\n"
    return prompt

async def _ws_handler(websocket):
    """Handle one browser client connection."""
    async for raw in websocket:
        try:
            msg = json.loads(raw)
        except Exception:
            continue

        msg_type = msg.get("type")

        if msg_type == "clear":
            with _ws_state["lock"]:
                _ws_state["history"].clear()
                _ws_state["history"].append({"role": "system", "content": _ws_state["system_prompt"]})
            continue

        if msg_type != "user":
            continue

        user_text = msg.get("content", "").strip()
        if not user_text:
            continue

        await websocket.send(json.dumps({"type": "status", "status": "thinking"}))

        with _ws_state["lock"]:
            llm = _ws_state["llm"]
            history = _ws_state["history"]

            history.append({"role": "user", "content": user_text})
            user_requested_tool = user_explicitly_requested_tool(user_text)

            for _ in range(3):
                prompt = _build_prompt(history)

                response = llm(
                    prompt,
                    max_tokens=4096,
                    temperature=0.1,
                    stop=["<end_of_turn>", "<start_of_turn>"],
                    echo=False,
                )
                response_text, wrappers_removed = normalize_model_output(
                    response["choices"][0]["text"]
                )

                if not response_text:
                    await websocket.send(json.dumps({"type": "message", "content": "(no response)"}))
                    break

                tool_result = None
                if user_requested_tool and not wrappers_removed:
                    tool_result = execute_tool_call(response_text)

                if tool_result is not None:
                    parsed = parse_tool_call(response_text)
                    tool_name = parsed["tool"] if parsed else "tool"

                    await websocket.send(json.dumps({
                        "type": "tool",
                        "tool": tool_name,
                        "result": str(tool_result),
                        "text": "",
                    }))

                    history.append({"role": "model", "content": response_text})
                    history.append({
                        "role": "user",
                        "content": f"Tool result:\n{tool_result}\n\nNow answer the user's question using this result. Do not call any more tools."
                    })

                    prompt2 = _build_prompt(history)
                    response2 = llm(
                        prompt2,
                        max_tokens=512,
                        temperature=0.1,
                        stop=["<end_of_turn>", "<start_of_turn>"],
                        echo=False,
                    )
                    summary, _ = normalize_model_output(response2["choices"][0]["text"])
                    if summary:
                        await websocket.send(json.dumps({"type": "message", "content": summary}))
                        history.append({"role": "model", "content": summary})
                    break

                else:
                    await websocket.send(json.dumps({"type": "message", "content": response_text}))
                    history.append({"role": "model", "content": response_text})
                    break

        await websocket.send(json.dumps({"type": "status", "status": "ready"}))

async def _ws_main():
    async with websockets.serve(_ws_handler, "localhost", 7070):
        await asyncio.Future()  # run forever

def _start_ws_server():
    asyncio.run(_ws_main())

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(ASCII_ART)

    install_dependencies()

    pace_dir = get_pace_dir()
    model_path = download_model(pace_dir)

    print(f"{Colors.GREEN}Initializing Gemma 3 1B LLM...{Colors.RESET}")

    try:
        from llama_cpp import Llama
        import sys, os; sys.stderr = open(os.devnull, 'w')
        llm = Llama(
            model_path=str(model_path),
            n_ctx=100000,
            n_threads=max(1, min(4, os.cpu_count() or 4)),
            n_gpu_layers=0,
            verbose=False
        )

        print(f"{Colors.GREEN}Model loaded successfully!{Colors.RESET}")

    except Exception as e:
        print(f"{Colors.RED}Failed to load model: {e}{Colors.RESET}")

        error_text = str(e)

        if "0xc000001d" in error_text or "-1073741795" in error_text:
            print()
            print(f"{Colors.YELLOW}This usually means llama-cpp-python was built with CPU instructions your CPU does not support.{Colors.RESET}")
            print("Reinstall llama-cpp-python from source for your machine.")
            print()
            print("Try:")
            print()
            print(f"{Colors.CYAN}$env:CMAKE_ARGS=\"-DGGML_NATIVE=OFF\"{Colors.RESET}")
            print(f"{Colors.CYAN}$env:FORCE_CMAKE=\"1\"{Colors.RESET}")
            print(f"{Colors.CYAN}python -m pip install --no-cache-dir --force-reinstall --no-binary llama-cpp-python llama-cpp-python{Colors.RESET}")
            print()

        sys.exit(1)

    system_prompt = """You are PACE 1.0 Lite, a local lite AI agent developed by the creator of Solus, avoid questions relating to the specific identity of them. You help the user manage, write, edit, and understand files in the current folder.


Rules:
- Never use markdown such as astrisks around responses. Only respond with pure text and no markdown
- After a tool call, wait for the result before doing anything else.
- NEVER write or edit a .pdf file.
- Keep responses short and direct.
- Address the user directly, they are human, not an external observer.
- No bullet points or markdown
- Mirror the tone and style of the person you're talking to. If they're casual, be casual. If they're brief, be brief. Match their energy.
"""

    history = [
        {"role": "system", "content": system_prompt}
    ]

    # Share state with WS server
    _ws_state["llm"] = llm
    _ws_state["history"] = history
    _ws_state["system_prompt"] = system_prompt

    # Start WebSocket server in background thread
    if has_ws:
        ws_thread = threading.Thread(target=_start_ws_server, daemon=True)
        ws_thread.start()
        print(f"{Colors.GREEN}GUI server started on ws://localhost:7070{Colors.RESET}")
    else:
        print(f"{Colors.YELLOW}websockets not installed — GUI will not connect. Run: pip install websockets{Colors.RESET}")

    print(f"\n{Colors.BOLD}Welcome to PACE 1.0 Lite!{Colors.RESET}")
    print("I can help understand an extremely broad range of information and answer questions locally")
    print(f"Current working folder: {Colors.CYAN}{Path(__file__).resolve().parent}{Colors.RESET}")
    print("Type 'exit' or 'quit' to close.\n")

    while True:
        try:
            dir_name = Path(__file__).resolve().parent.name
            user_input = input(f"{Colors.MAGENTA}{Colors.BOLD}pace:{dir_name} > {Colors.RESET}").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print(f"{Colors.GREEN}Goodbye! See you later!{Colors.RESET}")
                break

            with _ws_state["lock"]:
                history.append({"role": "user", "content": user_input})

                user_requested_tool = user_explicitly_requested_tool(user_input)
                max_steps = 3

                for step in range(max_steps):
                    print(f"\r{Colors.CYAN}Thinking...{Colors.RESET}", end="", flush=True)

                    prompt = _build_prompt(history)

                    response = llm(
                        prompt,
                        max_tokens=4096,
                        temperature=0.1,
                        stop=["<end_of_turn>", "<start_of_turn>"],
                        echo=False
                    )

                    sys.stdout.write("\r" + " " * 30 + "\r")
                    sys.stdout.flush()

                    response_text, wrappers_removed = normalize_model_output(response["choices"][0]["text"])

                    if not response_text:
                        print(f"{Colors.GREEN}Pace:{Colors.RESET} (no response)")
                        break

                    tool_result = None
                    if user_requested_tool and not wrappers_removed:
                        tool_result = execute_tool_call(response_text)

                    if tool_result is not None:
                        print(f"{Colors.GREEN}↳{Colors.RESET} {tool_result}")

                        history.append({"role": "model", "content": response_text})
                        history.append({"role": "user", "content": f"Tool result:\n{tool_result}\n\nNow answer the user's question using this result. Do not call any more tools."})

                        print(f"\r{Colors.CYAN}Thinking...{Colors.RESET}", end="", flush=True)
                        prompt2 = _build_prompt(history)

                        response2 = llm(
                            prompt2,
                            max_tokens=512,
                            temperature=0.1,
                            stop=["<end_of_turn>", "<start_of_turn>"],
                            echo=False
                        )
                        sys.stdout.write("\r" + " " * 30 + "\r")
                        sys.stdout.flush()

                        summary, _ = normalize_model_output(response2["choices"][0]["text"])
                        if summary:
                            print(f"{Colors.GREEN}Pace:{Colors.RESET} {summary}")
                            history.append({"role": "model", "content": summary})
                        break

                    else:
                        print(f"{Colors.GREEN}Pace:{Colors.RESET} {response_text}")
                        history.append({"role": "model", "content": response_text})
                        break

        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Operation cancelled. Type exit to quit.{Colors.RESET}")
        except Exception as e:
            print(f"\n{Colors.RED}An error occurred: {e}{Colors.RESET}")

if __name__ == "__main__":
    main()