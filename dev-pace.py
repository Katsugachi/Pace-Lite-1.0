# PACE 1.0: Local Agentic Terminal + GUI WebSocket Server
import os
import sys
import subprocess
import urllib.request
import urllib.parse
import re
import ast
import threading
import json
import asyncio
import time
import datetime
import html
import contextlib
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
MAX_BASIC_PROMPT_CHAR_COUNT = 12
MAX_RESEARCH_TEXT_LENGTH = 6500
INTERNET_CHECK_CACHE_SECONDS = 20
MAX_QUERY_TERMS = 10
MAX_RESULT_AGE_YEARS = 4
MAX_HISTORY_CHARS = 45000
WEB_RESEARCH_MIN_SOURCES = 5
WEB_RESEARCH_MAX_SOURCES = 6
WEB_SOURCE_FETCH_TIMEOUT_SECONDS = 8
WEB_SOURCE_MAX_BYTES = 180000
WEB_SOURCE_SNIPPET_CHARS = 420
MIN_HISTORY_CHAR_BUDGET = 1000
HISTORY_MESSAGE_OVERHEAD_CHARS = 32
CODE_RELATED_KEYWORDS_PATTERN = re.compile(
    r"\b(code|python|javascript|java|c\+\+|cpp|tutorial|error|bug|function|api|class|framework|syntax|compile|debug|library|package|module|import|export|variable|loop|array|database|sql|html|css|react|node|git)\b",
    re.IGNORECASE,
)
CDN_HINT_PATTERN = re.compile(r"\bcdn\b", re.IGNORECASE)
WEB_DEV_CONTEXT_PATTERN = re.compile(
    r"\b(html|css|javascript|js|typescript|ts|react|vue|angular|node|npm|library|framework|bootstrap|tailwind)\b",
    re.IGNORECASE,
)
QUERY_NOISE_PATTERN = re.compile(
    r"\b(can you|could you|would you|please|tell me|show me|help me|i need|i want|"
    r"search for|look up|find|what is|what are|how do i|how to)\b",
    re.IGNORECASE,
)
SEARCH_STOPWORDS = {
    "a", "an", "and", "the", "to", "for", "of", "on", "in", "at", "from", "with",
    "about", "into", "over", "after", "before", "by", "it", "this", "that", "these",
    "those", "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "can", "could", "would", "should", "please", "me", "my", "you", "your", "we", "our",
}
CURRENT_INFO_HINT_PATTERN = re.compile(
    r"\b(latest|current|recent|today|now|new|updated|update|up-to-date|as of)\b",
    re.IGNORECASE,
)
INTERNET_RESEARCH_TRIGGER_PATTERN = re.compile(
    r"\b(what is|what are|who is|when did|where is|latest|current|recent|today|news|price|release|version|documentation|docs|official|tutorial|guide|api|reference|cdn|install|setup|fix|error|troubleshoot|compare|difference)\b",
    re.IGNORECASE,
)
LOCAL_ONLY_INTENT_PATTERN = re.compile(
    r"\b(write|draft|compose|story|poem|song|lyrics|essay|narrative|fiction|scene|dialogue|imagine|brainstorm|summarize|paraphrase|rewrite)\b",
    re.IGNORECASE,
)
SUPER_BASIC_PROMPTS = {
    "hi", "hello", "hey", "yo", "sup", "what's up", "how are you",
    "thanks", "thank you", "ok", "okay", "cool", "nice", "bye", "goodbye",
    "howdy", "hiya", "cheers", "good morning", "good afternoon", "good evening", "see you"
}

# ── Coding-skill constants ────────────────────────────────────────────────────
EXECUTABLE_LANGUAGES = {"python", "javascript", "bash"}
LINTABLE_LANGUAGES = {"python", "javascript"}
CODE_EXEC_TIMEOUT = 15          # seconds per code execution
MAX_CODE_RETRY_ATTEMPTS = 3     # self-correction retries on failure
MAX_EXEC_OUTPUT_CHARS = 2000    # truncate long stdout/stderr
MAX_LINT_OUTPUT_CHARS = 1500    # truncate long lint output
MAX_FILE_CONTEXT_CHARS = 2000   # max chars injected per file
MAX_GREP_RESULTS_SHOWN = 5      # max file snippets returned by grep
MAX_GREP_SNIPPET_LINES = 6      # context lines around each grep hit

FILE_REF_PATTERN = re.compile(
    r'\b[\w][\w\-]*\.(?:py|js|ts|html|css|json|md|txt|sh|yaml|yml)\b',
    re.IGNORECASE,
)
SAFE_CODE_EXEC_BLOCKLIST = re.compile(
    r'(?:shutil\.rmtree|os\.remove|os\.unlink|os\.rmdir)\s*\(',
    re.IGNORECASE,
)

# Shared state for WebSocket server
_ws_state = {
    "llm": None,
    "history": [],
    "system_prompt": "",
    "internet_available": False,
    "internet_last_checked": 0.0,
    "internet_enabled": True,
    "startup_issue": "",
    # Re-entrant lock avoids deadlock when handlers refresh internet status
    # while already holding shared-state lock for history updates.
    "lock": threading.RLock(),
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
        return None

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
    print(f"{Colors.CYAN}python dev-pace.py{Colors.RESET}")
    print()
    return (
        "PACE is running without the local model because llama-cpp-python is not installed. "
        "Install llama-cpp-python, then restart PACE to enable full responses."
    )

def build_startup_issue_message(error_text=None):
    message = (
        "PACE is connected, but the local model is unavailable right now. "
        "Install or fix llama-cpp-python and restart PACE to enable full responses."
    )
    if error_text:
        return f"{message} Startup error: {error_text}"
    return message

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

def detect_internet_access(timeout=3):
    try:
        req = urllib.request.Request(
            "https://www.google.com/generate_204",
            headers={"User-Agent": "PACE-Lite/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception:
        return False

def get_internet_status(force=False):
    with _ws_state["lock"]:
        last_checked = _ws_state.get("internet_last_checked", 0.0)
        cached = _ws_state.get("internet_available", False)

    if not force and (time.time() - last_checked) < INTERNET_CHECK_CACHE_SECONDS:
        return cached

    current = detect_internet_access()

    with _ws_state["lock"]:
        _ws_state["internet_available"] = current
        _ws_state["internet_last_checked"] = time.time()

    return current

def get_internet_mode():
    with _ws_state["lock"]:
        return bool(_ws_state.get("internet_enabled", True))

def set_internet_mode(enabled):
    with _ws_state["lock"]:
        _ws_state["internet_enabled"] = bool(enabled)

def build_internet_status_payload(force=False):
    available = get_internet_status(force=force)
    enabled = get_internet_mode()
    return {
        "type": "internet_status",
        "available": available,
        "enabled": enabled,
        "active": enabled and available,
    }

def _normalize_terminal_command(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())

def parse_internet_mode_command(text):
    normalized = _normalize_terminal_command(text)
    if normalized in {"/internet", "internet", "/internet status", "internet status"}:
        return "status"
    if normalized in {"/internet on", "internet on"}:
        return "on"
    if normalized in {"/internet off", "internet off"}:
        return "off"
    if normalized in {"/internet toggle", "internet toggle"}:
        return "toggle"
    return None

def analyze_code_query(text):
    asks_for_cdn = bool(CDN_HINT_PATTERN.search(text or ""))
    has_web_dev_context = bool(WEB_DEV_CONTEXT_PATTERN.search(text or ""))
    has_code_keywords = bool(CODE_RELATED_KEYWORDS_PATTERN.search(text or ""))
    return has_code_keywords or (asks_for_cdn and has_web_dev_context), asks_for_cdn

def _flatten_related_topics(items):
    out = []
    for item in items or []:
        if isinstance(item, dict) and "Topics" in item:
            out.extend(_flatten_related_topics(item.get("Topics")))
        else:
            out.append(item)
    return out

def _clean_text(value):
    try:
        text = html.unescape(value or "")
    except Exception:
        text = value or ""
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()

def _truncate_with_ellipsis(text, max_chars):
    value = (text or "").strip()
    if max_chars <= 0:
        return ""
    if len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return value[:max_chars]
    return value[:max_chars - 3].rstrip() + "..."

def _strip_tag_block(html_text, tag_name):
    text = html_text or ""
    lower = text.lower()
    open_marker = f"<{tag_name}"
    close_marker = f"</{tag_name}"
    idx = 0
    out = []

    while True:
        start = lower.find(open_marker, idx)
        if start < 0:
            out.append(text[idx:])
            break
        out.append(text[idx:start])
        close_start = lower.find(close_marker, start + len(open_marker))
        if close_start < 0:
            break
        close_end = lower.find(">", close_start + len(close_marker))
        if close_end < 0:
            break
        idx = close_end + 1

    return "".join(out)

def _current_utc_date():
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")

def _current_utc_year():
    return datetime.datetime.now(datetime.UTC).year

def _extract_years(text):
    years = []
    for year_text in re.findall(r"\b(19\d{2}|20\d{2}|21\d{2})\b", text or ""):
        years.append(int(year_text))
    return years

def _format_search_focus(user_text):
    text = (user_text or "").strip()
    if not text:
        return ""

    text = QUERY_NOISE_PATTERN.sub(" ", text)
    text = re.sub(r"[`\"“”‘’]", " ", text)
    text = re.sub(r"[^a-zA-Z0-9+#.\-_\s]", " ", text)
    tokens = []
    for tok in text.split():
        t = tok.strip(" .,_-").lower()
        if len(t) < 2 and t not in {"c", "r"}:
            continue
        if t in SEARCH_STOPWORDS:
            continue
        tokens.append(tok.strip())

    if not tokens:
        return (user_text or "").strip()
    return " ".join(tokens[:MAX_QUERY_TERMS])

def _tokenize_for_relevance(text):
    tokens = re.findall(r"[a-z0-9#+._-]+", (text or "").lower())
    return {t for t in tokens if len(t) > 1 and t not in SEARCH_STOPWORDS}

def _score_search_hit(hit, relevance_terms, current_year):
    combined = " ".join([
        hit.get("title", ""),
        hit.get("snippet", ""),
        hit.get("url", ""),
    ])
    combined_lower = combined.lower()
    overlap = sum(1 for term in relevance_terms if term in combined_lower)
    years = [y for y in _extract_years(combined) if y <= current_year + 1]
    latest_year = max(years) if years else None
    age_years = (current_year - latest_year) if latest_year else None

    score = overlap * 3
    if latest_year is not None:
        if age_years <= 1:
            score += 3
        elif age_years <= 2:
            score += 2
        elif age_years <= MAX_RESULT_AGE_YEARS:
            score += 1
        else:
            score -= 5

    url_lower = (hit.get("url") or "").lower()
    if any(x in url_lower for x in ("docs.", "/docs", "developer", "wikipedia.org", ".gov", ".edu")):
        score += 1

    is_stale = latest_year is not None and age_years > MAX_RESULT_AGE_YEARS
    return {
        "score": score,
        "overlap": overlap,
        "latest_year": latest_year,
        "age_years": age_years,
        "is_stale": is_stale,
    }

def _rank_and_filter_hits(user_text, query, hits):
    current_year = _current_utc_year()
    relevance_terms = _tokenize_for_relevance(f"{_format_search_focus(user_text)} {query}")
    scored = []

    for hit in hits:
        meta = _score_search_hit(hit, relevance_terms, current_year)
        if meta["is_stale"] and meta["overlap"] < 3:
            continue
        item = dict(hit)
        item.update({
            "score": meta["score"],
            "overlap": meta["overlap"],
            "latest_year": meta["latest_year"],
            "age_years": meta["age_years"],
        })
        scored.append(item)

    if not scored:
        for hit in hits:
            meta = _score_search_hit(hit, relevance_terms, current_year)
            item = dict(hit)
            item.update({
                "score": meta["score"],
                "overlap": meta["overlap"],
                "latest_year": meta["latest_year"],
                "age_years": meta["age_years"],
            })
            scored.append(item)

    scored.sort(key=lambda x: (x.get("score", 0), x.get("overlap", 0)), reverse=True)
    return scored[:WEB_RESEARCH_MAX_SOURCES]

def _normalize_search_result_url(url):
    raw = (url or "").strip()
    if not raw:
        return ""

    if raw.startswith("//"):
        raw = "https:" + raw

    if raw.startswith("/l/?"):
        parsed = urllib.parse.urlparse(raw)
        target = urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0]
        raw = urllib.parse.unquote(target).strip()

    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme in {"http", "https"}:
        return raw
    return ""

def _extract_source_context_from_html(page_html):
    text = page_html or ""
    text = _strip_tag_block(text, "script")
    text = _strip_tag_block(text, "style")

    title_match = re.search(r"<title[^>]*>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL)
    title = _clean_text(title_match.group(1)) if title_match else ""

    desc_match = re.search(
        r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    description = _clean_text(desc_match.group(1)) if desc_match else ""

    summary_source = description
    if not summary_source:
        summary_source = _clean_text(text)
    summary = _truncate_with_ellipsis(summary_source, WEB_SOURCE_SNIPPET_CHARS)

    return {
        "title": title or "Untitled source",
        "summary": summary or "No extractable content.",
    }

def fetch_source_context(url):
    normalized = _normalize_search_result_url(url)
    if not normalized:
        return None

    try:
        req = urllib.request.Request(normalized, headers={"User-Agent": "PACE-Lite/1.0"})
        with urllib.request.urlopen(req, timeout=WEB_SOURCE_FETCH_TIMEOUT_SECONDS) as response:
            payload = response.read(WEB_SOURCE_MAX_BYTES).decode("utf-8", errors="replace")
        context = _extract_source_context_from_html(payload)
        context["url"] = normalized
        return context
    except Exception:
        return None

def search_duckduckgo(query, max_results=5):
    url = (
        "https://api.duckduckgo.com/?format=json&no_html=1&skip_disambig=1&q="
        + urllib.parse.quote(query)
    )
    req = urllib.request.Request(url, headers={"User-Agent": "PACE-Lite/1.0"})
    with urllib.request.urlopen(req, timeout=10) as response:
        payload = response.read().decode("utf-8", errors="replace")

    data = json.loads(payload)
    results = []

    abstract = _clean_text(data.get("AbstractText"))
    if abstract:
        results.append({
            "title": _clean_text(data.get("Heading")) or query,
            "snippet": abstract,
            "url": _normalize_search_result_url(data.get("AbstractURL", "")),
        })

    for item in data.get("Results", []):
        snippet = _clean_text(item.get("Text"))
        if not snippet:
            continue
        results.append({
            "title": snippet.split(" - ")[0][:120],
            "snippet": snippet,
            "url": _normalize_search_result_url(item.get("FirstURL", "")),
        })

    for item in _flatten_related_topics(data.get("RelatedTopics")):
        if not isinstance(item, dict):
            continue
        snippet = _clean_text(item.get("Text"))
        if not snippet:
            continue
        results.append({
            "title": snippet.split(" - ")[0][:120],
            "snippet": snippet,
            "url": _normalize_search_result_url(item.get("FirstURL", "")),
        })

    deduped = []
    seen = set()
    for item in results:
        key = (item.get("url", "").strip(), item.get("snippet", "").strip().lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= max_results:
            break

    return deduped

def is_super_basic_prompt(text):
    lowered = (text or "").strip().lower()
    if not lowered:
        return True

    if lowered in SUPER_BASIC_PROMPTS:
        return True

    if len(lowered) <= MAX_BASIC_PROMPT_CHAR_COUNT and re.fullmatch(r"[\w\s!?.,'’-]+", lowered, re.UNICODE):
        return True

    return False

def should_use_internet_research(text):
    prompt = (text or "").strip()
    if not prompt:
        return False

    if is_super_basic_prompt(prompt):
        return False

    has_trigger = bool(INTERNET_RESEARCH_TRIGGER_PATTERN.search(prompt))
    has_local_only_intent = bool(LOCAL_ONLY_INTENT_PATTERN.search(prompt))
    is_code_related, asks_for_cdn = analyze_code_query(prompt)
    has_current_hint = bool(CURRENT_INFO_HINT_PATTERN.search(prompt))
    asks_question = "?" in prompt

    if has_local_only_intent and not (has_trigger or has_current_hint or asks_for_cdn):
        return False

    if is_code_related or asks_for_cdn or has_current_hint:
        return True

    return has_trigger and asks_question

def build_search_queries(user_text):
    text = (user_text or "").strip()
    if not text:
        return []

    current_year = _current_utc_year()
    focus = _format_search_focus(text)
    explicit_years = [y for y in _extract_years(text) if y <= current_year + 1]
    has_current_hint = bool(CURRENT_INFO_HINT_PATTERN.search(text))

    is_code_related, asks_for_cdn = analyze_code_query(text)
    queries = []

    if explicit_years:
        target_year = str(max(explicit_years))
        queries.append(f"{focus} {target_year}")
    else:
        queries.append(f"{focus} {current_year}")

    if is_code_related:
        queries.append(f"{focus} official documentation {current_year}")
        if asks_for_cdn:
            queries.append(f"{focus} cdn integration guide {current_year}")
        else:
            queries.append(f"{focus} tutorial {current_year}")
    else:
        if has_current_hint or not is_super_basic_prompt(text):
            queries.append(f"{focus} latest updates {current_year}")
        queries.append(f"{focus} facts")

    unique = []
    seen = set()
    for q in queries:
        key = q.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(q.strip())
    return unique[:3]

def build_web_research(user_text, progress_cb=None):
    queries = build_search_queries(user_text)
    if not queries:
        return ""

    current_date = _current_utc_date()
    by_query = []
    candidate_hits = []
    url_to_queries = {}

    total = len(queries)
    for idx, query in enumerate(queries, start=1):
        if progress_cb:
            progress_cb({
                "phase": "searching",
                "step": idx,
                "total": total,
                "query": query,
                "message": f"Searching web ({idx}/{total}): {query}",
            })

        try:
            hits = search_duckduckgo(query, max_results=10)
            hits = _rank_and_filter_hits(user_text, query, hits)
        except Exception as e:
            hits = []
            if progress_cb:
                progress_cb({
                    "phase": "error",
                    "step": idx,
                    "total": total,
                    "query": query,
                    "message": f"Search failed for '{query}': {e}",
                })

        for hit in hits:
            url = (hit.get("url") or "").strip()
            if url:
                url_to_queries.setdefault(url, set()).add(query)
                item = dict(hit)
                item["query"] = query
                candidate_hits.append(item)

        by_query.append((query, hits))
        if progress_cb:
            progress_cb({
                "phase": "results",
                "step": idx,
                "total": total,
                "query": query,
                "message": f"Found {len(hits)} relevant references for '{query}'",
            })

    corroborated_urls = [
        url for url, matched_queries in url_to_queries.items() if len(matched_queries) > 1
    ][:5]

    candidate_hits.sort(key=lambda x: (x.get("score", 0), x.get("overlap", 0)), reverse=True)
    visited_sources = []
    seen_visited_urls = set()
    for hit in candidate_hits:
        if len(visited_sources) >= WEB_RESEARCH_MAX_SOURCES:
            break
        url = (hit.get("url") or "").strip()
        if not url or url in seen_visited_urls:
            continue
        seen_visited_urls.add(url)

        if progress_cb:
            progress_cb({
                "phase": "visiting",
                "step": len(visited_sources) + 1,
                "total": WEB_RESEARCH_MAX_SOURCES,
                "query": hit.get("query", ""),
                "message": f"Visiting source ({len(visited_sources) + 1}/{WEB_RESEARCH_MAX_SOURCES}): {url}",
            })

        try:
            source = fetch_source_context(url)
            if source:
                source["score"] = hit.get("score", 0)
                source["query"] = hit.get("query", "")
                visited_sources.append(source)
        except Exception as e:
            if progress_cb:
                progress_cb({
                    "phase": "visit_error",
                    "step": len(visited_sources) + 1,
                    "total": WEB_RESEARCH_MAX_SOURCES,
                    "query": hit.get("query", ""),
                    "message": f"Could not fetch source '{url}': {e}",
                })

    lines = []
    lines.append(f"Live web research (cross-referenced, current date UTC: {current_date}):")
    for query, hits in by_query:
        lines.append(f"Query: {query}")
        if not hits:
            lines.append("- No reliable references returned.")
            continue
        for hit in hits[:3]:
            title = _clean_text(hit.get("title", "")) or "Untitled"
            snippet = _clean_text(hit.get("snippet", "")) or "No snippet available."
            url = (hit.get("url") or "").strip()
            year_note = ""
            if hit.get("latest_year"):
                year_note = f" (latest year detected: {hit['latest_year']})"
            lines.append(f"- {title}: {snippet}{year_note}")
            if url:
                lines.append(f"  Source: {url}")

    if visited_sources:
        lines.append(f"Visited source pages ({len(visited_sources)} successful fetches, target {WEB_RESEARCH_MIN_SOURCES}-{WEB_RESEARCH_MAX_SOURCES}):")
        for source in visited_sources:
            lines.append(f"- {source.get('title', 'Untitled source')}: {source.get('summary', 'No extractable content.')}")
            lines.append(f"  Source: {source.get('url', '')}")
        if len(visited_sources) < WEB_RESEARCH_MIN_SOURCES:
            lines.append(
                f"Note: only {len(visited_sources)} source pages were fetchable; network/content restrictions may have limited source retrieval."
            )
    else:
        lines.append("Visited source pages: no source pages could be fetched.")

    if corroborated_urls:
        lines.append("Cross-reference matches (same source appeared in multiple searches):")
        for url in corroborated_urls:
            lines.append(f"- {url}")
    else:
        lines.append("Cross-reference matches: no repeated sources found across queries.")

    source_urls = []
    for source in visited_sources:
        url = (source.get("url") or "").strip()
        if url and url not in source_urls:
            source_urls.append(url)
    if not source_urls:
        for _, hits in by_query:
            for hit in hits:
                url = (hit.get("url") or "").strip()
                if url and url not in source_urls:
                    source_urls.append(url)
                if len(source_urls) >= WEB_RESEARCH_MAX_SOURCES:
                    break
            if len(source_urls) >= WEB_RESEARCH_MAX_SOURCES:
                break

    lines.append("Sources consulted:")
    if source_urls:
        for url in source_urls[:WEB_RESEARCH_MAX_SOURCES]:
            lines.append(f"- {url}")
    else:
        lines.append("- No source URLs available.")

    research_text = "\n".join(lines).strip()
    if len(research_text) > MAX_RESEARCH_TEXT_LENGTH:
        trimmed = research_text[:MAX_RESEARCH_TEXT_LENGTH]
        split_at = max(trimmed.rfind("\n"), trimmed.rfind(". "))
        # If split_at <= 0, no safe boundary was found and we keep raw truncation.
        if split_at > 0:
            trimmed = trimmed[:split_at].rstrip()
        research_text = trimmed + "\n... [truncated]"

    if progress_cb:
        progress_cb({
            "phase": "complete",
            "step": total,
            "total": total,
            "query": "",
            "message": "Web research complete.",
        })

    return research_text

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

def tool_run_command(command, headless=False):
    if headless:
        print(f"\n{Colors.BLUE}[GUI] Running shell command:{Colors.RESET} {Colors.YELLOW}{command}{Colors.RESET}")
    else:
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


def tool_execute_code(language, code):
    """Execute a code block in a subprocess and return exit_code/stdout/stderr."""
    import tempfile

    lang = _normalize_code_language(language)

    if lang == "python":
        ext, cmd_fn = ".py", lambda p: [sys.executable, p]
    elif lang in ("javascript", "typescript"):
        ext, cmd_fn = ".js", lambda p: ["node", p]
    elif lang == "bash":
        ext, cmd_fn = ".sh", lambda p: ["bash", p]
    else:
        return {
            "exit_code": -1, "stdout": "", "timed_out": False, "skipped": True,
            "stderr": f"Language '{language}' is not executable.",
        }

    if lang == "python" and SAFE_CODE_EXEC_BLOCKLIST.search(code or ""):
        return {
            "exit_code": -1, "stdout": "", "timed_out": False, "skipped": True,
            "stderr": "Skipped: code contains destructive file-system calls (os.remove / shutil.rmtree).",
        }

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=ext, delete=False, encoding="utf-8"
        ) as f:
            f.write(code or "")
            tmp_path = f.name

        try:
            result = subprocess.run(
                cmd_fn(tmp_path),
                capture_output=True,
                text=True,
                timeout=CODE_EXEC_TIMEOUT,
                cwd=Path(__file__).resolve().parent,
            )
            return {
                "exit_code": result.returncode,
                "stdout": result.stdout[:MAX_EXEC_OUTPUT_CHARS],
                "stderr": result.stderr[:MAX_EXEC_OUTPUT_CHARS],
                "timed_out": False,
                "skipped": False,
            }
        except subprocess.TimeoutExpired:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Execution timed out after {CODE_EXEC_TIMEOUT}s.",
                "timed_out": True,
                "skipped": False,
            }
        except FileNotFoundError as exc:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Runtime not found: {exc}",
                "timed_out": False,
                "skipped": True,
            }
    except Exception as exc:
        return {
            "exit_code": -1, "stdout": "", "timed_out": False, "skipped": False,
            "stderr": str(exc),
        }
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def tool_run_lint(language, code):
    """Run pylint/pyflakes (Python) or eslint (JS) and return the issues string."""
    import tempfile

    lang = _normalize_code_language(language)

    if lang not in LINTABLE_LANGUAGES:
        return ""

    tmp_path = None
    try:
        ext = ".py" if lang == "python" else ".js"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=ext, delete=False, encoding="utf-8"
        ) as f:
            f.write(code or "")
            tmp_path = f.name

        if lang == "python":
            # Try pylint first (errors/warnings only, no style)
            try:
                r = subprocess.run(
                    [sys.executable, "-m", "pylint", "--score=n",
                     "--disable=C,R,W0611,W0614", tmp_path],
                    capture_output=True, text=True, timeout=30,
                )
                out = (r.stdout + r.stderr).replace(tmp_path, "<code>").strip()
                if r.returncode != 0 and out and "No module named" not in out:
                    return out[:MAX_LINT_OUTPUT_CHARS]
                if "No module named" not in out:
                    return ""
            except FileNotFoundError:
                pass

            # Fall back to pyflakes
            try:
                r = subprocess.run(
                    [sys.executable, "-m", "pyflakes", tmp_path],
                    capture_output=True, text=True, timeout=15,
                )
                out = (r.stdout + r.stderr).replace(tmp_path, "<code>").strip()
                if r.returncode != 0 and out and "No module named" not in out:
                    return out[:MAX_LINT_OUTPUT_CHARS]
                return ""
            except FileNotFoundError:
                pass

        elif lang == "javascript":
            try:
                r = subprocess.run(
                    ["eslint", "--no-eslintrc", "--env", "node,es6",
                     "--rule", "semi: error",
                     "--rule", "no-undef: warn",
                     "--rule", "no-unused-vars: warn",
                     tmp_path],
                    capture_output=True, text=True, timeout=30,
                )
                out = (r.stdout + r.stderr).replace(tmp_path, "<code>").strip()
                if r.returncode != 0 and out and "No module named" not in out:
                    return out[:MAX_LINT_OUTPUT_CHARS]
            except FileNotFoundError:
                pass

    except Exception as exc:
        return f"Lint error: {exc}"
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return ""


def tool_grep_files(pattern, file_glob=None):
    """Search project files for a regex pattern, return annotated snippets."""
    base_dir = Path(__file__).resolve().parent

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return f"Error: Invalid regex pattern: {exc}"

    text_exts = {
        ".py", ".js", ".ts", ".html", ".css", ".json", ".md",
        ".txt", ".sh", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".xml",
    }

    if file_glob:
        try:
            candidates = list(base_dir.glob(file_glob))
        except Exception:
            candidates = list(base_dir.rglob("*"))
    else:
        candidates = list(base_dir.rglob("*"))

    results = []
    for fp in candidates:
        if not fp.is_file():
            continue
        parts = fp.relative_to(base_dir).parts
        if any(p.startswith(".") for p in parts) or PACE_DIR_NAME in parts:
            continue
        if fp.suffix.lower() not in text_exts:
            continue
        try:
            lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines):
            if regex.search(line):
                start = max(0, i - 2)
                end = min(len(lines), i + 3)   # 2 lines before + match + 2 lines after
                snippet = "\n".join(lines[start:end])
                rel = str(fp.relative_to(base_dir))
                results.append(f"{rel}:{i + 1}:\n{snippet}")
                if len(results) >= MAX_GREP_RESULTS_SHOWN:
                    break
        if len(results) >= MAX_GREP_RESULTS_SHOWN:
            break

    if not results:
        return f"No matches found for pattern '{pattern}'."
    return "\n\n".join(results)


def _auto_inject_file_context(user_text):
    """If the user explicitly mentions project filenames, return their contents."""
    if not user_text:
        return ""
    base_dir = Path(__file__).resolve().parent
    found = FILE_REF_PATTERN.findall(user_text)
    parts = []
    total = 0
    for fname in dict.fromkeys(found):  # deduplicate, preserve order
        fp = (base_dir / fname).resolve()
        if not fp.is_file() or not is_safe_path(base_dir, fp):
            continue
        try:
            content = fp.read_text(encoding="utf-8", errors="replace")
            if len(content) > MAX_FILE_CONTEXT_CHARS:
                content = content[:MAX_FILE_CONTEXT_CHARS] + "\n... [truncated]"
            snippet = f"--- Content of {fname} ---\n{content}\n--- End of File ---"
            parts.append(snippet)
            total += len(snippet)
            if total >= MAX_FILE_CONTEXT_CHARS * 3:
                break
        except Exception:
            pass
    return "\n\n".join(parts)



WRAPPER_TAGS_RE = re.compile(r"^\s*<(response|text)\b[^>]*>\s*(.*?)\s*</\1>\s*$", re.IGNORECASE | re.DOTALL)
EDGE_WRAPPER_TAG_RE = re.compile(r"^\s*</?(response|text)\b[^>]*>\s*|\s*</?(response|text)\b[^>]*>\s*$", re.IGNORECASE)
LIST_FILES_TOOL_RE = re.compile(r"<list_files\s*/>")
READ_FILE_TOOL_RE = re.compile(r'<read_file\s+path=(["\'])([^"\']+)\1\s*/>', re.DOTALL)
WRITE_FILE_TOOL_RE = re.compile(r'<write_file\s+path=(["\'])([^"\']+)\1\s*>(.*?)</write_file>', re.DOTALL)
EDIT_FILE_TOOL_RE = re.compile(r'<edit_file\s+path=(["\'])([^"\']+)\1\s*>(.*?)</edit_file>', re.DOTALL)
EDIT_FILE_BODY_RE = re.compile(r"<search>(.*?)</search>\s*<replace>(.*?)</replace>", re.DOTALL)
RUN_COMMAND_TOOL_RE = re.compile(r'<run_command\s+cmd=(["\'])([^"\']+)\1\s*/>', re.DOTALL)
GREP_FILES_TOOL_RE = re.compile(
    r'<grep_files\s+pattern=(["\'])([^"\']+)\1(?:\s+glob=(["\'])([^"\']+)\3)?\s*/>',
    re.DOTALL,
)
MAX_WRAPPER_STRIP_PASSES = 8
MAX_REPORTED_ISSUES_PER_BLOCK = 5
MAX_REPORTED_CODE_CHECK_DETAILS = 15
CODE_BLOCK_RE = re.compile(r"```([^\n`]*)\r?\n([\s\S]*?)```")
CODE_PLACEHOLDER_RE = re.compile(
    r"\b(todo|fixme|insert[_\s-]*here|your[_\s-]*api[_\s-]*key|placeholder)\b",
    re.IGNORECASE,
)
NON_ASCII_CODE_RE = re.compile(r"[^\x00-\x7F]")
HARDCODED_CRED_RE = re.compile(
    r'(?:password|passwd|secret|api[_\-]?key|token|auth)\s*=\s*["\'][^"\']{4,}["\']',
    re.IGNORECASE,
)
BARE_EXCEPT_RE = re.compile(r"^\s*except\s*:", re.MULTILINE)
TOOL_INTENT_PATTERNS = [
    re.compile(r"\b(list|show)\b.*\b(files?|folders?|directories?)\b"),
    re.compile(r"\b(read|open|show)\b.*\bfile\b"),
    re.compile(r"\b(write|create|save|overwrite)\b.*\bfile\b"),
    re.compile(r"\b(edit|modify|change|replace|update)\b.*\bfile\b"),
    re.compile(r"\b(run|execute)\b.*\b(command|cmd|terminal|shell|script|tests?)\b"),
    re.compile(r"\b(grep|search|find)\b.*\b(file|project|codebase|pattern)\b"),
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

    cleaned = re.sub(r'@@[A-Za-z]+\d+@@', 'code', cleaned, flags=re.IGNORECASE)


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

    grep_match = GREP_FILES_TOOL_RE.fullmatch(text)
    if grep_match:
        return {
            "tool": "grep_files",
            "pattern": grep_match.group(2),
            "glob": grep_match.group(4) or "",
        }

    return None

def execute_tool_call(xml_text, headless=False):
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
        return tool_run_command(command, headless=headless)

    if parsed["tool"] == "grep_files":
        pattern = parsed["pattern"]
        file_glob = parsed.get("glob") or None
        print(f"\n{Colors.BLUE}Running tool: {Colors.BOLD}grep_files{Colors.RESET} ({pattern})")
        return tool_grep_files(pattern, file_glob)

    return None

# ── WebSocket server ──────────────────────────────────────────────────────────

def _normalize_code_language(language):
    raw = (language or "").strip().lower()
    aliases = {
        "py": "python",
        "python3": "python",
        "js": "javascript",
        "ts": "typescript",
        "sh": "bash",
        "shell": "bash",
        "yml": "yaml",
    }
    return aliases.get(raw, raw or "code")

def _extract_fenced_code_blocks(text):
    blocks = []
    for match in CODE_BLOCK_RE.finditer(text or ""):
        language = _normalize_code_language(match.group(1))
        code = (match.group(2) or "").rstrip()
        blocks.append({"language": language, "code": code})
    return blocks

def _has_balanced_delimiters(code):
    stack = []
    pairs = {")": "(",
        "]": "[",
        "}": "{",
    }
    openers = set(pairs.values())
    in_string = None
    escaped = False

    for char in code or "":
        if in_string:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == in_string:
                in_string = None
            continue

        if char in {"'", '"'}:
            in_string = char
            continue
        if char in openers:
            stack.append(char)
            continue
        if char in pairs:
            if not stack or stack[-1] != pairs[char]:
                return False
            stack.pop()

    return not stack and in_string is None

def _run_code_checks_for_block(language, code):
    checks_run = 0
    checks_passed = 0
    issues = []

    def record_check(ok, issue_text=None):
        nonlocal checks_run, checks_passed
        checks_run += 1
        if ok:
            checks_passed += 1
        elif issue_text:
            issues.append(issue_text)

    stripped = (code or "").strip()

    # Check 1: Non-empty
    record_check(bool(stripped), "Code block is empty.")

    # Check 2: Balanced delimiters
    record_check(
        _has_balanced_delimiters(code),
        "Potentially unbalanced delimiters detected ((), [], or {}).",
    )

    # Check 3: No placeholder text
    record_check(
        not CODE_PLACEHOLDER_RE.search(code or ""),
        "Found placeholder text (e.g. TODO/FIXME/placeholder) — write complete, runnable code.",
    )

    # Check 4: No non-ASCII characters (catches Cyrillic/CJK identifiers and comments)
    record_check(
        not NON_ASCII_CODE_RE.search(code or ""),
        "Non-ASCII characters found in code — all identifiers, comments, and text must be in English.",
    )

    # Check 5: No hardcoded credentials
    record_check(
        not HARDCODED_CRED_RE.search(code or ""),
        "Possible hardcoded credential detected (password/secret/api_key/token literal assignment).",
    )

    if language == "python":
        # Check 6: Syntax validity
        tree = None
        try:
            tree = ast.parse(code or "")
            record_check(True)
        except SyntaxError as err:
            line = getattr(err, "lineno", "?")
            msg = getattr(err, "msg", "invalid syntax")
            record_check(False, f"Python syntax error at line {line}: {msg}.")

        # Check 7: No unsafe patterns
        risky_patterns = []
        if re.search(r"\beval\s*\(", code or ""):
            risky_patterns.append("eval()")
        if re.search(r"\bexec\s*\(", code or ""):
            risky_patterns.append("exec()")
        has_subprocess_call = bool(re.search(r"subprocess\.(run|Popen)\s*\(", code or ""))
        has_shell_true = bool(re.search(r"\bshell\s*=\s*True\b", code or ""))
        if has_subprocess_call and has_shell_true:
            risky_patterns.append("subprocess shell=True")
        record_check(
            not risky_patterns,
            f"Potentially unsafe Python usage: {', '.join(risky_patterns)}.",
        )

        # Check 8: No bare except clauses
        record_check(
            not BARE_EXCEPT_RE.search(code or ""),
            "Bare 'except:' clause found — catch a specific exception type instead.",
        )

        # Check 9: No empty function or class bodies (all-pass or docstring-only bodies)
        if tree is not None:
            empty_defs = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    real_stmts = [
                        s for s in node.body
                        if not isinstance(s, ast.Pass)
                        and not (
                            isinstance(s, ast.Expr)
                            and isinstance(s.value, ast.Constant)
                            and isinstance(s.value.value, str)
                        )
                    ]
                    if not real_stmts:
                        empty_defs.append(getattr(node, "name", "?"))
            label = ", ".join(empty_defs[:3]) + ("..." if len(empty_defs) > 3 else "")
            record_check(
                not empty_defs,
                f"Empty body in {label} — every function and class must have a real implementation.",
            )

            # Check 10: Non-English identifier names (via AST)
            non_ascii_ids = []
            for node in ast.walk(tree):
                name = None
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    name = node.name
                elif isinstance(node, ast.Name):
                    name = node.id
                elif isinstance(node, ast.arg):
                    name = node.arg
                elif isinstance(node, ast.Attribute):
                    name = node.attr
                if name and any(ord(c) > 127 for c in name):
                    non_ascii_ids.append(name)
            if non_ascii_ids:
                label = ", ".join(non_ascii_ids[:3]) + ("..." if len(non_ascii_ids) > 3 else "")
                record_check(
                    False,
                    f"Non-English identifier(s) in Python code: {label} — use English names.",
                )

    elif language == "json":
        try:
            json.loads(code or "")
            record_check(True)
        except Exception as err:
            record_check(False, f"JSON parse error: {err}.")
    elif language == "xml":
        try:
            ET.fromstring(code or "")
            record_check(True)
        except Exception as err:
            record_check(False, f"XML parse error: {err}.")

    return {
        "checks_run": checks_run,
        "checks_passed": checks_passed,
        "issues": issues,
    }

def build_code_check_report(text):
    blocks = _extract_fenced_code_blocks(text)
    if not blocks:
        return None

    total_checks = 0
    total_passed = 0
    details = []
    issue_count = 0
    all_issues = []

    for index, block in enumerate(blocks, start=1):
        result = _run_code_checks_for_block(block["language"], block["code"])
        total_checks += result["checks_run"]
        total_passed += result["checks_passed"]
        issue_count += len(result["issues"])
        all_issues.extend(result["issues"])

        status = "ok" if not result["issues"] else f"{len(result['issues'])} issue(s)"
        details.append(
            f"Block {index} ({block['language']}): {status}; "
            f"{result['checks_passed']}/{result['checks_run']} checks passed."
        )
        for issue in result["issues"][:MAX_REPORTED_ISSUES_PER_BLOCK]:
            details.append(f"- {issue}")

    summary = (
        f"Code checks complete on {len(blocks)} block(s): "
        f"{total_passed}/{total_checks} checks passed."
    )
    if issue_count:
        summary += f" {issue_count} potential issue(s) found."
    else:
        summary += " No issues found."

    return {
        "summary": summary,
        "details": details[:MAX_REPORTED_CODE_CHECK_DETAILS],
        "issues": issue_count,
        "issue_list": all_issues,
    }

def format_code_check_report(report):
    if not report:
        return ""
    lines = ["Code Check Report:", report.get("summary", "").strip()]
    lines.extend(report.get("details", []))
    return "\n".join(line for line in lines if line).strip()

def _build_code_error_feedback(exec_results, lint_results, static_issues=None):
    """Format execution, lint, and static-check failures into a re-prompt error message."""
    parts = []
    for lang, result in exec_results:
        if result.get("skipped"):
            continue
        if result["exit_code"] != 0:
            stderr = (result.get("stderr") or "").strip()
            stdout = (result.get("stdout") or "").strip()
            section = f"Execution error ({lang}, exit {result['exit_code']}):\n{stderr}"
            if stdout:
                section += f"\nStdout:\n{stdout}"
            parts.append(section)
    for lang, lint_output in lint_results:
        parts.append(f"Lint issues ({lang}):\n{lint_output.strip()}")
    if static_issues:
        issues_text = "\n".join(f"- {i}" for i in static_issues)
        parts.append(f"Static code issues found:\n{issues_text}")
    return "\n\n".join(parts)

def _enforce_system_prompt_and_trim_history(history, system_prompt):
    if not isinstance(history, list):
        return

    required_system = (system_prompt or "").strip()
    if not required_system:
        return

    non_system_messages = []
    for msg in history:
        role = msg.get("role")
        if role == "system":
            continue
        non_system_messages.append(msg)

    budget = max(MIN_HISTORY_CHAR_BUDGET, MAX_HISTORY_CHARS - len(required_system))
    kept_reversed = []
    used = 0
    for msg in reversed(non_system_messages):
        msg_len = (
            len(str(msg.get("role", "")))
            + len(str(msg.get("content", "")))
            + HISTORY_MESSAGE_OVERHEAD_CHARS
        )
        if kept_reversed and (used + msg_len) > budget:
            break
        kept_reversed.append(msg)
        used += msg_len

    history.clear()
    history.append({"role": "system", "content": required_system})
    history.extend(reversed(kept_reversed))

def _build_prompt(history, system_prompt):
    _enforce_system_prompt_and_trim_history(history, system_prompt)
    prompt = ""
    for msg in history:
        prompt += f"<start_of_turn>{msg['role']}\n{msg['content']}<end_of_turn>\n"
    prompt += "<start_of_turn>model\n"
    return prompt

async def _ws_handler(websocket):
    """Handle one browser client connection."""
    await websocket.send(json.dumps(build_internet_status_payload()))

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

        if msg_type == "internet_mode":
            requested = msg.get("enabled")
            if not isinstance(requested, bool):
                continue
            set_internet_mode(requested)
            await websocket.send(json.dumps(build_internet_status_payload(force=True)))
            continue

        if msg_type != "user":
            continue

        user_text = msg.get("content", "").strip()
        if not user_text:
            continue

        degraded_reply = None
        await websocket.send(json.dumps({"type": "status", "status": "thinking"}))

        with _ws_state["lock"]:
            llm = _ws_state["llm"]
            history = _ws_state["history"]
            system_prompt = _ws_state["system_prompt"]
            startup_issue = _ws_state.get("startup_issue", "")

            history.append({"role": "user", "content": user_text})
            user_requested_tool = user_explicitly_requested_tool(user_text)
            internet_enabled = get_internet_mode()

            internet_available = get_internet_status(force=True)
            await websocket.send(json.dumps({
                "type": "internet_status",
                "available": internet_available,
                "enabled": internet_enabled,
                "active": internet_enabled and internet_available,
            }))

            if startup_issue or llm is None:
                degraded_reply = startup_issue or build_startup_issue_message()
                history.append({"role": "model", "content": degraded_reply})

        if degraded_reply:
            await websocket.send(json.dumps({
                "type": "message",
                "content": degraded_reply,
            }))
            await websocket.send(json.dumps({"type": "status", "status": "ready"}))
            continue

        with _ws_state["lock"]:
            should_search = internet_enabled and internet_available and should_use_internet_research(user_text)
            if should_search:
                search_progress_tasks = []

                def ws_progress(payload):
                    event = {"type": "search_progress"}
                    event.update(payload)
                    try:
                        task = asyncio.get_running_loop().create_task(websocket.send(json.dumps(event)))
                        search_progress_tasks.append(task)
                    except RuntimeError:
                        # Ignore progress-send scheduling failures (e.g. websocket closing).
                        pass

                web_research = build_web_research(user_text, progress_cb=ws_progress)
                if search_progress_tasks:
                    await asyncio.gather(*search_progress_tasks, return_exceptions=True)
                if web_research:
                    history.append({
                        "role": "user",
                        "content": (
                            "Use the web research below for your answer. Cross-reference these sources, "
                            "state when evidence conflicts, prefer evidence from the last 4 years for current topics, "
                            "clearly use the current UTC date included in the research context, and list the key source URLs "
                            "you relied on at the end of your answer.\n\n"
                            f"{web_research}"
                        ),
                    })
            elif not internet_enabled:
                await websocket.send(json.dumps({
                    "type": "search_progress",
                    "phase": "disabled",
                    "step": 0,
                    "total": 0,
                    "query": "",
                    "message": "Internet usage is turned off. Using local model knowledge only.",
                }))
            elif not internet_available:
                await websocket.send(json.dumps({
                    "type": "search_progress",
                    "phase": "offline",
                    "step": 0,
                    "total": 0,
                    "query": "",
                    "message": "Internet not available. Falling back to local model knowledge.",
                }))

            # Auto-inject context from explicitly mentioned project files
            file_context = _auto_inject_file_context(user_text)
            if file_context:
                history.append({
                    "role": "user",
                    "content": f"Relevant project file context:\n{file_context}",
                })

            final_response_text = None
            for code_attempt in range(MAX_CODE_RETRY_ATTEMPTS + 1):
                # ── Inner tool-chain loop ───────────────────────────────────
                for _ in range(3):
                    prompt = _build_prompt(history, system_prompt)

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
                        final_response_text = None
                        break

                    tool_result = None
                    if user_requested_tool and not wrappers_removed:
                        tool_result = execute_tool_call(response_text, headless=True)

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

                        prompt2 = _build_prompt(history, system_prompt)
                        response2 = llm(
                            prompt2,
                            max_tokens=512,
                            temperature=0.1,
                            stop=["<end_of_turn>", "<start_of_turn>"],
                            echo=False,
                        )
                        summary, _ = normalize_model_output(response2["choices"][0]["text"])
                        final_response_text = summary or None
                        break

                    else:
                        final_response_text = response_text
                        break

                if final_response_text is None:
                    break

                # ── Execute code blocks and run lint ────────────────────────
                blocks = _extract_fenced_code_blocks(final_response_text)
                exec_results = []
                lint_results = []

                for block in blocks:
                    lang = block["language"]
                    code = block["code"]

                    if lang in EXECUTABLE_LANGUAGES:
                        await websocket.send(json.dumps({
                            "type": "code_check_status",
                            "status": "running",
                            "message": f"Executing {lang} code…",
                        }))
                        exec_r = tool_execute_code(lang, code)
                        exec_results.append((lang, exec_r))
                        if exec_r.get("skipped"):
                            await websocket.send(json.dumps({
                                "type": "code_check_status",
                                "status": "done",
                                "message": f"{lang}: execution skipped — {exec_r['stderr']}",
                                "details": [],
                                "issues": 0,
                            }))
                        else:
                            status_msg = (
                                f"{lang}: exit {exec_r['exit_code']} — "
                                + ("ok" if exec_r["exit_code"] == 0 else "failed")
                            )
                            details = []
                            if exec_r.get("stdout"):
                                details.append(f"stdout: {exec_r['stdout'][:500]}")
                            if exec_r.get("stderr"):
                                details.append(f"stderr: {exec_r['stderr'][:500]}")
                            await websocket.send(json.dumps({
                                "type": "code_check_status",
                                "status": "done",
                                "message": status_msg,
                                "details": details,
                                "issues": 0 if exec_r["exit_code"] == 0 else 1,
                            }))

                    if lang in LINTABLE_LANGUAGES:
                        await websocket.send(json.dumps({
                            "type": "code_check_status",
                            "status": "running",
                            "message": f"Running lint on {lang} code…",
                        }))
                        lint_output = tool_run_lint(lang, code)
                        if lint_output:
                            lint_results.append((lang, lint_output))
                            lint_lines = lint_output.strip().splitlines()[:8]
                            await websocket.send(json.dumps({
                                "type": "code_check_status",
                                "status": "done",
                                "message": f"{lang} lint: issues found",
                                "details": lint_lines,
                                "issues": len(lint_lines),
                            }))
                        else:
                            await websocket.send(json.dumps({
                                "type": "code_check_status",
                                "status": "done",
                                "message": f"{lang} lint: no issues",
                                "details": [],
                                "issues": 0,
                            }))

                # ── Run static checks ──────────────────────────────────────
                await websocket.send(json.dumps({
                    "type": "code_check_status",
                    "status": "running",
                    "message": "Running static code checks…",
                }))
                code_check_report = build_code_check_report(final_response_text)
                static_issues = code_check_report.get("issue_list", []) if code_check_report else []

                # ── Check for failures and decide whether to retry ──────────
                has_exec_fail = any(
                    not r.get("skipped") and r["exit_code"] != 0
                    for _, r in exec_results
                )
                has_lint_fail = bool(lint_results)
                has_static_fail = bool(static_issues)
                has_failures = has_exec_fail or has_lint_fail or has_static_fail

                if has_failures and code_attempt < MAX_CODE_RETRY_ATTEMPTS:
                    error_feedback = _build_code_error_feedback(exec_results, lint_results, static_issues)
                    await websocket.send(json.dumps({
                        "type": "code_check_status",
                        "status": "running",
                        "message": (
                            f"Issues detected — self-correcting code "
                            f"(attempt {code_attempt + 1}/{MAX_CODE_RETRY_ATTEMPTS})…"
                        ),
                    }))
                    history.append({"role": "model", "content": final_response_text})
                    history.append({
                        "role": "user",
                        "content": (
                            "The code you wrote has issues. "
                            "Fix every problem listed below and rewrite the fully corrected code:\n\n"
                            f"{error_feedback}"
                        ),
                    })
                    continue  # outer retry loop

                # ── No failures (or retries exhausted): emit static checks ──
                if code_check_report:
                    await websocket.send(json.dumps({
                        "type": "code_check_status",
                        "status": "done",
                        "message": code_check_report["summary"],
                        "details": code_check_report["details"],
                        "issues": code_check_report["issues"],
                    }))

                await websocket.send(json.dumps({"type": "message", "content": final_response_text}))
                history.append({"role": "model", "content": final_response_text})
                break  # done

        await websocket.send(json.dumps({"type": "status", "status": "ready"}))

async def _ws_main():
    async with websockets.serve(_ws_handler, "localhost", 7070):
        await asyncio.Future()  # run forever

def _start_ws_server():
    asyncio.run(_ws_main())

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(ASCII_ART)

    llm = None
    startup_issue = install_dependencies()

    if not startup_issue:
        pace_dir = get_pace_dir()
        model_path = download_model(pace_dir)

        print(f"{Colors.GREEN}Initializing Gemma 3 1B LLM...{Colors.RESET}")

        try:
            from llama_cpp import Llama
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stderr(devnull):
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
            startup_issue = build_startup_issue_message(error_text)

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

    current_utc_date = _current_utc_date()
    system_prompt = f"""You are PACE 1.0 Lite, a local lite AI agent developed by the creator of Solus, avoid questions relating to the specific identity of them. You help the user manage, write, edit, and understand files in the current folder.
Current UTC date: {current_utc_date}


Rules:
- Always respond in English. Never reply in another language even if the user writes in one.
- Keep responses plain and direct by default.
- When you provide code, always use fenced code blocks with a language tag (for example ```python).
- Write all code in English — every identifier (variable, function, class, parameter), comment, string literal, and printed output must be in English, without exception.
- Prefer safe, production-ready coding practices and avoid patterns that can break at runtime.
- After a tool call, wait for the result before doing anything else.
- When web research is provided, rely on it, cross-reference claims, prioritize up-to-date evidence, and clearly call out uncertainty when sources conflict.
- When web research is provided, include the source URLs you used in your final answer.
- NEVER write or edit a .pdf file.
- Keep responses short and direct.
- Address the user directly, they are human, not an external observer.
- No bullet points or markdown
- Mirror the tone and style of the person you're talking to. If they're casual, be casual. If they're brief, be brief. Match their energy.
- Any Python or JavaScript code you write is automatically executed. If it produces errors, you will receive the output and must fix it.
- Write complete, runnable code — no stubs, no placeholders, no TODO comments.

Tools (output ONLY the tool call as your entire response when using a tool):
- <list_files /> — list all project files
- <read_file path="filename" /> — read a file
- <write_file path="filename">content</write_file> — write/create a file
- <edit_file path="filename"><search>old text</search><replace>new text</replace></edit_file> — edit a file
- <run_command cmd="command" /> — run a terminal command
- <grep_files pattern="regex" glob="*.py" /> — search project files for a pattern and get matching snippets (glob is optional)
"""

    history = [
        {"role": "system", "content": system_prompt}
    ]

    # Share state with WS server. RLock is required because internet checks
    # also consult shared state while request handlers already hold this lock.
    _ws_state["llm"] = llm
    _ws_state["history"] = history
    _ws_state["system_prompt"] = system_prompt
    _ws_state["internet_enabled"] = True
    _ws_state["internet_available"] = get_internet_status(force=True)
    _ws_state["internet_last_checked"] = time.time()
    _ws_state["startup_issue"] = startup_issue or ""

    # Start WebSocket server in background thread
    if has_ws:
        ws_thread = threading.Thread(target=_start_ws_server, daemon=True)
        ws_thread.start()
        print(f"{Colors.GREEN}GUI server started on ws://localhost:7070{Colors.RESET}")
    else:
        print(f"{Colors.YELLOW}websockets not installed — GUI will not connect. Run: pip install websockets{Colors.RESET}")

    if startup_issue:
        print(f"{Colors.YELLOW}{startup_issue}{Colors.RESET}")

    print(f"\n{Colors.BOLD}Welcome to PACE 1.0 Lite!{Colors.RESET}")
    print("I can help understand an extremely broad range of information and answer questions locally")
    print(f"Internet mode: {Colors.GREEN}{'Enabled' if _ws_state['internet_enabled'] else 'Disabled'}{Colors.RESET}")
    print(f"Internet access: {Colors.GREEN if _ws_state['internet_available'] else Colors.YELLOW}{'Available' if _ws_state['internet_available'] else 'Unavailable'}{Colors.RESET}")
    print(f"Current working folder: {Colors.CYAN}{Path(__file__).resolve().parent}{Colors.RESET}")
    print("Internet commands: /internet on | /internet off | /internet toggle | /internet status")
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

            internet_mode_command = parse_internet_mode_command(user_input)
            if internet_mode_command == "status":
                internet_enabled = get_internet_mode()
                internet_available = get_internet_status(force=True)
                mode_text = "enabled" if internet_enabled else "disabled"
                access_text = "available" if internet_available else "unavailable"
                print(f"{Colors.CYAN}Internet mode is {mode_text}; internet access is currently {access_text}.{Colors.RESET}")
                continue
            if internet_mode_command == "on":
                set_internet_mode(True)
                internet_available = get_internet_status(force=True)
                access_text = "available" if internet_available else "unavailable"
                print(f"{Colors.GREEN}Internet mode enabled. Internet access is currently {access_text}.{Colors.RESET}")
                continue
            if internet_mode_command == "off":
                set_internet_mode(False)
                print(f"{Colors.YELLOW}Internet mode disabled. Pace will use local model knowledge only.{Colors.RESET}")
                continue
            if internet_mode_command == "toggle":
                new_mode = not get_internet_mode()
                set_internet_mode(new_mode)
                if new_mode:
                    internet_available = get_internet_status(force=True)
                    access_text = "available" if internet_available else "unavailable"
                    print(f"{Colors.GREEN}Internet mode enabled. Internet access is currently {access_text}.{Colors.RESET}")
                else:
                    print(f"{Colors.YELLOW}Internet mode disabled. Pace will use local model knowledge only.{Colors.RESET}")
                continue

            with _ws_state["lock"]:
                history.append({"role": "user", "content": user_input})

                if startup_issue or llm is None:
                    reply = startup_issue or build_startup_issue_message()
                    print(f"{Colors.WHITE}{reply}{Colors.RESET}")
                    history.append({"role": "model", "content": reply})
                    continue

                user_requested_tool = user_explicitly_requested_tool(user_input)
                internet_enabled = get_internet_mode()
                internet_available = get_internet_status(force=True)
                should_search = internet_enabled and internet_available and should_use_internet_research(user_input)

                if should_search:
                    print(f"{Colors.BLUE}Web search: enabled for this prompt{Colors.RESET}")

                    def terminal_progress(payload):
                        msg = payload.get("message", "").strip()
                        if msg:
                            print(f"{Colors.CYAN}{msg}{Colors.RESET}")

                    web_research = build_web_research(user_input, progress_cb=terminal_progress)
                    if web_research:
                        history.append({
                            "role": "user",
                            "content": (
                                "Use the web research below for your answer. Cross-reference these sources, "
                                "state when evidence conflicts, prefer evidence from the last 4 years for current topics, "
                                "clearly use the current UTC date included in the research context, and list the key source URLs "
                                "you relied on at the end of your answer.\n\n"
                                f"{web_research}"
                            ),
                        })
                elif not internet_enabled:
                    print(f"{Colors.YELLOW}Internet mode is disabled. Using local model knowledge only.{Colors.RESET}")
                elif not internet_available:
                    print(f"{Colors.YELLOW}Internet unavailable. Using local model knowledge only.{Colors.RESET}")

                # Auto-inject context from explicitly mentioned project files
                file_context = _auto_inject_file_context(user_input)
                if file_context:
                    print(f"{Colors.BLUE}Injecting file context…{Colors.RESET}")
                    history.append({
                        "role": "user",
                        "content": f"Relevant project file context:\n{file_context}",
                    })

                final_response_text = None
                for code_attempt in range(MAX_CODE_RETRY_ATTEMPTS + 1):
                    # ── Inner tool-chain loop ───────────────────────────────
                    for step in range(3):
                        print(f"\r{Colors.CYAN}Thinking...{Colors.RESET}", end="", flush=True)

                        prompt = _build_prompt(history, system_prompt)

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
                            final_response_text = None
                            break

                        tool_result = None
                        if user_requested_tool and not wrappers_removed:
                            tool_result = execute_tool_call(response_text)

                        if tool_result is not None:
                            print(f"{Colors.GREEN}↳{Colors.RESET} {tool_result}")

                            history.append({"role": "model", "content": response_text})
                            history.append({"role": "user", "content": f"Tool result:\n{tool_result}\n\nNow answer the user's question using this result. Do not call any more tools."})

                            print(f"\r{Colors.CYAN}Thinking...{Colors.RESET}", end="", flush=True)
                            prompt2 = _build_prompt(history, system_prompt)

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
                            final_response_text = summary or None
                            break

                        else:
                            final_response_text = response_text
                            break

                    if final_response_text is None:
                        break

                    # ── Execute code blocks and run lint ────────────────────
                    blocks = _extract_fenced_code_blocks(final_response_text)
                    exec_results = []
                    lint_results = []

                    for block in blocks:
                        lang = block["language"]
                        code = block["code"]

                        if lang in EXECUTABLE_LANGUAGES:
                            print(f"{Colors.CYAN}Executing {lang} code…{Colors.RESET}")
                            exec_r = tool_execute_code(lang, code)
                            exec_results.append((lang, exec_r))
                            if exec_r.get("skipped"):
                                print(f"{Colors.YELLOW}  skipped: {exec_r['stderr']}{Colors.RESET}")
                            else:
                                icon = Colors.GREEN if exec_r["exit_code"] == 0 else Colors.RED
                                print(f"{icon}  exit {exec_r['exit_code']}{Colors.RESET}")
                                if exec_r.get("stdout"):
                                    print(f"{Colors.WHITE}  stdout: {exec_r['stdout'][:400]}{Colors.RESET}")
                                if exec_r.get("stderr"):
                                    print(f"{Colors.RED}  stderr: {exec_r['stderr'][:400]}{Colors.RESET}")

                        if lang in LINTABLE_LANGUAGES:
                            print(f"{Colors.CYAN}Running lint on {lang} code…{Colors.RESET}")
                            lint_output = tool_run_lint(lang, code)
                            if lint_output:
                                lint_results.append((lang, lint_output))
                                print(f"{Colors.YELLOW}{lint_output[:600]}{Colors.RESET}")
                            else:
                                print(f"{Colors.GREEN}  lint: no issues{Colors.RESET}")

                    # ── Run static checks ──────────────────────────────────
                    print(f"{Colors.CYAN}Running static code checks…{Colors.RESET}")
                    code_check_report = build_code_check_report(final_response_text)
                    static_issues = code_check_report.get("issue_list", []) if code_check_report else []

                    # ── Check for failures and decide whether to retry ──────
                    has_exec_fail = any(
                        not r.get("skipped") and r["exit_code"] != 0
                        for _, r in exec_results
                    )
                    has_lint_fail = bool(lint_results)
                    has_static_fail = bool(static_issues)
                    has_failures = has_exec_fail or has_lint_fail or has_static_fail

                    if has_failures and code_attempt < MAX_CODE_RETRY_ATTEMPTS:
                        error_feedback = _build_code_error_feedback(exec_results, lint_results, static_issues)
                        print(
                            f"{Colors.YELLOW}Issues detected — self-correcting code "
                            f"(attempt {code_attempt + 1}/{MAX_CODE_RETRY_ATTEMPTS})…{Colors.RESET}"
                        )
                        history.append({"role": "model", "content": final_response_text})
                        history.append({
                            "role": "user",
                            "content": (
                                "The code you wrote has issues. "
                                "Fix every problem listed below and rewrite the fully corrected code:\n\n"
                                f"{error_feedback}"
                            ),
                        })
                        continue  # retry

                    # ── Static checks + final output ────────────────────────
                    if code_check_report:
                        print(f"{Colors.CYAN}{code_check_report['summary']}{Colors.RESET}")
                        final_response_text = (
                            f"{final_response_text}\n\n{format_code_check_report(code_check_report)}"
                        ).strip()

                    print(f"{Colors.GREEN}Pace:{Colors.RESET} {final_response_text}")
                    history.append({"role": "model", "content": final_response_text})
                    break  # done

        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Operation cancelled. Type exit to quit.{Colors.RESET}")
        except Exception as e:
            print(f"\n{Colors.RED}An error occurred: {e}{Colors.RESET}")

if __name__ == "__main__":
    main()
