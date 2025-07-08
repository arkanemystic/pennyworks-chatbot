# Backend logic for CSV-to-API operations and LLM intent detection
import os
import re
import tempfile
import importlib.util
import asyncio
from typing import Optional
import sys
import subprocess
import logging
import requests

# Dynamically add the absolute path to the 'csv2api' folder to sys.path for 'src' imports
csv2api_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'csv2api'))
if csv2api_path not in sys.path:
    sys.path.insert(0, csv2api_path)

# Import csv2api router (assume submodule is present)
CSV2API_ROUTER_PATH = os.path.abspath(os.path.join(csv2api_path, 'src', 'pipeline', 'router.py'))
spec = importlib.util.spec_from_file_location("router", CSV2API_ROUTER_PATH)
router = importlib.util.module_from_spec(spec)
spec.loader.exec_module(router)

# Dynamically import and execute csv2api/src/main.py as a module
import importlib.util
main_path = os.path.abspath(os.path.join(csv2api_path, 'src', 'main.py'))
main_spec = importlib.util.spec_from_file_location("csv2api_main", main_path)
csv2api_main = importlib.util.module_from_spec(main_spec)
main_spec.loader.exec_module(csv2api_main)

# Penny persona system prompt for DeepSeek R1
PENNY_SYSTEM_PROMPT = (
    "You are Penny, a friendly, respectful, and helpful AI assistant for crypto-based accounting firms. "
    "Your tone is enthusiastic, warm, and professional, always using PG-appropriate language. "
    "Stay on topic: accounting, bookkeeping, crypto transactions, and relevant finance topics. "
    "Never give speculative investment advice, always maintain user confidentiality, and defer to CSV2API for transaction routing. "
    "Behavior traits: Greet users warmly and show appreciation for their time. Clarify requests politely if unclear. "
    "Respond with concise, accurate financial help in crypto contexts. Use emoji sparingly for warmth (e.g., 🙂, 📊), but never excessively. "
    "Reaffirm safety, compliance, and professionalism in all interactions. Never generate or respond to inappropriate, off-topic, or NSFW content."
)

def build_penny_prompt(user_input: str) -> str:
    """
    Compose a prompt for DeepSeek R1 that includes Penny's persona and the user's message.
    """
    return f"{PENNY_SYSTEM_PROMPT}\n\nUser: {user_input}\nPenny:"

# Helper: Run csv2api router (SYNC version for UI)
def run_csv2api(csv_path: str, user_intent: Optional[str] = None):
    """
    Synchronous version for UI: Save uploaded_file (file-like object or str path) to a temp file if needed, run csv2api pipeline via subprocess, and return a summary for the chatbot UI.
    Always return a string, not a coroutine.
    """
    import tempfile
    import subprocess
    import os
    import logging
    logger = logging.getLogger("pennyworks")
    tmp_path = None
    try:
        # If csv_path is a str, treat as path; if it has .read, treat as file-like
        if hasattr(csv_path, 'read'):
            with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".csv") as tmp:
                tmp.write(csv_path.read())
                tmp_path = tmp.name
        else:
            tmp_path = csv_path
        result = subprocess.run(
            ["python3", "-m", "src.main", "-i", tmp_path],
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "csv2api")),
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            logger.error(f"csv2api failed: {result.stderr.strip()}")
            return f"csv2api failed: {result.stderr.strip()}"
        summary_lines = [
            line.strip() for line in result.stdout.splitlines()
            if ("Successfully processed" in line) or ("Failed rows" in line)
        ]
        if not summary_lines:
            summary_lines = ["csv2api completed, but no summary lines found."]
        logger.info(f"csv2api summary: {' | '.join(summary_lines)}")
        return "\n".join(summary_lines)
    except Exception as e:
        logger.error(f"Exception in run_csv2api: {e}")
        return f"Exception in run_csv2api: {e}"
    finally:
        # Only delete temp file if we created it
        if tmp_path and hasattr(csv_path, 'read') and os.path.exists(tmp_path):
            os.unlink(tmp_path)

# Helper: Use LLM to summarize/simplify user intent
LLM_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "deepseek-r1:latest"
def get_simplified_intent(user_input: str) -> str:
    prompt = build_penny_prompt(f"Summarize the following user request as a compact CSV-to-API instruction: {user_input}")
    response = requests.post(
        LLM_URL,
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
        timeout=180
    )
    response.raise_for_status()
    result = response.json()
    # Remove <think>...</think> tags
    content = re.sub(r'<think>.*?</think>', '', result.get("response", ""), flags=re.DOTALL).strip()
    return content

# Helper: Detect if user input is a CSV-to-API request
CSV2API_KEYWORDS = ["convert", "api", "endpoint", "expose", "rest", "serve", "csv", "query"]
# Heuristic keywords for transaction-related CSV2API routing
CSV2API_TRANSACTION_KEYWORDS = [
    "tag", "label", "expense", "categorize", "parse", "process", "analyze", "summarize",
    "receipt", "lookup", "details", "transactions", "hash"
]
def is_csv2api_intent(user_input: str) -> bool:
    text = user_input.lower()
    return any(word in text for word in CSV2API_KEYWORDS)

def should_use_csv2api(user_message: str) -> bool:
    """
    Decide if the user message should trigger csv2api routing.
    Returns True if the message contains transaction-related keywords, or explicitly mentions csv2api/csv2api caller, and is not a general/greeting message.
    """
    msg = user_message.lower().strip()
    # Ignore if message is a greeting or general question
    general_phrases = [
        "hello", "hi", "hey", "how are you", "who are you", "what can you do", "help", "thanks", "thank you",
        "about you", "about this", "what is this", "explain yourself", "your name"
    ]
    if any(phrase in msg for phrase in general_phrases):
        return False
    # Trigger if explicit csv2api mention
    if "csv2api" in msg:
        return True
    # Must contain at least one transaction-related keyword
    return any(kw in msg for kw in CSV2API_TRANSACTION_KEYWORDS)

def simplify_for_csv2api(user_message: str) -> str:
    """
    Simplify and sanitize the user message for csv2api routing.
    Returns a short, imperative command or summary.
    """
    # Remove polite prefixes/suffixes and keep only the core request
    msg = re.sub(r"^(please|can you|could you|would you|kindly|hey|hi|hello)[, ]*", "", user_message, flags=re.IGNORECASE)
    msg = re.sub(r"[, ]*(please|thanks|thank you)[.!]*$", "", msg, flags=re.IGNORECASE)
    # Truncate to first sentence or 120 chars
    msg = msg.strip().split(". ")[0][:120]
    # If still too vague, add a fallback
    if not any(kw in msg.lower() for kw in CSV2API_TRANSACTION_KEYWORDS):
        return "Process transactions in the uploaded CSV."
    return msg.strip()

# Helper: LLM chat with Penny persona

def penny_llm_chat(user_input: str) -> str:
    """
    Send a user message to DeepSeek R1 with Penny's persona prompt and return the response.
    """
    prompt = build_penny_prompt(user_input)
    response = requests.post(
        LLM_URL,
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
        timeout=180
    )
    response.raise_for_status()
    result = response.json()
    # Remove <think>...</think> tags
    content = re.sub(r'<think>.*?</think>', '', result.get("response", ""), flags=re.DOTALL).strip()
    return content
