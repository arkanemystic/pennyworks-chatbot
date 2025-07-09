# Backend logic for CSV-to-API operations and LLM intent detection
import os
import re
import tempfile
import asyncio
from typing import Optional
import sys
import subprocess
import logging
import requests

# Penny persona system prompt for DeepSeek R1
PENNY_SYSTEM_PROMPT = (
    "You are Penny, a friendly, respectful, and helpful AI assistant for crypto-based accounting firms. "
    "Your tone is enthusiastic, warm, and professional, always using PG-appropriate language. "
    "Stay on topic: accounting, bookkeeping, crypto transactions, and relevant finance topics. "
    "Never give speculative investment advice, always maintain user confidentiality, and defer to CSV2API for transaction routing. "
    "Behavior traits: Greet users warmly and show appreciation for their time. Clarify requests politely if unclear. "
    "Respond with concise, accurate financial help in crypto contexts. Use emoji sparingly for warmth (e.g., 🙂, 📊), but never excessively. "
    "Reaffirm safety, compliance, and professionalism in all interactions. Never generate or respond to inappropriate, off-topic, or NSFW content. "
    "ROUTING INSTRUCTIONS: "
    "- If csv_ready is 'true' and the user wants to process CSV data, respond with exactly 'ROUTE_TO_CSV2API' "
    "- If csv_ready is 'false' and the user wants to process CSV data, respond with exactly 'CSV_REQUIRED' "
    "- Otherwise, provide helpful accounting assistance."
)

def build_penny_prompt(user_input: str, csv_path: str = None, csv_validated: bool = False) -> str:
    """
    Compose a prompt for DeepSeek R1 that includes Penny's persona, file context, and the user's message.
    Uses csv_validated flag to ensure accurate context about CSV availability.
    """
    if csv_validated and csv_path:
        file_context = f"IMPORTANT: A CSV file named '{os.path.basename(csv_path)}' is currently uploaded and ready for processing. The file exists and is valid."
        csv_ready = True
    else:
        file_context = "No CSV file has been uploaded yet."
        csv_ready = False
    
    return (
        f"{PENNY_SYSTEM_PROMPT}\n\n"
        f"{file_context}\n"
        f"csv_ready: {str(csv_ready).lower()}\n\n"
        f"User: {user_input}\nPenny:"
    )

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

def penny_llm_chat(user_input: str, csv_path: str = None) -> str:
    """
    Send a user message to DeepSeek R1 with Penny's persona prompt and return the response.
    Includes CSV file context for correct routing.
    """
    prompt = build_penny_prompt(user_input, csv_path)
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

def penny_llm_chat_with_validation(user_input: str, csv_path: str = None, csv_validated: bool = False) -> str:
    """
    Send a user message to DeepSeek R1 with Penny's persona prompt and validated CSV context.
    """
    prompt = build_penny_prompt(user_input, csv_path, csv_validated)
    logging.info(f"[DEBUG] Prompt sent to LLM: {repr(prompt)}")
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

# ---
# CSV2API CLI Integration (per csv2api/README.md)
def run_csv2api_cli(input_file: str = None, prompt: str = None, debug: bool = False):
    """
    Call csv2api's llm_client.py CLI as a subprocess, passing --input and/or --prompt.
    Parse stdout for API call lines and summary lines, and return structured results.
    """
    import subprocess
    import json
    import shlex
    import os
    csv2api_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'csv2api'))
    llm_client_path = os.path.join(csv2api_dir, 'src', 'pipeline', 'llm_client.py')
    # Use sys.executable to ensure the correct Python interpreter is used
    python_exec = sys.executable
    cmd = [python_exec, llm_client_path]
    if input_file:
        cmd += ['--input', input_file]
    if prompt:
        cmd += ['--prompt', prompt]
    if debug:
        cmd += ['--debug']
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=csv2api_dir)
    api_calls = []
    summary = {}
    for line in result.stdout.splitlines():
        if line.startswith('API call made:'):
            api_json = line[len('API call made:'):].strip()
            try:
                api_calls.append(json.loads(api_json))
            except Exception:
                pass
        elif line.startswith('Successfully processed'):
            try:
                summary['success'] = int(line.split()[2])
            except Exception:
                pass
        elif line.startswith('Failed rows:'):
            try:
                summary['failed'] = int(line.split(':')[1].strip())
            except Exception:
                pass
        elif line.startswith('{') and prompt:
            # Prompt output is pretty-printed JSON
            try:
                api_calls.append(json.loads(line))
            except Exception:
                pass
    return {
        'api_calls': api_calls,
        'summary': summary,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'returncode': result.returncode
    }

def is_valid_csv_file(csv_path: str) -> bool:
    """
    Checks if the CSV file exists, is non-empty, and has a plausible header line.
    """
    if not csv_path or not os.path.exists(csv_path):
        return False
    if os.path.getsize(csv_path) == 0:
        return False
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            header = f.readline()
            # Basic check: at least 2 columns, comma-separated
            if ',' in header and len(header.strip().split(',')) >= 2:
                return True
    except Exception:
        return False
    return False

def normalize_routing_decision(raw: str) -> str:
    """
    Normalize LLM routing decision: lowercase, strip, replace common phrases with canonical tokens, remove punctuation, collapse whitespace.
    """
    import string
    s = raw.lower().strip()
    # Replace common CSV-required phrases
    for phrase in [
        "please upload a csv", "upload a csv", "csv required", "csv is required", "csv needed", "waiting for csv", "csv file required", "please upload csv", "upload csv file", "csv file needed"
    ]:
        s = s.replace(phrase, "csv_required")
    # Replace route-to-csv2api phrases
    for phrase in [
        "route to csv2api", "route_to_csv2api", "send to csv2api", "process with csv2api", "use csv2api", "run csv2api", "forward to csv2api"
    ]:
        s = s.replace(phrase, "route_to_csv2api")
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = ' '.join(s.split())
    return s

def run_csv2api_pipeline(uploaded_csv_path: str, timeout: int = 180) -> str:
    """
    Execute the csv2api pipeline and return formatted results.
    """
    csv2api_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'csv2api'))
    cmd = [sys.executable, '-m', 'src.main', '-i', uploaded_csv_path]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=csv2api_dir,
            timeout=timeout
        )
        csv2api_output = result.stdout.strip()
        
        # Parse API calls from output
        api_calls = []
        for line in csv2api_output.splitlines():
            line = line.strip()
            if line.startswith('{') and 'api_call' in line:
                try:
                    import json
                    api_obj = json.loads(line)
                    api_calls.append(api_obj)
                except Exception:
                    pass
            elif line.startswith('API call made:'):
                try:
                    import json
                    api_obj = json.loads(line[len('API call made:'):].strip())
                    api_calls.append(api_obj)
                except Exception:
                    pass
        
        # Summarize API call types and counts
        if api_calls:
            call_summary = {}
            for call in api_calls:
                if isinstance(call, dict):
                    api_name = call.get('api_call') or call.get('function') or 'API call'
                    call_summary[api_name] = call_summary.get(api_name, 0) + 1
            
            reply_lines = ["\u2705 Processed CSV using `csv2api`"]
            for api_name, count in call_summary.items():
                reply_lines.append(f"• {api_name} — {count} call{'s' if count != 1 else ''}")
            return '\n'.join(reply_lines)
        else:
            return f"\u2705 Processed CSV using `csv2api`.\nOutput:\n{csv2api_output}" if csv2api_output else "\u26A0\uFE0F No output from csv2api pipeline."
    
    except subprocess.TimeoutExpired:
        return "Sorry, the CSV processing took too long and timed out."
    except Exception as e:
        return f"An error occurred while processing the CSV: {e}"

def handle_routing_decision(uploaded_csv_path: str, user_input: str, timeout: int = 180) -> str:
    """
    Enhanced routing logic with explicit CSV validation before LLM call.
    """
    # Step 1: Validate CSV file is ready
    wait_for_csv_ready(uploaded_csv_path)
    csv_is_valid = is_valid_csv_file(uploaded_csv_path)
    
    log_csv_file_state(uploaded_csv_path, context="at routing decision")
    logging.info(f"[DEBUG] CSV validation result: {csv_is_valid}")

    # Step 2: Get LLM routing decision with validated CSV state
    llm_response = penny_llm_chat_with_validation(user_input, uploaded_csv_path, csv_is_valid)
    raw_routing_decision = llm_response.strip()
    norm_decision = normalize_routing_decision(raw_routing_decision)
    
    logging.info(f"[ROUTING] Raw LLM response: {repr(raw_routing_decision)}")
    logging.info(f"[ROUTING] Normalized decision: {repr(norm_decision)}")

    # Step 3: Parse routing intent
    if "route_to_csv2api" in norm_decision:
        routing_decision = "route_to_csv2api"
    elif "csv_required" in norm_decision:
        if csv_is_valid:
            logging.warning("[DEBUG] LLM routing mismatch: said 'csv_required' but CSV is valid. Forcing route to csv2api.")
            routing_decision = "route_to_csv2api"
        else:
            routing_decision = "no_csv"
    else:
        # Heuristic: If LLM output is empty/generic but CSV exists, force csv2api
        generic_responses = [
            '', 'ok', 'got it', 'sure', 'processing', 'let me check', 'working on it',
            'i am not sure', 'i do not understand', 'can you clarify', 'please upload',
            'please upload a csv', 'csv required', 'csv needed', 'waiting for csv',
        ]
        if csv_is_valid and (norm_decision in generic_responses or len(norm_decision) < 10):
            logging.warning("LLM gave generic/empty response, but CSV is valid. Forcing route to csv2api.")
            routing_decision = "route_to_csv2api"
        elif not csv_is_valid:
            routing_decision = "no_csv"
        else:
            routing_decision = "direct_response"
    
    logging.info(f"[ROUTING] Final routing decision: {routing_decision}")

    # Step 4: Act on routing decision
    if routing_decision == "no_csv":
        return "Please upload a CSV file to continue."
    elif routing_decision == "route_to_csv2api":
        if not csv_is_valid:
            return "No valid CSV file found. Please upload a CSV file first."
        # Run csv2api pipeline
        return run_csv2api_pipeline(uploaded_csv_path, timeout)
    else:
        return raw_routing_decision

def log_csv_file_state(csv_path: str, context: str):
    """
    Log the state of the CSV file (exists, size, timestamp) for debugging upload/routing timing issues.
    """
    import datetime
    exists = os.path.exists(csv_path)
    size = os.path.getsize(csv_path) if exists else 0
    now = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    logging.info(f"[INFO] File '{csv_path}' at {context} ({now}), exists={exists}, size={size}")


def wait_for_csv_ready(csv_path: str, max_wait: float = 1.0, interval: float = 0.2) -> bool:
    """
    Wait up to max_wait seconds for the CSV file to exist and have non-zero size.
    Returns True if file is ready, False otherwise.
    Logs when file becomes ready.
    """
    import time
    import datetime
    waited = 0.0
    while waited < max_wait:
        exists = os.path.exists(csv_path)
        size = os.path.getsize(csv_path) if exists else 0
        if exists and size > 0:
            now = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
            logging.info(f"[INFO] File became ready at {now}, size={size} bytes")
            return True
        time.sleep(interval)
        waited += interval
    # Final log if not ready
    now = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    logging.info(f"[INFO] File '{csv_path}' NOT ready after waiting ({now}), exists={exists}, size={size}")
    return False

def handle_user_input(user_input: str, uploaded_csv_path: str, timeout: int = 180) -> str:
    """
    Handles user input for CSV-to-API processing. Only processes CSV if user intent is explicit.
    Returns a summary of API calls or raw output. Handles errors and missing file cases.
    Logs file state before routing for investigation.
    """
    # Define trigger keywords for explicit processing
    TRIGGER_KEYWORDS = [
        'get transaction', 'get transactions', 'tag', 'process', 'extract', 'categorize', 'analyze', 'summarize',
        'parse', 'label', 'lookup', 'receipt', 'expense', 'csv2api', 'run csv', 'process csv', 'api', 'endpoint', 'serve'
    ]
    # Lowercase input for matching
    msg = user_input.lower().strip()
    # Ignore small talk/general
    GENERAL_PHRASES = [
        'hello', 'hi', 'hey', 'how are you', 'who are you', 'what can you do', 'help', 'thanks', 'thank you',
        'about you', 'about this', 'what is this', 'explain yourself', 'your name', 'good morning', 'good afternoon', 'good evening'
    ]
    if any(phrase in msg for phrase in GENERAL_PHRASES):
        return ""
    # Only trigger if explicit keyword present
    if not any(kw in msg for kw in TRIGGER_KEYWORDS):
        return ""
    # Log file state before routing
    log_csv_file_state(uploaded_csv_path, context="before routing")
    # Wait for file to be ready (if needed)
    wait_for_csv_ready(uploaded_csv_path)
    # Check for CSV file
    if not is_valid_csv_file(uploaded_csv_path):
        return "\u26A0\uFE0F Please upload a CSV file first."
    # Run csv2api
    csv2api_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'csv2api'))
    cmd = [sys.executable, '-m', 'src.main', '-i', uploaded_csv_path]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=csv2api_dir,
            timeout=timeout
        )
        output = result.stdout.strip()
        # Try to parse API calls from output (JSON lines or API call lines)
        api_calls = []
        for line in output.splitlines():
            line = line.strip()
            if line.startswith('{') and 'api_call' in line:
                try:
                    import json
                    api_obj = json.loads(line)
                    api_calls.append(api_obj)
                except Exception:
                    pass
            elif line.startswith('API call made:'):
                try:
                    import json
                    api_obj = json.loads(line[len('API call made:'):].strip())
                    api_calls.append(api_obj)
                except Exception:
                    pass
        # Summarize API call types and counts
        if api_calls:
            call_summary = {}
            for call in api_calls:
                if isinstance(call, dict):
                    api_name = call.get('api_call') or call.get('function') or 'API call'
                    call_summary[api_name] = call_summary.get(api_name, 0) + 1
            reply_lines = ["\u2705 Processed CSV using csv2api:"]
            for api_name, count in call_summary.items():
                reply_lines.append(f"• {api_name} — {count} call(s)")
            return '\n'.join(reply_lines)
        # If not JSON, return raw output in formatted block
        if output:
            return f"\u2705 Processed CSV using csv2api. Output:\n```\n{output}\n```"
        else:
            return "\u26A0\uFE0F No output from csv2api pipeline."
    except subprocess.TimeoutExpired:
        return "Sorry, the CSV processing took too long and timed out."
    except Exception as e:
        return f"An error occurred while processing the CSV: {e}"