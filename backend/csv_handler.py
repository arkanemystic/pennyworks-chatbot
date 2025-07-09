import os
import re
import subprocess
import sys
import logging
import requests
import json
import time
from typing import Optional, Dict, Any

# Penny persona system prompt
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

# LLM configuration
LLM_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "deepseek-r1:latest"

def is_valid_csv_file(csv_path: str) -> bool:
    """Check if CSV file exists and is valid"""
    if not csv_path or not os.path.exists(csv_path):
        return False
    if os.path.getsize(csv_path) == 0:
        return False
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            header = f.readline()
            if ',' in header and len(header.strip().split(',')) >= 2:
                return True
    except Exception:
        return False
    return False

def penny_llm_chat(user_input: str, csv_path: str = None) -> str:
    """Send message to LLM with Penny persona"""
    csv_ready = is_valid_csv_file(csv_path)
    
    if csv_ready and csv_path:
        file_context = f"IMPORTANT: A CSV file is currently uploaded and ready for processing. The file exists and is valid."
    else:
        file_context = "No CSV file has been uploaded yet."
    
    prompt = (
        f"{PENNY_SYSTEM_PROMPT}\n\n"
        f"{file_context}\n"
        f"csv_ready: {str(csv_ready).lower()}\n\n"
        f"User: {user_input}\nPenny:"
    )
    
    try:
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
    except Exception as e:
        logging.error(f"LLM request failed: {e}")
        return "I'm having trouble connecting to my AI assistant right now. Please try again."

def find_csv2api_executable() -> Optional[str]:
    """Find the correct csv2api executable path"""
    csv2api_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'csv2api'))
    
    # Common possible paths
    possible_paths = [
        os.path.join(csv2api_dir, 'src', 'main.py'),
        os.path.join(csv2api_dir, 'main.py'),
        os.path.join(csv2api_dir, 'csv2api.py'),
        os.path.join(csv2api_dir, 'src', 'pipeline', 'llm_client.py'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    logging.error(f"Could not find csv2api executable in {csv2api_dir}")
    return None

def run_csv2api_simple(csv_path: str, user_prompt: str = None) -> Dict[str, Any]:
    """Simple CSV2API execution with better error handling"""
    if not is_valid_csv_file(csv_path):
        return {
            'success': False,
            'error': 'Invalid CSV file',
            'output': 'Please upload a valid CSV file.'
        }
    
    executable = find_csv2api_executable()
    if not executable:
        return {
            'success': False,
            'error': 'CSV2API not found',
            'output': 'CSV2API module not found. Please check the installation.'
        }
    
    csv2api_dir = os.path.dirname(executable)
    cmd = [sys.executable, executable, '--input', csv_path]
    
    if user_prompt:
        cmd.extend(['--prompt', user_prompt])
    
    try:
        logging.info(f"Running CSV2API command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=csv2api_dir,
            timeout=180
        )
        
        if result.returncode == 0:
            return {
                'success': True,
                'output': result.stdout.strip() or 'Processing completed successfully.',
                'raw_output': result.stdout,
                'error': None
            }
        else:
            return {
                'success': False,
                'error': f'CSV2API failed with return code {result.returncode}',
                'output': f'Error: {result.stderr.strip() if result.stderr else "Unknown error"}',
                'raw_output': result.stdout,
                'raw_error': result.stderr
            }
    
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': 'Timeout',
            'output': 'CSV processing timed out. Please try with a smaller file.'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'output': f'An error occurred: {str(e)}'
        }

def handle_user_message(user_input: str, csv_path: str = None) -> str:
    """Main handler for user messages"""
    logging.info(f"Handling user message: {user_input}")
    logging.info(f"CSV path: {csv_path}")
    
    # Get LLM routing decision
    llm_response = penny_llm_chat(user_input, csv_path)
    logging.info(f"LLM response: {llm_response}")
    
    # Parse routing decision
    response_clean = llm_response.strip()
    
    if response_clean == 'CSV_REQUIRED':
        return "Please upload a CSV file to process your request."
    
    elif response_clean == 'ROUTE_TO_CSV2API':
        if not csv_path or not is_valid_csv_file(csv_path):
            return "Please upload a valid CSV file first."
        
        # Run CSV2API
        result = run_csv2api_simple(csv_path, user_input)
        
        if result['success']:
            return f"✅ CSV processed successfully!\n\n{result['output']}"
        else:
            return f"❌ CSV processing failed: {result['output']}"
    
    else:
        # Direct LLM response
        return llm_response

# Simplified functions for backward compatibility
def is_csv2api_intent(user_input: str) -> bool:
    """Check if user wants CSV2API processing"""
    keywords = ['process', 'analyze', 'csv', 'api', 'convert', 'extract', 'tag', 'categorize']
    return any(kw in user_input.lower() for kw in keywords)

def get_simplified_intent(user_input: str) -> str:
    """Get simplified intent for CSV2API"""
    return user_input.strip()

def run_csv2api_cli(input_file: str = None, prompt: str = None, debug: bool = False):
    """Legacy function for backward compatibility"""
    if input_file:
        result = run_csv2api_simple(input_file, prompt)
        return {
            'api_calls': [],
            'summary': {'success': 1 if result['success'] else 0, 'failed': 0 if result['success'] else 1},
            'stdout': result.get('raw_output', ''),
            'stderr': result.get('raw_error', ''),
            'returncode': 0 if result['success'] else 1
        }
    return {'api_calls': [], 'summary': {}, 'stdout': '', 'stderr': '', 'returncode': 1}