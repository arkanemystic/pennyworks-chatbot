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
    "- If csv_ready is 'true' and the user wants to process, analyze, or convert CSV data, respond with exactly 'ROUTE_TO_CSV2API' "
    "- If csv_ready is 'false' and the user wants to process CSV data, respond with exactly 'CSV_REQUIRED' "
    "- Otherwise, provide helpful accounting assistance."
)

# LLM configuration
LLM_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "deepseek-r1:latest"

def setup_logging():
    """Setup logging for csv_handler"""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('penny_csv_handler.log')
        ]
    )

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
    # Get the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    csv2api_dir = os.path.join(project_root, 'csv2api')
    
    logging.info(f"Looking for csv2api in: {csv2api_dir}")
    
    # Check if csv2api directory exists
    if not os.path.exists(csv2api_dir):
        logging.error(f"csv2api directory not found at: {csv2api_dir}")
        return None
    
    # Common possible paths for the main executable
    possible_paths = [
        os.path.join(csv2api_dir, 'main.py'),
        os.path.join(csv2api_dir, 'src', 'main.py'),
        os.path.join(csv2api_dir, 'csv2api.py'),
        os.path.join(csv2api_dir, 'app.py'),
        os.path.join(csv2api_dir, 'run.py'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logging.info(f"Found csv2api executable at: {path}")
            return path
    
    # If no main file found, list contents for debugging
    try:
        contents = os.listdir(csv2api_dir)
        logging.info(f"csv2api directory contents: {contents}")
    except Exception as e:
        logging.error(f"Could not list csv2api directory: {e}")
    
    logging.error(f"Could not find csv2api executable in {csv2api_dir}")
    return None

def run_csv2api_subprocess(csv_path: str, user_prompt: str = None) -> Dict[str, Any]:
    """Execute csv2api as a subprocess with proper error handling"""
    
    # Validate CSV file first
    if not is_valid_csv_file(csv_path):
        return {
            'success': False,
            'error': 'Invalid CSV file',
            'output': 'Please upload a valid CSV file.',
            'stdout': '',
            'stderr': 'Invalid CSV file provided'
        }
    
    # Find the csv2api executable
    executable = find_csv2api_executable()
    if not executable:
        return {
            'success': False,
            'error': 'CSV2API not found',
            'output': 'csv2api module not found. Please ensure the submodule is properly initialized.',
            'stdout': '',
            'stderr': 'csv2api executable not found'
        }
    
    # Prepare the command
    csv2api_dir = os.path.dirname(executable)
    cmd = [sys.executable, executable, '--csv', csv_path]
    
    # Add user prompt if provided
    if user_prompt:
        cmd.extend(['--query', user_prompt])
    
    # Alternative command formats to try
    alternative_commands = [
        [sys.executable, executable, '--input', csv_path],
        [sys.executable, executable, csv_path],
        [sys.executable, executable, '--file', csv_path],
    ]
    
    if user_prompt:
        alternative_commands = [
            [sys.executable, executable, '--csv', csv_path, '--query', user_prompt],
            [sys.executable, executable, '--input', csv_path, '--prompt', user_prompt],
            [sys.executable, executable, csv_path, user_prompt],
        ]
    
    # Try the main command first, then alternatives
    commands_to_try = [cmd] + alternative_commands
    
    for attempt, current_cmd in enumerate(commands_to_try, 1):
        try:
            logging.info(f"Attempt {attempt}: Running csv2api command: {' '.join(current_cmd)}")
            
            result = subprocess.run(
                current_cmd,
                capture_output=True,
                text=True,
                cwd=csv2api_dir,
                timeout=300,  # 5 minute timeout
                env=os.environ.copy()  # Pass environment variables
            )
            
            logging.info(f"csv2api return code: {result.returncode}")
            logging.info(f"csv2api stdout: {result.stdout}")
            if result.stderr:
                logging.warning(f"csv2api stderr: {result.stderr}")
            
            if result.returncode == 0:
                output = result.stdout.strip() if result.stdout else 'Processing completed successfully.'
                return {
                    'success': True,
                    'output': output,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'command': ' '.join(current_cmd)
                }
            else:
                # If this isn't the last attempt, continue to next command
                if attempt < len(commands_to_try):
                    logging.warning(f"Command {attempt} failed, trying next alternative...")
                    continue
                
                # Last attempt failed
                error_msg = result.stderr.strip() if result.stderr else f"Process failed with return code {result.returncode}"
                return {
                    'success': False,
                    'error': f'csv2api failed with return code {result.returncode}',
                    'output': f'Error: {error_msg}',
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'command': ' '.join(current_cmd)
                }
        
        except subprocess.TimeoutExpired:
            logging.error(f"csv2api process timed out after 5 minutes")
            return {
                'success': False,
                'error': 'Process timeout',
                'output': 'CSV processing timed out after 5 minutes. Please try with a smaller file.',
                'stdout': '',
                'stderr': 'Process timed out',
                'command': ' '.join(current_cmd)
            }
        
        except FileNotFoundError:
            logging.error(f"csv2api executable not found: {current_cmd[1]}")
            if attempt < len(commands_to_try):
                continue
            return {
                'success': False,
                'error': 'Executable not found',
                'output': f'csv2api executable not found at: {current_cmd[1]}',
                'stdout': '',
                'stderr': 'Executable not found',
                'command': ' '.join(current_cmd)
            }
        
        except Exception as e:
            logging.error(f"Unexpected error running csv2api: {e}")
            if attempt < len(commands_to_try):
                continue
            return {
                'success': False,
                'error': str(e),
                'output': f'An unexpected error occurred: {str(e)}',
                'stdout': '',
                'stderr': str(e),
                'command': ' '.join(current_cmd)
            }
    
    # If we get here, all attempts failed
    return {
        'success': False,
        'error': 'All command attempts failed',
        'output': 'Could not execute csv2api with any known command format.',
        'stdout': '',
        'stderr': 'All command attempts failed'
    }

def handle_user_message(user_input: str, csv_path: str = None) -> str:
    """Main handler for user messages with improved csv2api routing"""
    
    # Setup logging
    setup_logging()
    
    logging.info(f"Handling user message: {user_input}")
    logging.info(f"CSV path provided: {csv_path}")
    logging.info(f"CSV file valid: {is_valid_csv_file(csv_path)}")
    
    # Get LLM routing decision
    llm_response = penny_llm_chat(user_input, csv_path)
    logging.info(f"LLM response: {llm_response}")
    
    # Parse routing decision
    response_clean = llm_response.strip()
    
    # Handle routing decisions
    if response_clean == 'CSV_REQUIRED':
        logging.info("Routing decision: CSV_REQUIRED")
        return "Please upload a CSV file first to process your request! 📊"
    
    elif response_clean == 'ROUTE_TO_CSV2API':
        logging.info("Routing decision: ROUTE_TO_CSV2API")
        
        # Double-check CSV validity
        if not csv_path or not is_valid_csv_file(csv_path):
            logging.error("CSV required but not valid")
            return "Please upload a valid CSV file first! 📄"
        
        # Execute csv2api subprocess
        logging.info("Executing csv2api subprocess...")
        result = run_csv2api_subprocess(csv_path, user_input)
        
        # Format response based on result
        if result['success']:
            logging.info("csv2api executed successfully")
            response = f"✅ CSV processed successfully!\n\n{result['output']}"
            
            # Add debug info if stdout contains useful information
            if result['stdout'] and result['stdout'].strip() != result['output']:
                response += f"\n\n📋 Additional details:\n{result['stdout']}"
            
            return response
        else:
            logging.error(f"csv2api execution failed: {result['error']}")
            error_response = f"❌ CSV processing failed: {result['output']}"
            
            # Add debug information for troubleshooting
            if result.get('command'):
                error_response += f"\n\n🔧 Command used: `{result['command']}`"
            
            if result.get('stderr') and result['stderr'].strip():
                error_response += f"\n\n⚠️ Error details: {result['stderr']}"
            
            return error_response
    
    else:
        # Direct LLM response for non-routing cases
        logging.info("Routing decision: Direct LLM response")
        return llm_response

# Legacy compatibility functions
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
        result = run_csv2api_subprocess(input_file, prompt)
        return {
            'api_calls': [],
            'summary': {'success': 1 if result['success'] else 0, 'failed': 0 if result['success'] else 1},
            'stdout': result.get('stdout', ''),
            'stderr': result.get('stderr', ''),
            'returncode': 0 if result['success'] else 1
        }
    return {'api_calls': [], 'summary': {}, 'stdout': '', 'stderr': '', 'returncode': 1}

# Alias for backward compatibility
run_csv2api_simple = run_csv2api_subprocess