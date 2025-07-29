#!/usr/bin/env python3
"""
Updated Main Handler - Replace your existing handler code with this
Save this as: main_handler.py or update your existing handler file
"""

import os
import sys
import logging
from typing import Dict, Any, Optional
import subprocess

# Import the CSV processor
from csv_handler import CSVProcessor, format_transaction_response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PennyCSVHandler:
    def __init__(self):
        self.csv_processor = CSVProcessor()
        
    def handle_user_message(self, user_input: str, csv_path: str) -> str:
        """
        Main handler for user messages with CSV processing
        """
        logger.info("=" * 50)
        logger.info("NEW USER MESSAGE HANDLER SESSION")
        logger.info(f"User input: {user_input}")
        logger.info(f"CSV path: {csv_path}")
        
        # Check if CSV file exists
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            return "❌ CSV file not found. Please upload a valid CSV file."
        
        logger.info("CSV file exists: True")
        
        # Validate CSV file
        if not self.is_valid_csv(csv_path):
            logger.error("CSV file validation failed")
            return "❌ Invalid CSV file. Please ensure the file is a valid CSV format."
        
        logger.info("CSV file valid: True")
        
        # Get routing decision
        routing_decision = self.get_routing_decision(user_input, csv_path)
        logger.info(f"Routing decision: {routing_decision}")
        
        if routing_decision == "ROUTE_TO_CSV2API":
            return self.process_with_csv2api(csv_path, user_input)
        else:
            return self.generate_penny_response(user_input, csv_path)
    
    def is_valid_csv(self, csv_path: str) -> bool:
        """
        Validate if the file is a valid CSV
        """
        try:
            import pandas as pd
            df = pd.read_csv(csv_path, nrows=1)  # Just read first row to validate
            return len(df.columns) > 0
        except Exception as e:
            logger.error(f"CSV validation error: {e}")
            return False
    
    def get_routing_decision(self, user_input: str, csv_path: str) -> str:
        """
        Determine if request should be routed to CSV2API or handled directly
        Uses contextual analysis to avoid false positives from keywords
        """
        user_input_lower = user_input.lower()

        # First, detect informational queries about csv2api
        info_patterns = [
            'what can csv2api do',
            'what api calls',
            'how does csv2api work',
            'tell me about csv2api',
            'help with csv2api',
            'explain csv2api'
        ]
        if any(pattern in user_input_lower for pattern in info_patterns):
            logger.info("Detected informational query about csv2api")
            return "DIRECT_RESPONSE"

        # Then check for actual CSV processing intent
        action_verbs = ['process', 'analyze', 'extract', 'get', 'parse']
        target_nouns = ['transactions', 'data', 'events', 'records']
        csv_context = ['from the csv', 'in the csv', 'from this file', 'this data']

        # Count matching context patterns
        context_score = 0
        if any(verb in user_input_lower for verb in action_verbs):
            context_score += 1
        if any(noun in user_input_lower for noun in target_nouns):
            context_score += 1
        if any(ctx in user_input_lower for ctx in csv_context):
            context_score += 1

        # Route to CSV2API only if we have strong contextual evidence
        if context_score >= 2:
            logger.info(f"Found CSV processing intent (context score: {context_score}/3)")
            return "ROUTE_TO_CSV2API"
        
        logger.info(f"Insufficient context for CSV2API routing (score: {context_score}/3)")
        return "DIRECT_RESPONSE"
    
    def process_with_csv2api(self, csv_path: str, user_query: str) -> str:
        """
        Process CSV using csv2api with fallback
        """
        logger.info("ROUTING: ROUTE_TO_CSV2API")
        logger.info("Executing csv2api subprocess...")
        
        try:
            result = self.csv_processor.process_csv(csv_path, user_query)
            response = format_transaction_response(result)
            
            if result['success']:
                logger.info("CSV processing successful")
            else:
                logger.error("CSV processing failed")
                
            return response
            
        except Exception as e:
            logger.error(f"CSV processing error: {e}")
            return f"❌ Error processing CSV: {str(e)}"
    
    def generate_penny_response(self, user_input: str, csv_path: str) -> str:
        """
        Generate a direct response from Penny without CSV processing
        """
        logger.info("ROUTING: Direct LLM response")
        
        # This is where you'd integrate with your LLM (Ollama, OpenAI, etc.)
        # For now, providing a template response
        
        responses = {
            "hello": "Hello! I'm Penny, your crypto accounting assistant! 😊 I can help you process CSV files and analyze transactions. What would you like to do?",
            "help": "I can help you with:\n• Processing CSV transaction files\n• Extracting and analyzing transaction data\n• Filling account information\n• Generating reports\n\nJust upload a CSV file and tell me what you'd like to do!",
            "default": "I'm ready to help you with your CSV file! 📊 You can ask me to:\n• Get transactions from the CSV\n• Process the data\n• Analyze the transactions\n• Fill account information\n\nWhat would you like me to do?"
        }
        
        user_lower = user_input.lower()
        
        if "hello" in user_lower or "hi" in user_lower:
            return responses["hello"]
        elif "help" in user_lower:
            return responses["help"]
        else:
            return responses["default"]

# Integration with your existing system
def main():
    """
    Main entry point - replace your existing main function with this
    """
    handler = PennyCSVHandler()
    
    # Example usage - replace with your actual integration
    if len(sys.argv) >= 3:
        csv_path = sys.argv[1]
        user_query = sys.argv[2]
        
        response = handler.handle_user_message(user_query, csv_path)
        print(response)
    else:
        print("Usage: python main_handler.py <csv_path> <user_query>")

# Flask/FastAPI integration example
def create_flask_app():
    """
    Flask integration example
    """
    from flask import Flask, request, jsonify
    import tempfile
    
    app = Flask(__name__)
    handler = PennyCSVHandler()
    
    @app.route('/process_csv', methods=['POST'])
    def process_csv():
        try:
            # Get uploaded file
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400
            
            file = request.files['file']
            user_query = request.form.get('query', 'get transactions from csv')
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                file.save(tmp_file.name)
                
                # Process the file
                response = handler.handle_user_message(user_query, tmp_file.name)
                
                # Clean up
                os.unlink(tmp_file.name)
                
                return jsonify({'response': response})
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return app

if __name__ == "__main__":
    main()