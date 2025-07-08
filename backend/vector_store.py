def is_csv_analysis_intent(user_input: str) -> bool:
    """
    Heuristically determine if the user wants to analyze/process an uploaded CSV file.
    Looks for keywords/phrases like 'analyze expenses', 'summarize this csv', 'get transaction data', etc.
    """
    user_input = user_input.lower()
    csv_keywords = [
        'analyze expenses', 'summarize this csv', 'summarize csv', 'get transaction data',
        'analyze csv', 'process csv', 'extract data', 'show me insights', 'summarize file',
        'find patterns', 'csv report', 'csv summary', 'csv analysis', 'csv stats',
        'csv overview', 'csv breakdown', 'csv trends', 'csv insights', 'csv results',
        'csv api', 'convert csv', 'query csv', 'csv endpoint', 'csv to api'
    ]
    return any(kw in user_input for kw in csv_keywords)

def detect_csv2api_intent(message: str) -> bool:
    """
    Use a local or hosted LLM to determine if the user's message is requesting analysis or processing
    of a CSV file related to expenses or blockchain transactions. Returns True if the LLM responds
    with 'true' in the output, otherwise False.
    """
    import requests
    import re
    LLM_URL = "http://localhost:11434/api/generate"
    LLM_MODEL = "deepseek-r1:latest"
    prompt = (
        "Does the following message ask to analyze or process a CSV file containing expenses or blockchain transactions? "
        "Respond only with 'true' or 'false'.\nMessage: " + message
    )
    try:
        response = requests.post(
            LLM_URL,
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        content = re.sub(r'<think>.*?</think>', '', result.get("response", ""), flags=re.DOTALL).strip().lower()
        return "true" in content
    except Exception:
        return False

def get_chroma_collection(persistent: bool = True):
    """
    Initialize and return a ChromaDB collection named 'pennyworks'.
    If persistent=True, use a persistent directory; otherwise, use in-memory.
    """
    import chromadb
    from chromadb.config import Settings
    import os
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'chroma_db'))
    if persistent:
        client = chromadb.PersistentClient(path=db_path, settings=Settings(allow_reset=True))
    else:
        client = chromadb.Client()
    collection_name = "pennyworks"
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        return client.get_collection(collection_name)
    return client.create_collection(collection_name)