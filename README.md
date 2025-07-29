# Penny: Crypto Accounting Chatbot

A standalone Streamlit chatbot app for crypto accounting, featuring CSV upload, DeepSeek R1 (Ollama) LLM integration, ChromaDB vector store, and a friendly, professional Penny persona. Penny can analyze uploaded CSVs, answer accounting questions, and route data to csv2api as needed... all with structured logging and persistent chat history.

## Features
- **Conversational Chat UI** with persistent history and Penny persona
- **CSV Upload & Preview** for crypto accounting data
- **Local LLM Integration** (DeepSeek R1 via Ollama)
- **ChromaDB Vector Store** for persistent chat and CSV content storage
- **ChromaDB Visualizer** for exploring stored vectors
- **Structured Logging** of all user actions, routing, and LLM responses
- **csv2api Integration**: Calls csv2api as a subprocess for advanced CSV analysis (no code dependency)
- **Intent Detection** and prompt engineering for accurate, persona-driven responses
- **Test Scripts** for ChromaDB and logger sanity checks

## How to Run

1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Ensure Ollama and DeepSeek R1 are running locally.
3. Start the app:
   ```sh
   streamlit run app.py
   ```
4. (Optional) Visualize ChromaDB contents:
   ```sh
   streamlit run visualize_chromadb.py
   ```
5. (Optional) Run tests:
   ```sh
   python test_chromadb.py
   python test_logger.py
   ```

## Architecture
- **app.py**: Main Streamlit app (chat UI, CSV upload, LLM/csv2api integration, logging)
- **backend/csv_handler.py**: Persona prompt engineering, LLM chat, intent detection, sync csv2api runner
- **backend/vector_store.py**: ChromaDB vector store logic
- **utils/logger.py**: Custom structured logger
- **visualize_chromadb.py**: ChromaDB visualizer UI
- **csv2api/**: Submodule, called as a subprocess (no direct code dependency)

## Notes
- This app is fully independent of csv2api logic, but can call csv2api as a subprocess for advanced CSV analysis.
- All LLM chat and intent detection use the Penny persona prompt for consistent, professional responses.
- All major requirements (persona, sync/async, logging, ChromaDB, csv2api routing) are implemented.

---
For crypto accounting teams and professionals seeking a robust, privacy-friendly, and extensible AI assistant for CSV-based workflows.
