import os
import streamlit as st
import tempfile
import pandas as pd
from backend.csv_handler import handle_user_message, is_valid_csv_file
from backend.vector_store import get_chroma_collection
from utils.logger import setup_logger
from datetime import datetime

# Setup logger
logger = setup_logger()

st.set_page_config(page_title="Penny: Accounting Assistant")
st.title("Penny: Accounting Assistant")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "csv_path" not in st.session_state:
    st.session_state["csv_path"] = None

# CSV upload section
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.read())
        tmp.flush()
        tmp_name = tmp.name
    st.session_state["csv_path"] = tmp_name
    logger.info(f"Temporary CSV saved at: {tmp_name}")
    
    # Show success message
    st.success(f"CSV file '{uploaded_file.name}' uploaded successfully!")
    logger.info(f"CSV file uploaded: {uploaded_file.name}")
    
    # Preview CSV
    try:
        df = pd.read_csv(st.session_state["csv_path"])
        st.subheader("CSV Preview")
        st.dataframe(df.head())
        st.info(f"CSV contains {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        st.error(f"Could not preview CSV: {str(e)}")

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about your CSV or accounting..."):
    logger.info(f"User input: {prompt}")
    
    # Add user message to chat
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response from handler
    with st.spinner("Processing..."):
        response = handle_user_message(prompt, st.session_state["csv_path"])
    
    # Add assistant response to chat
    st.session_state["messages"].append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

# Status information
with st.sidebar:
    st.header("Status")
    csv_status = "✅ CSV uploaded" if st.session_state["csv_path"] and is_valid_csv_file(st.session_state["csv_path"]) else "❌ No CSV uploaded"
    st.write(csv_status)
    
    if st.session_state["csv_path"]:
        st.write(f"File path: {os.path.basename(st.session_state['csv_path'])}")
    
    st.header("Usage Tips")
    st.write("• Upload a CSV file first")
    st.write("• Ask me to 'process the CSV' or 'analyze transactions'")
    st.write("• I can help with accounting questions too!")

# ChromaDB Visualizer (simplified)
with st.expander("🔍 ChromaDB Knowledge Base", expanded=False):
    try:
        collection = get_chroma_collection()
        all_docs = collection.get(include=["documents", "metadatas"])
        
        if all_docs["documents"]:
            st.write(f"Total documents: {len(all_docs['documents'])}")
            
            # Show recent documents
            recent_docs = []
            for doc, meta in zip(all_docs["documents"], all_docs["metadatas"]):
                recent_docs.append({
                    "Text": doc[:100] + ("..." if len(doc) > 100 else ""),
                    "Metadata": str(meta) if meta else "None"
                })
            
            st.dataframe(pd.DataFrame(recent_docs))
        else:
            st.info("No documents in knowledge base yet.")
    except Exception as e:
        st.error(f"Could not load ChromaDB: {str(e)}")