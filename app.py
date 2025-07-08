import os
import streamlit as st
import re
import asyncio
import tempfile
import pandas as pd
import numpy as np
from backend.csv_handler import is_csv2api_intent, get_simplified_intent, run_csv2api, penny_llm_chat
from backend.vector_store import get_chroma_collection, is_csv_analysis_intent, detect_csv2api_intent
from sklearn.manifold import TSNE
from datetime import datetime
import logging
from utils.logger import setup_logger

# Setup logger
logger = setup_logger()

st.set_page_config(page_title="Penny: Accounting Assistant")
st.title("Penny: Accounting Assistant")

# Initialize chat history and CSV path
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "csv_path" not in st.session_state:
    st.session_state["csv_path"] = None

# CSV upload and preview
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    # Save uploaded file to a temp file and store the path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.read())
        st.session_state["csv_path"] = tmp.name
    # If the user has already entered a prompt, process immediately
    if "last_prompt" in st.session_state and st.session_state["last_prompt"]:
        user_prompt = st.session_state["last_prompt"]
        # Generate a context-aware confirmation message
        from backend.csv_handler import simplify_for_csv2api, run_csv2api
        short_cmd = simplify_for_csv2api(user_prompt)
        confirm_msg = f"Got it! We're {short_cmd.lower()} from your uploaded file now. Hang tight while we process everything."
        st.session_state["messages"].append({
            "role": "assistant",
            "content": confirm_msg
        })
        logger.info(confirm_msg)
        with st.chat_message("assistant"):
            st.markdown(confirm_msg)
        # Call csv2api and show result as a follow-up
        with st.spinner("Processing your CSV with csv2api..."):
            result = run_csv2api(st.session_state["csv_path"], short_cmd)
            followup_msg = f"Processing complete!\n{result}"
            st.session_state["messages"].append({"role": "assistant", "content": followup_msg})
            with st.chat_message("assistant"):
                st.markdown(followup_msg)
    else:
        st.session_state["messages"].append({
            "role": "assistant",
            "content": f"CSV file '{uploaded_file.name}' uploaded successfully. Please enter your request."
        })
    logger.info(f"CSV file '{uploaded_file.name}' uploaded successfully.")
    uploaded_file = None  # Prevent re-adding on rerun

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about CSV to API..."):
    logger.info(f"User input: {prompt}")
    file_uploaded = bool(st.session_state.get("csv_path"))
    logger.info(f"File uploaded: {file_uploaded}")
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.session_state["last_prompt"] = prompt
    with st.chat_message("user"):
        st.markdown(prompt)

    # If a CSV is already uploaded, process immediately
    if st.session_state["csv_path"]:
        from backend.csv_handler import simplify_for_csv2api, run_csv2api
        short_cmd = simplify_for_csv2api(prompt)
        confirm_msg = f"Got it! We're {short_cmd.lower()} from your uploaded file now. Hang tight while we process everything."
        st.session_state["messages"].append({
            "role": "assistant",
            "content": confirm_msg
        })
        logger.info(confirm_msg)
        with st.chat_message("assistant"):
            st.markdown(confirm_msg)
        with st.spinner("Processing your CSV with csv2api..."):
            result = run_csv2api(st.session_state["csv_path"], short_cmd)
            followup_msg = f"Processing complete!\n{result}"
            st.session_state["messages"].append({"role": "assistant", "content": followup_msg})
            with st.chat_message("assistant"):
                st.markdown(followup_msg)
    else:
        # No CSV yet, just store the prompt and wait for upload
        st.session_state["messages"].append({
            "role": "assistant",
            "content": "Please upload a CSV file to process your request."
        })

# --- ChromaDB Visualizer Section ---
with st.expander("🔍 Visualize ChromaDB Knowledge Base", expanded=False):
    collection = get_chroma_collection()
    # Remove 'ids' from include list, as ChromaDB always returns them
    all_docs = collection.get(include=["documents", "metadatas", "embeddings"])

    def parse_metadata(meta):
        if not meta:
            return {}
        return meta

    def filter_docs(docs, metadatas, ids, date_range, chain, purpose):
        filtered = []
        for doc, meta, doc_id in zip(docs, metadatas, ids):
            meta = parse_metadata(meta)
            ts = meta.get("timestamp")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts)
                except Exception:
                    continue
                if not (date_range[0] <= dt <= date_range[1]):
                    continue
            if chain and meta.get("chain") and meta.get("chain") != chain:
                continue
            if purpose and meta.get("purpose") and meta.get("purpose") != purpose:
                continue
            filtered.append((doc, meta, doc_id))
        return filtered

    st.subheader("ChromaDB Knowledge Visualizer")
    chains = list({meta.get("chain") for meta in all_docs["metadatas"] if meta and meta.get("chain")})
    purposes = list({meta.get("purpose") for meta in all_docs["metadatas"] if meta and meta.get("purpose")})
    min_date = min([datetime.fromisoformat(meta["timestamp"]) for meta in all_docs["metadatas"] if meta and meta.get("timestamp")], default=datetime(2020,1,1))
    max_date = max([datetime.fromisoformat(meta["timestamp"]) for meta in all_docs["metadatas"] if meta and meta.get("timestamp")], default=datetime.now())
    date_range = st.date_input("Date range", [min_date, max_date], key="chroma_date_range")
    chain = st.selectbox("Chain", ["All"] + chains, key="chroma_chain")
    purpose = st.selectbox("Purpose", ["All"] + purposes, key="chroma_purpose")

    filtered = filter_docs(
        all_docs["documents"],
        all_docs["metadatas"],
        all_docs["ids"],
        [datetime.combine(date_range[0], datetime.min.time()), datetime.combine(date_range[1], datetime.max.time())],
        None if chain == "All" else chain,
        None if purpose == "All" else purpose
    )

    st.markdown("**Stored Entries Table**")
    table_data = [
        {"ID": doc_id, "Text": doc[:100] + ("..." if len(doc) > 100 else ""), **meta}
        for doc, meta, doc_id in filtered
    ]
    if table_data:
        st.dataframe(pd.DataFrame(table_data))
    else:
        st.info("No entries match the current filters.")

    st.markdown("**Embedding Space Visualization (t-SNE)**")
    if filtered:
        embeddings = [all_docs["embeddings"][all_docs["documents"].index(doc)] for doc, _, _ in filtered]
        docs_short = [doc[:40] + ("..." if len(doc) > 40 else "") for doc, _, _ in filtered]
        if len(embeddings) > 1:
            tsne = TSNE(n_components=2, random_state=42)
            emb_2d = tsne.fit_transform(np.array(embeddings))
            df_plot = pd.DataFrame({"x": emb_2d[:,0], "y": emb_2d[:,1], "label": docs_short})
            st.scatter_chart(df_plot, x="x", y="y", color=None)
            for i, row in df_plot.iterrows():
                st.text(f"{row['label']} @ ({row['x']:.2f}, {row['y']:.2f})")
        else:
            st.info("Not enough data for embedding visualization.")
    else:
        st.info("Not enough data for embedding visualization.")
