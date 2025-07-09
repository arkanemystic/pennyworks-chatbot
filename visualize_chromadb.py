import streamlit as st
import pandas as pd
import numpy as np
from backend.vector_store import get_chroma_collection
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from datetime import datetime

st.set_page_config(page_title="ChromaDB Knowledge Visualizer")
st.title("ChromaDB Knowledge Visualizer")

# Load collection and all documents
collection = get_chroma_collection()
all_docs = collection.get(include=["documents", "metadatas", "embeddings", "ids"])

def parse_metadata(meta):
    # Flatten and normalize metadata for DataFrame
    if not meta:
        return {}
    return meta

def filter_docs(docs, metadatas, date_range, chain, purpose):
    filtered = []
    for doc, meta in zip(docs, metadatas):
        meta = parse_metadata(meta)
        # Date filter
        ts = meta.get("timestamp")
        if ts:
            try:
                dt = datetime.fromisoformat(ts)
            except Exception:
                continue
            if not (date_range[0] <= dt <= date_range[1]):
                continue
        # Chain filter
        if chain and meta.get("chain") and meta.get("chain") != chain:
            continue
        # Purpose filter
        if purpose and meta.get("purpose") and meta.get("purpose") != purpose:
            continue
        filtered.append((doc, meta))
    return filtered

# Sidebar filters
st.sidebar.header("Filters")
chains = list({meta.get("chain") for meta in all_docs["metadatas"] if meta and meta.get("chain")})
purposes = list({meta.get("purpose") for meta in all_docs["metadatas"] if meta and meta.get("purpose")})

min_date = min([datetime.fromisoformat(meta["timestamp"]) for meta in all_docs["metadatas"] if meta and meta.get("timestamp")], default=datetime(2020,1,1))
max_date = max([datetime.fromisoformat(meta["timestamp"]) for meta in all_docs["metadatas"] if meta and meta.get("timestamp")], default=datetime.now())
date_range = st.sidebar.date_input("Date range", [min_date, max_date])
chain = st.sidebar.selectbox("Chain", ["All"] + chains)
purpose = st.sidebar.selectbox("Purpose", ["All"] + purposes)

# Filtered docs
filtered = filter_docs(
    all_docs["documents"],
    all_docs["metadatas"],
    [datetime.combine(date_range[0], datetime.min.time()), datetime.combine(date_range[1], datetime.max.time())],
    None if chain == "All" else chain,
    None if purpose == "All" else purpose
)

# Table view
st.subheader("Stored Entries Table")
table_data = [
    {"Text": doc[:100] + ("..." if len(doc) > 100 else ""), **meta}
    for doc, meta in filtered
]
if table_data:
    st.dataframe(pd.DataFrame(table_data))
else:
    st.info("No entries match the current filters.")

# Embedding visualization (t-SNE)
st.subheader("Embedding Space Visualization (t-SNE)")
if filtered:
    embeddings = [all_docs["embeddings"][all_docs["documents"].index(doc)] for doc, _ in filtered]
    docs_short = [doc[:40] + ("..." if len(doc) > 40 else "") for doc, _ in filtered]
    if len(embeddings) > 1:
        tsne = TSNE(n_components=2, random_state=42)
        emb_2d = tsne.fit_transform(np.array(embeddings))
        df_plot = pd.DataFrame({"x": emb_2d[:,0], "y": emb_2d[:,1], "label": docs_short})
        st.scatter_chart(df_plot, x="x", y="y", color=None)
        for i, row in df_plot.iterrows():
            st.text(f"{row['label']} @ ({row['x']:.2f}, {row['y']:.2f})")
    else:
        st.info("Need at least 2 embeddings for t-SNE visualization.")
else:
    st.info("Not enough data for embedding visualization.")