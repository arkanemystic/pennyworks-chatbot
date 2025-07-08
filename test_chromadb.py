import chromadb

# 1. Create a ChromaDB client with default (ephemeral) storage
client = chromadb.Client()

# 2. Create a collection called "test_collection"
collection = client.create_collection("test_collection")

# 3. Add two documents: "hello world" and "goodbye world", each with metadata and a unique ID
collection.add(
    documents=["hello world", "goodbye world"],
    metadatas=[{"source": "test"}, {"source": "test"}],
    ids=["id1", "id2"]
)

# 4. Query the collection with the term "hello" and return the top 2 results
results = collection.query(query_texts=["hello"], n_results=2)

# 5. Print the query results
print(results)
