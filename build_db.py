import json
import numpy as np
import chromadb

# Load the techniques and the embeddings you already computed in v1.
with open("techniques.json", "r") as f:
    techniques = json.load(f)
vectors = np.load("embeddings.npz")["vectors"]

# Create a persistent vector database on disk (survives restarts).
client = chromadb.PersistentClient(path="./chroma_db")

# Rebuild cleanly every run: drop the old collection if it exists.
try:
    client.delete_collection("attack_techniques")
except Exception:
    pass

# A collection is like a table. We configure it to measure cosine distance.
collection = client.get_or_create_collection(
    name="attack_techniques",
    metadata={"hnsw:space": "cosine"},
)

# Hand Chroma your existing vectors, the text, and metadata for each technique.
collection.add(
    ids=[str(i) for i in range(len(techniques))],
    embeddings=vectors.tolist(),
    documents=[t["text"] for t in techniques],
    metadatas=[{"id": t["id"] or "N/A", "name": t["name"]} for t in techniques],
)

print(f"Indexed {collection.count()} techniques into Chroma.")