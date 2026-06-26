import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the techniques we parsed in stage 1.
with open("techniques.json", "r") as f:
    techniques = json.load(f)

# Load a small, local embedding model. First run downloads it (~90 MB).
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Pull out just the text of each technique, in order.
texts = [t["text"] for t in techniques]

# Turn all 697 documents into vectors. This is the core step.
print(f"Embedding {len(texts)} techniques...")
vectors = model.encode(texts, show_progress_bar=True)

# Save the vectors so we never have to re-embed.
np.savez("embeddings.npz", vectors=vectors)

print("\nDone.")
print("Shape of the vector array:", vectors.shape)
print("\n--- What one embedding looks like ---")
print("First technique:", techniques[0]["id"], "-", techniques[0]["name"])
print("Its vector (first 8 of", len(vectors[0]), "numbers):")
print(vectors[0][:8])