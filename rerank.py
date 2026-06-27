import os
import json

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- Load data, vector DB, and BOTH models ---
with open("techniques.json", "r") as f:
    techniques = json.load(f)
id_to_tech = {t["id"]: t for t in techniques}

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("attack_techniques")

# Bi-encoder: fast, for the first broad retrieval pass.
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Cross-encoder: slow but accurate, for reranking the shortlist.
# First run downloads it (~80 MB).
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_search(question, first_stage_k=20, final_k=5):
    # STAGE 1: fast bi-encoder retrieves a broad candidate set.
    qvec = bi_encoder.encode(question).tolist()
    results = collection.query(
        query_embeddings=[qvec],
        n_results=first_stage_k,
        include=["documents", "metadatas"],
    )
    candidate_docs = results["documents"][0]
    candidate_ids = [m["id"] for m in results["metadatas"][0]]

    # STAGE 2: cross-encoder scores each (question, document) pair together.
    pairs = [(question, doc) for doc in candidate_docs]
    scores = cross_encoder.predict(pairs)

    # Sort candidates by the cross-encoder's relevance score, highest first.
    ranked = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)
    return [tech_id for tech_id, _ in ranked[:final_k]]


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "how do threat actors perform a golden ticket attack?"
    print(f'Question: "{q}"\n')
    print("Reranked top 5:", rerank_search(q))
    