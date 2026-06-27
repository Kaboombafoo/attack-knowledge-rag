import os
import json
import re
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from collections import Counter


# --- Load everything once ---
with open("techniques.json", "r") as f:
    techniques = json.load(f)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("attack_techniques")
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Build the BM25 keyword index over the same technique text ---
# BM25 works on tokenized text (just lowercased word lists here).

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "by", "as", "is", "are", "be", "may", "can", "this", "that", "it", "its",
    "from", "at", "their", "they", "which", "such", "via", "into",
}

def tokenize(text):
    # lowercase, split on non-letters/numbers, drop stopwords and 1-char tokens
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]

tokenized_corpus = [tokenize(t["text"]) for t in techniques]
bm25 = BM25Okapi(tokenized_corpus)
# Count how many documents each token appears in (document frequency).
doc_freq = Counter()
for toks in tokenized_corpus:
    for w in set(toks):
        doc_freq[w] += 1

NUM_DOCS = len(tokenized_corpus)

def has_rare_term(question):
    """True if the query contains a low-frequency (discriminating) token."""
    for w in tokenize(question):
        # appears in under ~3% of documents → rare, strong keyword anchor
        if doc_freq.get(w, 0) < NUM_DOCS * 0.03:
            return True
    return False

def dense_search(question, k=10):
    """Vector (meaning) search → returns a ranked list of technique indices."""
    qvec = model.encode(question).tolist()
    results = collection.query(query_embeddings=[qvec], n_results=k, include=["metadatas"])
    # Map returned IDs back to their index in `techniques`.
    id_to_idx = {t["id"]: i for i, t in enumerate(techniques)}
    return [id_to_idx[m["id"]] for m in results["metadatas"][0] if m["id"] in id_to_idx]


def sparse_search(question, k=10):
    """BM25 (keyword) search → returns a ranked list of technique indices."""
    scores = bm25.get_scores(tokenize(question))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return ranked[:k]

def reciprocal_rank_fusion(ranked_lists, weights, k=60):
    """Fuse ranked lists by rank position, each list scaled by its weight."""
    fused = {}
    for ranked, weight in zip(ranked_lists, weights):
        for rank, idx in enumerate(ranked):
            fused[idx] = fused.get(idx, 0) + weight * (1 / (k + rank))
    return sorted(fused, key=lambda i: fused[i], reverse=True)

def hybrid_search(question, k=5):
    """Dense by default; bring in BM25 only when the query has a rare exact term."""
    dense = dense_search(question)

    if not has_rare_term(question):
        # Purely conceptual query — dense alone is the champion here.
        return [techniques[i]["id"] for i in dense[:k]]

    # Query has a discriminating term — let BM25 help.
    sparse = sparse_search(question)
    fused = reciprocal_rank_fusion([dense, sparse], weights=[1.0, 0.5])
    return [techniques[i]["id"] for i in fused[:k]]



if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "how do threat actors perform a golden ticket attack?"
    print(f'Question: "{q}"\n')
    print("Hybrid top 5:", hybrid_search(q))