import os
import json

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import chromadb
from sentence_transformers import SentenceTransformer
from rerank import rerank_search
K = 5  # we check whether the right technique is in the top K retrieved

# Load the golden test set and connect to the vector DB.
with open("golden_set.json", "r") as f:
    golden = json.load(f)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("attack_techniques")
model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_ids(question, k=K):
    """Return the top-k retrieved technique IDs using bi-encoder + cross-encoder rerank."""
    return rerank_search(question, final_k=k)


def hits(expected, retrieved):
    """Did any expected ID appear in the retrieved list? (handles sub-techniques)"""
    for exp in expected:
        for got in retrieved:
            # exact match, or the retrieved ID is a sub-technique of the expected parent
            if got == exp or got.startswith(exp + ".") or exp.startswith(got + "."):
                return True
    return False


real = [g for g in golden if g["difficulty"] != "expected-fail"]
fails = [g for g in golden if g["difficulty"] == "expected-fail"]

print(f"Evaluating recall@{K} on {len(real)} standard questions:\n")

passed = 0
for g in real:
    retrieved = retrieve_ids(g["question"])
    ok = hits(g["expected"], retrieved)
    passed += ok
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] expected {g['expected']}  got {retrieved}")

recall = passed / len(real)
print(f"\n>>> Recall@{K}: {passed}/{len(real)} = {recall:.1%}\n")

print(f"--- Known-hard cases (tracked separately, expected to struggle) ---\n")
for g in fails:
    retrieved = retrieve_ids(g["question"])
    ok = hits(g["expected"], retrieved)
    mark = "hit" if ok else "miss"
    print(f"[{mark}] {g['question'][:60]}...")
    print(f"       wanted any of {g['expected']}, got {retrieved}")