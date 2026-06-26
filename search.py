import os
import sys
import json
import numpy as np

# Silence the harmless Hugging Face "unauthenticated" warning.
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# --- Load the knowledge base and its cached embeddings ---
with open("techniques.json", "r") as f:
    techniques = json.load(f)
vectors = np.load("embeddings.npz")["vectors"]

# Same model used to create the embeddings (must match).
model = SentenceTransformer("all-MiniLM-L6-v2")

SCORE_THRESHOLD = 0.35   # below this, retrieval is too weak to trust


def cosine_similarity(query_vec, all_vecs):
    """How aligned the query is with each stored vector (1.0 = identical)."""
    query_norm = query_vec / np.linalg.norm(query_vec)
    all_norms = all_vecs / np.linalg.norm(all_vecs, axis=1, keepdims=True)
    return all_norms @ query_norm


def answer(question):
    # 1. RETRIEVE: embed the question, score every technique, take the top 5.
    query_vec = model.encode(question)
    scores = cosine_similarity(query_vec, vectors)
    top_k = np.argsort(scores)[::-1][:5]
    top_score = scores[top_k[0]]

    print(f'Question: "{question}"\n')

    # 2. GUARDRAIL: if even the best match is weak, refuse honestly.
    if top_score < SCORE_THRESHOLD:
        print(f"(Top match scored only {top_score:.3f} — retrieval is weak here.)")
        print("I don't have a strong match for that in the ATT&CK technique data.")
        return

    # 3. GENERATE: hand the retrieved techniques to Claude as grounding context.
    context = "\n\n---\n\n".join(techniques[i]["text"] for i in top_k)
    client = Anthropic()
    system_prompt = (
        "You are a security assistant. Answer the user's question using ONLY the "
        "MITRE ATT&CK techniques provided as context. Cite the technique IDs (e.g. "
        "T1053) you used. If the context does not contain the answer, say so plainly "
        "instead of guessing."
    )
    user_message = f"Context (retrieved ATT&CK techniques):\n\n{context}\n\nQuestion: {question}"

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    print("Retrieved techniques:", ", ".join(techniques[i]["id"] for i in top_k))
    print(f"(top match score: {top_score:.3f})\n")
    print(response.content[0].text)


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "How do attackers maintain persistence?"
    answer(question)