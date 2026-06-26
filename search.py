import os
import sys
import json

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import chromadb
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# Connect to the vector database we built (no re-embedding the corpus).
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("attack_techniques")

# Same model as before — we still need it to embed the QUESTION.
model = SentenceTransformer("all-MiniLM-L6-v2")

SCORE_THRESHOLD = 0.35   # on similarity (1 - distance); below this, refuse


def answer(question):
    # 1. RETRIEVE: embed the question, ask Chroma for the 5 nearest techniques.
    query_vec = model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=5,
        include=["documents", "distances", "metadatas"],
    )

    docs = results["documents"][0]
    distances = results["distances"][0]
    metas = results["metadatas"][0]

    # Chroma returns DISTANCE; convert to the similarity you know.
    top_similarity = 1 - distances[0]

    print(f'Question: "{question}"\n')

    # 2. GUARDRAIL: if even the nearest match is weak, refuse honestly.
    if top_similarity < SCORE_THRESHOLD:
        print(f"(Top match similarity only {top_similarity:.3f} — retrieval is weak here.)")
        print("I don't have a strong match for that in the ATT&CK technique data.")
        return

    # 3. GENERATE: hand the retrieved techniques to Claude as grounding context.
    context = "\n\n---\n\n".join(docs)
    anthropic_client = Anthropic()
    system_prompt = (
        "You are a security assistant. Answer the user's question using ONLY the "
        "MITRE ATT&CK techniques provided as context. Cite the technique IDs (e.g. "
        "T1053) you used. If the context does not contain the answer, say so plainly "
        "instead of guessing."
    )
    user_message = f"Context (retrieved ATT&CK techniques):\n\n{context}\n\nQuestion: {question}"

    response = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    retrieved_ids = ", ".join(m["id"] for m in metas)
    print("Retrieved techniques:", retrieved_ids)
    print(f"(top match similarity: {top_similarity:.3f})\n")
    print(response.content[0].text)


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "How do attackers maintain persistence?"
    answer(question)