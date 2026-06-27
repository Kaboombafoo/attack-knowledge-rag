import os
import sys
import json

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("attack_techniques")

bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

SCORE_THRESHOLD = 0.35
FIRST_STAGE_K = 20
FINAL_K = 5


def answer(question):
    qvec = bi_encoder.encode(question).tolist()
    results = collection.query(
        query_embeddings=[qvec],
        n_results=FIRST_STAGE_K,
        include=["documents", "distances", "metadatas"],
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    top_similarity = 1 - results["distances"][0][0]

    print(f'Question: "{question}"\n')

    if top_similarity < SCORE_THRESHOLD:
        print(f"(Top match similarity only {top_similarity:.3f} — retrieval is weak here.)")
        print("I don't have a strong match for that in the ATT&CK technique data.")
        return

    pairs = [(question, doc) for doc in docs]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(docs, metas, scores), key=lambda x: x[2], reverse=True)
    top = ranked[:FINAL_K]

    top_docs = [d for d, m, s in top]
    top_ids = [m["id"] for d, m, s in top]

    context = "\n\n---\n\n".join(top_docs)
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

    print("Retrieved techniques (reranked):", ", ".join(top_ids))
    print(f"(stage-1 top similarity: {top_similarity:.3f})\n")
    print(response.content[0].text)


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "How do attackers maintain persistence?"
    answer(question)