# ATT&CK Knowledge RAG

A retrieval-augmented generation (RAG) system that answers security questions
from the live MITRE ATT&CK knowledge base. Ask a plain-English question and it
retrieves the most relevant ATT&CK techniques by meaning, then generates an
answer grounded in those techniques — with citations, and an honest refusal when
it has no good match.

I built this to understand how semantic search and RAG actually work — the pipeline is assembled from primitives (a local embedding model, hand-computed cosine similarity, a direct LLM call) rather than a RAG framework or vector database. 

## How it works

The pipeline has four stages:

1. **Ingest** — parses the official ATT&CK STIX dataset (697 enterprise
   techniques), skipping revoked/deprecated entries, into clean text documents.
2. **Embed** — runs each technique through a local embedding model
   (`all-MiniLM-L6-v2`), turning its meaning into a 384-dimension vector. Vectors
   are cached so embedding only happens once.
3. **Retrieve** — embeds the user's question with the same model and queries a
   Chroma vector database (cosine) for the nearest techniques. (v1 computed this
   by hand with NumPy; v2 swaps in a real vector DB so retrieval scales beyond a
   brute-force scan.)
4. **Generate** — passes the retrieved techniques to Claude as context, with a
   strict instruction to answer *only* from them and cite technique IDs.

The embedding model is frozen and general-purpose — it positions the ATT&CK
documents using language understanding it learned during its own training; the
dataset is not used to train it.

## Honest behavior on weak matches

If the top retrieval score falls below a threshold, the system refuses to answer
rather than forcing a response from poor matches — because a RAG answer is only as
good as what retrieval found. Retrieval quality is visible in the score, so the
guardrail is driven by it.

## Limitations

- **Naive RAG.** Single similarity pass — no reranking, no query expansion.
- **Technique-level coverage only.** The knowledge base contains individual
  techniques, not ATT&CK's higher-level *tactics*. So broad, category-level
  questions (e.g. "lateral movement" as a whole) often score low and get refused
  — not because the topic is invalid, but because no single technique document is
  a close match. This is a coverage gap, not a topic gap.
- **No source-text cleanup.** ATT&CK descriptions still contain inline
  `(Citation: ...)` markers.

## Roadmap

- Add tactic-level and group/software documents to close the coverage gap.
- Add reranking to improve retrieval precision.
- Clean inline `(Citation: ...)` markers from source text.

## Running it

Requires Python 3.11+ and an Anthropic API key.

\`\`\`bash
python -m venv .venv
source .venv/bin/activate
pip install sentence-transformers numpy anthropic python-dotenv requests
\`\`\`

Create a `.env` file (never committed):

\`\`\`
ANTHROPIC_API_KEY=your-anthropic-key
\`\`\`

Build the knowledge base, then query it:

\`\`\`bash
curl -L -o enterprise-attack.json https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json
python ingest.py     # parse techniques
python embed.py      # build embeddings (one time)
python search.py "how do attackers steal credentials from memory"
\`\`\`

## Built with

Python · sentence-transformers (`all-MiniLM-L6-v2`) · Chroma (vector DB) · Anthropic API (Claude Haiku) · MITRE ATT&CK (STIX)

ATT&CK® is a registered trademark of The MITRE Corporation. Data used under MITRE's terms.