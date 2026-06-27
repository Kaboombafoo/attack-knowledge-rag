# ATT&CK Knowledge RAG

A retrieval-augmented generation (RAG) system that answers security questions
from the real MITRE ATT&CK knowledge base — and an evaluation harness that
measures how well it actually retrieves.

Ask a plain-English question and it retrieves the most relevant ATT&CK techniques
by meaning, reranks them for precision, and generates a grounded, cited answer —
refusing honestly when it has no strong match.

I built this to understand how semantic search, RAG, and retrieval evaluation
actually work. The pipeline is assembled from primitives (a local embedding
model, a vector database, a cross-encoder reranker, a direct LLM call) rather than
a RAG framework.

## How it works

1. **Ingest** — parses the official ATT&CK STIX dataset (697 enterprise
   techniques), skipping revoked/deprecated entries, into clean text documents.
2. **Embed** — runs each technique through a local embedding model
   (`all-MiniLM-L6-v2`) into a 384-dimension vector. Vectors are cached.
3. **Retrieve** — embeds the question and queries a Chroma vector database
   (cosine) for a broad candidate set.
4. **Rerank** — a cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores the
   candidates by reading the question and each technique *together*, then keeps
   the best five. This fixes cases where similar techniques (e.g. Golden Ticket
   vs. Silver Ticket) blur under embedding-only search.
5. **Generate** — passes the reranked techniques to Claude with a strict
   instruction to answer *only* from them and cite technique IDs.

A guardrail refuses to answer when the best retrieval score is weak, rather than
forcing an answer from poor matches.

## Evaluation

Retrieval quality is measured, not guessed. `evaluate.py` runs a hand-built
golden set of 38 questions (each tagged with the ATT&CK technique that should be
retrieved) and reports recall@5.

The development of the retriever was driven by this harness:

| Stage | Recall@5 |
|-------|----------|
| Dense retrieval (baseline) | 76.5% |
| Hybrid search (dense + BM25) | tested, no improvement — rejected |
| Dense + cross-encoder reranker | **97.1%** |

Two findings worth noting:

- **Hybrid search did not help here.** I hypothesized that adding BM25 keyword
  search would improve exact-term retrieval, and tested it three ways (naive
  fusion, weighted fusion, adaptive query-routing). None beat dense-only on this
  corpus — ATT&CK descriptions are conceptual prose, which plays to embeddings'
  strength. I rejected the added complexity rather than ship it. The experiment
  is preserved in `hybrid.py`.
- **Auditing the benchmark mattered.** Several apparent failures were stale labels
  in my golden set caused by ATT&CK's technique renumbering — cases where the
  system retrieved correctly but the answer key was outdated. Verifying every
  label against the current dataset is part of the harness (a label-validation
  sweep), and is why the final number is trustworthy.

The one remaining failure at 97.1% is a genuine sub-technique-precision limit, not
a benchmark error.

## Limitations & roadmap

- **Conceptual prose only.** Strong on technique-level questions; weak on
  category-level ("lateral movement" as a whole) and set queries ("list all
  techniques that...") — see the expected-fail cases in the golden set.
- **Sub-technique precision** is the main remaining failure mode.
- Roadmap: add tactic/group/software documents to close the category gap;
  experiment with domain-fine-tuned embeddings.

## Running it

Requires Python 3.11+ and an Anthropic API key.

\`\`\`bash
python -m venv .venv
source .venv/bin/activate
pip install sentence-transformers chromadb rank-bm25 numpy anthropic python-dotenv requests
\`\`\`

Create a `.env` file (never committed):

\`\`\`
ANTHROPIC_API_KEY=your-anthropic-key
\`\`\`

Build the knowledge base, then query it:

\`\`\`bash
curl -L -o enterprise-attack.json https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json
python ingest.py      # parse techniques
python embed.py       # build embeddings (one time)
python build_db.py    # load into the vector database
python search.py "how do attackers steal credentials from memory"
python evaluate.py    # run the retrieval evaluation
\`\`\`

## Built with

Python · sentence-transformers · Chroma (vector DB) · cross-encoder reranker · Anthropic API (Claude Haiku) · MITRE ATT&CK (STIX)

ATT&CK® is a registered trademark of The MITRE Corporation.