# BugSense AI
AI-powered Visual Bug Intelligence Platform for automated bug triage using Multimodal AI, RAG, Vector Databases, and Large Language Models.

## Overview

BugSense AI is a multimodal bug triage platform designed to automate software bug analysis by combining screenshots, error logs, and textual descriptions.

Instead of manually checking whether a bug already exists and writing tickets, the system retrieves similar historical bugs, reasons over the retrieved context, predicts severity, identifies the affected component, suggests possible root causes, and generates structured bug reports.

Project is currently under active development.

---

## Current Features

* Semantic text embeddings using **BAAI/bge-small-en-v1.5**
* Bug report preprocessing pipeline
* ChromaDB vector storage (cosine space)
* Semantic similarity retrieval with a duplicate-detection benchmark
* pytest suite covering the storage, ingestion, embedding, and retrieval layers

---

## Quick Start

```bash
pip install -e ".[dev]"

python scripts/build_vector_index.py --reset   # build the index
python scripts/evaluate_retrieval.py           # measure Recall@k / MRR
pytest -q                                      # run the suite
```

`--reset` is required the first time, and any time the collection's distance space
changes: Chroma cannot convert an existing collection, and `ChromaStore` refuses to
query one built with the wrong space rather than return silently wrong similarities.

---

## Evaluation

`dataset/gitbugs/firefox/firefox_bugs-combined.csv` maps each bug to the bug it
duplicates. Filtering to pairs where both bugs are indexed yields ~3,460 ground-truth
queries. `scripts/evaluate_retrieval.py` reports Recall@1/@5/@10 and MRR against them,
and writes a timestamped JSON to `eval_results/`.

Treat that number as the contract. Any retrieval change — reranking, hybrid search,
chunking, a larger embedding model, CLIP — is worth keeping only if it moves Recall@5
or MRR.

---

## Current Pipeline

```text
CSV row ──► DatasetMapper ──► BugReport ──► Preprocessor
                 │                               │
        LogExtractor pulls the             TextCleaner
        error signature out of                   │
        the description                          ▼
                                        build_retrieval_text()
                                   "Title: … Error Log: … Description: …"
                                                 │
                                                 ▼
                                    TextEmbedder.encode_documents()
                                          384-D, normalized
                                                 │
                                                 ▼
                                    ChromaDB (hnsw:space=cosine)
                                                 │
   query text ──► encode_query() ────────────────┤
   (BGE prefix)                                  ▼
                                             Retriever
                                    similarity = 1 - distance
```

Error Log precedes Description in the embedded document because the model truncates
past 512 tokens and roughly a third of these documents exceed that. The shortest,
highest-signal field goes first.

`build_retrieval_text()` is shared by the indexer and every query path. When the
template was inlined in two places it drifted, and queries landed in a different
region of the space than the documents they were meant to match.

---

## Project Structure

```text
BugSense/
│
├── src/
│   ├── document_builder.py   # shared index-time / query-time template
│   ├── config.py
│   ├── ingestion/            # CSV streaming + row mapping
│   ├── preprocessing/        # cleaning + error-log extraction
│   ├── embeddings/
│   ├── indexing/
│   ├── storage/
│   └── retrieval/
│
├── scripts/
│   ├── build_vector_index.py
│   └── evaluate_retrieval.py
│
├── tests/
├── data/                     # vector store, gitignored, rebuildable
├── pyproject.toml
└── README.md
```

---

## Tech Stack

* Python
* PyTorch
* Sentence Transformers
* BAAI/bge-small-en-v1.5
* ChromaDB
* NumPy
* Pandas
* FastAPI *(Upcoming)*
* Next.js *(Upcoming)*

---

## Roadmap

* Image embeddings using CLIP
* Multimodal embedding fusion
* Duplicate bug detection
* Root cause analysis
* Qwen-powered reasoning
* AI ticket generation
* FastAPI backend
* Next.js frontend
* Docker & Cloud deployment