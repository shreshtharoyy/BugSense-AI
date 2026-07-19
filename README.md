# BugSense AI
AI-powered Visual Bug Intelligence Platform for automated bug triage using Multimodal AI, RAG, Vector Databases, and Large Language Models.

## Overview

BugSense AI is a multimodal bug triage platform designed to automate software bug analysis by combining screenshots, error logs, and textual descriptions.

Instead of manually checking whether a bug already exists and writing tickets, the system retrieves similar historical bugs, reasons over the retrieved context, predicts severity, identifies the affected component, suggests possible root causes, and generates structured bug reports.

Project is currently under active development.

---

## Current Features

* Semantic text embeddings using **BAAI/bge-small-en-v1.5**
* Token-aware document chunking (max 4 chunks/bug, parent aggregation)
* Bug report preprocessing pipeline
* ChromaDB vector storage (cosine space)
* Semantic similarity retrieval with a duplicate-detection benchmark
* Hybrid dense + BM25 retrieval (RRF)
* pytest suite covering the storage, ingestion, embedding, and retrieval layers

---

## Quick Start

```bash
pip install -e ".[dev]"

python scripts/build_vector_index.py --reset   # build the index
python scripts/evaluate_retrieval.py           # measure Recall@k / MRR
python scripts/try_retrieval.py --sample 3     # quick manual sanity check
pytest -q                                      # run the suite
```

`--reset` is required the first time, and any time the collection's distance space
changes: Chroma cannot convert an existing collection, and `ChromaStore` refuses to
query one built with the wrong space rather than return silently wrong similarities.

### Run the product shell

Backend:

```bash
uvicorn backend.main:app --reload
```

Frontend:

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
```

The frontend expects the FastAPI server at `http://127.0.0.1:8000` by default.

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
                                      chunk_retrieval_text()
                           (chunk 0 keeps Error Log, later chunks keep tail)
                                                 │
                                                 ▼
                                    TextEmbedder.encode_documents()
                                          384-D, normalized
                                                 │
                                                 ▼
                                    ChromaDB (hnsw:space=cosine)
                                   ids = {bug_id}::{chunk_index}
                                                 │
   query text ──► encode_query() ────────────────┤
   (BGE prefix)                                  ▼
                                             Retriever
                              max-aggregate chunks → parent bug_id
                                    similarity = 1 - distance
```

Long descriptions are split into overlapping token windows so the model no longer
silently drops the tail past 512 tokens. Chunk 0 keeps the Error Log with the title,
later chunks keep Title + Description only, and the last chunk always preserves the
description tail. The retriever collapses chunk hits to unique parent bugs (max
similarity / max RRF) before returning results.

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
│   ├── chunking.py           # token windows over Description
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
│   ├── evaluate_retrieval.py
│   └── try_retrieval.py
│
├── backend/                  # FastAPI app for retrieval endpoints
├── frontend/                 # Next.js + Tailwind product shell
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
* FastAPI
* Next.js
* Tailwind CSS

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