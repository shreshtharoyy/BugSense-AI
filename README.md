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
* ChromaDB vector storage
* Semantic similarity retrieval
* Integration testing for the complete text retrieval pipeline

---

## Current Pipeline

```text
Raw Bug Report
      │
      ▼
 BugFactory
      │
      ▼
  BugReport
      │
      ▼
 Preprocessor
      │
      ▼
 TextCleaner
      │
      ▼
 TextEmbedder (BGE)
      │
      ▼
384-D Semantic Embedding
      │
      ▼
 ChromaDB Vector Store
      │
      ▼
 Semantic Retriever
```

---

## Project Structure

```text
BugSense/
│
├── src/
│   ├── preprocessing/
│   ├── embeddings/
│   ├── storage/
│   └── retrieval/
│
├── tests/
├── data/
├── requirements.txt
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