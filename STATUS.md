# BugSense-AI — Project Status & Roadmap

_Living status doc. Read top-to-bottom to get from "where are we" to "what's left."_
_Last updated: 2026-07-16._

---

## What this is

Multimodal bug-triage platform. Core feature today: **duplicate bug detection** — given a bug report, retrieve the most similar historical bugs from a vector index. Dataset: 24,824 Firefox bugs (text only; no screenshots yet).

---

## TL;DR — where we are right now

- Pipeline **works end-to-end** and is **measured** (not guessed).
- **Champion: chunked hybrid dense+BM25 @ weight 0.3 on `bge-small`** — Recall@5 **0.4023**, MRR **0.3033**.
- Unchunked hybrid (0.398 / 0.299) is the previous champion; kept on the board for comparison.
- `bge-base` rejected. Chunking v2 adopted after measurement.
- Git: you own all commit/push actions.

---

## Retrieval scoreboard — the source of truth

Measured by `scripts/evaluate_retrieval.py` over **3,460 labeled duplicate pairs** (corpus 24,824 parents / 37,914 chunks, cosine Chroma). Every future change is judged against this.

### bge-small-en-v1.5 (384-d), one vector per bug — previous champion

| Approach | Recall@1 | Recall@5 | Recall@10 | MRR | Verdict |
|---|---|---|---|---|---|
| Baseline (pure dense) | 0.2217 | 0.3980 | 0.4630 | 0.2950 | reference |
| Cross-encoder reranker | 0.1387 | 0.2763 | 0.3610 | 0.1987 | ❌ **−12pp, dropped** |
| Hybrid, equal weight (1.0) | 0.2223 | 0.3919 | 0.4659 | 0.2930 | tie |
| Hybrid @ 0.3 | 0.2280 | 0.3980 | 0.4679 | 0.2990 | previous champ |

### bge-base-en-v1.5 (768-d) — rejected 2026-07-15

| Approach | Recall@1 | Recall@5 | Recall@10 | MRR | Verdict |
|---|---|---|---|---|---|
| Baseline (pure dense) | 0.2139 | 0.3850 | 0.4538 | 0.2875 | ❌ **−1.3pp R@5 vs small** |
| Hybrid @ 0.3 | 0.2217 | 0.3864 | 0.4566 | 0.2916 | ❌ still under small |

### bge-small + chunking v2 — adopted 2026-07-16

Index: **24,824 parents → 37,914 chunks** (~1.5 chunks/bug).

| Approach | Recall@1 | Recall@5 | Recall@10 | MRR | Verdict |
|---|---|---|---|---|---|
| Baseline (chunked dense) | 0.2228 | 0.4009 | 0.4656 | 0.2966 | +0.3pp R@5 vs unchunked baseline |
| **Hybrid @ 0.3 (chunked)** | **0.2301** | **0.4023** | **0.4777** | **0.3033** | ✅ **new champion** (+0.4pp R@5 / +0.4pp MRR vs unchunked hybrid) |

**Chunking v2:** `CHUNK_TOKENS=480`, `MAX_CHUNKS_PER_BUG=4`, Error Log **only on chunk 0**, later chunks Title+Description, last chunk = description tail. `MAX_DESCRIPTION_LENGTH=4000`, `BATCH_SIZE=32`. Eval loads Chroma in batches of 500 (full `get()` hits SQLite “too many SQL variables”).

---

## The pipeline

```
Raw bug → BugFactory → Preprocessor (clean + extract error log)
        → build_retrieval_text ("Title: … Error Log: … Description: …")
        → chunk_retrieval_text (token windows)
        → bge-small embedding (384-d, cosine)  ─┐
                                               ├─► RRF fuse (chunk ids) ─► max per parent ─► top-k
          BM25 keyword index (bm25s) ──────────┘
        → ChromaDB (cosine space)  ─►  Retriever  ─►  Evaluate (Recall@k, MRR)
```

Key modules: `src/chunking.py`, `src/retrieval/retriever.py`, `src/retrieval/bm25_index.py`, `src/storage/chroma_store.py`, `src/embeddings/text_embedder.py`, `scripts/evaluate_retrieval.py`.

---

## Commands (PowerShell, from repo root)

Long-running commands are run by **you**, in your own terminal.

```powershell
$env:HF_HUB_OFFLINE = "1"

.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe -u scripts\build_vector_index.py --reset
.\venv\Scripts\python.exe -u scripts\evaluate_retrieval.py
.\venv\Scripts\python.exe -u scripts\evaluate_retrieval.py --hybrid
```

Results: `eval_results/*.json` (gitignored).

---

## Done ✅

- [x] Correctness fixes, eval harness, GPU (`torch 2.13.0+cu126`), cosine index.
- [x] Baseline + hybrid@0.3 recorded on bge-small.
- [x] Reranker evaluated and dropped.
- [x] Hybrid BM25+dense adopted (weight 0.3).
- [x] **bge-base measured and rejected** (config rolled back to small).
- [x] **Chunking v2 measured and adopted** (new champion).
- [x] Eval Chroma pagination fix (SQL variable limit).

---

## Next — pick a lever

1. ~~Chunking~~ — done; champion updated.

2. **FastAPI + microservice** ⭐ — product shape now that retrieval quality is settled.

3. **CLIP / multimodal** — 🚫 blocked (no screenshot data).

4. ~~Stronger embedding model~~ — base tried, lost; skip large for now.

Also later (README): severity prediction, root-cause, LLM reasoning, ticket generation, Next.js, Docker.

---

## Immediate pending action

None for retrieval. Next product step is FastAPI (or another retrieval lever if you want more quality headroom).

---

## Key decisions & gotchas

- **Cosine space is mandatory.** Switching space or embedding layout requires `--reset`.
- **Chunk ids ≠ bug ids.** Resume uses `existing_parent_ids()`; eval corpus/hits use parent ids; Chroma vectors are `{bug_id}::{i}`.
- **Eval harness is the referee.** Beat Recall@5 **0.4023** / MRR **0.3033** or drop the change.
- **Long tasks run by you** in your terminal.
- **Git:** you commit/push; agent does not.
- `eval_results/` and `data/` are gitignored.
