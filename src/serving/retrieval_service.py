from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.config import RRF_BM25_WEIGHT, TOP_K
from src.retrieval.retriever import RetrievedBug, Retriever
from src.storage.chroma_store import ChromaStore

DUPLICATES_CSV = Path("dataset/gitbugs/firefox/firefox_bugs-combined.csv")
_GET_BATCH = 500
_SNIPPET_CHARS = 280


def _iter_collection(store: ChromaStore, include: list[str]):
    all_ids = store.collection.get(include=[])["ids"]
    for start in range(0, len(all_ids), _GET_BATCH):
        batch_ids = all_ids[start : start + _GET_BATCH]
        page = store.collection.get(ids=batch_ids, include=include)
        yield page["ids"], page.get("documents"), page.get("metadatas")


def _load_duplicate_targets(corpus_ids: set[str]) -> dict[str, str]:
    if not DUPLICATES_CSV.exists():
        return {}

    frame = pd.read_csv(DUPLICATES_CSV)
    frame["Duplicates"] = pd.to_numeric(frame["Duplicates"], errors="coerce")
    frame = frame.dropna(subset=["Duplicates"])

    targets: dict[str, str] = {}
    for issue_id, duplicate_of in zip(frame["Issue id"], frame["Duplicates"]):
        query_id = str(int(issue_id))
        target_id = str(int(duplicate_of))
        if query_id in corpus_ids and target_id in corpus_ids:
            targets[query_id] = target_id
    return targets


def _fetch_parent_document(store: ChromaStore, parent_id: str) -> tuple[str, str]:
    page = store.collection.get(
        where={"parent_bug_id": parent_id},
        include=["documents", "metadatas"],
    )
    if not page["ids"]:
        page = store.collection.get(
            ids=[f"{parent_id}::0"],
            include=["documents", "metadatas"],
        )
    if not page["ids"]:
        return "", ""

    pieces: list[tuple[int, str, str]] = []
    for chunk_id, document, metadata in zip(
        page["ids"],
        page.get("documents") or [],
        page.get("metadatas") or [],
    ):
        metadata = metadata or {}
        index = int(metadata.get("chunk_index", 0))
        title = str(metadata.get("title") or "")
        pieces.append((index, title, document or ""))

    pieces.sort(key=lambda item: item[0])
    title = pieces[0][1] if pieces else ""
    if len(pieces) == 1:
        return title, pieces[0][2]
    return title, "\n".join(text for _, _, text in pieces)


def _build_retriever(store: ChromaStore, hybrid: bool) -> Retriever:
    if not hybrid:
        return Retriever(store)

    from src.retrieval.bm25_index import BM25Index

    corpus_ids: list[str] = []
    corpus_docs: list[str] = []
    for ids, documents, _ in _iter_collection(store, include=["documents"]):
        corpus_ids.extend(ids)
        corpus_docs.extend(documents or [""] * len(ids))

    bm25 = BM25Index.from_documents(corpus_ids, corpus_docs)
    return Retriever(store, bm25=bm25, bm25_weight=RRF_BM25_WEIGHT)


def _snippet(text: str, size: int = _SNIPPET_CHARS) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= size:
        return compact
    return compact[: size - 3].rstrip() + "..."


def _serialize_result(result: RetrievedBug) -> dict:
    score = result.rrf_score if result.rrf_score is not None else result.similarity
    return {
        "bug_id": result.bug_id,
        "title": result.title,
        "project": result.project,
        "resolution": result.resolution,
        "created_at": result.created_at,
        "resolved_at": result.resolved_at,
        "document": result.document,
        "snippet": _snippet(result.document),
        "similarity": result.similarity,
        "rrf_score": result.rrf_score,
        "score": score,
    }


class RetrievalService:
    def __init__(self, hybrid: bool = True) -> None:
        self.hybrid = hybrid
        self.store = ChromaStore()
        if self.store.count() == 0:
            raise RuntimeError(
                "Vector index is empty. Run scripts/build_vector_index.py --reset first."
            )
        self.retriever = _build_retriever(self.store, hybrid=hybrid)
        self.corpus_ids = self.store.existing_parent_ids()
        self.duplicate_map = _load_duplicate_targets(self.corpus_ids)

    def corpus_summary(self) -> dict:
        return {
            "mode": "hybrid" if self.hybrid else "dense",
            "parent_bugs": len(self.corpus_ids),
            "chunk_vectors": self.store.count(),
        }

    def get_bug(self, bug_id: str) -> dict | None:
        if bug_id not in self.corpus_ids:
            return None
        title, query_text = _fetch_parent_document(self.store, bug_id)
        if not query_text:
            return None
        return {
            "bug_id": bug_id,
            "title": title,
            "query_text": query_text,
            "snippet": _snippet(query_text),
            "known_duplicate_bug_id": self.duplicate_map.get(bug_id),
        }

    def search_by_bug_id(self, bug_id: str, top_k: int = TOP_K) -> dict:
        bug = self.get_bug(bug_id)
        if bug is None:
            raise KeyError(f"Bug #{bug_id} is not in the index.")

        results = self.retriever.retrieve(
            bug["query_text"],
            top_k=top_k,
            min_similarity=-1.0,
            exclude_ids=[bug_id],
        )
        known_duplicate = bug["known_duplicate_bug_id"]
        duplicate_rank = next(
            (index + 1 for index, row in enumerate(results) if row.bug_id == known_duplicate),
            None,
        )
        return {
            "query": bug,
            "results": [_serialize_result(row) for row in results],
            "ground_truth_found": duplicate_rank is not None,
            "ground_truth_rank": duplicate_rank,
        }

    def search_by_text(self, query_text: str, top_k: int = TOP_K) -> dict:
        query_text = query_text.strip()
        if not query_text:
            raise ValueError("Query text cannot be empty.")

        results = self.retriever.retrieve(
            query_text,
            top_k=top_k,
            min_similarity=-1.0,
        )
        return {
            "query": {
                "title": "",
                "query_text": query_text,
                "snippet": _snippet(query_text),
            },
            "results": [_serialize_result(row) for row in results],
            "ground_truth_found": None,
            "ground_truth_rank": None,
        }


@lru_cache(maxsize=1)
def get_retrieval_service() -> RetrievalService:
    return RetrievalService(hybrid=True)
