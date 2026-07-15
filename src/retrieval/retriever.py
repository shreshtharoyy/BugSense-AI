from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from typing import TYPE_CHECKING, Protocol

from src.config import (
    HYBRID_POOL,
    RERANK_POOL,
    RRF_BM25_WEIGHT,
    RRF_K,
    SIMILARITY_THRESHOLD,
    TOP_K,
)
from src.embeddings.text_embedder import TextEmbedder
from src.storage.chroma_store import ChromaStore

if TYPE_CHECKING:
    from src.retrieval.bm25_index import BM25Index


class Reranker(Protocol):
    """Structural interface for a second stage. Retriever depends on this, not on the
    concrete cross-encoder, so the reranking package stays optional and test stubs work
    without subclassing."""

    def rerank(
        self, query_text: str, candidates: list[RetrievedBug], top_k: int
    ) -> list[RetrievedBug]: ...


def _as_str(value) -> str | None:
    """Chroma widens metadata values to str | int | float | bool | list | None."""
    return value if isinstance(value, str) else None


@dataclass(frozen=True)
class RetrievedBug:
    bug_id: str
    similarity: float
    title: str
    project: str
    resolution: str | None
    created_at: str | None
    resolved_at: str | None
    document: str
    # Set by the reranker; None means the row came straight from dense retrieval.
    rerank_score: float | None = None
    # Set by hybrid retrieval; None outside the hybrid path.
    rrf_score: float | None = None


class Retriever:
    def __init__(
        self,
        store: ChromaStore,
        embedder: TextEmbedder | None = None,
        reranker: Reranker | None = None,
        rerank_pool: int = RERANK_POOL,
        bm25: BM25Index | None = None,
        hybrid_pool: int = HYBRID_POOL,
        bm25_weight: float = RRF_BM25_WEIGHT,
    ) -> None:
        self.store = store
        self.embedder = embedder if embedder is not None else TextEmbedder()
        self.reranker = reranker
        self.rerank_pool = rerank_pool
        self.bm25 = bm25
        self.hybrid_pool = hybrid_pool
        self.bm25_weight = bm25_weight

    @staticmethod
    def _to_similarity(distance: float) -> float:
        # Valid only because the collection is built in cosine space, where Chroma
        # returns `1 - cos`. Under the default l2 space this returns squared euclidean
        # distance and the result is not a similarity at all -- it goes negative below
        # cos 0.5. ChromaStore.CONFIGURATION pins the space; do not unpin it.
        return 1.0 - distance

    def _row(
        self,
        bug_id: str,
        metadata,
        document,
        similarity: float,
        rrf_score: float | None = None,
    ) -> RetrievedBug:
        metadata = metadata or {}
        return RetrievedBug(
            bug_id=bug_id,
            similarity=similarity,
            title=_as_str(metadata.get("title")) or "",
            project=_as_str(metadata.get("project")) or "",
            resolution=_as_str(metadata.get("resolution")),
            created_at=_as_str(metadata.get("created_at")),
            resolved_at=_as_str(metadata.get("resolved_at")),
            document=document or "",
            rrf_score=rrf_score,
        )

    def retrieve(
        self,
        query_text: str,
        top_k: int = TOP_K,
        min_similarity: float = SIMILARITY_THRESHOLD,
        exclude_ids: Iterable[str] = (),
    ) -> list[RetrievedBug]:
        if self.bm25 is not None:
            # RRF scores are not cosine similarities, so min_similarity does not apply.
            return self._hybrid(query_text, top_k, exclude_ids)

        exclude = set(exclude_ids)

        # With a reranker, fetch a larger dense pool for it to reorder; without one, the
        # dense order is final so top_k is all we need.
        fetch_k = self.rerank_pool if self.reranker else top_k

        # Over-fetch so that excluded ids and threshold filtering cannot starve fetch_k.
        n_results = fetch_k + len(exclude)

        embedding = self.embedder.encode_query(query_text)
        results = self.store.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=min(n_results, max(self.store.count(), 1)),
        )

        # Chroma types documents/metadatas/distances as optional on QueryResult.
        ids = results["ids"][0]
        documents = (results["documents"] or [[]])[0]
        metadatas = (results["metadatas"] or [[]])[0]
        distances = (results["distances"] or [[]])[0]

        candidates: list[RetrievedBug] = []
        for bug_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            if bug_id in exclude:
                continue

            similarity = self._to_similarity(distance)
            if similarity < min_similarity:
                continue

            candidates.append(self._row(bug_id, metadata, document, similarity))

            if len(candidates) == fetch_k:
                break

        if self.reranker is not None:
            return self.reranker.rerank(query_text, candidates, top_k)
        return candidates[:top_k]

    def _dense_ranking(
        self, query_text: str, pool: int, exclude: set[str]
    ) -> tuple[list[str], dict[str, float]]:
        embedding = self.embedder.encode_query(query_text)
        results = self.store.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=min(pool + len(exclude), max(self.store.count(), 1)),
        )
        ids = results["ids"][0]
        distances = (results["distances"] or [[]])[0]

        ranked: list[str] = []
        distance_by_id: dict[str, float] = {}
        for bug_id, distance in zip(ids, distances):
            if bug_id in exclude:
                continue
            distance_by_id[bug_id] = distance
            ranked.append(bug_id)
            if len(ranked) == pool:
                break
        return ranked, distance_by_id

    def _hybrid(
        self, query_text: str, top_k: int, exclude_ids: Iterable[str]
    ) -> list[RetrievedBug]:
        assert self.bm25 is not None
        exclude = set(exclude_ids)
        pool = self.hybrid_pool

        dense_ids, distance_by_id = self._dense_ranking(query_text, pool, exclude)
        bm25_ids = [
            bug_id
            for bug_id in self.bm25.query(query_text, pool + len(exclude))
            if bug_id not in exclude
        ][:pool]

        # Reciprocal Rank Fusion. rank is 1-based; a doc found by only one retriever still
        # contributes its share, so a strong BM25 hit can pull up a weak dense hit.
        rrf: dict[str, float] = {}
        for rank, bug_id in enumerate(dense_ids, start=1):
            rrf[bug_id] = rrf.get(bug_id, 0.0) + 1.0 / (RRF_K + rank)
        for rank, bug_id in enumerate(bm25_ids, start=1):
            rrf[bug_id] = rrf.get(bug_id, 0.0) + self.bm25_weight / (RRF_K + rank)

        ranked = sorted(rrf, key=lambda bug_id: rrf[bug_id], reverse=True)[:top_k]
        if not ranked:
            return []

        # One fetch for the final set: bm25-only ids have no dense row to reuse.
        got = self.store.collection.get(ids=ranked, include=["documents", "metadatas"])
        meta_by_id = dict(zip(got["ids"], got["metadatas"] or []))
        doc_by_id = dict(zip(got["ids"], got["documents"] or []))

        rows: list[RetrievedBug] = []
        for bug_id in ranked:
            distance = distance_by_id.get(bug_id)
            similarity = self._to_similarity(distance) if distance is not None else 0.0
            rows.append(
                self._row(
                    bug_id,
                    meta_by_id.get(bug_id),
                    doc_by_id.get(bug_id),
                    similarity,
                    rrf_score=rrf[bug_id],
                )
            )
        return rows
