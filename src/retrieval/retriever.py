from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from typing import TYPE_CHECKING, Protocol

from src.chunking import parent_id_from_chunk_id
from src.config import (
    CHUNK_OVERFETCH,
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


def _parent_id(chunk_id: str, metadata) -> str:
    if metadata:
        parent = metadata.get("parent_bug_id")
        if parent is not None:
            return str(parent)
    return parent_id_from_chunk_id(chunk_id)


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
        chunk_overfetch: int = CHUNK_OVERFETCH,
    ) -> None:
        self.store = store
        self.embedder = embedder if embedder is not None else TextEmbedder()
        self.reranker = reranker
        self.rerank_pool = rerank_pool
        self.bm25 = bm25
        self.hybrid_pool = hybrid_pool
        self.bm25_weight = bm25_weight
        self.chunk_overfetch = chunk_overfetch

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
        # dense order is final so top_k is all we need. CHUNK_OVERFETCH compensates for
        # multiple chunks collapsing onto the same parent.
        parent_pool = self.rerank_pool if self.reranker else top_k
        chunk_pool = parent_pool * self.chunk_overfetch

        # Over-fetch so that excluded parents and threshold filtering cannot starve the pool.
        n_results = chunk_pool + len(exclude) * self.chunk_overfetch

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

        # Chroma returns chunks sorted best-first; first time we see a parent is max sim.
        candidates: list[RetrievedBug] = []
        seen_parents: set[str] = set()
        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            parent = _parent_id(chunk_id, metadata)
            if parent in exclude or parent in seen_parents:
                continue

            similarity = self._to_similarity(distance)
            if similarity < min_similarity:
                continue

            seen_parents.add(parent)
            candidates.append(self._row(parent, metadata, document, similarity))

            if len(candidates) == parent_pool:
                break

        if self.reranker is not None:
            return self.reranker.rerank(query_text, candidates, top_k)
        return candidates[:top_k]

    def _dense_ranking(
        self, query_text: str, pool: int, exclude: set[str]
    ) -> tuple[list[str], dict[str, float]]:
        """Return ranked *chunk* ids and their distances (for hybrid fusion)."""
        embedding = self.embedder.encode_query(query_text)
        results = self.store.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=min(
                pool + len(exclude) * self.chunk_overfetch,
                max(self.store.count(), 1),
            ),
        )
        ids = results["ids"][0]
        distances = (results["distances"] or [[]])[0]
        metadatas = (results["metadatas"] or [[]])[0]

        ranked: list[str] = []
        distance_by_id: dict[str, float] = {}
        for chunk_id, distance, metadata in zip(ids, distances, metadatas):
            if _parent_id(chunk_id, metadata) in exclude:
                continue
            distance_by_id[chunk_id] = distance
            ranked.append(chunk_id)
            if len(ranked) == pool:
                break
        return ranked, distance_by_id

    def _hybrid(
        self, query_text: str, top_k: int, exclude_ids: Iterable[str]
    ) -> list[RetrievedBug]:
        assert self.bm25 is not None
        exclude = set(exclude_ids)
        # Fuse over a chunk pool large enough that parent collapse still fills top_k.
        pool = self.hybrid_pool * self.chunk_overfetch

        dense_ids, distance_by_id = self._dense_ranking(query_text, pool, exclude)
        bm25_ids = [
            chunk_id
            for chunk_id in self.bm25.query(
                query_text, pool + len(exclude) * self.chunk_overfetch
            )
            if parent_id_from_chunk_id(chunk_id) not in exclude
        ][:pool]

        # Reciprocal Rank Fusion over chunk ids, then collapse to parents by max RRF.
        rrf: dict[str, float] = {}
        for rank, chunk_id in enumerate(dense_ids, start=1):
            rrf[chunk_id] = rrf.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        for rank, chunk_id in enumerate(bm25_ids, start=1):
            rrf[chunk_id] = rrf.get(chunk_id, 0.0) + self.bm25_weight / (RRF_K + rank)

        ranked_chunks = sorted(rrf, key=lambda chunk_id: rrf[chunk_id], reverse=True)
        if not ranked_chunks:
            return []

        # Resolve parent metadata for the ranked chunk list; stop once top_k parents filled.
        # Chunks are RRF-sorted best-first, so the first time we see a parent is its max.
        got = self.store.collection.get(
            ids=ranked_chunks, include=["documents", "metadatas"]
        )
        meta_by_id = dict(zip(got["ids"], got["metadatas"] or []))
        doc_by_id = dict(zip(got["ids"], got["documents"] or []))

        rows: list[RetrievedBug] = []
        seen_parents: set[str] = set()

        for chunk_id in ranked_chunks:
            parent = _parent_id(chunk_id, meta_by_id.get(chunk_id))
            if parent in exclude or parent in seen_parents:
                continue
            seen_parents.add(parent)

            distance = distance_by_id.get(chunk_id)
            similarity = self._to_similarity(distance) if distance is not None else 0.0
            rows.append(
                self._row(
                    parent,
                    meta_by_id.get(chunk_id),
                    doc_by_id.get(chunk_id),
                    similarity,
                    rrf_score=rrf[chunk_id],
                )
            )
            if len(rows) == top_k:
                break
        return rows
