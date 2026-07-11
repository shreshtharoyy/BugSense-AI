from dataclasses import dataclass
from collections.abc import Iterable

from src.config import SIMILARITY_THRESHOLD, TOP_K
from src.embeddings.text_embedder import TextEmbedder
from src.storage.chroma_store import ChromaStore


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


class Retriever:
    def __init__(self, store: ChromaStore, embedder: TextEmbedder | None = None) -> None:
        self.store = store
        self.embedder = embedder if embedder is not None else TextEmbedder()

    @staticmethod
    def _to_similarity(distance: float) -> float:
        # Valid only because the collection is built in cosine space, where Chroma
        # returns `1 - cos`. Under the default l2 space this returns squared euclidean
        # distance and the result is not a similarity at all -- it goes negative below
        # cos 0.5. ChromaStore.CONFIGURATION pins the space; do not unpin it.
        return 1.0 - distance

    def retrieve(
        self,
        query_text: str,
        top_k: int = TOP_K,
        min_similarity: float = SIMILARITY_THRESHOLD,
        exclude_ids: Iterable[str] = (),
    ) -> list[RetrievedBug]:
        exclude = set(exclude_ids)

        # Over-fetch so that excluded ids and threshold filtering cannot starve top_k.
        n_results = top_k + len(exclude)

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

        retrieved: list[RetrievedBug] = []
        for bug_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            if bug_id in exclude:
                continue

            similarity = self._to_similarity(distance)
            if similarity < min_similarity:
                continue

            metadata = metadata or {}
            retrieved.append(
                RetrievedBug(
                    bug_id=bug_id,
                    similarity=similarity,
                    title=_as_str(metadata.get("title")) or "",
                    project=_as_str(metadata.get("project")) or "",
                    resolution=_as_str(metadata.get("resolution")),
                    created_at=_as_str(metadata.get("created_at")),
                    resolved_at=_as_str(metadata.get("resolved_at")),
                    document=document or "",
                )
            )

            if len(retrieved) == top_k:
                break

        return retrieved
